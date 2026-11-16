#!/usr/bin/env python3
"""Free model aggregator -- make.fun free-model-aggregator-01 entry.

One server, two client wires (OpenAI Chat Completions, Anthropic Messages),
ten simulated free-tier providers on three upstream wires (openai, anthropic,
gemini). Python 3.13 standard library only.

Design (from the kit's RULES.txt / env / scorer, read before writing):
- The brief ($MAKEFUN_PROMPT_FILE) holds three JSON documents: the provider
  directory, the sealed quality table (rating_milli + capabilities per free
  model) and the workload manifest (20 requests, in order, with requirements).
- At startup we probe each provider's catalog route in its own wire and build
  the one catalog document; `python3 main.py --list` prints it, and the
  register body carries the same document.
- Each client request names model "auto". We rank every free model that meets
  the request's requirements by rating_milli (best first, ties by the brief's
  provider order then model id), try the best, and on a 402/429/5xx retry that
  provider once (honouring retry-after) and then fail over to the next
  candidate. A provider that fails three calls in a row is treated as spent
  for the rest of the trial (a one-off fail_call never reaches three in a row).
- JSON-mode requests from an OpenAI client are never routed to an
  anthropic-wire provider: that wire cannot carry response_format, so its echo
  would report json_mode absent and the request would fail.
- Responses are translated back into the wire the client used, with the
  provider's text whole, its usage counts as that wire reports them, and the
  stop reason mapped to the client's family (stop/end_turn/STOP are one
  family). OpenAI streams get a usage chunk (we ask the upstream for
  include_usage) so the client's usage check can pass.
- Deterministic: every choice derives from the brief, the requests and the
  providers' answers. No wall clock, no unseeded randomness.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600
PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 15.0
MAX_BODY_BYTES = 32 * 1024 * 1024
DEAD_AFTER_CONSECUTIVE = 3
ANTHROPIC_VERSION = "2023-06-01"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def number_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


# --------------------------------------------------------------------------
# Brief parsing: find every JSON object in the text that carries a `schema`.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _str(v):
    return v if isinstance(v, str) else ""


def _provider_of(v):
    if not is_record(v):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = _str(v.get("path")) or _str(v.get("base_path")) or _str(v.get("base_url")) or _str(v.get("base"))
    return {
        "id": pid,
        "name": _str(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _str(v.get("key_env")),
        "key": _str(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = _str(v.get("model"))
    provider = _str(v.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
        "context_length": number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def read_brief(path):
    brief = {"directory": {"providers": []}, "ratings": None, "workload": None, "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = documents_in(text)
    directory_doc = None
    for d in docs:
        if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list):
            directory_doc = d
            break
    if directory_doc is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    providers = []
    for v in directory_doc["providers"]:
        p = _provider_of(v)
        if p is not None:
            providers.append(p)
    brief["directory"]["providers"] = providers
    ratings_doc = None
    for d in docs:
        if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list):
            ratings_doc = d
            break
    if ratings_doc is not None:
        rows = []
        for v in ratings_doc["ratings"]:
            r = _rating_of(v)
            if r is not None:
                rows.append(r)
        best = []
        if isinstance(ratings_doc.get("best"), list):
            for v in ratings_doc["best"]:
                if is_record(v) and _str(v.get("id")) != "":
                    best.append({"id": v["id"], "rating_milli": number_or_none(v.get("rating_milli")) or 0})
        brief["ratings"] = {"ratings": rows, "best": best}
    workload_doc = None
    for d in docs:
        if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list):
            workload_doc = d
            break
    if workload_doc is not None:
        manifest = []
        for v in workload_doc["manifest"]:
            if not is_record(v) or _str(v.get("id")) == "":
                continue
            req = v.get("requires") if is_record(v.get("requires")) else {}
            manifest.append({
                "id": v["id"],
                "api": "anthropic" if v.get("api") == "anthropic" else "openai",
                "stream": v.get("stream") is True,
                "weight": 2 if v.get("weight") == 2 else 1,
                "requires": {
                    "tools": req.get("tools") is True,
                    "vision": req.get("vision") is True,
                    "json": req.get("json") is True,
                    **({"min_context": req["min_context"]} if isinstance(req.get("min_context"), (int, float)) else {}),
                },
            })
        brief["workload"] = {"manifest": manifest}
    return brief


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for r in ratings["ratings"]:
        if r["provider"] == provider_id and r["model"] == model_id:
            return r
    return None


# --------------------------------------------------------------------------
# Catalog probe: each provider's catalog route, in its own wire.
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v) else None
    if not is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = _str(v.get("name")).removeprefix("models/")
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    context = number_or_none(v.get("context_length"))
    if context is None:
        context = number_or_none(v.get("context_window"))
    if context is None:
        context = number_or_none(v.get("inputTokenLimit"))
    free = bool_or_none(v.get("free"))
    if free is None:
        free = _zero_priced(v)
    return {
        "id": mid,
        "display_name": _str(v.get("display_name")) or _str(v.get("displayName")) or mid,
        "free": free,
        "context_length": context,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def probe_provider(env_url, provider, timeout_s=PROBE_TIMEOUT_S):
    if not env_url:
        return {"provider": provider, "ok": False, "status": 0, "models": [], "error": None}
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status < 200 or resp.status >= 300:
                return {"provider": provider, "ok": False, "status": resp.status, "models": [], "error": "GET %s answered HTTP %s" % (url, resp.status)}
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"provider": provider, "ok": False, "status": 0, "models": [], "error": "GET %s failed: %s" % (url, exc)}
    if not is_record(body):
        return {"provider": provider, "ok": False, "status": 200, "models": [], "error": "GET %s answered a non-object" % url}
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    models = []
    if isinstance(listing, list):
        for v in listing:
            m = _model_in(provider["wire"], v)
            if m is not None:
                models.append(m)
    return {"provider": provider, "ok": True, "status": 200, "models": models, "error": None}


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] is True


def capabilities_of(model, row):
    if row is not None:
        return {
            "context_length": row["context_length"] if row["context_length"] is not None else model["context_length"],
            "tools": row["tools"],
            "vision": row["vision"],
            "json": row["json"],
        }
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    if requires.get("min_context") is not None:
        ctx = caps.get("context_length")
        if ctx is None or ctx < requires["min_context"]:
            return False
    return True


def build_catalog(probes, ratings):
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            ctx = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row), "context_length": ctx})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(probes, ratings):
    """Free models, in the brief's provider order, ids sorted: GET /v1/models."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({"id": model["id"], "provider": probe["provider"]["id"], "display_name": model["display_name"]})
    return out


# --------------------------------------------------------------------------
# Wire translation: request bodies.
# The providers only echo what they parsed (system text, last user text,
# tool names, max_tokens, stops, image sha, json mode), so every translation
# must preserve exactly those fields -- nothing else in the body matters to
# the echo, but the echo is what the client scores.
# --------------------------------------------------------------------------

def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "\n".join(parts)
    return ""


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return "" if content is None else json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def _image_part_to_anthropic(image_url):
    url = image_url.get("url") if is_record(image_url) else None
    if not isinstance(url, str):
        return None
    if url.startswith("data:"):
        try:
            header, data = url[5:].split(",", 1)
            media = header.split(";")[0] or "image/png"
        except ValueError:
            media, data = "image/png", ""
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    messages = []
    system_parts = []
    last_user_idx = -1
    for m in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(m):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            t = text_of(content)
            if t != "":
                system_parts.append(t)
        elif role == "assistant":
            blocks = []
            t = text_of(content)
            if t != "":
                blocks.append({"type": "text", "text": t})
            for tc in m.get("tool_calls") if isinstance(m.get("tool_calls"), list) else []:
                if not is_record(tc):
                    continue
                fn = tc.get("function") if is_record(tc.get("function")) else {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") if isinstance(tc.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": _arguments_of(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id") if isinstance(m.get("tool_call_id"), str) else "",
                "content": tool_result_text(content),
            }
            # Tool results ride in a user message; attach them to the last user
            # message so the provider's "last user text" stays the client's.
            if last_user_idx >= 0:
                messages[last_user_idx]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
                last_user_idx = len(messages) - 1
        else:  # user (and anything else: treat as user)
            parts = []
            if isinstance(content, str):
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for p in content:
                    if not is_record(p):
                        continue
                    if p.get("type") == "text" and isinstance(p.get("text"), str):
                        parts.append({"type": "text", "text": p["text"]})
                    elif p.get("type") == "image_url":
                        part = _image_part_to_anthropic(p.get("image_url"))
                        if part is not None:
                            parts.append(part)
            if parts:
                messages.append({"role": "user", "content": parts})
                last_user_idx = len(messages) - 1
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = number_or_none(body.get("max_completion_tokens"))
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop"), list) and body["stop"]:
        stops = [s for s in body["stop"] if isinstance(s, str)]
        if stops:
            out["stop_sequences"] = stops
    elif isinstance(body.get("stop"), str):
        out["stop_sequences"] = [body["stop"]]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = {"type": "auto"}
        elif choice.get("type") == "any":
            out["tool_choice"] = {"type": "any"}
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "tool", "name": choice["name"]}
    elif choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}, ensure_ascii=False, separators=(",", ":")),
                    },
                })
            text = text_of(content)
            messages.append({
                "role": "assistant",
                "content": text if text != "" else None,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": tool_result_text(block.get("content")),
                })
                continue
            part = _part_of_anthropic_block(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        stops = [s for s in body["stop_sequences"] if isinstance(s, str)]
        if stops:
            out["stop"] = stops
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _part_of_anthropic_block(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _openai_messages_to_gemini_contents(body):
    system_parts = []
    contents = []
    for m in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(m):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            t = text_of(content)
            if t != "":
                system_parts.append(t)
        elif role == "assistant":
            t = text_of(content)
            if t != "":
                contents.append({"role": "model", "parts": [{"text": t}]})
        elif role == "tool":
            # Tool results carry no echo on the Gemini wire (it reads no tool
            # role); skipping them keeps the last user text exactly as sent.
            continue
        else:  # user
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for p in content:
                    if not is_record(p):
                        continue
                    if p.get("type") == "text" and isinstance(p.get("text"), str):
                        parts.append({"text": p["text"]})
                    elif p.get("type") == "image_url" and is_record(p.get("image_url")):
                        url = p["image_url"].get("url")
                        if not isinstance(url, str):
                            continue
                        if url.startswith("data:"):
                            try:
                                header, data = url[5:].split(",", 1)
                                media = header.split(";")[0] or "image/png"
                            except ValueError:
                                media, data = "image/png", ""
                            parts.append({"inlineData": {"mimeType": media, "data": data}})
                        else:
                            parts.append({"fileData": {"fileUri": url}})
            if parts:
                contents.append({"role": "user", "parts": parts})
    return system_parts, contents


def _tools_to_gemini_declarations(tools):
    decls = []
    for raw in tools if isinstance(tools, list) else []:
        if not is_record(raw):
            continue
        if is_record(raw.get("function")):
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decls.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        elif isinstance(raw.get("name"), str):
            decls.append({
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            })
    return decls


def _gemini_config(body, max_tokens_key="max_tokens"):
    config = {}
    max_tokens = number_or_none(body.get(max_tokens_key))
    if max_tokens is None:
        max_tokens = number_or_none(body.get("max_completion_tokens"))
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stops = None
    if isinstance(body.get("stop"), list):
        stops = [s for s in body["stop"] if isinstance(s, str)]
    elif isinstance(body.get("stop"), str):
        stops = [body["stop"]]
    elif isinstance(body.get("stop_sequences"), list):
        stops = [s for s in body["stop_sequences"] if isinstance(s, str)]
    if stops:
        config["stopSequences"] = stops
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    return config


def openai_to_gemini(body, model):
    system_parts, contents = _openai_messages_to_gemini_contents(body)
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": t} for t in system_parts]}
    decls = _tools_to_gemini_declarations(body.get("tools"))
    if decls:
        out["tools"] = [{"functionDeclarations": decls}]
    config = _gemini_config(body)
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body, model):
    system_parts = []
    system = text_of(body.get("system"))
    if system != "":
        system_parts.append(system)
    contents = []
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            t = text_of(content)
            if t != "":
                contents.append({"role": "model", "parts": [{"text": t}]})
            continue
        if not isinstance(content, list):
            if isinstance(content, str):
                contents.append({"role": "user", "parts": [{"text": content}]})
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                    parts.append({"inlineData": {"mimeType": media, "data": source["data"]}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"fileData": {"fileUri": source["url"]}})
            # tool_result blocks: no echo on the Gemini wire, skip them.
        if parts:
            contents.append({"role": "user", "parts": parts})
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": t} for t in system_parts]}
    decls = _tools_to_gemini_declarations(body.get("tools"))
    if decls:
        out["tools"] = [{"functionDeclarations": decls}]
    config = _gemini_config(body)
    if config:
        out["generationConfig"] = config
    return out


# --------------------------------------------------------------------------
# Wire translation: responses. The provider's answer text ("ANSWER ECHO...")
# is passed whole; only the envelope changes. Usage is always the provider's
# own counts, and the stop reason is mapped into the client's family.
# --------------------------------------------------------------------------

def _num_or_zero(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def stop_reason_of_openai(finish):
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def finish_reason_of_anthropic(stop):
    if stop == "max_tokens":
        return "length"
    if stop == "tool_use":
        return "tool_calls"
    if stop == "refusal":
        return "content_filter"
    return "stop"


def finish_reason_of_gemini(reason):
    if reason == "MAX_TOKENS":
        return "length"
    if reason == "SAFETY":
        return "content_filter"
    return "stop"


def _openai_completion(model, text, finish, prompt_tokens, completion_tokens):
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _anthropic_message(model, text, stop_reason, input_tokens, output_tokens):
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _split_text(text, size=13):
    if text == "":
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def _sse_event(event, data):
    text = "data: %s\n\n" % json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if event is not None:
        text = "event: %s\n%s" % (event, text)
    return text


def openai_stream_events(model, text, finish, prompt_tokens, completion_tokens):
    """OpenAI SSE: role chunk, content deltas, finish chunk, usage, [DONE]."""
    def chunk(delta, finish_reason):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    out = [_sse_event(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_text(text):
        out.append(_sse_event(None, chunk({"content": piece}, None)))
    out.append(_sse_event(None, chunk({}, finish)))
    out.append(_sse_event(None, {
        "id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }))
    out.append("data: [DONE]\n\n")
    return "".join(out)


def anthropic_stream_events(model, text, stop_reason, input_tokens, output_tokens):
    """Anthropic SSE event order: start, block start, deltas, block stop,
    message_delta, message_stop."""
    out = []
    out.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}},
    }))
    out.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""},
    }))
    for piece in _split_text(text):
        out.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece},
        }))
    out.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    out.append(_sse_event("message_delta", {
        "type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }))
    out.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(out)


def openai_json_to_anthropic(answer, model):
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = []
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) else "toolu_unknown",
            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
            "input": _arguments_of(fn.get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = answer.get("usage") if is_record(answer.get("usage")) else {}
    return {
        "id": "msg_%s" % (answer.get("id") if isinstance(answer.get("id"), str) else "agg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of_openai(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": _num_or_zero(usage.get("prompt_tokens")),
                  "output_tokens": _num_or_zero(usage.get("completion_tokens"))},
    }


def anthropic_json_to_openai(answer, model):
    text = ""
    if isinstance(answer.get("content"), list):
        parts = []
        for block in answer["content"]:
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        text = "".join(parts)
    usage = answer.get("usage") if is_record(answer.get("usage")) else {}
    prompt = _num_or_zero(usage.get("input_tokens"))
    completion = _num_or_zero(usage.get("output_tokens"))
    return _openai_completion(model, text, finish_reason_of_anthropic(answer.get("stop_reason")), prompt, completion)


def gemini_json_to_openai(answer, model):
    text, usage = _gemini_answer_parts(answer)
    finish = "stop"
    candidates = answer.get("candidates") if isinstance(answer.get("candidates"), list) else []
    if candidates and is_record(candidates[0]):
        finish = finish_reason_of_gemini(candidates[0].get("finishReason"))
    return _openai_completion(model, text, finish, usage[0], usage[1])


def gemini_json_to_anthropic(answer, model):
    text, usage = _gemini_answer_parts(answer)
    return _anthropic_message(model, text, "end_turn", usage[0], usage[1])


def _gemini_answer_parts(answer):
    text = ""
    candidates = answer.get("candidates") if isinstance(answer.get("candidates"), list) else []
    if candidates and is_record(candidates[0]):
        content = candidates[0].get("content")
        if is_record(content) and isinstance(content.get("parts"), list):
            parts = []
            for p in content["parts"]:
                if is_record(p) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            text = "".join(parts)
    meta = answer.get("usageMetadata") if is_record(answer.get("usageMetadata")) else {}
    return text, (_num_or_zero(meta.get("promptTokenCount")), _num_or_zero(meta.get("candidatesTokenCount")))


def _parse_sse_frames(text):
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append({"event": event, "data": "\n".join(data)})
    return frames


def openai_sse_to_anthropic(upstream_text, model):
    """An OpenAI SSE stream as Anthropic events (buffered: usage is known
    before the first event is emitted)."""
    text = ""
    prompt = 0
    completion = 0
    finish = "end_turn"
    for frame in _parse_sse_frames(upstream_text):
        payload = frame["data"].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            prompt = _num_or_zero(chunk["usage"].get("prompt_tokens"))
            completion = _num_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choice.get("finish_reason"), str):
            finish = stop_reason_of_openai(choice["finish_reason"])
    return anthropic_stream_events(model, text, finish, prompt, completion)


def anthropic_sse_to_openai(upstream_text, model):
    """An Anthropic SSE stream as OpenAI events (buffered)."""
    text = ""
    prompt = 0
    completion = 0
    finish = "stop"
    for frame in _parse_sse_frames(upstream_text):
        try:
            data = json.loads(frame["data"])
        except Exception:
            continue
        if not is_record(data):
            continue
        event = frame["event"]
        if event == "message_start":
            message = data.get("message") if is_record(data.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            prompt = _num_or_zero(usage.get("input_tokens"))
        elif event == "content_block_delta":
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text += delta["text"]
        elif event == "message_delta":
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                finish = finish_reason_of_anthropic(delta["stop_reason"])
            usage = data.get("usage") if is_record(data.get("usage")) else {}
            completion = _num_or_zero(usage.get("output_tokens"))
    return openai_stream_events(model, text, finish, prompt, completion)


def gemini_sse_to_openai(upstream_text, model):
    text, usage, finish = _gemini_sse_parts(upstream_text)
    return openai_stream_events(model, text, finish, usage[0], usage[1])


def gemini_sse_to_anthropic(upstream_text, model):
    text, usage, finish = _gemini_sse_parts(upstream_text)
    return anthropic_stream_events(model, text, "end_turn", usage[0], usage[1])


def _gemini_sse_parts(upstream_text):
    text = ""
    prompt = 0
    completion = 0
    finish = "stop"
    for frame in _parse_sse_frames(upstream_text):
        try:
            data = json.loads(frame["data"])
        except Exception:
            continue
        if not is_record(data):
            continue
        candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
        for cand in candidates:
            if not is_record(cand):
                continue
            content = cand.get("content")
            if is_record(content) and isinstance(content.get("parts"), list):
                for p in content["parts"]:
                    if is_record(p) and isinstance(p.get("text"), str):
                        text += p["text"]
            if isinstance(cand.get("finishReason"), str):
                finish = finish_reason_of_gemini(cand["finishReason"])
        meta = data.get("usageMetadata") if is_record(data.get("usageMetadata")) else {}
        if meta:
            prompt = _num_or_zero(meta.get("promptTokenCount"))
            completion = _num_or_zero(meta.get("candidatesTokenCount"))
    return text, (prompt, completion), finish


# --------------------------------------------------------------------------
# Provider calls: one attempt, and the failover loop over ranked candidates.
# --------------------------------------------------------------------------

class ProviderResult:
    def __init__(self, ok, status, body, retry_after, error=None):
        self.ok = ok
        self.status = status
        self.body = body          # parsed JSON or raw SSE text
        self.retry_after = retry_after
        self.error = error


def _retry_after_of(status, headers, body):
    if status != 429:
        return None
    header = headers.get("retry-after") or headers.get("Retry-After")
    if header is not None:
        try:
            return float(header)
        except ValueError:
            pass
    if is_record(body):
        err = body.get("error")
        if is_record(err) and isinstance(err.get("retry_after"), (int, float)):
            return float(err["retry_after"])
    return None


def call_provider_once(env_url, provider, client_api, body, model, stream):
    """One call to one provider on its own wire. Returns ProviderResult
    whose body is the answer already translated into the client's wire
    (a dict for JSON, a string of SSE for streams)."""
    base = join_base(env_url, provider["base"])
    wire = provider["wire"]
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    if stream:
        headers["accept"] = "text/event-stream"
    if wire == "openai":
        url = base + "/v1/chat/completions"
        if client_api == "openai":
            upstream = dict(body)
            upstream["model"] = model
            if stream:
                upstream["stream_options"] = {"include_usage": True}
        else:
            upstream = anthropic_to_openai(body, model)
    elif wire == "anthropic":
        url = base + "/v1/messages"
        if client_api == "anthropic":
            upstream = dict(body)
            upstream["model"] = model
        else:
            upstream = openai_to_anthropic(body, model)
    else:  # gemini
        method = "streamGenerateContent" if stream else "generateContent"
        url = base + "/v1beta/models/%s:%s" % (urllib.parse.quote(model, safe=""), method)
        if stream:
            url += "?alt=sse"
        if client_api == "anthropic":
            upstream = anthropic_to_gemini(body, model)
        else:
            upstream = openai_to_gemini(body, model)
    data = json.dumps(upstream, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT_S) as resp:
            raw = resp.read()
            status = resp.status
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
        resp_headers = {k.lower(): v for k, v in exc.headers.items()}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = None
        return ProviderResult(False, status, parsed, _retry_after_of(status, resp_headers, parsed))
    except Exception as exc:
        return ProviderResult(False, 0, None, None, error=str(exc))
    if status < 200 or status >= 300:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = None
        return ProviderResult(False, status, parsed, _retry_after_of(status, resp_headers, parsed))
    text = raw.decode("utf-8")
    if stream:
        return ProviderResult(True, status, _translate_stream(wire, client_api, text, model), None)
    try:
        answer = json.loads(text)
    except Exception:
        return ProviderResult(False, status, None, None, error="the provider answered a non-JSON body")
    if not is_record(answer):
        return ProviderResult(False, status, None, None, error="the provider answered a non-object")
    return ProviderResult(True, status, _translate_json(wire, client_api, answer, model), None)


def _translate_json(wire, client_api, answer, model):
    if wire == client_api:
        return answer
    if wire == "openai" and client_api == "anthropic":
        return openai_json_to_anthropic(answer, model)
    if wire == "anthropic" and client_api == "openai":
        return anthropic_json_to_openai(answer, model)
    if wire == "gemini" and client_api == "openai":
        return gemini_json_to_openai(answer, model)
    if wire == "gemini" and client_api == "anthropic":
        return gemini_json_to_anthropic(answer, model)
    return answer


def _translate_stream(wire, client_api, text, model):
    if wire == client_api:
        return text
    if wire == "openai" and client_api == "anthropic":
        return openai_sse_to_anthropic(text, model)
    if wire == "anthropic" and client_api == "openai":
        return anthropic_sse_to_openai(text, model)
    if wire == "gemini" and client_api == "openai":
        return gemini_sse_to_openai(text, model)
    if wire == "gemini" and client_api == "anthropic":
        return gemini_sse_to_anthropic(text, model)
    return text


class Router:
    """Ranked candidates, failover, and per-provider spent tracking."""

    def __init__(self, env_url, probes, ratings):
        self.env_url = env_url
        self.probes = probes
        self.ratings = ratings
        self.consecutive_failures = {}
        self.dead = set()
        self._candidates_cache = {}

    def candidates(self, requires, client_api, body):
        key = (json.dumps(requires, sort_keys=True), client_api,
               _is_json_mode_request(client_api, body))
        if key in self._candidates_cache:
            return self._candidates_cache[key]
        json_mode = _is_json_mode_request(client_api, body)
        out = []
        for order, probe in enumerate(self.probes):
            if not probe["ok"]:
                continue
            provider = probe["provider"]
            # An anthropic-wire provider cannot carry an OpenAI JSON-mode
            # request: its echo would report json_mode absent.
            if json_mode and client_api == "openai" and provider["wire"] == "anthropic":
                continue
            for model in probe["models"]:
                row = rating_of_model(self.ratings, provider["id"], model["id"])
                if not model_free(model, row):
                    continue
                caps = capabilities_of(model, row)
                if not meets_requirements(requires, caps):
                    continue
                rating = row["rating_milli"] if row is not None else 0
                out.append((provider, model, rating, order))
        out.sort(key=lambda c: (-c[2], c[3], c[1]["id"]))
        result = [(p, m) for (p, m, _r, _o) in out]
        self._candidates_cache[key] = result
        return result

    def serve(self, client_api, body, requires):
        """Try candidates in rank order; return (ok, result)."""
        tried = set()
        for provider, model in self.candidates(requires, client_api, body):
            pid = provider["id"]
            if pid in tried or pid in self.dead:
                continue
            stream = body.get("stream") is True
            result = call_provider_once(self.env_url, provider, client_api, body, model["id"], stream)
            if result.ok:
                self.consecutive_failures[pid] = 0
                return True, result
            tried.add(pid)
            if result.status in (402, 429) or result.status >= 500:
                self.consecutive_failures[pid] = self.consecutive_failures.get(pid, 0) + 1
                if self.consecutive_failures[pid] >= DEAD_AFTER_CONSECUTIVE:
                    self.dead.add(pid)
                    log("%s looks spent after %d consecutive refusals" % (pid, DEAD_AFTER_CONSECUTIVE))
                # One retry on the same provider: a one-off fail_call passes
                # on the next call number, a spent budget fails again.
                if result.retry_after:
                    time.sleep(min(float(result.retry_after), 1.0))
                retry = call_provider_once(self.env_url, provider, client_api, body, model["id"], stream)
                if retry.ok:
                    self.consecutive_failures[pid] = 0
                    return True, retry
                self.consecutive_failures[pid] = self.consecutive_failures.get(pid, 0) + 1
                if self.consecutive_failures[pid] >= DEAD_AFTER_CONSECUTIVE:
                    self.dead.add(pid)
                    log("%s looks spent after %d consecutive refusals" % (pid, DEAD_AFTER_CONSECUTIVE))
            elif result.status in (400, 401):
                log("call to %s/%s refused HTTP %s" % (pid, model["id"], result.status))
            else:
                log("call to %s/%s failed: %s" % (pid, model["id"], result.error or result.status))
        return False, None


def _is_json_mode_request(client_api, body):
    if client_api != "openai":
        return False
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    return fmt is not None and fmt.get("type") in ("json_object", "json_schema")


# --------------------------------------------------------------------------
# HTTP server: the two client routes and the model list.
# --------------------------------------------------------------------------

def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def openai_model_list(models):
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]} for m in models],
    }


def anthropic_model_list(models):
    return {
        "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": "2026-01-01T00:00:00Z"} for m in models],
        "has_more": False,
    }


def openai_error(message, type_, code):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


class Aggregator:
    def __init__(self):
        self.brief = read_brief(PROMPT_FILE)
        if self.brief["error"] is not None:
            log(self.brief["error"])
        self.probes = []
        self.catalog = {"providers": []}
        self.router = None
        self.manifest_index = 0
        self.handle_lock = threading.Lock()

    def load(self):
        providers = self.brief["directory"]["providers"]
        for provider in providers:
            probe = probe_provider(ENV_URL, provider)
            if not probe["ok"] and probe["error"] is not None:
                log(probe["error"])
            self.probes.append(probe)
        self.catalog = build_catalog(self.probes, self.brief["ratings"])
        self.router = Router(ENV_URL, self.probes, self.brief["ratings"])

    def requirements_for(self, body):
        manifest = self.brief["workload"]["manifest"] if self.brief["workload"] is not None else []
        if self.manifest_index < len(manifest):
            entry = manifest[self.manifest_index]
            self.manifest_index += 1
            return entry["requires"]
        return requirements_from_body(body)

    def models_list(self, req_headers):
        models = served_models(self.probes, self.brief["ratings"])
        if req_headers.get("x-api-key") is not None:
            return anthropic_model_list(models)
        return openai_model_list(models)


def requirements_from_body(body):
    vision = False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    return {
        "tools": tools,
        "vision": vision,
        "json": fmt is not None and fmt.get("type") in ("json_object", "json_schema"),
    }


def make_handler(agg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                return None
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw.decode("utf-8"))
            except Exception:
                return None
            return value if is_record(value) else None

        def _send_json(self, status, value):
            data = _json_bytes(value)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_sse(self, text):
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream; charset=utf-8")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "keep-alive")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, wire, status, type_, message):
            if wire == "anthropic":
                self._send_json(status, anthropic_error(type_, message))
            else:
                self._send_json(status, openai_error(message, type_, None))

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/v1/models":
                self._send_json(200, agg.models_list(self.headers))
                return
            self._send_error("openai", 404, "not_found", "no route GET %s" % url.path)

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/v1/chat/completions":
                self._handle_chat("openai")
                return
            if url.path == "/v1/messages":
                self._handle_chat("anthropic")
                return
            self._send_error("anthropic" if url.path == "/v1/messages" else "openai",
                             404, "not_found", "no route POST %s" % url.path)

        def _handle_chat(self, wire):
            body = self._read_json()
            if body is None:
                self._send_error(wire, 400, "invalid_request_error", "the request body is not one JSON object")
                return
            with agg.handle_lock:
                requires = agg.requirements_for(body)
                stream = body.get("stream") is True
                ok, result = agg.router.serve(wire, body, requires)
            if not ok:
                self._send_error(wire, 503, "overloaded_error" if wire == "anthropic" else "no_free_model",
                                 "no free model on the directory meets this request")
                return
            if stream:
                self._send_sse(result.body)
            else:
                self._send_json(200, result.body)

    return Handler


def register(port, catalog):
    req = urllib.request.Request(
        ENV_URL + "/register",
        data=_json_bytes({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError("POST %s/register answered HTTP %s" % (ENV_URL, resp.status))
            reply = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("POST %s/register failed: %s" % (ENV_URL, exc))
    if is_record(reply) and isinstance(reply.get("requests"), int):
        log("registered: the workload holds %d requests" % reply["requests"])


def poll_work(server):
    while True:
        time.sleep(POLL_INTERVAL_S)
        try:
            req = urllib.request.Request(ENV_URL + "/work", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status != 200:
                    continue
                work = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            log("the workload is done; exiting 0")
            server.shutdown()
            return


def main():
    agg = Aggregator()
    agg.load()
    if "--list" in sys.argv:
        text = json.dumps(agg.catalog, ensure_ascii=False, separators=(",", ":")) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    handler = make_handler(agg)
    server = ThreadingHTTPServer(("0.0.0.0", 0), handler)
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(agg.catalog["providers"]), len(served_models(agg.probes, agg.brief["ratings"]))))
    try:
        register(port, agg.catalog)
    except RuntimeError as exc:
        log(str(exc))
        return 1
    poller = threading.Thread(target=poll_work, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
