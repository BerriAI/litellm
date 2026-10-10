from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

from integration._support.openai_wire import chat_stream, responses_stream
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

AZURE_KEY: Final = "synthetic-azure-key"
TAVILY_KEY: Final = "synthetic-tavily-key"
API_VERSION: Final = "2025-04-01-preview"
SEARCH_TOOL: Final = "litellm_web_search"
SEARCH_TARGET: Final = "/tavily/search"
STREAM_OPTIONS_REFUSAL: Final = "The 'stream_options' parameter is only allowed when 'stream' is enabled."
FIRST_USAGE: Final = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
FOLLOWUP_USAGE: Final = {"prompt_tokens": 23, "completion_tokens": 5, "total_tokens": 28}
RESPONSES_FIRST_USAGE: Final = {"input_tokens": 13, "output_tokens": 6, "total_tokens": 19}
RESPONSES_FOLLOWUP_USAGE: Final = {"input_tokens": 29, "output_tokens": 4, "total_tokens": 33}
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])
_CHAT_TARGET: Final = re.compile(r"^/openai/deployments/([^/]+)/chat/completions$")

Mode: TypeAlias = Literal["search", "plain", "search-down", "followup-fails", "missing"]


def deployment(mode: Mode) -> str:
    return f"dep-{uuid.uuid4().hex}-{mode}"


def answer(name: str) -> str:
    return f"answer-{name}"


def plain(name: str) -> str:
    return f"plain-{name}"


def query(name: str) -> str:
    return f"q-{name}"


def _minted(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _azure_error(status: int, message: str, *, param: str | None = None) -> Reply:
    return Reply(
        status=status,
        body=json.dumps(
            {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": None}}
        ).encode(),
    )


def _refusal(body: Mapping[str, JsonValue]) -> Reply | None:
    if body.get("stream_options") is not None and body.get("stream") is not True:
        return _azure_error(400, STREAM_OPTIONS_REFUSAL, param="stream_options")
    unknown: Final = tuple(key for key in body if key.startswith("_"))
    if unknown:
        return _azure_error(400, f"Unrecognized request argument supplied: {unknown[0]}")
    return None


def _offers_search(body: Mapping[str, JsonValue]) -> bool:
    tools: Final = _ITEMS.validate_python(body.get("tools") or [])
    return any(
        tool.get("name") == SEARCH_TOOL
        or JSON_OBJECT.validate_python(tool.get("function") or {}).get("name") == SEARCH_TOOL
        for tool in tools
    )


def _chat_completion(identity: str, message: Mapping[str, JsonValue], finish: str, usage: Mapping[str, int]) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": dict(message), "finish_reason": finish}],
                "usage": dict(usage),
            }
        ).encode()
    )


def _chat(name: str, body: Mapping[str, JsonValue]) -> Reply:
    if body.get("stream") is True:
        return Reply(
            content_type="text/event-stream", chunks=chat_stream(_minted(f"chatcmpl-{name}-s-"), "gpt-4o", plain(name))
        )
    messages: Final = _ITEMS.validate_python(body["messages"])
    if any(message.get("role") == "tool" for message in messages):
        if name.endswith("-followup-fails"):
            return _azure_error(500, "scripted follow-up failure")
        return _chat_completion(
            _minted(f"chatcmpl-{name}-2-"), {"role": "assistant", "content": answer(name)}, "stop", FOLLOWUP_USAGE
        )
    if name.endswith("-plain") or not _offers_search(body):
        return _chat_completion(
            _minted(f"chatcmpl-{name}-1-"), {"role": "assistant", "content": plain(name)}, "stop", FIRST_USAGE
        )
    call: Final = {
        "id": f"call_{name[4:36]}",
        "type": "function",
        "function": {"name": SEARCH_TOOL, "arguments": json.dumps({"query": query(name)})},
    }
    message: Final = JSON_OBJECT.validate_python({"role": "assistant", "content": None, "tool_calls": [call]})
    return _chat_completion(_minted(f"chatcmpl-{name}-1-"), message, "tool_calls", FIRST_USAGE)


def _response(identity: str, name: str, output: Sequence[Mapping[str, JsonValue]], usage: Mapping[str, int]) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": name,
                "output": [dict(item) for item in output],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": dict(usage),
            }
        ).encode()
    )


def _text_item(identity: str, text: str) -> Mapping[str, JsonValue]:
    return {
        "id": f"msg_{identity}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _responses(body: Mapping[str, JsonValue]) -> Reply:
    name: Final = str(body["model"])
    if body.get("stream") is True:
        return Reply(
            content_type="text/event-stream", chunks=responses_stream(_minted(f"resp_{name}_s_"), name, plain(name))
        )
    items: Final = _ITEMS.validate_python(body["input"]) if isinstance(body.get("input"), list) else []
    if any(item.get("type") == "function_call_output" for item in items):
        if name.endswith("-followup-fails"):
            return _azure_error(500, "scripted follow-up failure")
        return _response(
            _minted(f"resp_{name}_2_"), name, (_text_item(f"{name}_2", answer(name)),), RESPONSES_FOLLOWUP_USAGE
        )
    if name.endswith("-plain") or not _offers_search(body):
        return _response(
            _minted(f"resp_{name}_1_"), name, (_text_item(f"{name}_1", plain(name)),), RESPONSES_FIRST_USAGE
        )
    call: Final = {
        "type": "function_call",
        "id": f"fc_{name[4:36]}",
        "call_id": f"call_{name[4:36]}",
        "name": SEARCH_TOOL,
        "arguments": json.dumps({"query": query(name)}),
        "status": "completed",
    }
    return _response(_minted(f"resp_{name}_1_"), name, (call,), RESPONSES_FIRST_USAGE)


def _search(request: Request, body: Mapping[str, JsonValue]) -> Reply:
    if request.headers.get("authorization") != f"Bearer {TAVILY_KEY}":
        return Reply(status=401, body=b'{"detail":{"error":"Unauthorized"}}')
    text: Final = str(body.get("query", ""))
    if text.endswith("-search-down"):
        return Reply(status=500, body=b'{"detail":{"error":"scripted search outage"}}')
    return Reply(
        body=json.dumps(
            {
                "query": text,
                "results": [
                    {"title": f"title-{text}", "url": "https://example.test/result", "content": f"snippet-{text}"}
                ],
            }
        ).encode()
    )


def respond(request: Request) -> Reply:
    target: Final = urlsplit(request.target).path
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = JSON_OBJECT.validate_json(request.body or b"{}")
    if target == SEARCH_TARGET:
        return _search(request, body)
    if request.headers.get("api-key") != AZURE_KEY:
        return Reply(
            status=401, body=b'{"error":{"code":"401","message":"Access denied due to invalid subscription key."}}'
        )
    chat: Final = _CHAT_TARGET.match(target)
    if chat is not None and chat.group(1).endswith("-missing"):
        return Reply(
            status=404,
            body=b'{"error":{"code":"DeploymentNotFound","message":"The API deployment for this resource does not exist."}}',
        )
    refused: Final = _refusal(body)
    if refused is not None:
        return refused
    if chat is not None:
        return _chat(chat.group(1), body)
    if target.endswith("/responses"):
        return _responses(body)
    return Reply(status=404, body=b'{"error":{"code":"404","message":"Resource not found"}}')


@dataclass(frozen=True, slots=True)
class Received:
    model_calls: tuple[Mapping[str, JsonValue], ...]
    searches: tuple[Mapping[str, JsonValue], ...]


def _named(request: Request, name: str) -> bool:
    return name in request.target or name in request.body.decode(errors="replace")


def received(drained: Sequence[Request], name: str) -> Received:
    requests: Final = tuple(request for request in drained if request.method == "POST" and _named(request, name))
    return Received(
        model_calls=tuple(
            JSON_OBJECT.validate_json(request.body)
            for request in requests
            if urlsplit(request.target).path != SEARCH_TARGET
        ),
        searches=tuple(
            JSON_OBJECT.validate_json(request.body)
            for request in requests
            if urlsplit(request.target).path == SEARCH_TARGET
        ),
    )
