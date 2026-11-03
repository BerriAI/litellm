from __future__ import annotations

import base64
import json
import os
import re
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlsplit

from integration._support import claude_code as cc
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
THOUGHT: Final = "plan the answer"
USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}
CHAT_USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}
CLAUDE_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 20, "output_tokens": 7}
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])
MINTED_ID: Final = re.compile(r"^rs_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_INNER_ID: Final = re.compile(r"response_id:([^;]+)")
_WRAPPER_PREFIX: Final = "litellm:custom_llm_provider:"
_PROXY_WRAPPED_PREFIX: Final = "litellm_proxy:responses_api:response_id:"


def signature(marker: str) -> str:
    return f"sig-{marker}"


def answer(marker: str | None) -> str:
    return "ok" if marker is None else f"answer marker-{marker}"


def newest_marker(text: str) -> str | None:
    found: Final = MARKER.findall(text)
    return str(found[-1]) if found else None


def error(status: int, message: str, code: str) -> Reply:
    body: Final = {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}}
    return Reply(status=status, body=json.dumps(body).encode())


def sse(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def chat_sse(frame: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def thinking_json(marker: str) -> str:
    return json.dumps([{"type": "thinking", "thinking": THOUGHT, "signature": signature(marker)}])


def minted_item(marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"type": "reasoning", "id": f"rs_{uuid.uuid4()}", "encrypted_content": thinking_json(marker), **extra}


def agents_sdk_history(marker: str, *reasoning: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": "Pick a city and look up its weather."},
        *reasoning,
        {
            "type": "message",
            "id": f"msg_{uuid.uuid4()}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Prague", "annotations": []}],
        },
        {"type": "function_call", "call_id": "call_weather", "name": "weather", "arguments": '{"city": "Prague"}'},
        {"type": "function_call_output", "call_id": "call_weather", "output": '{"celsius": 18}'},
        {"role": "user", "content": f"Now answer marker-{marker}"},
    ]


def without(history: Sequence[dict[str, JsonValue]], dropped: Sequence[dict[str, JsonValue]]) -> list[JsonValue]:
    return [item for item in history if all(item is not gone for gone in dropped)]


def reasoning_items(body: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [item for item in ITEMS.validate_python(body["input"]) if item.get("type") == "reasoning"]


def _decoded_wrapper(value: str) -> str | None:
    try:
        decoded: Final = base64.b64decode(value.removeprefix("resp_"), validate=True).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded if decoded.startswith(_WRAPPER_PREFIX) else None


def response_identities(value: str) -> frozenset[str]:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

    salt: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
    opened: Final = decrypt_if_encrypted_with(value.removeprefix("resp_"), salt)
    sealed: Final = opened is not None and opened.startswith(_PROXY_WRAPPED_PREFIX)
    wrapped: Final = opened.removeprefix(_PROXY_WRAPPED_PREFIX).split(";", 1)[0] if sealed and opened else value
    decoded: Final = _decoded_wrapper(wrapped)
    if decoded is None:
        return frozenset({wrapped})
    inner: Final = _INNER_ID.search(decoded)
    assert inner is not None, decoded
    return frozenset({wrapped, inner.group(1)})


def same_response(left: str, right: str) -> bool:
    return bool(response_identities(left) & response_identities(right))


@dataclass(frozen=True, slots=True)
class ResponsesVendor:
    claude_model: str = cc.OPUS
    pause_between_chunks: float = 0
    minted: deque[str] = field(default_factory=deque)

    def respond(self, request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": [{"id": "gpt-5.6", "object": "model"}]}).encode())
        body: Final = JSON_OBJECT.validate_json(request.body)
        if path.endswith("/messages"):
            return self._claude(body)
        if path.endswith("/chat/completions"):
            return self._chat(body)
        assert path.endswith("/responses"), request.target
        verdict: Final = self._verdict(body)
        return verdict if verdict is not None else self._responses(body)

    def _verdict(self, body: Mapping[str, JsonValue]) -> Reply | None:
        received: Final = body.get("input")
        if isinstance(received, str):
            return None
        items: Final = ITEMS.validate_python(received)
        if not items and "previous_response_id" not in body:
            return error(
                400, 'One of "input" or "previous_response_id" must be provided.', "missing_required_parameter"
            )
        for index, item in enumerate(items):
            if item.get("type") != "reasoning":
                continue
            item_id: Final = item.get("id")
            if item_id is not None and not isinstance(item_id, str):
                return error(400, f"Invalid type for 'input[{index}].id': expected a string.", "invalid_type")
            if "summary" not in item:
                return error(
                    400, f"Missing required parameter: 'input[{index}].summary'.", "missing_required_parameter"
                )
            if item_id == "":
                return error(400, f"Invalid 'input[{index}].id': empty string.", "invalid_value")
            if isinstance(item_id, str) and item_id not in self.minted:
                return error(404, f"Item with id '{item_id}' not found.", "invalid_request_error")
        return None

    def _responses(self, body: Mapping[str, JsonValue]) -> Reply:
        marker: Final = newest_marker(json.dumps(body))
        tag: Final = uuid.uuid4().hex
        self.minted.append(f"rs_{tag}")
        reasoning: Final[dict[str, JsonValue]] = {
            "id": f"rs_{tag}",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": f"gAAAAA-vendor-{tag}",
        }
        message: Final[dict[str, JsonValue]] = {
            "id": f"msg_{tag}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": answer(marker), "annotations": []}],
        }
        response: Final[dict[str, JsonValue]] = {
            "id": f"resp_{tag}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": body["model"],
            "output": [reasoning, message],
            "usage": USAGE,
        }
        if body.get("stream") is not True:
            return Reply(body=json.dumps(response).encode())
        events: Final[tuple[dict[str, JsonValue], ...]] = (
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**response, "status": "in_progress", "output": []},
            },
            {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": reasoning},
            {"type": "response.output_item.done", "sequence_number": 2, "output_index": 0, "item": reasoning},
            {
                "type": "response.output_item.added",
                "sequence_number": 3,
                "output_index": 1,
                "item": {**message, "content": []},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 4,
                "item_id": f"msg_{tag}",
                "output_index": 1,
                "content_index": 0,
                "delta": answer(marker),
            },
            {"type": "response.output_item.done", "sequence_number": 5, "output_index": 1, "item": message},
            {"type": "response.completed", "sequence_number": 6, "response": response},
        )
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(sse(event) for event in events),
            pause_between_chunks=self.pause_between_chunks,
        )

    def _chat(self, body: Mapping[str, JsonValue]) -> Reply:
        marker: Final = newest_marker(json.dumps(body))
        tag: Final = uuid.uuid4().hex
        if body.get("stream") is not True:
            return Reply(
                body=json.dumps(
                    {
                        "id": f"chatcmpl-{tag}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": answer(marker)},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": CHAT_USAGE,
                    }
                ).encode()
            )
        chunk: Final[dict[str, JsonValue]] = {
            "id": f"chatcmpl-{tag}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
        }
        frames: Final[tuple[dict[str, JsonValue], ...]] = (
            {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer(marker)}}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": CHAT_USAGE},
        )
        return Reply(
            content_type="text/event-stream",
            chunks=(*(chat_sse(frame) for frame in frames), b"data: [DONE]\n\n"),
            pause_between_chunks=self.pause_between_chunks,
        )

    def _claude(self, body: Mapping[str, JsonValue]) -> Reply:
        marker: Final = newest_marker(json.dumps(body))
        content: Final = (
            {"type": "thinking", "thinking": THOUGHT, "signature": signature(marker or "")},
            {"type": "text", "text": answer(marker)},
        )
        identity: Final = f"msg_{uuid.uuid4().hex}"
        if body.get("stream") is True:
            return Reply(
                content_type="text/event-stream",
                chunks=cc.message_stream(identity, self.claude_model, content, CLAUDE_USAGE),
                pause_between_chunks=self.pause_between_chunks,
            )
        return Reply(body=cc.message_reply(identity, self.claude_model, content, CLAUDE_USAGE))
