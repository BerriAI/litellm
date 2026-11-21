"""Shared rig for the `test_dec1_otherprov2_*` files: round two of the widely used providers beyond the big six.

It reuses the round one rig (`dec1_otherprov_support.py`: the recording loopback `peer`, `deployment`, `chat`, the
stream readers and the spend row reader) and adds what round two needs: an upstream error reply in the OpenAI
error shape, a tool call answer, an OpenAI Responses API answer for the xAI `/v1/responses` route, a raw SSE stream
built from literal chunks, and a reader for the error a stream ends with.

Every deployment points its `api_base` at the loopback peer, so no request can reach a real provider host.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Final

import httpx
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov_support import (
    REPLY_TEXT,
    Call,
    chat_json,
    json_reply,
    marker,
    sse,
    usage,
)
from pydantic import JsonValue

TOOL_CALL_ID: Final = "call_dec1otherprov2"
TOOL_ARGUMENTS: Final = '{"city": "Oslo"}'
RESPONSES_INPUT_TOKENS: Final = 13
RESPONSES_OUTPUT_TOKENS: Final = 5


def tool_call(name: str = "get_weather", identity: str = TOOL_CALL_ID) -> dict[str, JsonValue]:
    """One OpenAI-shaped tool call with `TOOL_ARGUMENTS`."""
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": TOOL_ARGUMENTS}}


def tool_call_reply(call: Call, finish_reason: str = "tool_calls") -> Reply:
    """A chat answer whose only content is one call to `get_weather`."""
    return json_reply(
        chat_json(call.body, content=None, finish_reason=finish_reason, extra_message={"tool_calls": [tool_call()]})
    )


def upstream_error(status: int, message: str, error_type: str = "invalid_request_error") -> Reply:
    """The OpenAI error envelope most OpenAI-compatible providers answer a failed call with."""
    return json_reply({"error": {"message": message, "type": error_type, "code": None}}, status=status)


def raw_stream(*chunks: Mapping[str, JsonValue]) -> Reply:
    """An SSE stream of exactly these JSON chunks followed by `[DONE]`, nothing added."""
    return Reply(content_type="text/event-stream", chunks=(*(sse(chunk) for chunk in chunks), b"data: [DONE]\n\n"))


def chunk(body: Mapping[str, JsonValue], **fields: JsonValue) -> dict[str, JsonValue]:
    """One chat chunk with a fixed id and whatever top-level fields the test gives (`choices`, `usage`, ...)."""
    return {
        "id": "chatcmpl-dec1-otherprov2",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": body.get("model"),
        **fields,
    }


def delta_choice(delta: Mapping[str, JsonValue], finish_reason: str | None = None) -> list[JsonValue]:
    return [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}]


def responses_reply(body: Mapping[str, JsonValue], text: str = REPLY_TEXT) -> Reply:
    """An OpenAI Responses API answer (`object: response`) with one output message and 13 in, 5 out tokens."""
    identity: Final = f"resp_dec1_{uuid.uuid4().hex}"
    return json_reply(
        {
            "id": identity,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": body.get("model"),
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": RESPONSES_INPUT_TOKENS,
                "output_tokens": RESPONSES_OUTPUT_TOKENS,
                "total_tokens": RESPONSES_INPUT_TOKENS + RESPONSES_OUTPUT_TOKENS,
            },
        }
    )


def responses(gateway: Gateway, model: str, **extra: JsonValue) -> httpx.Response:
    """POST /v1/responses with a fresh marker as the input."""
    return gateway.request("POST", "/v1/responses", {"model": model, "input": marker(), **extra})


def usage_with(**extra: JsonValue) -> dict[str, JsonValue]:
    """The default 11 prompt plus 7 completion usage block with extra provider fields."""
    return {**usage(), **extra}


def stream_text(gateway: Gateway, body: Mapping[str, JsonValue]) -> tuple[int, str]:
    """Send a streaming chat request and return the status and the whole raw answer text, error frames included."""
    with gateway.client.stream(
        "POST", "/v1/chat/completions", json=dict(body), headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        return response.status_code, response.read().decode()


def message_roles(sent: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(str(object_value(message)["role"]) for message in list_value(sent["messages"]))
