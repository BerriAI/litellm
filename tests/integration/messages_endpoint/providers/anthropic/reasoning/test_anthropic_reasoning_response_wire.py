import json
import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL: Final = "claude-haiku-4-5"
_THINKING: Final = "the user wants a single word"
_SIGNATURE: Final = "EqQBCkgIBRABGAIiQLz"
_REDACTED: Final = "EmwKAhgBEgy3va3pzixlit"


def _reasoning_stream(identity: str) -> tuple[bytes, ...]:
    return (
        cc.sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": _MODEL,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            },
        ),
        cc.sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": _THINKING}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": _SIGNATURE}},
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        cc.sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "redacted_thinking", "data": _REDACTED},
            },
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
        cc.sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "PONG"}},
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 2}),
        cc.sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 30},
            },
        ),
        cc.sse_frame("message_stop", {"type": "message_stop"}),
    )


def _reasoning_message(identity: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": _MODEL,
        "content": [
            {"type": "thinking", "thinking": _THINKING, "signature": _SIGNATURE},
            {"type": "redacted_thinking", "data": _REDACTED},
            {"type": "text", "text": "PONG"},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 30},
    }


def _content_events(stream: str) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(event for event in cc.sse_events(stream) if event[0].startswith("content_block_"))


def _client_body(stream: bool) -> dict[str, JsonValue]:
    return {
        **cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"),
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "stream": stream,
    }


def test_streamed_thinking_signature_and_redacted_thinking_reach_the_client_unchanged(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    upstream_frames: Final = _reasoning_stream(identity)

    def respond(request: Request) -> Reply:
        return Reply(chunks=upstream_frames, content_type="text/event-stream")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", {**_client_body(stream=True), "model": model})
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
    assert _content_events(response.text) == _content_events(b"".join(upstream_frames).decode()), response.text


def test_non_streamed_thinking_signature_and_redacted_thinking_reach_the_client_unchanged(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    upstream_message: Final = _reasoning_message(identity)

    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(upstream_message).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", {**_client_body(stream=False), "model": model})
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
    assert response.json()["content"] == upstream_message["content"], response.text
