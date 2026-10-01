import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL: Final = "claude-haiku-4-5"
_CONTENT: Final = (
    {"type": "thinking", "thinking": "the user wants a single word", "signature": "EqQBCkgIBRABGAIiQLz"},
    {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzixlit"},
    {"type": "text", "text": "PONG"},
)
_THINKING_TOKENS: Final = {"thinking_tokens": 20}
_USAGE: Final = {"input_tokens": 12, "output_tokens": 30, "output_tokens_details": _THINKING_TOKENS}


def _client_body(stream: bool) -> dict[str, JsonValue]:
    return {
        **cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"),
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "stream": stream,
    }


def _content_events(stream: str) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(event for event in cc.sse_events(stream) if event[0].startswith("content_block_"))


def _streamed_thinking_tokens(stream: str) -> tuple[object, ...]:
    return tuple(
        data["usage"].get("output_tokens_details")
        for event, data in cc.sse_events(stream)
        if event == "message_delta" and isinstance(data["usage"], dict)
    )


def test_streamed_thinking_blocks_and_thinking_token_count_reach_the_client_unchanged(gateway: Gateway) -> None:
    upstream_frames: Final = cc.message_stream(f"msg_{uuid.uuid4().hex}", _MODEL, _CONTENT, _USAGE)

    def respond(request: Request) -> Reply:
        return Reply(chunks=upstream_frames, content_type="text/event-stream")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", {**_client_body(stream=True), "model": model})
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
    assert _content_events(response.text) == _content_events(b"".join(upstream_frames).decode()), response.text
    assert _streamed_thinking_tokens(response.text) == (_THINKING_TOKENS,), response.text


def test_non_streamed_thinking_blocks_and_thinking_token_count_reach_the_client_unchanged(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(body=cc.message_reply(f"msg_{uuid.uuid4().hex}", _MODEL, _CONTENT, _USAGE))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", {**_client_body(stream=False), "model": model})
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
    assert response.json()["content"] == list(_CONTENT), response.text
    assert response.json()["usage"]["output_tokens_details"] == _THINKING_TOKENS, response.text
