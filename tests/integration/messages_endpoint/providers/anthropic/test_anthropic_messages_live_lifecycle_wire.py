import json
import threading
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "claude-sonnet-4-5-20250929"
_API_KEY: Final = "synthetic-anthropic-key"


def _sse(event: str, payload: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def test_messages_stream_message_start_reaches_client_before_content_without_fallback(
    gateway: Gateway,
) -> None:
    """With no fallback able to take over, the proxy must not hold lifecycle
    frames back for a retry that cannot happen: message_start reaches the
    client while the upstream is still thinking."""
    gate: Final = threading.Event()
    head: Final = _sse("message_start", {"type": "message_start", "message": {"id": "msg_live_1"}}) + _sse(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )
    tail: Final = (
        _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        )
        + _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        + _sse(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
        )
        + _sse("message_stop", {"type": "message_stop"})
    )
    prompt: Final = "live-lifecycle-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == _API_KEY
        body: Final = json.loads(request.body)
        assert body["model"] == _MODEL
        assert body["stream"] is True
        assert body["messages"] == [{"role": "user", "content": prompt}]
        return Reply(content_type="text/event-stream", chunks=(head, tail), gate_after_first=gate)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            lines = response.iter_lines()
            first_event: Final = next(
                json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")
            )
            assert first_event["type"] == "message_start"
            gate.set()
            events: Final = (first_event,) + tuple(
                json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")
            )
    assert tuple(event["type"] for event in events) == (
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ), f"observed events: {events!r}"
    assert [request.target for request in wire.drain()] == ["/v1/messages"]
