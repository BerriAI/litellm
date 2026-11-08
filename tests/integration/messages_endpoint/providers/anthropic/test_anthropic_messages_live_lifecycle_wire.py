import threading
import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.anthropic_sse import CONTENT_TAIL, message_start, sse
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL: Final = "claude-sonnet-5-5"
_API_KEY: Final = "synthetic-anthropic-key"


@pytest.mark.parametrize("block_type", ("text", "thinking"))
def test_messages_stream_lifecycle_reaches_client_before_content_without_fallback(
    gateway: Gateway,
    block_type: str,
) -> None:
    gate: Final = threading.Event()
    block_start: Final[Mapping[str, JsonValue]] = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": block_type, block_type: ""},
    }
    head: Final = (
        message_start("msg_live_1", _MODEL)
        + sse(
            "content_block_start",
            block_start,
        )
        + sse("ping", {"type": "ping"})
    )
    delta: Final[Mapping[str, JsonValue]] = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": f"{block_type}_delta", block_type: "Hello"},
    }
    tail: Final = (
        sse(
            "content_block_delta",
            delta,
        )
        + CONTENT_TAIL
    )
    prompt: Final = "live-lifecycle-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == _API_KEY
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body == {
            "model": _MODEL,
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": prompt}],
        }
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
            timeout=10,
        ) as response:
            assert response.status_code == 200, response.read().decode()
            events: Final = (
                JSON_OBJECT.validate_json(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            )
            try:
                start: Final = next(events)
                assert start["type"] == "message_start"
                assert next(events) == block_start
                assert next(events) == {"type": "ping"}
            finally:
                gate.set()
            remaining: Final = tuple(events)
    assert tuple(event["type"] for event in remaining) == (
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ), f"observed events: {remaining!r}"
    assert remaining[0] == delta
    assert [request.target for request in wire.drain()] == ["/v1/messages"]
