import json
import time
import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.wire import Reply, Request, wire_server

_UPSTREAM_STALL_SECONDS: Final = 4.0
_CONFIGURED_TIMEOUT_SECONDS: Final = 1.0


@pytest.mark.covers("providers.anthropic_messages.configured_timeout_aborts_stalled_upstream")
def test_messages_endpoint_honors_configured_timeout_against_stalled_upstream(gateway: Gateway) -> None:
    prompt: Final = "stall-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "claude-sonnet-4-5-20250929"
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert body["max_tokens"] == 16
        assert "timeout" not in body
        time.sleep(_UPSTREAM_STALL_SECONDS)
        return Reply(
            body=json.dumps(
                {
                    "id": "msg_stalled",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "too late"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            timeout=_CONFIGURED_TIMEOUT_SECONDS,
        )
        started: Final = time.monotonic()
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]},
        )
        elapsed: Final = time.monotonic() - started
        assert response.status_code == 408, response.text
        assert elapsed < _UPSTREAM_STALL_SECONDS, f"timed out only after {elapsed:.2f}s: {response.text}"
        assert len(wire.drain()) == 1
