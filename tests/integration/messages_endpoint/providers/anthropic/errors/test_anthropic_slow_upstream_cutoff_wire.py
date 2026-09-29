import json
import time
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.provider_wire.anthropic.messages_request_timeout_reaches_transport")
def test_anthropic_messages_slow_upstream_is_cut_off_at_the_deployment_request_timeout(gateway: Gateway) -> None:
    identity: Final = "anthropic-timeout-" + uuid.uuid4().hex
    prompt: Final = f"slow answer {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        body: Final = json.loads(request.body)
        assert body["model"] == "claude-sonnet-4-5-20250929"
        assert body["max_tokens"] == 16
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert not {
            "timeout",
            "request_timeout",
            "stream_chunk_size",
            "litellm_params",
            "litellm_metadata",
            "rpm",
            "tpm",
        }.intersection(body)
        time.sleep(1.5)
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "late"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            request_timeout=0.3,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]},
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 408, response.text
        assert "Timeout" in response.json()["error"]["message"], response.text
        assert eventually(wire.drain, lambda requests: len(requests) == 1, seconds=5, return_last_on_timeout=True)
