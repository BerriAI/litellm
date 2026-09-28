import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


def _responses_stream(identity: str, text: str) -> tuple[bytes, ...]:
    completed: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.3-codex",
        "output": [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    events: Final = (
        {"type": "response.created", "response": {**completed, "status": "in_progress", "output": [], "usage": None}},
        {
            "type": "response.output_text.delta",
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.completed", "response": completed},
    )
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


@pytest.mark.covers("providers.responses_bridge.always_include_stream_usage_keeps_include_usage_off_the_responses_wire")
def test_messages_stream_with_always_include_stream_usage_omits_include_usage_from_responses_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "responses-stream-options-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        return Reply(content_type="text/event-stream", chunks=_responses_stream(identity, "usage control"))

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"].update({"always_include_stream_usage": True})
    path: Final = tmp_path / "always_include_stream_usage.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        wire_server(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-5.3-codex", api_base=wire.url, api_key="synthetic-openai-key")
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": f"count the usage {identity}"}],
            },
        )
        assert response.status_code == 200, response.text
        assert "event: message_stop" in response.text, response.text
        requests: Final = wire.drain()
        assert len(requests) == 1, response.text
        assert json.loads(requests[0].body) == {
            "model": "gpt-5.3-codex",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"count the usage {identity}"}],
                }
            ],
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 64,
            "stream": True,
        }, response.text
