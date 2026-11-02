import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "accounts/fireworks/models/glm-5p3"
_API_KEY: Final = "synthetic-fireworks-key"
_STOP: Final = "</block>"
_ANSWER: Final = "<verdict>allow</verdict>"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@pytest.mark.covers(
    "providers.anthropic_messages_adapter.stop_sequences_and_disabled_thinking_reach_openai_compatible_provider_as_stop_and_reasoning_effort"
)
def test_messages_stop_sequences_to_fireworks_are_sent_as_stop_not_stop_sequences(gateway: Gateway) -> None:
    prompt: Final = "classify this tool call " + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "stop_sequences" not in body, body
        assert body["stop"] == [_STOP], body
        assert body["reasoning_effort"] == "none", body
        assert body["model"] == _MODEL, body
        assert body["messages"] == [{"role": "user", "content": prompt}], body
        return Reply(
            body=json.dumps(
                {
                    "id": "fw-classifier",
                    "object": "chat.completion",
                    "created": 1,
                    "model": _MODEL,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": _ANSWER}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fireworks_ai/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
                "stop_sequences": [_STOP],
                "thinking": {"type": "disabled"},
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["content"] == [{"type": "text", "text": _ANSWER}], response.text
        assert payload["stop_reason"] == "end_turn", response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
