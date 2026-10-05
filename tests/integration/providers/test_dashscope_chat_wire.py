import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "qwen3.7-plus"
_API_KEY: Final = "synthetic-dashscope-key"
_PROMPT: Final = "What is 3^3?"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "27"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
        }
    ).encode()


@pytest.mark.covers("other.provider_wire.dashscope.reasoning_effort_reaches_provider")
def test_dashscope_chat_forwards_reasoning_effort_none_to_the_provider(gateway: Gateway) -> None:
    identity: Final = f"dashscope-reasoning-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": _BACKEND,
            "messages": [{"role": "user", "content": _PROMPT}],
            "reasoning_effort": "none",
        }
        return Reply(body=_completion(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"dashscope/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "reasoning_effort": "none"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "27", "provider_specific_fields": {"refusal": None}},
                "provider_specific_fields": {},
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
