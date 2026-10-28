import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "qwen3-strict-tools"
_API_KEY: Final = "synthetic-hosted-vllm-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "record_code",
            "description": "Record a ten letter code",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "object",
                        "properties": {"value": {"type": "string", "pattern": "^Z{10}$"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }
]


@pytest.mark.parametrize("drop_params", [True, False])
def test_hosted_vllm_strict_tool_schema_reaches_the_wire_as_sent(gateway: Gateway, drop_params: bool) -> None:
    identity: Final = f"hosted-vllm-strict-tools-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["tools"] == _TOOLS, body["tools"]
        assert body["tool_choice"] == "auto", body
        return chat_reply(identity, _BACKEND, "recorded", stream=False)

    with wire_server(answering_model_discovery(respond)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY, drop_params=drop_params
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Record the code ZZZZZZZZZZ."}],
                "tools": _TOOLS,
                "tool_choice": "auto",
                "num_retries": 0,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"] == identity
