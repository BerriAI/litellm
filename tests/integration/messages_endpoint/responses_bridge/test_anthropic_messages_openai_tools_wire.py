import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-openai-key"
_TOOL_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "include_forecast": {"type": "boolean"},
    },
    "required": ["city"],
}


@pytest.mark.covers("providers.anthropic_messages_bridge.optional_tool_properties_stay_optional_on_the_wire")
def test_messages_tool_with_optional_properties_reaches_openai_responses_non_strict(gateway: Gateway) -> None:
    identity: Final = f"messages-optional-tool-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = json.loads(request.body)
        assert body["model"] == _BACKEND, body
        assert body["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "strict": False,
                "description": "Current weather for a city",
                "parameters": _TOOL_SCHEMA,
            }
        ], body["tools"]
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": _BACKEND,
                    "output": [
                        {
                            "type": "function_call",
                            "id": f"fc_{identity}",
                            "call_id": f"call_{identity}",
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 30, "output_tokens": 9, "total_tokens": 39},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Current weather for a city",
                        "input_schema": _TOOL_SCHEMA,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
        body: Final = response.json()
        assert body["stop_reason"] == "tool_use", response.text
        assert body["content"] == [
            {
                "type": "tool_use",
                "id": f"call_{identity}",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ], response.text
