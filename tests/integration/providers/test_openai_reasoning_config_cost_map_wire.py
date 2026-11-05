"""A GPT reasoning series the name markers do not know, declared only by a cost-map row, forwards
`reasoning_effort: "none"` alongside function tools instead of dropping it on the plain chat config."""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.templates.openai import OPENAI_CHAT_AGENT_RESPONSE
from pydantic import JsonValue

SERIES: Final = "gpt-7-nova"
MODEL_INFO: Final = {
    "mode": "chat",
    "input_cost_per_token": 1e-6,
    "output_cost_per_token": 1e-6,
    "supports_function_calling": True,
    "supports_reasoning": True,
    "supports_none_reasoning_effort": True,
}
WEATHER_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
)
TOOL_CALL_REPLY: Final = {
    **OPENAI_CHAT_AGENT_RESPONSE,
    "id": "chatcmpl-nova",
    "created": 1700000000,
    "model": SERIES,
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
}
ROWS: Final = (
    ("openai", f"openai/{SERIES}", "/v1/chat/completions"),
    ("azure", f"azure/{SERIES}", "/openai/v1/chat/completions"),
)


def _deploy(scenario: Scenario, model: str, api_base: str) -> str:
    if model.startswith("azure/"):
        return scenario.model(model=model, api_base=api_base, api_version="v1", model_info=MODEL_INFO)
    return scenario.model(model=model, api_base=f"{api_base}/v1", model_info=MODEL_INFO)


@pytest.mark.parametrize(("provider_name", "model", "target"), ROWS, ids=[row[0] for row in ROWS])
def test_cost_map_only_series_forwards_reasoning_effort_none_with_tools(
    provider_name: str, model: str, target: str, gateway: Gateway, scenario: Scenario, provider: SharedProvider
) -> None:
    deployment: Final = _deploy(scenario, model, provider.wire.url)
    provider.expect(Reply(body=json.dumps(TOOL_CALL_REPLY).encode()))
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": deployment,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [dict(WEATHER_TOOL)],
            "reasoning_effort": "none",
        },
    )
    assert response.status_code == 200, response.text
    received: Final = provider.received()
    assert [(request.method, request.target) for request in received] == [("POST", target)]
    body: Final = JSON_OBJECT.validate_json(received[0].body)
    assert body["reasoning_effort"] == "none", body
    tools: Final = body["tools"]
    assert isinstance(tools, list), body
    assert [object_value(object_value(tool)["function"])["name"] for tool in tools] == ["get_weather"], body
