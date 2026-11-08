"""A chat completions ``tool_choice`` that forces an OpenAI hosted tool, such as ``{"type": "web_search"}``, reaches the
Responses API unchanged, and is a 400 before any provider call where the provider cannot enforce it."""

from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value
from integration._support.openai_wire import responses_reply
from integration._support.provider import SharedProvider
from pydantic import JsonValue

_RESPONSES_MODEL: Final = "openai/responses/gpt-5.6-sol"
_ANTHROPIC_MODEL: Final = "anthropic/claude-haiku-4-5"
_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}


def _choice_id(tool_choice: dict[str, JsonValue]) -> str:
    return str(tool_choice["type"])


def _chat_request(model: str, tool_choice: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
        "tools": [_WEATHER_TOOL, {"type": tool_choice["type"]}],
        "tool_choice": tool_choice,
    }


@pytest.mark.parametrize(
    "tool_choice",
    [{"type": "web_search"}, {"type": "web_search_preview"}, {"type": "file_search"}],
    ids=_choice_id,
)
def test_hosted_tool_choice_reaches_the_responses_api_unchanged(
    gateway: Gateway, provider: SharedProvider, tool_choice: dict[str, JsonValue]
) -> None:
    provider.expect(responses_reply("resp_hosted_tool_choice", "gpt-5.6-sol", "It is 18C in Paris.", stream=False))
    response: Final = gateway.request("POST", "/v1/chat/completions", _chat_request(_RESPONSES_MODEL, tool_choice))
    assert response.status_code == 200, response.text
    assert "It is 18C in Paris." in response.text, response.text
    received: Final = provider.received()
    assert [(request.method, request.target) for request in received] == [("POST", "/responses")]
    sent: Final = JSON_OBJECT.validate_json(received[0].body)
    assert sent["tool_choice"] == tool_choice
    assert {"type": tool_choice["type"]} in list_value(sent["tools"]), sent["tools"]


def test_hosted_tool_choice_is_a_400_before_any_call_where_the_provider_cannot_enforce_it(
    gateway: Gateway, provider: SharedProvider
) -> None:
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", _chat_request(_ANTHROPIC_MODEL, {"type": "web_search"})
    )
    assert response.status_code == 400, response.text
    assert "web_search" in response.text and "only supported for" in response.text, response.text
    assert provider.received() == ()
