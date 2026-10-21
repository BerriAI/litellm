from __future__ import annotations

from types import MappingProxyType
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    LiteLLMParamsBody,
    OutMessage,
    ThinkingParam,
    ToolCall,
)
from passthrough_client import PassthroughClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

GEMINI_BACKEND: Final = "gemini/gemini-3.5-flash-lite"
MISTRAL_BACKEND: Final = "mistral/mistral-medium-3.5"
ANTHROPIC_BACKEND: Final = "anthropic/claude-haiku-4-5"
BEDROCK_CONVERSE_BACKEND: Final = "bedrock/converse/us.anthropic.claude-sonnet-5-5"
BEDROCK_LEGACY_THINKING_BACKEND: Final = "bedrock/converse/us.anthropic.claude-sonnet-4-6"

PROMPT: Final = "What is the weather in Paris and in Tokyo? Use the get_weather tool for each city."
CITY_TEMPERATURES: Final = MappingProxyType({"paris": "22", "tokyo": "31"})
THINKING: Final = ThinkingParam(type="enabled", budget_tokens=1024)

WEATHER_TOOL: Final = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
)


class _WeatherArgs(BaseModel):
    location: str


def _api_key_params(backend: str, env: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=backend, api_key=f"os.environ/{env}")


def _bedrock_params(backend: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=backend,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="os.environ/AWS_REGION",
    )


def _register(client: PassthroughClient, resources: ResourceManager, params: LiteLLMParamsBody) -> tuple[str, str]:
    model = f"e2e-chat-tool-loop-{unique_marker()}"
    model_id = client.proxy.create_model(model, params)
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model, resources.key()


def _choice(client: PassthroughClient, key: str, body: ChatBody) -> tuple[OutMessage, str | None]:
    response = unwrap(client.proxy.chat(key, body))
    choice = response.choices[0] if response.choices else None
    assert choice is not None and choice.message is not None, f"chat returned no message: {response}"
    return choice.message, choice.finish_reason


def _city_for(call: ToolCall) -> str:
    location = _WeatherArgs.model_validate_json(call.function.arguments or "").location.lower()
    city = next((city for city in CITY_TEMPERATURES if city in location), None)
    assert city is not None, f"get_weather called for a city the prompt never named: {location!r}"
    return city


def _assert_tool_results_reach_the_model(
    client: PassthroughClient, key: str, model: str, *, thinking: ThinkingParam | None, tool_choice: str | None
) -> None:
    first, finish_reason = _choice(
        client,
        key,
        ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=PROMPT)],
            tools=[WEATHER_TOOL],
            tool_choice=tool_choice,
            thinking=thinking,
            max_tokens=2048,
        ),
    )
    calls = tuple(call for call in first.tool_calls or () if call.function.name == "get_weather")
    assert calls and all(call.id for call in calls), f"model returned no addressable get_weather call: {first}"
    assert finish_reason == "tool_calls", f"a tool-calling turn must finish with tool_calls, got {finish_reason!r}"
    if thinking is not None:
        assert first.thinking_blocks, f"thinking was enabled but no thinking blocks came back: {first}"
    cities = tuple(_city_for(call) for call in calls)
    assert set(cities) == set(CITY_TEMPERATURES), (
        f"expected a get_weather call for every city {sorted(CITY_TEMPERATURES)}, got calls for {cities}"
    )
    temperatures = tuple(CITY_TEMPERATURES[city] for city in cities)

    answer, _ = _choice(
        client,
        key,
        ChatBody(
            model=model,
            messages=[
                ChatMessage(role="user", content=PROMPT),
                ChatAssistantTurn(
                    content=first.content, thinking_blocks=first.thinking_blocks, tool_calls=first.tool_calls
                ),
                *(
                    ChatToolResultTurn(tool_call_id=call.id or "", content=f"{temperature} degrees C and sunny")
                    for call, temperature in zip(calls, temperatures)
                ),
            ],
            tools=[WEATHER_TOOL],
            thinking=thinking,
            max_tokens=2048,
        ),
    )
    content = answer.content or ""
    assert all(temperature in content for temperature in temperatures), (
        f"the answer ignored the tool results {temperatures}: {content!r}"
    )


class TestChatToolResultRoundTrip:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.GEMINI,),
            models=(GEMINI_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_gemini(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model, key = _register(client, resources, _api_key_params(GEMINI_BACKEND, "GEMINI_API_KEY"))
        _assert_tool_results_reach_the_model(client, key, model, thinking=None, tool_choice="required")

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.MISTRAL,),
            models=(MISTRAL_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_mistral(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model, key = _register(client, resources, _api_key_params(MISTRAL_BACKEND, "MISTRAL_API_KEY"))
        _assert_tool_results_reach_the_model(client, key, model, thinking=None, tool_choice="required")

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_CONVERSE_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_bedrock_converse(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model, key = _register(client, resources, _bedrock_params(BEDROCK_CONVERSE_BACKEND))
        _assert_tool_results_reach_the_model(client, key, model, thinking=None, tool_choice="required")

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING, Capability.REASONING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_anthropic_with_extended_thinking(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model, key = _register(client, resources, _api_key_params(ANTHROPIC_BACKEND, "ANTHROPIC_API_KEY"))
        _assert_tool_results_reach_the_model(client, key, model, thinking=THINKING, tool_choice=None)

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_LEGACY_THINKING_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING, Capability.REASONING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_bedrock_converse_with_extended_thinking(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, _bedrock_params(BEDROCK_LEGACY_THINKING_BACKEND))
        _assert_tool_results_reach_the_model(client, key, model, thinking=THINKING, tool_choice=None)
