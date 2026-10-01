from collections.abc import Sequence

import pytest
from anthropic import Anthropic, Omit, omit
from anthropic.types import MessageParam, ToolParam
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

# Not the family's latest: on gemini-3.x, countTokens leaves out tool and image overhead generateContent bills,
# so count != bill for reasons outside litellm (measured against Vertex AI countTokens/generateContent, 2026-09-26)
BACKEND_MODEL = "gemini/gemini-2.5-flash"
GEMINI_API_KEY = "os.environ/GEMINI_API_KEY"

WEATHER_TOOL: ToolParam = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def _provision(proxy: ProxyClient, resources: ResourceManager) -> str:
    model_name = f"e2e-count-tokens-gemini-{unique_marker()}"
    model_id = proxy.create_model(
        model_name,
        LiteLLMParamsBody(model=BACKEND_MODEL, api_key=GEMINI_API_KEY),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name


def _assert_count_matches_bill(
    client: Anthropic,
    model: str,
    messages: Sequence[MessageParam],
    system: str | Omit = omit,
    tools: Sequence[ToolParam] | Omit = omit,
) -> None:
    counted = client.messages.count_tokens(model=model, messages=messages, system=system, tools=tools)
    billed = client.messages.create(
        model=model, max_tokens=16, messages=messages, system=system, tools=tools, extra_body=NO_PROXY_CACHE
    )

    assert counted.input_tokens == billed.usage.input_tokens > 0, (
        f"/v1/messages/count_tokens returned {counted.input_tokens} but /v1/messages billed "
        f"{billed.usage.input_tokens} input tokens for the same request"
    )


class TestMessagesCountTokens:
    @pytest.mark.covers("llm.messages.gemini.count_tokens.nonstream.works")
    def test_count_tokens_gemini_matches_billed_input_tokens(
        self, proxy: ProxyClient, resources: ResourceManager, scoped_key: str, sdk: SdkClients
    ) -> None:
        _assert_count_matches_bill(
            sdk.anthropic(scoped_key),
            _provision(proxy, resources),
            [{"role": "user", "content": f"hello world {unique_marker()}"}],
        )

    @pytest.mark.covers("llm.messages.gemini.count_tokens.nonstream.works")
    def test_count_tokens_gemini_counts_system_and_tools_like_the_bill(
        self, proxy: ProxyClient, resources: ResourceManager, scoped_key: str, sdk: SdkClients
    ) -> None:
        _assert_count_matches_bill(
            sdk.anthropic(scoped_key),
            _provision(proxy, resources),
            [{"role": "user", "content": f"What is the weather in Paris? {unique_marker()}"}],
            system="You are a helpful assistant",
            tools=[WEATHER_TOOL],
        )

    @pytest.mark.covers("llm.messages.gemini.count_tokens.nonstream.works")
    def test_count_tokens_gemini_counts_tool_history_like_the_bill(
        self, proxy: ProxyClient, resources: ResourceManager, scoped_key: str, sdk: SdkClients
    ) -> None:
        _assert_count_matches_bill(
            sdk.anthropic(scoped_key),
            _provision(proxy, resources),
            [
                {"role": "user", "content": f"What is the weather in Paris? {unique_marker()}"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "text", "text": "One moment."},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]},
            ],
            tools=[WEATHER_TOOL],
        )
