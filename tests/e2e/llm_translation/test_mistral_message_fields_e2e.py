from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from openai import OpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

BACKEND: Final = "mistral/ministral-3b-latest"
TOOL_CALL_ID: Final = "D681PevKs"
WEATHER_TOOL: Final[ChatCompletionToolParam] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
SUBJECT: Final = Subject(
    domain=Domain.LLM_TRANSLATION,
    route=Route.CHAT_COMPLETIONS,
    providers=(Provider.MISTRAL,),
    models=(BACKEND,),
    mode=Mode.NONSTREAM,
)
TOOL_SUBJECT: Final = Subject(
    domain=Domain.LLM_TRANSLATION,
    route=Route.CHAT_COMPLETIONS,
    providers=(Provider.MISTRAL,),
    models=(BACKEND,),
    capabilities=(Capability.FUNCTION_CALLING,),
    mode=Mode.NONSTREAM,
)


class _CopilotAssistantMessage(ChatCompletionAssistantMessageParam):
    cot_id: str
    cot_summary: str


class _UserMessageWithMetadata(ChatCompletionUserMessageParam):
    metadata: dict[str, str]


class _NamedToolMessage(ChatCompletionToolMessageParam):
    name: str


class _PrefixAssistantMessage(ChatCompletionAssistantMessageParam):
    prefix: bool


def _register(
    proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, *, drop_params: bool | None
) -> tuple[str, OpenAI]:
    model: Final = f"e2e-mistral-message-fields-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model, LiteLLMParamsBody(model=BACKEND, api_key="os.environ/MISTRAL_API_KEY", drop_params=drop_params)
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, sdk.openai(resources.key())


def _assert_answers(
    client: OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    *,
    extra_body: Mapping[str, object] = NO_PROXY_CACHE,
    tools: Sequence[ChatCompletionToolParam] = (),
) -> None:
    response: Final = client.chat.completions.create(
        model=model,
        messages=list(messages),
        max_tokens=32,
        tools=list(tools) if tools else omit,
        extra_body=dict(extra_body),
    )
    assert response.choices and response.choices[0].message.content, f"Mistral returned no answer: {response!r}"


class TestMistralClientOnlyMessageFields:
    @meta(SUBJECT)
    def test_deployment_drop_params_lets_a_copilot_conversation_reach_mistral(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, client = _register(proxy, resources, sdk, drop_params=True)
        _assert_answers(
            client,
            model,
            [
                {"role": "user", "content": f"Say hi. {unique_marker()}"},
                _CopilotAssistantMessage(
                    role="assistant", content="Hi!", cot_id="cot-1f6b2c", cot_summary="Greet the user briefly."
                ),
                {"role": "user", "content": "Now say bye in one word."},
            ],
        )

    @meta(SUBJECT)
    def test_request_drop_params_lets_user_message_metadata_reach_mistral(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, client = _register(proxy, resources, sdk, drop_params=None)
        _assert_answers(
            client,
            model,
            [
                _UserMessageWithMetadata(
                    role="user", content=f"Say hi. {unique_marker()}", metadata={"source": "chat"}
                ),
                {"role": "assistant", "content": "Hi!"},
                _UserMessageWithMetadata(role="user", content="Now say bye in one word.", metadata={"source": "chat"}),
            ],
            extra_body={**NO_PROXY_CACHE, "drop_params": True},
        )

    @meta(TOOL_SUBJECT)
    def test_tool_round_trip_with_tool_name_still_reaches_mistral_when_drop_params_is_off(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, client = _register(proxy, resources, sdk, drop_params=None)
        _assert_answers(
            client,
            model,
            [
                {"role": "user", "content": f"What is the weather in Paris? {unique_marker()}"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": TOOL_CALL_ID,
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                _NamedToolMessage(
                    role="tool", tool_call_id=TOOL_CALL_ID, name="get_weather", content="22 degrees C and sunny"
                ),
            ],
            tools=[WEATHER_TOOL],
        )

    @meta(SUBJECT)
    def test_assistant_prefix_turn_still_reaches_mistral_when_drop_params_is_off(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, client = _register(proxy, resources, sdk, drop_params=None)
        _assert_answers(
            client,
            model,
            [
                {"role": "user", "content": f"Name the capital of France. {unique_marker()}"},
                _PrefixAssistantMessage(role="assistant", content="The capital of France is", prefix=True),
            ],
        )
