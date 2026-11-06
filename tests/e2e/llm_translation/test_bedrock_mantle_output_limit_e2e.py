"""Live e2e for output-token limits on a Bedrock Mantle GPT-5.6 chat deployment.

A deployment registered with model_info.mode "chat" and an explicit Mantle
OpenAI-compatible api_base is served through Mantle chat completions. Mantle
rejects max_tokens for the GPT-5.x models there and only accepts
max_completion_tokens, so a caller that sets an output limit, or a deployment
that stores one, must still get a completion back instead of the upstream 400
"Unsupported parameter: 'max_tokens' is not supported with this model."
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from llm_translation.sdk_clients import NO_PROXY_CACHE, SdkClients
from models import LiteLLMParamsBody, ModelInfoBody, ModelNewBody
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall
from passthrough_client import PassthroughClient
from pydantic import BaseModel, ConfigDict

pytestmark: Final = pytest.mark.e2e

MANTLE_BACKEND: Final = "bedrock_mantle/openai.gpt-5.6-terra"
MANTLE_CHAT_API_BASE: Final = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
OUTPUT_LIMIT: Final = 1000
PROMPT: Final = "reply with one word"
WEATHER_PROMPT: Final = "What is the weather in Paris?"
WEATHER_TOOL: Final[ChatCompletionToolParam] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _register_mantle_chat_model(
    client: PassthroughClient, resources: ResourceManager, max_tokens: int | None = None
) -> str:
    model: Final = f"e2e-mantle-chat-output-limit-{unique_marker()}"
    model_id: Final = client.proxy.register_model(
        ModelNewBody(
            model_name=model,
            litellm_params=LiteLLMParamsBody(
                model=MANTLE_BACKEND,
                custom_llm_provider="bedrock_mantle",
                api_key="os.environ/AWS_BEARER_TOKEN_BEDROCK",
                api_base=MANTLE_CHAT_API_BASE,
                reasoning_effort="none",
                max_tokens=max_tokens,
            ),
            model_info=ModelInfoBody(mode="chat", supported_endpoints=["/v1/chat/completions", "/v1/responses"]),
        )
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model


def _user_message(text: str) -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": text}


def _assert_completion_within_limit(completion: ChatCompletion) -> None:
    assert completion.choices[0].message.content, f"empty completion: {completion.model_dump_json()[:400]}"
    assert completion.usage is not None, f"completion carries no usage: {completion.model_dump_json()[:400]}"
    assert completion.usage.completion_tokens <= OUTPUT_LIMIT, f"limit not applied: {completion.usage}"


class _WeatherArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    city: str


def _streamed_choices(chunks: tuple[ChatCompletionChunk, ...]) -> Iterator[Choice]:
    for chunk in chunks:
        yield from chunk.choices


def _streamed_tool_calls(chunks: tuple[ChatCompletionChunk, ...]) -> Iterator[ChoiceDeltaToolCall]:
    for choice in _streamed_choices(chunks):
        yield from choice.delta.tool_calls or ()


def _streamed_tool_call(chunks: tuple[ChatCompletionChunk, ...]) -> tuple[str, str]:
    calls: Final = tuple(call.function for call in _streamed_tool_calls(chunks) if call.function is not None)
    name: Final = "".join(call.name or "" for call in calls)
    arguments: Final = "".join(call.arguments or "" for call in calls)
    return name, arguments


class TestBedrockMantleChatOutputLimit:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK_MANTLE,),
            models=(MANTLE_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_client_max_tokens_returns_completion(
        self, client: PassthroughClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register_mantle_chat_model(client, resources)
        completion: Final = sdk.openai(resources.key()).chat.completions.create(
            model=model,
            messages=[_user_message(PROMPT)],
            max_tokens=OUTPUT_LIMIT,
            extra_body=NO_PROXY_CACHE,
        )
        _assert_completion_within_limit(completion)

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK_MANTLE,),
            models=(MANTLE_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_client_max_completion_tokens_returns_completion(
        self, client: PassthroughClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register_mantle_chat_model(client, resources)
        completion: Final = sdk.openai(resources.key()).chat.completions.create(
            model=model,
            messages=[_user_message(PROMPT)],
            max_completion_tokens=OUTPUT_LIMIT,
            extra_body=NO_PROXY_CACHE,
        )
        _assert_completion_within_limit(completion)

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK_MANTLE,),
            models=(MANTLE_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING, Capability.TOOL_CHOICE),
            mode=Mode.STREAM,
        )
    )
    def test_deployment_max_tokens_streams_tool_call_completion(
        self, client: PassthroughClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register_mantle_chat_model(client, resources, max_tokens=OUTPUT_LIMIT)
        chunks: Final = tuple(
            sdk.openai(resources.key()).chat.completions.create(
                model=model,
                messages=[_user_message(WEATHER_PROMPT)],
                stream=True,
                stream_options={"include_usage": True},
                tools=[WEATHER_TOOL],
                tool_choice="required",
                extra_body=NO_PROXY_CACHE,
            )
        )
        finish_reasons: Final = tuple(choice.finish_reason for choice in _streamed_choices(chunks))
        assert "tool_calls" in finish_reasons, f"stream did not finish on a tool call: {finish_reasons}"
        assert any(chunk.usage is not None for chunk in chunks), "stream carried no usage chunk"
        name, arguments = _streamed_tool_call(chunks)
        assert name == "get_weather", f"streamed tool call named {name!r}: {chunks[:5]}"
        assert _WeatherArgs.model_validate_json(arguments).city.strip(), (
            f"streamed tool call arguments missing city: {arguments!r}"
        )
