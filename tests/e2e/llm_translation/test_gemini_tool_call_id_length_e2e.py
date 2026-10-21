from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from llm_translation.sdk_clients import NO_PROXY_CACHE, SdkClients
from models import LiteLLMParamsBody
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

BACKEND_MODEL: Final = "gemini/gemini-3.5-flash"
GEMINI_API_KEY: Final = "os.environ/GEMINI_API_KEY"
# bedrock-agentcore HarnessToolUseId shape in botocore 1.43.6, read 2026-10-20
AGENTCORE_TOOL_USE_ID: Final = re.compile(r"[a-zA-Z0-9_-]{1,64}")
WEATHER_TOOL: Final[ChatCompletionToolParam] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _subject(mode: Mode) -> Subject:
    return Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.CHAT_COMPLETIONS,
        providers=(Provider.GEMINI,),
        models=(BACKEND_MODEL,),
        mode=mode,
        capabilities=(Capability.FUNCTION_CALLING, Capability.TOOL_CHOICE),
    )


def _register_model(client: PassthroughClient, resources: ResourceManager) -> str:
    model_name: Final = f"e2e-gemini-tool-call-id-{unique_marker()}"
    model_id: Final = client.proxy.create_model(
        model_name,
        LiteLLMParamsBody(
            model=BACKEND_MODEL,
            api_key=GEMINI_API_KEY,
            thought_signature_in_tool_call_id=False,
        ),
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model_name


def _user_message() -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": f"Use get_weather for San Francisco and return its result. {unique_marker()}"}


def _tool_calls(completion: ChatCompletion) -> tuple[ChatCompletionMessageFunctionToolCall, ...]:
    assert completion.choices, f"Gemini returned no choices: {completion.model_dump_json()[:400]}"
    calls: Final = tuple(
        call
        for call in completion.choices[0].message.tool_calls or ()
        if isinstance(call, ChatCompletionMessageFunctionToolCall)
    )
    assert calls, f"Gemini returned no tool calls for tool_choice='required': {completion.model_dump_json()[:400]}"
    return calls


def _assert_agentcore_accepts_ids(ids: tuple[str, ...]) -> None:
    rejected: Final = tuple(call_id for call_id in ids if not AGENTCORE_TOOL_USE_ID.fullmatch(call_id))
    assert not rejected, f"tool call id(s) AgentCore would reject: {tuple(call_id[:80] for call_id in rejected)}"


def _streamed_tool_calls(chunks: tuple[ChatCompletionChunk, ...]) -> Iterator[ChoiceDeltaToolCall]:
    for chunk in chunks:
        for choice in chunk.choices:
            yield from choice.delta.tool_calls or ()


def _short_id_tool_call(call: ChatCompletionMessageFunctionToolCall) -> ChatCompletionMessageFunctionToolCallParam:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments},
    }


def _tool_result(call: ChatCompletionMessageFunctionToolCall) -> ChatCompletionToolMessageParam:
    return {"role": "tool", "tool_call_id": call.id, "content": '{"temperature": 21, "condition": "sunny"}'}


class TestGeminiToolCallIdLength:
    @pytest.mark.covers(
        "llm.chat_completions.gemini.tool_use.nonstream.works",
        exercised_on=["chat_completions"],
    )
    @meta(_subject(Mode.NONSTREAM))
    def test_non_streaming_tool_call_id_fits_agentcore(
        self, client: PassthroughClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register_model(client, resources)
        completion: Final = sdk.openai(resources.key()).chat.completions.create(
            model=model,
            messages=[_user_message()],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=128,
            extra_body=NO_PROXY_CACHE,
        )

        _assert_agentcore_accepts_ids(tuple(call.id for call in _tool_calls(completion)))

    @pytest.mark.covers(
        "llm.chat_completions.gemini.tool_use.stream.works",
        exercised_on=["chat_completions"],
    )
    @meta(_subject(Mode.STREAM))
    def test_streaming_tool_call_id_fits_agentcore(
        self, client: PassthroughClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register_model(client, resources)
        chunks: Final[tuple[ChatCompletionChunk, ...]] = tuple(
            sdk.openai(resources.key()).chat.completions.create(
                model=model,
                messages=[_user_message()],
                tools=[WEATHER_TOOL],
                tool_choice="required",
                max_tokens=128,
                stream=True,
                extra_body=NO_PROXY_CACHE,
            )
        )
        ids: Final = tuple(call.id for call in _streamed_tool_calls(chunks) if call.id)
        assert ids, f"Gemini stream returned no tool call ids in {len(chunks)} chunks"

        _assert_agentcore_accepts_ids(ids)

    @pytest.mark.covers(
        "llm.chat_completions.gemini.multi_turn.nonstream.works",
        exercised_on=["chat_completions"],
    )
    @meta(_subject(Mode.NONSTREAM))
    def test_tool_call_round_trip_succeeds(
        self, client: PassthroughClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register_model(client, resources)
        openai_client: Final = sdk.openai(resources.key())
        question: Final = _user_message()
        first: Final = openai_client.chat.completions.create(
            model=model,
            messages=[question],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=128,
            extra_body=NO_PROXY_CACHE,
        )
        calls: Final = _tool_calls(first)
        assistant: Final[ChatCompletionAssistantMessageParam] = {
            "role": "assistant",
            "tool_calls": [_short_id_tool_call(call) for call in calls],
        }

        second: Final = openai_client.chat.completions.create(
            model=model,
            messages=[question, assistant, *(_tool_result(call) for call in calls)],
            tools=[WEATHER_TOOL],
            tool_choice="none",
            max_tokens=128,
            extra_body=NO_PROXY_CACHE,
        )

        assert second.choices, f"Gemini turn 2 returned no choices: {second.model_dump_json()[:400]}"
        answer: Final = second.choices[0].message.content
        assert answer and answer.strip(), f"Gemini turn 2 returned an empty answer: {second.model_dump_json()[:400]}"
