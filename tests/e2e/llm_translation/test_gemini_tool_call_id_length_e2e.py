from __future__ import annotations

from itertools import chain
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import Success, unwrap
from lifecycle import ResourceManager
from models import (
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    LiteLLMParamsBody,
    ToolCall,
)
from passthrough_client import PassthroughClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

BACKEND_MODEL: Final = "gemini/gemini-3.5-flash"
GEMINI_API_KEY: Final = "os.environ/GEMINI_API_KEY"
MAX_TOOL_CALL_ID_LENGTH: Final = 128
WEATHER_TOOL: Final = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
)


class _StreamToolCall(BaseModel):
    id: str | None = None


class _StreamDelta(BaseModel):
    tool_calls: tuple[_StreamToolCall, ...] = ()


class _StreamChoice(BaseModel):
    delta: _StreamDelta


class _StreamChunk(BaseModel):
    choices: tuple[_StreamChoice, ...] = ()


def _register_model(client: PassthroughClient, resources: ResourceManager) -> str:
    model_name: Final = f"e2e-gemini-tool-call-id-{unique_marker()}"
    model_id: Final = client.proxy.create_model(
        model_name,
        LiteLLMParamsBody(model=BACKEND_MODEL, api_key=GEMINI_API_KEY),
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model_name


def _tool_request(model: str, *, stream: bool = False) -> ChatBody:
    return ChatBody(
        model=model,
        messages=(
            ChatMessage(
                role="user",
                content=f"Use get_weather for San Francisco and return its result. {unique_marker()}",
            ),
        ),
        tools=(WEATHER_TOOL,),
        tool_choice="required",
        max_tokens=128,
        stream=stream,
    )


def _tool_calls(response: ChatResponse) -> tuple[ToolCall, ...]:
    assert response.choices, f"Gemini returned no choices: {response}"
    message: Final = response.choices[0].message
    calls: Final = tuple(message.tool_calls or ()) if message else ()
    assert calls, f"Gemini returned no tool calls for tool_choice='required': {response}"
    return calls


def _tool_call_id(call: ToolCall) -> str:
    assert call.id, f"Gemini returned a tool call without an id: {call}"
    return call.id


def _assert_id_lengths(ids: tuple[str, ...]) -> None:
    lengths: Final = tuple(len(call_id) for call_id in ids)
    assert all(length <= MAX_TOOL_CALL_ID_LENGTH for length in lengths), (
        f"tool call id length(s)={lengths}; each id must be at most "
        f"{MAX_TOOL_CALL_ID_LENGTH} characters"
    )


def _stream_tool_calls(events: list[str]) -> tuple[_StreamToolCall, ...]:
    chunks: Final = tuple(_StreamChunk.model_validate_json(event) for event in events)
    choices: Final = chain.from_iterable(chunk.choices for chunk in chunks)
    return tuple(chain.from_iterable(choice.delta.tool_calls for choice in choices))


class TestGeminiToolCallIdLength:
    @pytest.mark.covers(
        "llm.chat_completions.gemini.tool_use.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_non_streaming_tool_call_id_fits_128_chars(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model: Final = _register_model(client, resources)
        response: Final = unwrap(
            client.proxy.chat(resources.key(), _tool_request(model))
        )
        calls: Final = _tool_calls(response)
        ids: Final = tuple(_tool_call_id(call) for call in calls)

        _assert_id_lengths(ids)

    @pytest.mark.covers(
        "llm.chat_completions.gemini.tool_use.stream.works",
        exercised_on=["chat_completions"],
    )
    def test_streaming_tool_call_id_fits_128_chars(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model: Final = _register_model(client, resources)
        result: Final = client.proxy.chat_stream(
            resources.key(), _tool_request(model, stream=True)
        )
        assert result.ok, f"Gemini chat stream failed: {result}"
        assert result.is_streaming, f"Gemini response was not SSE: {result.content_type!r}"
        assert result.stream_error is None, f"Gemini stream contained an error: {result.stream_error}"

        calls: Final = _stream_tool_calls(result.stream_events)
        assert calls, f"Gemini stream returned no tool call deltas: {result.stream_events[:5]}"
        ids: Final = tuple(call.id for call in calls if call.id)
        assert ids, f"Gemini stream tool call deltas contained no ids: {calls}"

        _assert_id_lengths(ids)

    @pytest.mark.covers(
        "llm.chat_completions.gemini.multi_turn.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_tool_call_round_trip_succeeds(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model: Final = _register_model(client, resources)
        key: Final = resources.key()
        first_request: Final = _tool_request(model)
        first_response: Final = unwrap(client.proxy.chat(key, first_request))
        calls: Final = _tool_calls(first_response)
        first_message: Final = first_response.choices[0].message
        assert first_message is not None, f"Gemini returned no assistant message: {first_response}"
        ids: Final = tuple(_tool_call_id(call) for call in calls)

        second_result: Final = client.proxy.chat(
            key,
            ChatBody(
                model=model,
                messages=(
                    *first_request.messages,
                    ChatAssistantTurn(
                        content=first_message.content,
                        reasoning_content=first_message.reasoning_content,
                        tool_calls=first_message.tool_calls,
                    ),
                    *tuple(
                        ChatToolResultTurn(
                            tool_call_id=call_id,
                            content='{"temperature": 21, "condition": "sunny"}',
                        )
                        for call_id in ids
                    ),
                ),
                tools=(WEATHER_TOOL,),
                tool_choice="none",
                max_tokens=128,
            ),
        )
        assert isinstance(second_result, Success), f"Gemini turn 2 failed: {second_result}"
        assert second_result.status_code == 200, (
            f"Gemini turn 2 returned HTTP {second_result.status_code}, expected 200"
        )
        second_response: Final = second_result.data
        assert second_response.choices, f"Gemini turn 2 returned no choices: {second_response}"
        answer: Final = second_response.choices[0].message
        assert answer is not None and answer.content and answer.content.strip(), (
            f"Gemini turn 2 returned an empty answer: {second_response}"
        )
