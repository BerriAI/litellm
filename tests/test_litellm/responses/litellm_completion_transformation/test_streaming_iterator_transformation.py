import json
from collections.abc import Iterator, Sequence
from typing import Final, cast
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ReasoningSummaryPartDoneEvent,
    ReasoningSummaryTextDeltaEvent,
    ReasoningSummaryTextDoneEvent,
    ResponseCompletedEvent,
    ResponsePartAddedEvent,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

CHAT_COMPLETION_ID: Final = "chatcmpl-77d33d09-effa-4cd2-9c0d-c742d4358256"
RESPONSE_ID_EVENT_TYPES: Final = frozenset({"response.created", "response.in_progress", "response.completed"})


def _chunk(
    content: str | None,
    finish_reason: str | None = None,
    *,
    reasoning_content: str | None = None,
    tool_calls: Sequence[ChatCompletionDeltaToolCall] = (),
    chunk_id: str = CHAT_COMPLETION_ID,
) -> ModelResponseStream:
    return ModelResponseStream(
        id=chunk_id,
        created=1748575031,
        model="claude-haiku-4-5",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=list(tool_calls) if tool_calls else None,
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def _tool_delta(
    arguments: str,
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
) -> ChatCompletionDeltaToolCall:
    return ChatCompletionDeltaToolCall(
        index=index,
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


class _FakeStreamWrapper:
    def __init__(self, chunks: Sequence[ModelResponseStream]) -> None:
        self._chunks: Iterator[ModelResponseStream] = iter(chunks)
        self.logging_obj = MagicMock()

    def __iter__(self) -> "_FakeStreamWrapper":
        return self

    def __next__(self) -> ModelResponseStream:
        return next(self._chunks)

    def __aiter__(self) -> "_FakeStreamWrapper":
        return self

    async def __anext__(self) -> ModelResponseStream:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _build_iterator(
    chunks: Sequence[ModelResponseStream],
    *,
    responses_api_request: ResponsesAPIOptionalRequestParams | None = None,
    final_response: ModelResponse | None = None,
) -> LiteLLMCompletionStreamingIterator:
    iterator: Final = LiteLLMCompletionStreamingIterator(
        model="claude-haiku-4-5",
        litellm_custom_stream_wrapper=cast(litellm.CustomStreamWrapper, _FakeStreamWrapper(chunks)),
        request_input="What is the weather in San Francisco?",
        responses_api_request=responses_api_request if responses_api_request is not None else {},
        custom_llm_provider="anthropic",
        litellm_metadata={},
    )
    if final_response is not None:
        iterator.litellm_model_response = final_response
    return iterator


async def _collect_events(
    iterator: LiteLLMCompletionStreamingIterator, *, async_mode: bool
) -> list[BaseLiteLLMOpenAIResponseObject]:
    if async_mode:
        return [event async for event in iterator]
    return list(iterator)


def _response_ids(events: Sequence[BaseLiteLLMOpenAIResponseObject]) -> tuple[str, ...]:
    return tuple(event.response.id for event in events if getattr(event, "type", None) in RESPONSE_ID_EVENT_TYPES)


def _event_item(event: OutputItemAddedEvent | OutputItemDoneEvent) -> BaseLiteLLMOpenAIResponseObject:
    assert event.item is not None
    return event.item


def _assert_completed_output_alignment(events: Sequence[BaseLiteLLMOpenAIResponseObject]) -> None:
    completed_events: Final = tuple(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert len(completed_events) == 1
    completed: Final = completed_events[0]
    assert events[-1] is completed
    added_events: Final = tuple(event for event in events if isinstance(event, OutputItemAddedEvent))
    done_events: Final = tuple(event for event in events if isinstance(event, OutputItemDoneEvent))
    expected: Final = tuple((index, item.type, item.id) for index, item in enumerate(completed.response.output))
    assert (
        tuple((event.output_index, _event_item(event).type, _event_item(event).id) for event in added_events)
        == expected
    )
    assert (
        tuple(
            (event.output_index, event.item.type, event.item.id)
            for event in sorted(done_events, key=lambda event: event.output_index)
        )
        == expected
    )
    assert len({item.id for item in completed.response.output}) == len(expected)
    item_indexes: Final = tuple((index, item.id) for index, item in enumerate(completed.response.output))
    for event in events:
        if hasattr(event, "item_id"):
            assert (event.output_index, event.item_id) in item_indexes
    for added in added_events:
        assert added.item is not None
        done: Final = next(event for event in done_events if event.item.id == added.item.id)
        assert events.index(added) < events.index(done)
        for event in events:
            if getattr(event, "item_id", None) == added.item.id:
                assert event.output_index == added.output_index
                assert events.index(added) < events.index(event) < events.index(done)
        if added.item.type == "message":
            assert added.item.id.startswith("msg_")
        elif added.item.type == "reasoning":
            assert added.item.id.startswith("rs_")
    serialized: Final = tuple(json.loads(event.model_dump_json()) for event in events)
    assert tuple(event["sequence_number"] for event in serialized) == tuple(range(1, len(events) + 1))


def _assert_tool_lifecycle(
    events: Sequence[BaseLiteLLMOpenAIResponseObject],
    *,
    call_id: str,
    arguments: str,
    output_index: int,
    name: str,
    namespace: str | None = None,
) -> None:
    added_events: Final = tuple(
        event
        for event in events
        if isinstance(event, OutputItemAddedEvent)
        and event.item is not None
        and event.item.type == "function_call"
        and event.item.call_id == call_id
    )
    assert len(added_events) == 1
    added: Final = added_events[0]
    assert added.item is not None
    assert added.output_index == output_index
    assert added.item.id == f"fc_{call_id}"
    deltas: Final = tuple(
        event
        for event in events
        if isinstance(event, FunctionCallArgumentsDeltaEvent) and event.item_id == added.item.id
    )
    arguments_done: Final = tuple(
        event
        for event in events
        if isinstance(event, FunctionCallArgumentsDoneEvent) and event.item_id == added.item.id
    )
    item_done: Final = next(
        event for event in events if isinstance(event, OutputItemDoneEvent) and event.item.id == added.item.id
    )
    assert "".join(event.delta for event in deltas) == arguments
    assert len(arguments_done) == 1
    assert arguments_done[0].arguments == arguments
    assert events.index(added) < events.index(arguments_done[0]) < events.index(item_done)
    assert all(events.index(added) < events.index(event) < events.index(arguments_done[0]) for event in deltas)
    completed: Final = next(event for event in events if isinstance(event, ResponseCompletedEvent))
    snapshot: Final = completed.response.output[output_index]
    for item in (added.item, item_done.item, snapshot):
        assert item.id == added.item.id
        assert item.call_id == call_id
        assert item.name == name
        assert getattr(item, "namespace", None) == namespace
    assert item_done.item.arguments == snapshot.arguments == arguments


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
@pytest.mark.parametrize("combined", (False, True))
async def test_reasoning_and_content_preserve_ordered_lifecycle(async_mode: bool, combined: bool) -> None:
    chunks: Final = (
        (_chunk("Here is the answer", reasoning_content="First, let me analyze..."),)
        if combined
        else (
            _chunk("", reasoning_content="First, let me analyze..."),
            _chunk("Here is the answer"),
        )
    )
    events: Final = await _collect_events(
        _build_iterator((*chunks, _chunk(None, finish_reason="stop"))), async_mode=async_mode
    )
    payloads: Final = tuple(
        event for event in events if isinstance(event, (ReasoningSummaryTextDeltaEvent, OutputTextDeltaEvent))
    )
    assert tuple((event.type, event.delta) for event in payloads) == (
        (ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA, "First, let me analyze..."),
        (ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, "Here is the answer"),
    )
    assert tuple(event.type for event in events) == (
        ResponsesAPIStreamEvents.RESPONSE_CREATED,
        ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.RESPONSE_PART_ADDED,
        ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
        ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE,
        ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
        ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    )
    reasoning_part: Final = next(event for event in events if isinstance(event, ResponsePartAddedEvent))
    assert reasoning_part.item_id == payloads[0].item_id
    assert reasoning_part.output_index == payloads[0].output_index == 0
    assert getattr(reasoning_part, "summary_index", None) == 0
    assert reasoning_part.part == {"type": "summary_text", "text": ""}
    reasoning_done: Final = next(event for event in events if isinstance(event, ReasoningSummaryTextDoneEvent))
    assert reasoning_done.text == "First, let me analyze..."
    summary_done: Final = next(event for event in events if isinstance(event, ReasoningSummaryPartDoneEvent))
    assert summary_done.part.text == reasoning_done.text
    content_part: Final = next(event for event in events if isinstance(event, ContentPartAddedEvent))
    assert content_part.item_id == payloads[1].item_id
    assert content_part.output_index == payloads[1].output_index == 1
    content_done: Final = next(event for event in events if isinstance(event, ContentPartDoneEvent))
    assert content_done.part.type == "output_text"
    assert content_done.part.text == "Here is the answer"
    completed: Final = next(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert tuple(item.type for item in completed.response.output) == ("reasoning", "message")
    assert completed.response.output[0].content[0].text == "First, let me analyze..."
    assert completed.response.output[1].content[0].text == "Here is the answer"
    _assert_completed_output_alignment(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
async def test_content_only_first_payload_preserves_lifecycle_and_ids(async_mode: bool) -> None:
    events: Final = await _collect_events(
        _build_iterator(
            (
                _chunk("Hel", chunk_id="chatcmpl-first"),
                _chunk("lo", chunk_id="chatcmpl-second"),
                _chunk("!", finish_reason="stop", chunk_id="chatcmpl-third"),
            )
        ),
        async_mode=async_mode,
    )
    deltas: Final = tuple(event for event in events if isinstance(event, OutputTextDeltaEvent))
    assert tuple(event.delta for event in deltas) == ("Hel", "lo", "!")
    assert tuple(event.type for event in events) == (
        ResponsesAPIStreamEvents.RESPONSE_CREATED,
        ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
        ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    )
    text_done: Final = next(event for event in events if isinstance(event, OutputTextDoneEvent))
    content_done: Final = next(event for event in events if isinstance(event, ContentPartDoneEvent))
    assert text_done.text == "Hello!"
    assert content_done.part.text == "Hello!"
    completed: Final = next(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert tuple(item.type for item in completed.response.output) == ("message",)
    assert completed.response.output[0].content[0].text == "Hello!"
    _assert_completed_output_alignment(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
async def test_reasoning_deltas_and_completed_snapshot_share_one_item_id(async_mode: bool) -> None:
    events: Final = await _collect_events(
        _build_iterator(
            (
                _chunk(None, reasoning_content="thinking "),
                _chunk(None, reasoning_content="about fruit"),
                _chunk("apple", finish_reason="stop"),
            )
        ),
        async_mode=async_mode,
    )
    deltas: Final = tuple(event for event in events if isinstance(event, ReasoningSummaryTextDeltaEvent))
    assert tuple(event.delta for event in deltas) == ("thinking ", "about fruit")
    assert deltas[0].item_id == deltas[1].item_id
    assert len(tuple(event for event in events if isinstance(event, ResponsePartAddedEvent))) == 1
    done: Final = next(event for event in events if isinstance(event, ReasoningSummaryTextDoneEvent))
    assert done.text == "thinking about fruit"
    completed: Final = next(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert completed.response.output[0].content[0].text == done.text
    _assert_completed_output_alignment(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
@pytest.mark.parametrize("combined", (False, True))
async def test_reasoning_content_and_tools_have_distinct_output_indexes(async_mode: bool, combined: bool) -> None:
    tool: Final = _tool_delta("{}", call_id="call_reasoning_tool", name="lookup")
    chunks: Final = (
        (_chunk("The answer", reasoning_content="Checking the tool...", tool_calls=(tool,)),)
        if combined
        else (
            _chunk("", reasoning_content="Checking the tool..."),
            _chunk("The answer"),
            _chunk(None, tool_calls=(tool,)),
        )
    )
    events: Final = await _collect_events(
        _build_iterator((*chunks, _chunk(None, finish_reason="tool_calls"))), async_mode=async_mode
    )
    assert tuple(
        (event.output_index, _event_item(event).type) for event in events if isinstance(event, OutputItemAddedEvent)
    ) == ((0, "reasoning"), (1, "message"), (2, "function_call"))
    assert tuple(event.delta for event in events if isinstance(event, ReasoningSummaryTextDeltaEvent)) == (
        "Checking the tool...",
    )
    assert tuple(event.delta for event in events if isinstance(event, OutputTextDeltaEvent)) == ("The answer",)
    _assert_tool_lifecycle(events, call_id="call_reasoning_tool", arguments="{}", output_index=2, name="lookup")
    _assert_completed_output_alignment(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
@pytest.mark.parametrize("call_id", ("call_1", "toolu_01AbCdEf"))
async def test_tool_call_delta_is_emitted_as_responses_events(async_mode: bool, call_id: str) -> None:
    events: Final = await _collect_events(
        _build_iterator(
            (
                _chunk(None, tool_calls=(_tool_delta('{"x":1}', call_id=call_id, name="do_thing"),)),
                _chunk(None, finish_reason="tool_calls"),
            )
        ),
        async_mode=async_mode,
    )
    _assert_tool_lifecycle(events, call_id=call_id, arguments='{"x":1}', output_index=0, name="do_thing")
    assert not any(isinstance(event, ContentPartAddedEvent) for event in events)
    _assert_completed_output_alignment(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
async def test_tool_then_text_indexes_match_completed_output(async_mode: bool) -> None:
    events: Final = await _collect_events(
        _build_iterator(
            (
                _chunk(None, tool_calls=(_tool_delta("{}", call_id="call_first", name="lookup"),)),
                _chunk("Let me check", finish_reason="tool_calls"),
            )
        ),
        async_mode=async_mode,
    )
    assert tuple(
        (event.output_index, _event_item(event).type) for event in events if isinstance(event, OutputItemAddedEvent)
    ) == ((0, "function_call"), (1, "message"))
    assert tuple(event.delta for event in events if isinstance(event, OutputTextDeltaEvent)) == ("Let me check",)
    _assert_tool_lifecycle(events, call_id="call_first", arguments="{}", output_index=0, name="lookup")
    _assert_completed_output_alignment(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
@pytest.mark.parametrize("call_id", ("call_2", "toolu_01AbCdEf"))
async def test_tool_calls_present_only_in_final_response_are_emitted_before_completed(
    async_mode: bool, call_id: str
) -> None:
    response: Final = ModelResponse(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="claude-haiku-4-5",
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {"id": call_id, "type": "function", "function": {"name": "do_thing", "arguments": '{"y":2}'}}
                    ],
                ),
            )
        ],
    )
    events: Final = await _collect_events(_build_iterator((), final_response=response), async_mode=async_mode)
    _assert_tool_lifecycle(events, call_id=call_id, arguments='{"y":2}', output_index=0, name="do_thing")
    _assert_completed_output_alignment(events)


def test_tool_call_arguments_are_chunked_to_match_openai_behavior() -> None:
    arguments: Final = '{"param1": "value1", "param2": "value2", "param3": "value3"}'
    events: Final = list(
        _build_iterator(
            (
                _chunk(None, tool_calls=(_tool_delta(arguments, call_id="call_test", name="test_function"),)),
                _chunk(None, finish_reason="tool_calls"),
            )
        )
    )
    deltas: Final = tuple(event for event in events if isinstance(event, FunctionCallArgumentsDeltaEvent))
    assert all(0 < len(event.delta) <= 10 for event in deltas)
    _assert_tool_lifecycle(events, call_id="call_test", arguments=arguments, output_index=0, name="test_function")
    _assert_completed_output_alignment(events)


def test_tool_call_delta_without_id_uses_index_mapping() -> None:
    events: Final = list(
        _build_iterator(
            (
                _chunk(None, tool_calls=(_tool_delta('{"lo', call_id="call_abc123", name="get_weather"),)),
                _chunk(None, tool_calls=(_tool_delta('cation":'),)),
                _chunk(None, tool_calls=(_tool_delta(' "New'),)),
                _chunk(None, finish_reason="tool_calls", tool_calls=(_tool_delta(' York"}'),)),
            )
        )
    )
    _assert_tool_lifecycle(
        events, call_id="call_abc123", arguments='{"location": "New York"}', output_index=0, name="get_weather"
    )
    _assert_completed_output_alignment(events)


def test_parallel_tool_calls_without_ids_use_index_mapping() -> None:
    events: Final = list(
        _build_iterator(
            (
                _chunk(
                    None,
                    tool_calls=(
                        _tool_delta('{"x":', index=0, call_id="call_a", name="tool_a"),
                        _tool_delta('{"y":', index=1, call_id="call_b", name="tool_b"),
                    ),
                ),
                _chunk(
                    None,
                    finish_reason="tool_calls",
                    tool_calls=(_tool_delta("1}", index=0), _tool_delta("2}", index=1)),
                ),
            )
        )
    )
    _assert_tool_lifecycle(events, call_id="call_a", arguments='{"x":1}', output_index=0, name="tool_a")
    _assert_tool_lifecycle(events, call_id="call_b", arguments='{"y":2}', output_index=1, name="tool_b")
    _assert_completed_output_alignment(events)


def test_reused_index_with_new_call_id_does_not_misroute_arguments() -> None:
    final_response: Final = ModelResponse(
        id=CHAT_COMPLETION_ID,
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {"id": "call_a", "type": "function", "function": {"name": "tool_a", "arguments": '{"a":'}},
                        {"id": "call_b", "type": "function", "function": {"name": "tool_b", "arguments": '{"b":'}},
                    ],
                ),
            )
        ],
    )
    events: Final = list(
        _build_iterator(
            (
                _chunk(None, tool_calls=(_tool_delta('{"a":', call_id="call_a", name="tool_a"),)),
                _chunk(None, tool_calls=(_tool_delta('{"b":', call_id="call_b", name="tool_b"),)),
                _chunk(None, finish_reason="tool_calls", tool_calls=(_tool_delta("1}"),)),
            ),
            final_response=final_response,
        )
    )
    _assert_tool_lifecycle(events, call_id="call_a", arguments='{"a":', output_index=0, name="tool_a")
    _assert_tool_lifecycle(events, call_id="call_b", arguments='{"b":', output_index=1, name="tool_b")
    _assert_completed_output_alignment(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
async def test_streaming_events_share_the_chat_completion_response_id(async_mode: bool) -> None:
    events: Final = await _collect_events(
        _build_iterator((_chunk("Hello"), _chunk("!", finish_reason="stop"))), async_mode=async_mode
    )
    response_ids: Final = _response_ids(events)
    assert len(response_ids) == 3
    assert len(set(response_ids)) == 1
    decoded: Final = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_ids[0])
    assert decoded["response_id"] == CHAT_COMPLETION_ID
    assert decoded["custom_llm_provider"] == "anthropic"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", (False, True))
async def test_streaming_response_id_falls_back_when_upstream_yields_nothing(async_mode: bool) -> None:
    events: Final = await _collect_events(_build_iterator(()), async_mode=async_mode)
    response_ids: Final = _response_ids(events)
    assert len(response_ids) >= 2
    assert len(set(response_ids)) == 1
    assert response_ids[0].startswith("resp_")


def test_completed_event_restores_usage_hidden_by_stream_options_none() -> None:
    final_chunk: Final = _chunk("", finish_reason="stop")
    final_chunk._hidden_params = {"usage": Usage(prompt_tokens=117, completion_tokens=5, total_tokens=122)}
    events: Final = list(_build_iterator((_chunk("the document says hello"), final_chunk)))
    completed: Final = next(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert completed.response.usage.input_tokens == 117
    assert completed.response.usage.output_tokens == 5


def test_object_tool_call_arguments_stream_as_valid_json() -> None:
    iterator: Final = _build_iterator(())
    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "id": "call_obj",
                "type": "function",
                "function": {"name": "shell", "arguments": {"command": "ls", "flags": ["-l"]}},
            }
        ]
    )
    events: Final = tuple(iterator)
    streamed_arguments: Final = "".join(
        event.delta for event in events if isinstance(event, FunctionCallArgumentsDeltaEvent)
    )
    assert json.loads(streamed_arguments) == {"command": "ls", "flags": ["-l"]}


@pytest.mark.parametrize("tool_name", ("collaboration__spawn_agent", "spawn_agent"))
@pytest.mark.parametrize("final_only", (False, True))
def test_streaming_namespace_tool_calls_restore_responses_namespace(tool_name: str, final_only: bool) -> None:
    request: Final[ResponsesAPIOptionalRequestParams] = {
        "tools": [
            {
                "type": "namespace",
                "name": "collaboration",
                "tools": [{"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}}],
            }
        ]
    }
    arguments: Final = '{"task_name":"input_test"}'
    response: Final = ModelResponse(
        id=CHAT_COMPLETION_ID,
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_namespace",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments},
                        }
                    ],
                ),
            )
        ],
    )
    chunks: Final = (
        ()
        if final_only
        else (
            _chunk(
                None,
                finish_reason="tool_calls",
                tool_calls=(_tool_delta(arguments, call_id="call_namespace", name=tool_name),),
            ),
        )
    )
    events: Final = list(
        _build_iterator(chunks, responses_api_request=request, final_response=response if final_only else None)
    )
    _assert_tool_lifecycle(
        events,
        call_id="call_namespace",
        arguments=arguments,
        output_index=0,
        name="spawn_agent",
        namespace="collaboration",
    )
    _assert_completed_output_alignment(events)


def test_streaming_flat_namespace_tool_call_keeps_flat_name() -> None:
    request: Final[ResponsesAPIOptionalRequestParams] = {
        "tools": [
            {
                "type": "namespace",
                "name": "mcp__node_repl",
                "description": "Run JavaScript",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
            }
        ]
    }
    events: Final = list(
        _build_iterator(
            (
                _chunk(
                    None,
                    finish_reason="tool_calls",
                    tool_calls=(_tool_delta('{"code":"1+1"}', call_id="call_flat", name="mcp__node_repl"),),
                ),
            ),
            responses_api_request=request,
        )
    )
    _assert_tool_lifecycle(
        events, call_id="call_flat", arguments='{"code":"1+1"}', output_index=0, name="mcp__node_repl"
    )
    _assert_completed_output_alignment(events)


def test_streaming_top_level_function_collision_stays_unnamespaced() -> None:
    request: Final[ResponsesAPIOptionalRequestParams] = {
        "tools": [
            {"type": "function", "name": "run", "parameters": {"type": "object"}},
            {
                "type": "namespace",
                "name": "admin",
                "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}],
            },
        ]
    }
    events: Final = list(
        _build_iterator(
            (
                _chunk(
                    None,
                    finish_reason="tool_calls",
                    tool_calls=(_tool_delta("{}", call_id="call_top_level", name="run"),),
                ),
            ),
            responses_api_request=request,
        )
    )
    _assert_tool_lifecycle(events, call_id="call_top_level", arguments="{}", output_index=0, name="run")
    _assert_completed_output_alignment(events)


def test_completed_response_preserves_stream_finish_reason() -> None:
    events: Final = list(_build_iterator((_chunk("Partial answer", finish_reason="content_filter"),)))
    completed: Final = next(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert completed.response.status == "incomplete"
    assert completed.response.output[0].status == "incomplete"


def test_streamed_named_tool_choice_is_echoed_in_responses_api_shape() -> None:
    events: Final = list(
        _build_iterator(
            (
                _chunk(
                    None,
                    finish_reason="tool_calls",
                    tool_calls=(_tool_delta('{"command":"pwd"}', call_id="call_pwd", name="run_command"),),
                ),
            ),
            responses_api_request={
                "tools": [{"type": "function", "name": "run_command", "parameters": {"type": "object"}}],
                "tool_choice": {"type": "function", "name": "run_command"},
            },
        )
    )
    response_events: Final = tuple(event for event in events if getattr(event, "type", None) in RESPONSE_ID_EVENT_TYPES)
    assert tuple(event.type for event in response_events) == (
        "response.created",
        "response.in_progress",
        "response.completed",
    )
    assert tuple(event.response.tool_choice for event in response_events) == (
        {"type": "function", "name": "run_command"},
        {"type": "function", "name": "run_command"},
        {"type": "function", "name": "run_command"},
    )


def test_streamed_unrecognized_tool_choice_is_echoed_as_auto() -> None:
    events: Final = list(
        _build_iterator(
            (
                _chunk(
                    None,
                    finish_reason="tool_calls",
                    tool_calls=(_tool_delta('{"command":"pwd"}', call_id="call_pwd", name="run_command"),),
                ),
            ),
            responses_api_request={
                "tools": [{"type": "function", "name": "run_command", "parameters": {"type": "object"}}],
                "tool_choice": "any",
            },
        )
    )
    response_events: Final = tuple(event for event in events if getattr(event, "type", None) in RESPONSE_ID_EVENT_TYPES)
    assert tuple(event.response.tool_choice for event in response_events) == ("auto", "auto", "auto")
