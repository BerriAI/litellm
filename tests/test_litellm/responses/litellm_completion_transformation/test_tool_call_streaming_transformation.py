"""
Tests for streaming tool-calls in Responses API transformation.

Ensures that when the underlying chat-completions stream includes tool_calls deltas,
LiteLLM emits Responses API streaming events (output_item.added + function_call_arguments.*).

Also ensures that tool calls that only appear in the final built response still get emitted
before response.completed.
"""

from unittest.mock import AsyncMock

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)


class _StaticSyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0
        self.logging_obj = AsyncMock()

    def __next__(self):
        if self._index >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class _StaticAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0
        self.logging_obj = AsyncMock()

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def test_tool_call_delta_is_emitted_as_responses_events():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    # A streaming chunk with tool_calls delta but no text
    chunk = ModelResponseStream(
        id="chunk-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "do_thing", "arguments": '{"x":1}'},
                        }
                    ],
                ),
            )
        ],
    )

    evt1 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
    assert evt1 is not None
    assert evt1.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert evt1.output_index == 1

    # The arguments are now chunked, so we get the first delta chunk
    evt2 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
    assert evt2 is not None
    assert evt2.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    assert evt2.item_id == "call_1"
    assert evt2.output_index == 1
    # The delta will be a chunk of the arguments, not the full arguments
    assert len(evt2.delta) <= 10  # Chunks are max 10 characters


def test_sync_iterator_emits_message_then_text_for_same_chunk():
    chunk = ModelResponseStream(
        id="chunk-sync-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="DeepSeek final answer",
                ),
            )
        ],
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=_StaticSyncStream([chunk]),
        request_input="Test input",
        responses_api_request={},
    )

    evt1 = next(iterator)
    evt2 = next(iterator)
    evt3 = next(iterator)

    assert evt1.type == ResponsesAPIStreamEvents.RESPONSE_CREATED
    assert evt2.type == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS
    assert evt3.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED

    evt4 = next(iterator)
    assert evt4.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
    assert evt4.delta == "DeepSeek final answer"


def test_sync_iterator_emits_message_item_when_text_arrives_after_tool_call():
    tool_chunk = ModelResponseStream(
        id="chunk-tool-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_late_text",
                            "type": "function",
                            "function": {"name": "do_thing", "arguments": "{}"},
                        }
                    ],
                ),
            )
        ],
    )
    text_chunk = ModelResponseStream(
        id="chunk-text-1",
        created=124,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="Final answer after tool call",
                ),
            )
        ],
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=_StaticSyncStream([tool_chunk, text_chunk]),
        request_input="Test input",
        responses_api_request={},
    )

    seen_events = []
    for _ in range(8):
        seen_events.append(next(iterator))

    message_item_events = [
        evt
        for evt in seen_events
        if evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        and getattr(getattr(evt, "item", None), "type", None) == "message"
    ]
    text_delta_events = [
        evt
        for evt in seen_events
        if evt.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
    ]

    assert len(message_item_events) == 1
    assert len(text_delta_events) == 1
    assert text_delta_events[0].delta == "Final answer after tool call"


def test_sync_iterator_emits_reasoning_done_events_before_completion():
    reasoning_chunk = ModelResponseStream(
        id="chunk-reasoning-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="",
                    reasoning_content="step1 ",
                ),
            )
        ],
    )
    text_chunk = ModelResponseStream(
        id="chunk-text-1",
        created=124,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="answer",
                ),
            )
        ],
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=_StaticSyncStream([reasoning_chunk, text_chunk]),
        request_input="Test input",
        responses_api_request={},
    )

    seen_events = []
    while True:
        try:
            seen_events.append(next(iterator))
        except StopIteration:
            break

    event_types = [event.type for event in seen_events]
    assert ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE in event_types
    assert ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE in event_types

    reasoning_item_done_indexes = [
        idx
        for idx, event in enumerate(seen_events)
        if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        and getattr(getattr(event, "item", None), "type", None) == "reasoning"
    ]
    response_completed_index = event_types.index(
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    )

    assert len(reasoning_item_done_indexes) == 1
    assert reasoning_item_done_indexes[0] < response_completed_index


@pytest.mark.asyncio
async def test_async_mixed_reasoning_chunk_closes_reasoning_item_before_completion():
    mixed_chunk = ModelResponseStream(
        id="chunk-mixed-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="final answer",
                    reasoning_content="thinking...",
                ),
            )
        ],
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=_StaticAsyncStream([mixed_chunk]),
        request_input="Test input",
        responses_api_request={},
    )

    seen_events = []
    while True:
        try:
            seen_events.append(await iterator.__anext__())
        except StopAsyncIteration:
            break

    event_types = [event.type for event in seen_events]
    assert ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE in event_types
    assert ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE in event_types

    reasoning_item_done_indexes = [
        idx
        for idx, event in enumerate(seen_events)
        if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        and getattr(getattr(event, "item", None), "type", None) == "reasoning"
    ]
    response_completed_index = event_types.index(
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    )

    assert len(reasoning_item_done_indexes) == 1
    assert reasoning_item_done_indexes[0] < response_completed_index


def test_reasoning_delta_item_id_matches_reasoning_output_item_id():
    chunk = ModelResponseStream(
        id="chunk-reasoning-id-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="",
                    reasoning_content="thinking...",
                ),
            )
        ],
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    iterator._ensure_output_item_for_chunk(chunk)
    output_item_added_event = iterator._pending_response_events.pop(0)
    reasoning_delta_event = (
        iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
    )

    assert output_item_added_event is not None
    assert reasoning_delta_event is not None
    assert output_item_added_event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert (
        reasoning_delta_event.type
        == ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA
    )
    assert output_item_added_event.item.type == "reasoning"
    assert reasoning_delta_event.item_id == output_item_added_event.item.id


def test_tool_calls_present_only_in_final_response_are_emitted_before_completed():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    # Construct a final ModelResponse with tool_calls on the message.
    # We bypass the stream builder and directly set iterator.litellm_model_response.
    response = ModelResponse(
        id="resp-1",
        created=123,
        model="test-model",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "do_thing", "arguments": '{"y":2}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ],
    )
    iterator.litellm_model_response = response

    # First common_done_event_logic call should yield tool events, not response.completed.
    evt1 = iterator.common_done_event_logic(sync_mode=True)
    assert evt1.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert evt1.output_index == 1

    # Now delta events are emitted (arguments split into chunks)
    # Collect all delta events
    delta_events = []
    while True:
        evt = iterator.common_done_event_logic(sync_mode=True)
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA:
            delta_events.append(evt)
        else:
            break

    # Verify we got delta events
    assert len(delta_events) > 0
    # Verify they reconstruct the original arguments
    concatenated_args = "".join(evt.delta for evt in delta_events)
    assert concatenated_args == '{"y":2}'

    # The last event should be FUNCTION_CALL_ARGUMENTS_DONE
    assert evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE
    assert evt.item_id == "call_2"
    assert evt.output_index == 1
    assert evt.arguments == '{"y":2}'

    evt_final = iterator.common_done_event_logic(sync_mode=True)
    assert evt_final.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
    assert evt_final.output_index == 1


def test_tool_call_arguments_are_chunked_to_match_openai_behavior():
    """
    Test that large tool call arguments are split into smaller chunks (size 10)
    to replicate OpenAI's native streaming behavior.

    This is especially important for providers like Bedrock that send complete
    arguments at once, which need to be split to match OpenAI's token-by-token streaming.
    """
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    # Create a chunk with a large arguments string that should be split
    large_arguments = (
        '{"param1": "value1", "param2": "value2", "param3": "value3"}'  # 67 chars
    )
    chunk = ModelResponseStream(
        id="chunk-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "test_function",
                                "arguments": large_arguments,
                            },
                        }
                    ],
                ),
            )
        ],
    )

    # Process the chunk once - it queues all events internally
    evt = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)

    # First event should be OUTPUT_ITEM_ADDED
    assert evt is not None
    assert evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert evt.output_index == 1
    assert hasattr(evt, "__dict__") and "sequence_number" in evt.__dict__

    # Collect all remaining delta events from the pending queue by creating empty chunks
    delta_events = []
    empty_chunk = ModelResponseStream(
        id="chunk-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(role="assistant", content=""),
            )
        ],
    )

    # Keep draining pending events (expected: ceil(67 / 10) = 7 delta events)
    while iterator._pending_tool_events:
        evt = iterator._transform_chat_completion_chunk_to_response_api_chunk(
            empty_chunk
        )
        if evt and evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA:
            delta_events.append(evt)

    # Verify multiple delta events were created (at least 6 chunks for 67 chars)
    assert len(delta_events) >= 6  # 67 chars split into chunks of max 10 chars each

    # Verify each delta is at most 10 characters
    for evt in delta_events:
        assert len(evt.delta) <= 10
        assert evt.item_id == "call_test"
        assert evt.output_index == 1
        assert hasattr(evt, "__dict__") and "sequence_number" in evt.__dict__

    # Verify all deltas concatenated equal the original arguments
    concatenated = "".join(evt.delta for evt in delta_events)
    assert concatenated == large_arguments

    # Verify sequence numbers are increasing
    sequence_numbers = [evt.__dict__["sequence_number"] for evt in delta_events]
    assert sequence_numbers == sorted(sequence_numbers)
    assert len(set(sequence_numbers)) == len(sequence_numbers)  # All unique


def test_tool_call_delta_without_id_uses_index_mapping():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    chunks = [
        [
            {
                "index": 0,
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"lo'},
            }
        ],
        [{"index": 0, "type": "function", "function": {"arguments": 'cation":'}}],
        [{"index": 0, "type": "function", "function": {"arguments": ' "New'}}],
        [{"index": 0, "type": "function", "function": {"arguments": ' York"}'}}],
    ]

    for tool_calls in chunks:
        iterator._queue_tool_call_delta_events(tool_calls)

    all_events = []
    while iterator._pending_tool_events:
        all_events.append(iterator._pending_tool_events.pop(0))

    delta_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    ]
    streamed_arguments = "".join(evt.delta for evt in delta_events)

    assert streamed_arguments == '{"location": "New York"}'

    output_item_added_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    ]
    assert len(output_item_added_events) == 1
    assert output_item_added_events[0].item.id == "call_abc123"


def test_parallel_tool_calls_without_ids_use_index_mapping():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "id": "call_a",
                "type": "function",
                "function": {"name": "tool_a", "arguments": '{"x":'},
            },
            {
                "index": 1,
                "id": "call_b",
                "type": "function",
                "function": {"name": "tool_b", "arguments": '{"y":'},
            },
        ]
    )
    iterator._queue_tool_call_delta_events(
        [
            {"index": 0, "type": "function", "function": {"arguments": "1}"}},
            {"index": 1, "type": "function", "function": {"arguments": "2}"}},
        ]
    )

    all_events = []
    while iterator._pending_tool_events:
        all_events.append(iterator._pending_tool_events.pop(0))

    output_item_added_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    ]
    assert len(output_item_added_events) == 2

    delta_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    ]
    arguments_by_call_id = {}
    for evt in delta_events:
        arguments_by_call_id.setdefault(evt.item_id, "")
        arguments_by_call_id[evt.item_id] += evt.delta

    assert arguments_by_call_id["call_a"] == '{"x":1}'
    assert arguments_by_call_id["call_b"] == '{"y":2}'


def test_reused_index_with_new_call_id_marks_fallback_ambiguous():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "id": "call_a",
                "type": "function",
                "function": {"name": "tool_a", "arguments": '{"a":'},
            }
        ]
    )
    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "id": "call_b",
                "type": "function",
                "function": {"name": "tool_b", "arguments": '{"b":'},
            }
        ]
    )
    # Ambiguous chunk: index reused and id missing. We should skip fallback rather than misroute.
    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 0,
                "type": "function",
                "function": {"arguments": "1}"},
            }
        ]
    )

    all_events = []
    while iterator._pending_tool_events:
        all_events.append(iterator._pending_tool_events.pop(0))

    delta_events = [
        evt
        for evt in all_events
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    ]
    arguments_by_call_id = {}
    for evt in delta_events:
        arguments_by_call_id.setdefault(evt.item_id, "")
        arguments_by_call_id[evt.item_id] += evt.delta

    assert arguments_by_call_id["call_a"] == '{"a":'
    assert arguments_by_call_id["call_b"] == '{"b":'
    assert arguments_by_call_id["call_a"] != '{"a":1}'
    assert arguments_by_call_id["call_b"] != '{"b":1}'
