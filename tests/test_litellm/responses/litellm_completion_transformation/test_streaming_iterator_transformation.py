"""
Tests for the Responses API streaming bridge in
litellm/responses/litellm_completion_transformation/streaming_iterator.py.

Ensures that when the underlying chat-completions stream includes tool_calls deltas,
LiteLLM emits Responses API streaming events (output_item.added + function_call_arguments.*).

Also ensures that tool calls that only appear in the final built response still get emitted
before response.completed, and that every event of a bridged stream carries the response id
spend tracking stores, so a follow-up previous_response_id still finds the conversation.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

CHAT_COMPLETION_ID = "chatcmpl-77d33d09-effa-4cd2-9c0d-c742d4358256"
RESPONSE_ID_EVENT_TYPES = frozenset(
    {"response.created", "response.in_progress", "response.completed"}
)


def _chunk(content: str, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="claude-haiku-4-5",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
    )


class _FakeStreamWrapper:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.logging_obj = MagicMock()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _build_iterator(chunks) -> LiteLLMCompletionStreamingIterator:
    return LiteLLMCompletionStreamingIterator(
        model="claude-haiku-4-5",
        litellm_custom_stream_wrapper=_FakeStreamWrapper(chunks),
        request_input="What is the weather in San Francisco?",
        responses_api_request={},
        custom_llm_provider="anthropic",
        litellm_metadata={},
    )


def _response_ids(events) -> list[str]:
    return [
        event.response.id
        for event in events
        if getattr(event, "type", None) in RESPONSE_ID_EVENT_TYPES
    ]


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
    assert evt2.item_id == "fc_call_1"
    assert evt2.output_index == 1
    # The delta will be a chunk of the arguments, not the full arguments
    assert len(evt2.delta) <= 10  # Chunks are max 10 characters


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
    assert evt.item_id == "fc_call_2"
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
        assert evt.item_id == "fc_call_test"
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
    assert output_item_added_events[0].item.id == "fc_call_abc123"
    assert output_item_added_events[0].item.call_id == "call_abc123"


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

    assert arguments_by_call_id["fc_call_a"] == '{"x":1}'
    assert arguments_by_call_id["fc_call_b"] == '{"y":2}'


def test_final_tool_events_and_completed_snapshot_reuse_streamed_call_identity():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )
    streamed_ids = ["call_stream_a", "call_stream_b"]
    streamed_item_ids = [f"fc_{call_id}" for call_id in streamed_ids]
    terminal_ids = ["call_terminal_a", "call_terminal_b"]
    terminal_item_ids = [f"fc_{call_id}" for call_id in terminal_ids]

    iterator._queue_tool_call_delta_events(
        [
            {
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {
                    "name": f"tool_{index}",
                    "arguments": f'{{"value":{index}',
                },
            }
            for index, call_id in reversed(list(enumerate(streamed_ids)))
        ]
    )
    # Simulate delivery of all incremental events before the terminal aggregate arrives.
    iterator._pending_tool_events.clear()

    iterator.litellm_model_response = ModelResponse(
        id="chatcmpl-terminal",
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
                            "id": terminal_id,
                            "type": "function",
                            "function": {
                                "name": f"tool_{index}",
                                "arguments": f'{{"value":{index}}}',
                            },
                        }
                        for index, terminal_id in enumerate(terminal_ids)
                    ],
                },
            }
        ],
    )

    final_events = []
    for _ in range(20):
        event = iterator.common_done_event_logic()
        final_events.append(event)
        if event.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
            break
    else:
        pytest.fail("response.completed was not emitted")

    final_tool_items = [
        event.item
        for event in final_events
        if event.type
        in {
            ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        }
        and getattr(event.item, "type", None) == "function_call"
    ]
    assert [item.id for item in final_tool_items] == streamed_item_ids
    assert [item.call_id for item in final_tool_items] == streamed_ids

    argument_event_ids = [
        event.item_id
        for event in final_events
        if event.type
        in {
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
        }
    ]
    assert argument_event_ids
    assert set(argument_event_ids) == set(streamed_item_ids)

    completed = final_events[-1]
    completed_calls = [item for item in completed.response.output if item.type == "function_call"]
    assert [item.id for item in completed_calls] == streamed_item_ids
    assert [item.call_id for item in completed_calls] == streamed_ids
    assert not set(terminal_ids + terminal_item_ids) & {
        item_id for item in final_tool_items + completed_calls for item_id in (item.id, item.call_id)
    }


def test_final_events_preserve_distinct_streamed_item_id_and_call_id():
    from litellm.responses.litellm_completion_transformation.custom_tools import (
        build_tool_call_item_kwargs,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )
    terminal_response = ModelResponse(
        id="chatcmpl-terminal",
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
                            "id": "call_terminal",
                            "type": "function",
                            "function": {"name": "tool", "arguments": '{"value":1}'},
                        }
                    ],
                },
            }
        ],
    )

    def distinct_item_id_builder(call_id, *args, **kwargs):
        item_kwargs = build_tool_call_item_kwargs(call_id, *args, **kwargs)
        if call_id == "call_stream":
            item_kwargs["id"] = "fc_stream"
        return item_kwargs

    with patch(
        "litellm.responses.litellm_completion_transformation.streaming_iterator.build_tool_call_item_kwargs",
        side_effect=distinct_item_id_builder,
    ):
        iterator._queue_tool_call_delta_events(
            [
                {
                    "index": 0,
                    "id": "call_stream",
                    "type": "function",
                    "function": {"name": "tool", "arguments": '{"value":'},
                }
            ]
        )
        streamed_added = next(
            event
            for event in iterator._pending_tool_events
            if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        )
        assert (streamed_added.item.id, streamed_added.item.call_id) == (
            "fc_stream",
            "call_stream",
        )
        assert {
            event.item_id
            for event in iterator._pending_tool_events
            if event.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
        } == {"fc_stream"}
        iterator._pending_tool_events.clear()
        iterator._queue_final_tool_call_done_events(terminal_response)

    original_transform = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response
    )

    def terminal_response_with_distinct_item_id(*args, **kwargs):
        response = original_transform(*args, **kwargs)
        terminal_call = next(item for item in response.output if item.type == "function_call")
        terminal_call.id = "fc_terminal"
        terminal_call.call_id = "call_terminal"
        return response

    with patch.object(
        LiteLLMCompletionResponsesConfig,
        "transform_chat_completion_response_to_responses_api_response",
        side_effect=terminal_response_with_distinct_item_id,
    ):
        completed = iterator._emit_response_completed_event(terminal_response)

    assert completed is not None
    argument_events = [
        event
        for event in iterator._pending_tool_events
        if event.type
        in {
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
        }
    ]
    assert argument_events
    assert {event.item_id for event in argument_events} == {"fc_stream"}
    final_item = next(
        event.item
        for event in iterator._pending_tool_events
        if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
    )
    assert (final_item.id, final_item.call_id) == ("fc_stream", "call_stream")
    completed_call = next(item for item in completed.response.output if item.type == "function_call")
    assert (completed_call.id, completed_call.call_id) == ("fc_stream", "call_stream")


def test_terminal_only_tool_calls_keep_terminal_identity():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )
    terminal_ids = ["call_terminal_a", "call_terminal_b"]
    terminal_item_ids = [f"fc_{call_id}" for call_id in terminal_ids]
    response = ModelResponse(
        id="chatcmpl-terminal-only",
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
                            "id": call_id,
                            "type": "function",
                            "function": {"name": f"tool_{index}", "arguments": "{}"},
                        }
                        for index, call_id in enumerate(terminal_ids)
                    ],
                },
            }
        ],
    )

    iterator._queue_final_tool_call_done_events(response)
    added_items = [
        event.item
        for event in iterator._pending_tool_events
        if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    ]

    assert [item.id for item in added_items] == terminal_item_ids
    assert [item.call_id for item in added_items] == terminal_ids


def test_terminal_only_call_is_not_conflated_with_later_streamed_call():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )
    iterator._queue_tool_call_delta_events(
        [
            {
                "index": 1,
                "id": "call_streamed",
                "type": "function",
                "function": {"name": "streamed_tool", "arguments": "{}"},
            }
        ]
    )
    iterator._pending_tool_events.clear()
    response = ModelResponse(
        id="chatcmpl-mixed",
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
                            "id": "call_terminal_only",
                            "type": "function",
                            "function": {"name": "terminal_tool", "arguments": "{}"},
                        },
                        {
                            "id": "call_terminal_drifted",
                            "type": "function",
                            "function": {"name": "streamed_tool", "arguments": "{}"},
                        },
                    ],
                },
            }
        ],
    )

    iterator._queue_final_tool_call_done_events(response)
    added_or_done_items = [
        event.item
        for event in iterator._pending_tool_events
        if event.type
        in {
            ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        }
    ]
    completed = iterator._emit_response_completed_event(response)

    assert completed is not None
    assert {item.id for item in added_or_done_items} == {
        "fc_call_terminal_only",
        "fc_call_streamed",
    }
    completed_calls = [item for item in completed.response.output if item.type == "function_call"]
    assert [item.id for item in completed_calls] == [
        "fc_call_terminal_only",
        "fc_call_streamed",
    ]
    assert [item.call_id for item in completed_calls] == [
        "call_terminal_only",
        "call_streamed",
    ]


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

    assert arguments_by_call_id["fc_call_a"] == '{"a":'
    assert arguments_by_call_id["fc_call_b"] == '{"b":'
    assert arguments_by_call_id["fc_call_a"] != '{"a":1}'
    assert arguments_by_call_id["fc_call_b"] != '{"b":1}'


@pytest.mark.asyncio
async def test_streaming_events_share_the_chat_completion_response_id():
    """
    Every event of a bridged stream has to carry the same id, and that id has to decode
    to the chat completion id spend tracking stores as `request_id`. Otherwise a
    follow-up `previous_response_id` matches no session and the conversation is dropped.
    """
    iterator = _build_iterator([_chunk("Hello"), _chunk("!", finish_reason="stop")])

    events = [event async for event in iterator]

    response_ids = _response_ids(events)
    assert len(response_ids) == 3
    assert len(set(response_ids)) == 1
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_ids[0])
    assert decoded["response_id"] == CHAT_COMPLETION_ID
    assert decoded["custom_llm_provider"] == "anthropic"


def test_sync_streaming_events_share_the_chat_completion_response_id():
    iterator = _build_iterator([_chunk("Hello"), _chunk("!", finish_reason="stop")])

    events = list(iterator)

    response_ids = _response_ids(events)
    assert len(response_ids) == 3
    assert len(set(response_ids)) == 1
    assert (
        ResponsesAPIRequestUtils._decode_responses_api_response_id(response_ids[0])["response_id"]
        == CHAT_COMPLETION_ID
    )


@pytest.mark.asyncio
async def test_streaming_emits_every_chunk_after_priming_the_response_id():
    iterator = _build_iterator(
        [_chunk("Hel"), _chunk("lo"), _chunk("!", finish_reason="stop")]
    )

    events = [event async for event in iterator]

    deltas = "".join(
        event.delta for event in events if getattr(event, "type", None) == "response.output_text.delta"
    )
    assert deltas == "Hello!"


@pytest.mark.asyncio
async def test_streaming_response_id_falls_back_when_upstream_yields_nothing():
    iterator = _build_iterator([])

    events = [event async for event in iterator]

    response_ids = _response_ids(events)
    assert response_ids
    assert len(set(response_ids)) == 1
    assert response_ids[0].startswith("resp_")


def test_object_tool_call_arguments_stream_as_valid_json():
    """A provider that sends decoded object arguments must still stream valid JSON.

    `str()` on a dict yields a Python repr with single quotes, which clients
    parsing function_call_arguments reject with errors like
    "Expecting ',' delimiter".
    """
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
                "id": "call_obj",
                "type": "function",
                "function": {"name": "shell", "arguments": {"command": "ls", "flags": ["-l"]}},
            }
        ]
    )

    streamed_arguments = "".join(
        evt.delta
        for evt in iterator._pending_tool_events
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    )

    assert json.loads(streamed_arguments) == {"command": "ls", "flags": ["-l"]}


def test_streamed_anthropic_tool_call_events_correlate_on_normalized_item_id():
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    response = ModelResponse(
        id="resp-anthropic",
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
                            "id": "toolu_01AbCdEf",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ],
    )
    iterator.litellm_model_response = response

    events = []
    while True:
        evt = iterator.common_done_event_logic(sync_mode=True)
        events.append(evt)
        if evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
            break

    added = [e for e in events if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED]
    deltas = [e for e in events if e.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA]
    dones = [e for e in events if e.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE]
    item_dones = [e for e in events if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE]

    assert len(added) == 1 and len(dones) == 1 and len(item_dones) == 1 and deltas
    assert added[0].item.id == "fc_toolu_01AbCdEf"
    assert added[0].item.call_id == "toolu_01AbCdEf"
    assert item_dones[0].item.id == "fc_toolu_01AbCdEf"
    assert item_dones[0].item.call_id == "toolu_01AbCdEf"
    for evt in deltas + dones:
        assert evt.item_id == added[0].item.id
