"""
Test reasoning content preservation in Responses API transformation
"""

from collections.abc import Iterator
from typing import Final, cast
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ContentPartAddedEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    ReasoningSummaryTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsePartAddedEvent,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)


class _ChunkStream:
    logging_obj: None = None

    def __init__(self, chunks: tuple[ModelResponseStream, ...]) -> None:
        self._chunks: Iterator[ModelResponseStream] = iter(chunks)

    def __next__(self) -> ModelResponseStream:
        return next(self._chunks)

    async def __anext__(self) -> ModelResponseStream:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _event_item_type(event: OutputItemAddedEvent | OutputItemDoneEvent) -> str | None:
    return getattr(event.item, "type", None) if event.item is not None else None


def _mixed_reasoning_content_chunks() -> tuple[ModelResponseStream, ModelResponseStream]:
    mixed_chunk = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content="Here is the answer",
                    role="assistant",
                    reasoning_content="First, let me analyze...",
                ),
            )
        ],
    )
    finish_chunk = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content=None),
            )
        ],
    )
    return mixed_chunk, finish_chunk


def _reasoning_tool_chunks() -> tuple[ModelResponseStream, ModelResponseStream, ModelResponseStream]:
    reasoning_chunk: Final = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="", role="assistant", reasoning_content="Checking the tool..."),
            )
        ],
    )
    content_chunk: Final = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="The answer", role="assistant"),
            )
        ],
    )
    tool_chunk: Final = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="tool_calls",
                index=0,
                delta=Delta(
                    content=None,
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_reasoning_tool",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                ),
            )
        ],
    )
    return reasoning_chunk, content_chunk, tool_chunk


def _separate_reasoning_content_chunks() -> tuple[ModelResponseStream, ModelResponseStream, ModelResponseStream]:
    return (
        ModelResponseStream(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="", role="assistant", reasoning_content="First, let me analyze..."),
                )
            ],
        ),
        ModelResponseStream(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="Here is the answer", role="assistant"),
                )
            ],
        ),
        ModelResponseStream(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
            choices=[StreamingChoices(finish_reason="stop", index=0, delta=Delta(content=None))],
        ),
    )


def _completed_output_types(events: list[BaseLiteLLMOpenAIResponseObject]) -> tuple[str | None, ...]:
    completed_event: Final = next(event for event in events if isinstance(event, ResponseCompletedEvent))
    return tuple(getattr(item, "type", None) for item in completed_event.response.output)


def _done_output_items_by_index(
    events: list[BaseLiteLLMOpenAIResponseObject],
) -> tuple[tuple[int, str | None], ...]:
    done_events: Final = tuple(event for event in events if isinstance(event, OutputItemDoneEvent))
    return tuple((event.output_index, _event_item_type(event)) for event in done_events)


def _assert_completed_output_alignment(events: list[BaseLiteLLMOpenAIResponseObject]) -> None:
    completed_event: Final = next(event for event in events if isinstance(event, ResponseCompletedEvent))
    done_events: Final = tuple(event for event in events if isinstance(event, OutputItemDoneEvent))
    assert tuple((event.output_index, _event_item_type(event)) for event in done_events) == tuple(
        (index, getattr(item, "type", None)) for index, item in enumerate(completed_event.response.output)
    )
    added_events: Final = tuple(event for event in events if isinstance(event, OutputItemAddedEvent))
    assert all(
        any(added.output_index == done.output_index for added in added_events[: events.index(done)])
        for done in done_events
    )


def _assert_mixed_reasoning_content_events(events: list[BaseLiteLLMOpenAIResponseObject]) -> None:
    payload_events = [
        event for event in events if isinstance(event, (ReasoningSummaryTextDeltaEvent, OutputTextDeltaEvent))
    ]
    assert [event.type for event in payload_events] == [
        ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
    ]
    assert [event.delta for event in payload_events] == [
        "First, let me analyze...",
        "Here is the answer",
    ]
    reasoning_delta = payload_events[0]
    text_delta = payload_events[1]
    reasoning_item_added = next(
        event for event in events if isinstance(event, OutputItemAddedEvent) and _event_item_type(event) == "reasoning"
    )
    reasoning_part_added_events: Final = tuple(event for event in events if isinstance(event, ResponsePartAddedEvent))
    assert len(reasoning_part_added_events) == 1
    reasoning_part_added: Final = reasoning_part_added_events[0]
    reasoning_item_done = next(
        event for event in events if isinstance(event, OutputItemDoneEvent) and _event_item_type(event) == "reasoning"
    )
    message_item_added = next(
        event for event in events if isinstance(event, OutputItemAddedEvent) and _event_item_type(event) == "message"
    )
    content_part_added = next(event for event in events if isinstance(event, ContentPartAddedEvent))
    assert reasoning_part_added.item_id == reasoning_delta.item_id == getattr(reasoning_item_added.item, "id", None)
    assert reasoning_part_added.output_index == reasoning_delta.output_index == reasoning_item_added.output_index == 0
    assert getattr(reasoning_part_added, "summary_index", None) == 0
    assert reasoning_part_added.part == {"type": "summary_text", "text": ""}
    assert text_delta.output_index == message_item_added.output_index == 1
    assert _done_output_items_by_index(events) == ((0, "reasoning"), (1, "message"))
    assert _completed_output_types(events) == ("reasoning", "message")
    assert text_delta.item_id == getattr(message_item_added.item, "id", None)
    assert all(
        event.output_index == reasoning_item_added.output_index
        for event in events
        if getattr(event, "item_id", None) == reasoning_delta.item_id
        or getattr(getattr(event, "item", None), "id", None) == reasoning_delta.item_id
    )
    assert all(
        event.output_index == message_item_added.output_index
        for event in events
        if getattr(event, "item_id", None) == text_delta.item_id
        or getattr(getattr(event, "item", None), "id", None) == text_delta.item_id
    )
    assert events.index(reasoning_item_added) < events.index(reasoning_part_added) < events.index(reasoning_delta)
    assert events.index(reasoning_delta) < events.index(reasoning_item_done)
    assert events.index(reasoning_item_done) < events.index(message_item_added)
    assert events.index(message_item_added) < events.index(content_part_added)
    assert events.index(content_part_added) < events.index(text_delta)
    sequence_numbers: Final = tuple(event.model_dump()["sequence_number"] for event in events)
    assert sequence_numbers == tuple(range(1, len(events) + 1))
    _assert_completed_output_alignment(events)


class TestReasoningContentStreaming:
    """Test reasoning content preservation during streaming"""

    def test_reasoning_content_in_delta(self):
        """Test that reasoning content is preserved in streaming deltas"""
        # Setup
        chunk = ModelResponseStream(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="",
                        role="assistant",
                        reasoning_content="Let me think about this problem...",
                    ),
                )
            ],
        )

        mock_stream = AsyncMock()

        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        # Execute
        transformed_chunk = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)

        # Assert
        assert transformed_chunk.delta == "Let me think about this problem..."
        assert transformed_chunk.type == "response.reasoning_summary_text.delta"

    def test_mixed_content_and_reasoning(self):
        """Test handling of both content and reasoning content"""
        # Setup
        chunk = ModelResponseStream(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="Here is the answer",
                        role="assistant",
                        reasoning_content="First, let me analyze...",
                    ),
                )
            ],
        )

        mock_stream = AsyncMock()
        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        # Execute
        transformed_chunk = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)

        # Assert
        assert transformed_chunk.delta == "First, let me analyze..."
        assert transformed_chunk.type == "response.reasoning_summary_text.delta"

    def test_mixed_content_and_reasoning_sync_iterator(self):
        chunks = _mixed_reasoning_content_chunks()
        mock_stream = cast(litellm.CustomStreamWrapper, _ChunkStream(chunks))
        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        events = list(iterator)

        _assert_mixed_reasoning_content_events(events)

    @pytest.mark.asyncio
    async def test_mixed_content_and_reasoning_async_iterator(self):
        chunks = _mixed_reasoning_content_chunks()
        mock_stream = cast(litellm.CustomStreamWrapper, _ChunkStream(chunks))
        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        events = [event async for event in iterator]

        _assert_mixed_reasoning_content_events(events)

    def test_separate_reasoning_then_content_closes_reasoning_first(self):
        chunks: Final = _separate_reasoning_content_chunks()
        mock_stream: Final = cast(litellm.CustomStreamWrapper, _ChunkStream(chunks))
        iterator: Final = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        events: Final = list(iterator)
        reasoning_done: Final = next(
            event
            for event in events
            if isinstance(event, OutputItemDoneEvent) and _event_item_type(event) == "reasoning"
        )
        message_added: Final = next(
            event
            for event in events
            if isinstance(event, OutputItemAddedEvent) and _event_item_type(event) == "message"
        )
        content_part_added: Final = next(event for event in events if isinstance(event, ContentPartAddedEvent))
        text_delta: Final = next(event for event in events if isinstance(event, OutputTextDeltaEvent))
        assert events.index(reasoning_done) < events.index(message_added)
        assert events.index(message_added) < events.index(content_part_added) < events.index(text_delta)
        assert reasoning_done.output_index == 0
        assert message_added.output_index == content_part_added.output_index == text_delta.output_index == 1
        _assert_completed_output_alignment(events)

    def test_reasoning_and_tool_indexes_match_completed_output(self):
        chunks: Final = _reasoning_tool_chunks()
        mock_stream: Final = cast(litellm.CustomStreamWrapper, _ChunkStream(chunks))
        iterator: Final = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        events: Final = list(iterator)
        assert _done_output_items_by_index(events) == (
            (0, "reasoning"),
            (1, "message"),
            (2, "function_call"),
        )
        assert _completed_output_types(events) == ("reasoning", "message", "function_call")
        assert any(isinstance(event, OutputTextDeltaEvent) and event.output_index == 1 for event in events)
        _assert_completed_output_alignment(events)

    def test_no_reasoning_content(self):
        """Test handling when no reasoning content is present"""
        # Setup
        chunk = ModelResponseStream(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="Regular content only",
                        role="assistant",
                    ),
                )
            ],
        )

        mock_stream = AsyncMock()
        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        # Execute
        transformed_chunk = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)

        # Assert
        assert transformed_chunk.delta == "Regular content only"
        assert transformed_chunk.type == "response.output_text.delta"
        assert transformed_chunk.output_index == 0


class TestReasoningContentFinalResponse:
    """Test reasoning content preservation in final response transformation"""

    def test_reasoning_content_in_final_response(self):
        """Test that reasoning content is included in final response"""
        # Setup
        response = ModelResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Here is my answer",
                        role="assistant",
                        reasoning_content="Let me think step by step about this problem...",
                    ),
                )
            ],
        )

        # Execute
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test input",
                responses_api_request={},
                chat_completion_response=response,
            )
        )

        # Assert
        assert hasattr(responses_api_response, "output")
        assert len(responses_api_response.output) > 0

        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) > 0, "No reasoning item found in output"

        reasoning_item = reasoning_items[0]
        assert reasoning_item.content[0].text == "Let me think step by step about this problem..."

    def test_no_reasoning_content_in_response(self):
        """Test handling when no reasoning content in response"""
        # Setup
        response = ModelResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Simple answer",
                        role="assistant",
                    ),
                )
            ],
        )

        # Execute
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test input",
                responses_api_request={},
                chat_completion_response=response,
            )
        )

        # Assert
        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) == 0, "Should have no reasoning items when no reasoning content present"

    def test_multiple_choices_with_reasoning(self):
        """Test handling multiple choices, first with reasoning content"""
        # Setup
        response = ModelResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="First answer",
                        role="assistant",
                        reasoning_content="Reasoning for first answer",
                    ),
                ),
                Choices(
                    finish_reason="stop",
                    index=1,
                    message=Message(
                        content="Second answer",
                        role="assistant",
                        reasoning_content="Reasoning for second answer",
                    ),
                ),
            ],
        )

        # Execute
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input="Test input",
                responses_api_request={},
                chat_completion_response=response,
            )
        )

        # Assert
        reasoning_items = [item for item in responses_api_response.output if item.type == "reasoning"]
        assert len(reasoning_items) == 1, "Should have exactly one reasoning item"
        assert reasoning_items[0].content[0].text == "Reasoning for first answer"


def test_streaming_chunk_id_raw():
    """Test that streaming chunk IDs are raw (not encoded) to match OpenAI format"""
    chunk = ModelResponseStream(
        id="chunk-123",
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="Hello", role="assistant"),
            )
        ],
    )

    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
        custom_llm_provider="openai",
        litellm_metadata={"model_info": {"id": "gpt-4"}},
    )

    result = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)

    # Streaming chunk IDs should be raw (like OpenAI's msg_xxx format)
    assert result.item_id == "chunk-123"  # Should be raw, not encoded
    assert not result.item_id.startswith("resp_")  # Should NOT have resp_ prefix
