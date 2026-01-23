"""
Test reasoning content preservation in Responses API transformation
"""

from unittest.mock import AsyncMock, Mock

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)


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
        transformed_chunk = (
            iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
        )

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
        transformed_chunk = (
            iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
        )

        # Assert
        assert transformed_chunk.delta == "First, let me analyze..."
        assert transformed_chunk.type == "response.reasoning_summary_text.delta"

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
        transformed_chunk = (
            iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
        )

        # Assert
        assert transformed_chunk.delta == "Regular content only"
        assert transformed_chunk.type == "response.output_text.delta"

    def test_reasoning_delta_item_id_is_stable(self):
        """Test reasoning deltas share a stable item_id across the stream"""
        chunk1 = ModelResponseStream(
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
                        reasoning_content="First thought",
                    ),
                )
            ],
        )

        chunk2 = ModelResponseStream(
            id="test-id-2",
            created=1234567891,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="",
                        role="assistant",
                        reasoning_content="Second thought",
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

        evt1 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk1)
        evt2 = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk2)

        assert evt1 is not None
        assert evt2 is not None
        assert evt1.item_id == evt2.item_id

    @pytest.mark.asyncio
    async def test_reasoning_done_emitted_before_output_text_delta(self):
        """Ensure reasoning_summary_text.done is emitted before output_text.delta"""
        chunk_reasoning = ModelResponseStream(
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
                        reasoning_content="Thinking...",
                    ),
                )
            ],
        )

        chunk_text = ModelResponseStream(
            id="test-id",
            created=1234567891,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="Final answer",
                        role=None,
                    ),
                )
            ],
        )

        mock_stream = AsyncMock()
        mock_stream.logging_obj = Mock()
        mock_stream.__anext__.side_effect = [
            chunk_reasoning,
            chunk_text,
            StopAsyncIteration,
        ]

        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        events = []
        try:
            while True:
                events.append(await iterator.__anext__())
                if len(events) > 20:
                    break
        except StopAsyncIteration:
            pass

        event_types = [evt.type for evt in events]
        assert (
            "response.reasoning_summary_text.done" in event_types
        ), "Expected reasoning_summary_text.done event"
        assert (
            "response.output_text.delta" in event_types
        ), "Expected output_text.delta event"

        done_index = event_types.index("response.reasoning_summary_text.done")
        text_index = event_types.index("response.output_text.delta")
        assert done_index < text_index

    @pytest.mark.asyncio
    async def test_message_item_waits_until_reasoning_done_on_empty_chunk(self):
        """Ensure message output_item.added is delayed until reasoning is closed."""
        chunk_reasoning = ModelResponseStream(
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
                        reasoning_content="Thinking...",
                    ),
                )
            ],
        )

        chunk_empty = ModelResponseStream(
            id="test-id",
            created=1234567891,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="",
                        role="assistant",
                        annotations=[
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "start_index": 0,
                                    "end_index": 5,
                                    "url": "https://example.com",
                                    "title": "Example",
                                },
                            }
                        ],
                    ),
                )
            ],
        )

        chunk_text = ModelResponseStream(
            id="test-id",
            created=1234567892,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="Final answer",
                        role=None,
                    ),
                )
            ],
        )

        mock_stream = AsyncMock()
        mock_stream.logging_obj = Mock()
        mock_stream.__anext__.side_effect = [
            chunk_reasoning,
            chunk_empty,
            chunk_text,
            StopAsyncIteration,
        ]

        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        events = []
        try:
            while True:
                events.append(await iterator.__anext__())
                if len(events) > 50:
                    break
        except StopAsyncIteration:
            pass

        message_added = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
            and getattr(e.item, "type", None) == "message"
        ][0]
        reasoning_done = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
            and getattr(e.item, "type", None) == "reasoning"
        ][0]

        assert events.index(reasoning_done) < events.index(message_added)

    @pytest.mark.asyncio
    async def test_reasoning_lifecycle_events_order_and_indexes(self):
        """Validate reasoning lifecycle events and output_index alignment"""
        chunk_reasoning = ModelResponseStream(
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
                        reasoning_content="Thinking...",
                    ),
                )
            ],
        )

        chunk_text = ModelResponseStream(
            id="test-id",
            created=1234567891,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="Final answer",
                        role=None,
                    ),
                )
            ],
        )

        mock_stream = AsyncMock()
        mock_stream.logging_obj = Mock()
        mock_stream.__anext__.side_effect = [
            chunk_reasoning,
            chunk_text,
            StopAsyncIteration,
        ]

        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        events = []
        try:
            while True:
                events.append(await iterator.__anext__())
                if len(events) > 30:
                    break
        except StopAsyncIteration:
            pass

        event_types = [evt.type for evt in events]
        assert ResponsesAPIStreamEvents.RESPONSE_CREATED in event_types
        assert ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS in event_types
        assert ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA in event_types
        assert ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE in event_types
        assert ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE in event_types
        assert ResponsesAPIStreamEvents.RESPONSE_PART_ADDED in event_types
        assert ResponsesAPIStreamEvents.CONTENT_PART_ADDED in event_types
        assert ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE in event_types
        assert ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA in event_types

        reasoning_added = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
            and getattr(e.item, "type", None) == "reasoning"
        ][0]
        message_added = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
            and getattr(e.item, "type", None) == "message"
        ][0]

        assert reasoning_added.output_index == 0
        assert isinstance(getattr(reasoning_added.item, "summary", None), list)
        assert reasoning_added.item.summary == []
        assert message_added.output_index == 1

        reasoning_done = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
            and getattr(e.item, "type", None) == "reasoning"
        ][0]
        assert events.index(reasoning_done) < events.index(message_added)

        part_added = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.RESPONSE_PART_ADDED
        ][0]
        assert part_added.output_index == 0
        assert part_added.part.get("type") == "summary_text"
        assert getattr(part_added, "summary_index", None) == 0

        reasoning_delta = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA
        ][0]
        assert reasoning_delta.output_index == 0
        assert getattr(reasoning_delta, "summary_index", None) == 0
        assert events.index(part_added) < events.index(reasoning_delta)

        content_part_added = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED
        ][0]
        assert content_part_added.output_index == 1
        assert content_part_added.item_id == message_added.item.id
        assert events.index(message_added) < events.index(content_part_added)

        output_text_delta = [
            e for e in events
            if e.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        ][0]
        assert output_text_delta.output_index == 1
        assert output_text_delta.item_id == message_added.item.id
        assert events.index(content_part_added) < events.index(output_text_delta)

        sequence_numbers = [getattr(e, "sequence_number", None) for e in events]
        assert None not in sequence_numbers
        assert sequence_numbers == sorted(sequence_numbers)

    @pytest.mark.asyncio
    async def test_sequence_numbers_monotonic_with_tools_and_annotations(self):
        """Validate sequence_number ordering across reasoning, tools, and annotations"""
        chunk_reasoning = ModelResponseStream(
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
                        reasoning_content="Thinking...",
                    ),
                )
            ],
        )

        chunk_tool = ModelResponseStream(
            id="test-id",
            created=1234567891,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="",
                        role="assistant",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "do_thing",
                                    "arguments": '{"x":1}',
                                },
                            }
                        ],
                    ),
                )
            ],
        )

        chunk_annotations = ModelResponseStream(
            id="test-id",
            created=1234567892,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="",
                        role="assistant",
                        annotations=[
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "start_index": 0,
                                    "end_index": 5,
                                    "url": "https://example.com",
                                    "title": "Example",
                                },
                            }
                        ],
                    ),
                )
            ],
        )

        chunk_text = ModelResponseStream(
            id="test-id",
            created=1234567893,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="Final answer",
                        role=None,
                    ),
                )
            ],
        )

        mock_stream = AsyncMock()
        mock_stream.logging_obj = Mock()
        mock_stream.__anext__.side_effect = [
            chunk_reasoning,
            chunk_tool,
            chunk_annotations,
            chunk_text,
            StopAsyncIteration,
        ]

        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=mock_stream,
            request_input="Test input",
            responses_api_request={},
        )

        events = []
        try:
            while True:
                events.append(await iterator.__anext__())
                if len(events) > 60:
                    break
        except StopAsyncIteration:
            pass

        event_types = [evt.type for evt in events]
        assert ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA in event_types
        assert ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED in event_types

        sequence_numbers = [getattr(e, "sequence_number", None) for e in events]
        assert None not in sequence_numbers
        assert sequence_numbers == sorted(sequence_numbers)


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
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="Test input",
            responses_api_request={},
            chat_completion_response=response,
        )

        # Assert
        assert hasattr(responses_api_response, "output")
        assert len(responses_api_response.output) > 0

        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert len(reasoning_items) > 0, "No reasoning item found in output"

        reasoning_item = reasoning_items[0]
        assert (
            reasoning_item.summary[0].text
            == "Let me think step by step about this problem..."
        )

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
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="Test input",
            responses_api_request={},
            chat_completion_response=response,
        )

        # Assert
        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert (
            len(reasoning_items) == 0
        ), "Should have no reasoning items when no reasoning content present"

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
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="Test input",
            responses_api_request={},
            chat_completion_response=response,
        )

        # Assert
        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert len(reasoning_items) == 1, "Should have exactly one reasoning item"
        assert reasoning_items[0].summary[0].text == "Reasoning for first answer"


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
