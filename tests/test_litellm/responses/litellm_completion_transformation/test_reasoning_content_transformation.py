"""
Test reasoning content preservation in Responses API transformation
"""

from unittest.mock import AsyncMock

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
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
            reasoning_item.content[0].text
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
