"""
Unit tests for streaming finish_reason with tool calls.

Tests that when streaming responses contain tool calls, the final chunk
has finish_reason="tool_calls" instead of "stop".
"""

import pytest
from unittest.mock import MagicMock
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import GenericStreamingChunk


class TestStreamingFinishReasonToolCalls:
    """Test cases for streaming finish_reason with tool calls."""

    def create_mock_logging_obj(self):
        """Create a mock logging object for tests."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"litellm_params": {}}
        logging_obj.completion_start_time = None
        logging_obj._update_completion_start_time = MagicMock()
        logging_obj.async_success_handler = MagicMock()
        logging_obj.success_handler = MagicMock()
        logging_obj.failure_handler = MagicMock()
        return logging_obj

    def test_openai_streaming_with_tool_calls_stop_finish_reason(self):
        """Test OpenAI streaming chunks with tool calls and stop finish_reason."""
        logging_obj = self.create_mock_logging_obj()

        # Create test chunks - tool calls followed by stop finish_reason
        chunks = [
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            content=None,
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id="call_test",
                                    function=ChoiceDeltaToolCallFunction(
                                        arguments="", name="get_weather"
                                    ),
                                    type="function",
                                )
                            ],
                        ),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            content=None,
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaToolCallFunction(
                                        arguments='{"location": "San Francisco"}'
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(delta=ChoiceDelta(content=None), finish_reason="stop", index=0)
                ],
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
            ),
        ]

        # Create stream wrapper
        stream = CustomStreamWrapper(
            completion_stream=iter(chunks),
            model="gpt-4",
            logging_obj=logging_obj,
            custom_llm_provider="openai",
        )

        # Process chunks and collect finish reasons
        finish_reasons = []
        for chunk in stream:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].finish_reason
            ):
                finish_reasons.append(chunk.choices[0].finish_reason)

        # Verify finish_reason is "tool_calls" not "stop"
        assert len(finish_reasons) == 1
        assert finish_reasons[0] == "tool_calls"

    def test_generic_streaming_with_tool_use(self):
        """Test generic streaming chunks (Cohere/Anthropic style) with tool use."""
        logging_obj = self.create_mock_logging_obj()

        # Create test chunks using GenericStreamingChunk format
        chunks = [
            {
                "text": "",
                "tool_use": {
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    },
                },
                "is_finished": False,
                "finish_reason": None,
                "usage": None,
                "index": 0,
            },
            {
                "text": "",
                "tool_use": None,
                "is_finished": True,
                "finish_reason": "stop",
                "usage": None,
                "index": 0,
            },
        ]

        # Create stream wrapper
        stream = CustomStreamWrapper(
            completion_stream=iter(chunks),
            model="claude-3",
            logging_obj=logging_obj,
            custom_llm_provider="anthropic",
        )

        # Process chunks and collect finish reasons
        finish_reasons = []
        for chunk in stream:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].finish_reason
            ):
                finish_reasons.append(chunk.choices[0].finish_reason)

        # Verify finish_reason is "tool_calls" not "stop"
        assert len(finish_reasons) == 1
        assert finish_reasons[0] == "tool_calls"

    def test_streaming_without_tool_calls_keeps_stop(self):
        """Test that finish_reason remains 'stop' when no tool calls are present."""
        logging_obj = self.create_mock_logging_obj()

        # Create test chunks without tool calls
        chunks = [
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content="Hello", role="assistant"),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content=" world!"),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(delta=ChoiceDelta(content=None), finish_reason="stop", index=0)
                ],
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
            ),
        ]

        # Create stream wrapper
        stream = CustomStreamWrapper(
            completion_stream=iter(chunks),
            model="gpt-4",
            logging_obj=logging_obj,
            custom_llm_provider="openai",
        )

        # Process chunks and collect finish reasons
        finish_reasons = []
        for chunk in stream:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].finish_reason
            ):
                finish_reasons.append(chunk.choices[0].finish_reason)

        # Verify finish_reason remains "stop" without tool calls
        assert len(finish_reasons) == 1
        assert finish_reasons[0] == "stop"

    def test_function_call_finish_reason(self):
        """Test that function_call also triggers tool_calls finish_reason."""
        logging_obj = self.create_mock_logging_obj()

        # Create test chunks with function_call (legacy format)
        chunks = [
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            content=None,
                            role="assistant",
                            function_call={"name": "get_weather", "arguments": ""},
                        ),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=1234567890,
                model="gpt-3.5-turbo",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            content=None,
                            function_call={"arguments": '{"location": "NYC"}'},
                        ),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=1234567890,
                model="gpt-3.5-turbo",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="test-id",
                choices=[
                    Choice(delta=ChoiceDelta(content=None), finish_reason="stop", index=0)
                ],
                created=1234567890,
                model="gpt-3.5-turbo",
                object="chat.completion.chunk",
            ),
        ]

        # Create stream wrapper
        stream = CustomStreamWrapper(
            completion_stream=iter(chunks),
            model="gpt-3.5-turbo",
            logging_obj=logging_obj,
            custom_llm_provider="openai",
        )

        # Process chunks and collect finish reasons
        finish_reasons = []
        for chunk in stream:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].finish_reason
            ):
                finish_reasons.append(chunk.choices[0].finish_reason)

        # Verify finish_reason is "tool_calls" for function_call too
        assert len(finish_reasons) == 1
        assert finish_reasons[0] == "tool_calls"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])