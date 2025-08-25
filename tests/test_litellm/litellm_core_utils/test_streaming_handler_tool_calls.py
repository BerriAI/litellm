"""Test that streaming_handler correctly sets finish_reason for tool calls"""
import json
import os
import sys
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.utils import ModelResponseListIterator


class TestStreamingHandlerToolCallFinishReason:
    """Test that streaming handler correctly handles finish_reason for tool calls"""
    
    def test_tool_calls_finish_reason_override(self):
        """Test that finish_reason is overridden to 'tool_calls' when tool calls are present"""
        # Create chunks that simulate a tool call streaming response
        chunks = [
            # First chunk - assistant role
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            role="assistant",
                            content=None,
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Second chunk - tool call begins
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content=None,
                            tool_calls=[
                                ChatCompletionToolCallChunk(
                                    index=0,
                                    id="call_abc123",
                                    type="function",
                                    function=ChatCompletionToolCallFunctionChunk(
                                        name="get_weather",
                                        arguments=""
                                    )
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Third chunk - tool call arguments
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content=None,
                            tool_calls=[
                                ChatCompletionToolCallChunk(
                                    index=0,
                                    function=ChatCompletionToolCallFunctionChunk(
                                        arguments='{"location": "San Francisco"}'
                                    )
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Final chunk - finish with "stop" (should be overridden to "tool_calls")
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=None),
                        finish_reason="stop",  # This should be overridden to "tool_calls"
                    )
                ],
            ),
        ]
        
        # Create stream wrapper
        completion_stream = ModelResponseListIterator(model_responses=chunks)
        stream_wrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model="gpt-4",
            logging_obj=MagicMock(),
            custom_llm_provider="openai",
        )
        
        # Process chunks and verify finish_reason
        processed_chunks = list(stream_wrapper)
        
        # Check that tool_call flag was set
        assert stream_wrapper.tool_call is True
        
        # Check that the last chunk has finish_reason "tool_calls"
        last_chunk = processed_chunks[-1]
        assert last_chunk.choices[0].finish_reason == "tool_calls"
        
    def test_function_call_finish_reason_override(self):
        """Test that finish_reason is overridden for legacy function_call format"""
        # Create chunks with legacy function_call format
        chunks = [
            # First chunk - assistant role
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-3.5-turbo",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            role="assistant",
                            content=None,
                            function_call={"name": "get_weather", "arguments": ""},
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Second chunk - function arguments
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-3.5-turbo",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content=None,
                            function_call={"arguments": '{"location": "New York"}'},
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Final chunk - finish with "stop"
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-3.5-turbo",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=None),
                        finish_reason="stop",  # Should be overridden to "tool_calls"
                    )
                ],
            ),
        ]
        
        # Create stream wrapper
        completion_stream = ModelResponseListIterator(model_responses=chunks)
        stream_wrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model="gpt-3.5-turbo",
            logging_obj=MagicMock(),
            custom_llm_provider="openai",
        )
        
        # Process chunks
        processed_chunks = list(stream_wrapper)
        
        # Check that tool_call flag was set
        assert stream_wrapper.tool_call is True
        
        # Check that the last chunk has finish_reason "tool_calls"
        last_chunk = processed_chunks[-1]
        assert last_chunk.choices[0].finish_reason == "tool_calls"
        
    def test_no_tool_calls_finish_reason_unchanged(self):
        """Test that finish_reason remains 'stop' when no tool calls are present"""
        # Create chunks without tool calls
        chunks = [
            # First chunk - assistant role with content
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            role="assistant",
                            content="The weather in ",
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Second chunk - more content
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content="San Francisco is nice today.",
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Final chunk - finish with "stop"
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=None),
                        finish_reason="stop",  # Should remain "stop"
                    )
                ],
            ),
        ]
        
        # Create stream wrapper
        completion_stream = ModelResponseListIterator(model_responses=chunks)
        stream_wrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model="gpt-4",
            logging_obj=MagicMock(),
            custom_llm_provider="openai",
        )
        
        # Process chunks
        processed_chunks = list(stream_wrapper)
        
        # Check that tool_call flag was NOT set
        assert stream_wrapper.tool_call is False
        
        # Check that the last chunk still has finish_reason "stop"
        last_chunk = processed_chunks[-1]
        assert last_chunk.choices[0].finish_reason == "stop"
        
    def test_other_finish_reasons_unchanged(self):
        """Test that non-stop finish reasons are not modified"""
        # Create chunks with tool calls but different finish_reason
        chunks = [
            # First chunk - tool call
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            role="assistant",
                            tool_calls=[
                                ChatCompletionToolCallChunk(
                                    index=0,
                                    id="call_xyz",
                                    type="function",
                                    function=ChatCompletionToolCallFunctionChunk(
                                        name="test_function",
                                        arguments='{"test": true}'
                                    )
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Final chunk - finish with "length" (should NOT be changed)
            ModelResponseStream(
                id="test-id",
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=None),
                        finish_reason="length",  # Should remain "length"
                    )
                ],
            ),
        ]
        
        # Create stream wrapper
        completion_stream = ModelResponseListIterator(model_responses=chunks)
        stream_wrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model="gpt-4",
            logging_obj=MagicMock(),
            custom_llm_provider="openai",
        )
        
        # Process chunks
        processed_chunks = list(stream_wrapper)
        
        # Check that tool_call flag was set
        assert stream_wrapper.tool_call is True
        
        # Check that the last chunk still has finish_reason "length"
        last_chunk = processed_chunks[-1]
        assert last_chunk.choices[0].finish_reason == "length"