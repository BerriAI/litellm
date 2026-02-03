"""
Test for GitHub Issue #20347: Anthropic streaming silently completes with empty content
instead of raising exception on upstream errors.

This test verifies that when an upstream Anthropic streaming response ends abnormally
(e.g., only message_start is received, no content, no message_stop), the proxy should
detect this and raise an error to the client instead of completing "successfully".
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.pass_through_endpoints.streaming_handler import (
    AnthropicStreamValidator,
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


class TestAnthropicStreamingErrorHandling:
    """Test cases for detecting incomplete/error streaming responses from Anthropic."""

    def test_error_event_in_stream_raises_exception(self):
        """
        Test that when an error event is received in the stream, it is detected
        and an AnthropicError is raised during chunk parsing.

        This is the core of issue #20347 - error events should not be silently swallowed.
        """
        # Simulate chunks that include an error event
        error_chunks = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet-20240229","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
            'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"Internal server error"}}',
        ]

        anthropic_iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )

        # First chunk should parse fine
        result = anthropic_iterator.convert_str_chunk_to_generic_chunk(error_chunks[0])
        assert result is not None

        # Second chunk (error event) should raise AnthropicError
        with pytest.raises(AnthropicError) as exc_info:
            anthropic_iterator.convert_str_chunk_to_generic_chunk(error_chunks[1])

        assert "Internal server error" in str(exc_info.value)

    def test_build_streaming_response_with_error_event(self):
        """
        Test that _build_complete_streaming_response properly handles error events.

        The code should catch AnthropicError and return None to indicate the error.
        This ensures errors in the logging pipeline don't crash silently in background tasks.
        """
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}

        # Chunks that include an error event
        chunks_with_error = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet-20240229","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
            'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"Internal server error"}}',
        ]

        # The implementation catches AnthropicError and returns None
        # The error is logged but not propagated (prevents silent failures in background tasks)
        result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=chunks_with_error,
            litellm_logging_obj=mock_logging_obj,
            model="claude-3-sonnet-20240229",
        )

        # Should return None to indicate an error occurred
        assert result is None

    def test_incomplete_stream_missing_message_stop(self):
        """
        Test that a stream that ends without message_stop is detected as incomplete.

        Scenario: message_start is received, but the connection closes before
        message_stop arrives. This should be detected as an error.
        """
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}

        # Incomplete stream - has message_start but no content and no message_stop
        incomplete_chunks = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet-20240229","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
        ]

        result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=incomplete_chunks,
            litellm_logging_obj=mock_logging_obj,
            model="claude-3-sonnet-20240229",
        )

        # BUG: Currently this returns a "valid" response even though stream was incomplete
        # The result has no content because no content_block_delta was received
        # After fix, this should either:
        # 1. Raise an error indicating incomplete stream, OR
        # 2. Return a response marked as incomplete/error

    def test_stream_with_empty_content_after_message_start(self):
        """
        Test case from issue #20347: stream receives message_start but no content.

        From the issue logs:
        ```
        [EVENT] message_start - input_tokens: 22
        Stream completed WITHOUT exception
        Stream completed but content is EMPTY!
        ```
        """
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}

        # Stream with message_start and message_stop but no content
        empty_content_chunks = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet-20240229","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":22,"output_tokens":0}}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]

        result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=empty_content_chunks,
            litellm_logging_obj=mock_logging_obj,
            model="claude-3-sonnet-20240229",
        )

        # BUG: This appears to be a "successful" response with empty content
        # The client has no way to know something went wrong
        # After fix, empty content with 0 output tokens should be flagged as an error

    def test_normal_successful_stream(self):
        """
        Test that a normal successful stream is processed correctly (baseline test).
        """
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}

        # Normal successful stream
        successful_chunks = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet-20240229","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":22,"output_tokens":1}}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]

        result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=successful_chunks,
            litellm_logging_obj=mock_logging_obj,
            model="claude-3-sonnet-20240229",
        )

        # This should succeed and have content
        assert result is not None


class TestStreamingHandlerErrorPropagation:
    """Test that errors are properly propagated through the streaming handler."""

    @pytest.mark.asyncio
    async def test_chunk_processor_with_error_response(self):
        """
        Test that chunk_processor yields all chunks including error events.

        The issue is that even though error events are yielded to the client,
        the Anthropic SDK doesn't seem to recognize them as errors in some cases.
        """
        # Create a mock httpx response that yields chunks including an error
        mock_response = MagicMock()

        async def mock_aiter_bytes():
            # Yield message_start
            yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet-20240229","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
            # Yield error event
            yield b'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"Internal server error"}}\n\n'

        mock_response.aiter_bytes = mock_aiter_bytes

        # Mock other dependencies
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        mock_passthrough_handler = MagicMock()

        collected_chunks = []

        # Mock asyncio.create_task to prevent background task execution
        with patch('asyncio.create_task'):
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=mock_response,
                request_body={"model": "claude-3-sonnet-20240229"},
                litellm_logging_obj=mock_logging_obj,
                endpoint_type="anthropic",
                start_time=datetime.now(),
                passthrough_success_handler_obj=mock_passthrough_handler,
                url_route="/anthropic/v1/messages",
            ):
                collected_chunks.append(chunk)

        # Verify both chunks were yielded (including the error)
        assert len(collected_chunks) == 2
        # Error event should be in the second chunk
        assert b"error" in collected_chunks[1]
        assert b"Internal server error" in collected_chunks[1]


class TestAnthropicErrorEventParsing:
    """Test the parsing of Anthropic error events."""

    def test_error_event_format(self):
        """
        Test that Anthropic error events are parsed correctly and raise AnthropicError.
        """
        error_event = 'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"Internal server error"}}'

        iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )

        with pytest.raises(AnthropicError) as exc_info:
            iterator.convert_str_chunk_to_generic_chunk(error_event)

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.message

    def test_overloaded_error_event(self):
        """
        Test handling of Anthropic overloaded_error events.
        """
        error_event = 'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'

        iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )

        with pytest.raises(AnthropicError) as exc_info:
            iterator.convert_str_chunk_to_generic_chunk(error_event)

        assert "Overloaded" in exc_info.value.message

    def test_rate_limit_error_event(self):
        """
        Test handling of rate_limit_error events.
        """
        error_event = 'event: error\ndata: {"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}'

        iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )

        with pytest.raises(AnthropicError) as exc_info:
            iterator.convert_str_chunk_to_generic_chunk(error_event)

        assert "Rate limited" in exc_info.value.message


class TestAnthropicStreamValidator:
    """Test the AnthropicStreamValidator class for detecting errors and incomplete streams."""

    def test_validator_detects_error_event(self):
        """Test that the validator detects error events in the stream."""
        validator = AnthropicStreamValidator()

        # Process a normal message_start
        chunk1 = b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123"}}\n\n'
        is_error, error_msg = validator.process_chunk(chunk1)
        assert is_error is False
        assert error_msg is None
        assert validator.received_message_start is True

        # Process an error event
        chunk2 = b'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"Internal server error"}}\n\n'
        is_error, error_msg = validator.process_chunk(chunk2)
        assert is_error is True
        assert error_msg == "Internal server error"
        assert validator.received_error is True
        assert validator.error_type == "api_error"

    def test_validator_detects_message_stop(self):
        """Test that the validator tracks message_stop for proper termination."""
        validator = AnthropicStreamValidator()

        # Process message_start
        chunk1 = b'event: message_start\ndata: {"type":"message_start"}\n\n'
        validator.process_chunk(chunk1)
        assert validator.received_message_start is True
        assert validator.is_stream_complete() is False

        # Process message_stop
        chunk2 = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        validator.process_chunk(chunk2)
        assert validator.received_message_stop is True
        assert validator.is_stream_complete() is True

    def test_validator_incomplete_stream_error(self):
        """Test that get_incomplete_stream_error returns proper SSE format."""
        validator = AnthropicStreamValidator()

        error_chunk = validator.get_incomplete_stream_error()

        # Verify it's valid SSE format
        assert error_chunk.startswith(b"event: error\n")
        assert b"data:" in error_chunk
        assert b"incomplete_stream_error" in error_chunk
        assert b"message_stop" in error_chunk

    def test_validator_handles_multiple_events_in_chunk(self):
        """Test that the validator correctly parses multiple events in a single chunk."""
        validator = AnthropicStreamValidator()

        # Single chunk with multiple events
        multi_event_chunk = b'event: message_start\ndata: {"type":"message_start"}\n\nevent: content_block_start\ndata: {"type":"content_block_start"}\n\n'
        is_error, error_msg = validator.process_chunk(multi_event_chunk)

        assert is_error is False
        assert validator.received_message_start is True

    def test_validator_handles_overloaded_error(self):
        """Test handling of overloaded_error type."""
        validator = AnthropicStreamValidator()

        chunk = b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
        is_error, error_msg = validator.process_chunk(chunk)

        assert is_error is True
        assert error_msg == "Overloaded"
        assert validator.error_type == "overloaded_error"


class TestChunkProcessorWithStreamValidator:
    """Test the chunk_processor method with stream validation."""

    @pytest.mark.asyncio
    async def test_chunk_processor_yields_error_for_incomplete_stream(self):
        """
        Test that chunk_processor yields an error event when stream ends without message_stop.

        This is the core fix for issue #20347.
        """
        mock_response = MagicMock()

        async def mock_aiter_bytes():
            # Only yield message_start, then connection "closes"
            yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet-20240229","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
            # No message_stop - simulates connection drop

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        mock_passthrough_handler = MagicMock()

        collected_chunks = []

        with patch("asyncio.create_task"):
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=mock_response,
                request_body={"model": "claude-3-sonnet-20240229"},
                litellm_logging_obj=mock_logging_obj,
                endpoint_type=EndpointType.ANTHROPIC,
                start_time=datetime.now(),
                passthrough_success_handler_obj=mock_passthrough_handler,
                url_route="/anthropic/v1/messages",
            ):
                collected_chunks.append(chunk)

        # Should have 2 chunks: message_start + synthetic error
        assert len(collected_chunks) == 2

        # Last chunk should be the incomplete stream error
        error_chunk = collected_chunks[-1]
        assert b"event: error" in error_chunk
        assert b"incomplete_stream_error" in error_chunk
        assert b"message_stop" in error_chunk

    @pytest.mark.asyncio
    async def test_chunk_processor_no_error_for_complete_stream(self):
        """Test that a complete stream doesn't get an error event appended."""
        mock_response = MagicMock()

        async def mock_aiter_bytes():
            yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123"}}\n\n'
            yield b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"Hello"}}\n\n'
            yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        mock_passthrough_handler = MagicMock()

        collected_chunks = []

        with patch("asyncio.create_task"):
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=mock_response,
                request_body={"model": "claude-3-sonnet-20240229"},
                litellm_logging_obj=mock_logging_obj,
                endpoint_type=EndpointType.ANTHROPIC,
                start_time=datetime.now(),
                passthrough_success_handler_obj=mock_passthrough_handler,
                url_route="/anthropic/v1/messages",
            ):
                collected_chunks.append(chunk)

        # Should only have 3 chunks (no synthetic error)
        assert len(collected_chunks) == 3

        # No error chunk should be appended
        for chunk in collected_chunks:
            assert b"incomplete_stream_error" not in chunk

    @pytest.mark.asyncio
    async def test_chunk_processor_no_duplicate_error_for_upstream_error(self):
        """
        Test that when upstream sends an error event, we don't add another error.

        The upstream error is already in the stream, so we shouldn't append our own.
        """
        mock_response = MagicMock()

        async def mock_aiter_bytes():
            yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123"}}\n\n'
            yield b'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"Internal server error"}}\n\n'

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        mock_passthrough_handler = MagicMock()

        collected_chunks = []

        with patch("asyncio.create_task"):
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=mock_response,
                request_body={"model": "claude-3-sonnet-20240229"},
                litellm_logging_obj=mock_logging_obj,
                endpoint_type=EndpointType.ANTHROPIC,
                start_time=datetime.now(),
                passthrough_success_handler_obj=mock_passthrough_handler,
                url_route="/anthropic/v1/messages",
            ):
                collected_chunks.append(chunk)

        # Should have exactly 2 chunks (message_start + upstream error)
        # No additional error chunk should be added
        assert len(collected_chunks) == 2

        # Verify the error is the upstream one, not our synthetic one
        error_chunk = collected_chunks[-1]
        assert b"api_error" in error_chunk
        assert b"Internal server error" in error_chunk
        assert b"incomplete_stream_error" not in error_chunk

    @pytest.mark.asyncio
    async def test_chunk_processor_no_validation_for_openai_endpoint(self):
        """Test that OpenAI endpoint type doesn't get Anthropic validation."""
        mock_response = MagicMock()

        async def mock_aiter_bytes():
            # Incomplete OpenAI-style stream (no done event)
            yield b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}\n\n'

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        mock_passthrough_handler = MagicMock()

        collected_chunks = []

        with patch("asyncio.create_task"):
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=mock_response,
                request_body={"model": "gpt-4"},
                litellm_logging_obj=mock_logging_obj,
                endpoint_type=EndpointType.OPENAI,
                start_time=datetime.now(),
                passthrough_success_handler_obj=mock_passthrough_handler,
                url_route="/openai/v1/chat/completions",
            ):
                collected_chunks.append(chunk)

        # Should only have 1 chunk - no Anthropic error appended
        assert len(collected_chunks) == 1
        assert b"incomplete_stream_error" not in collected_chunks[0]

    @pytest.mark.asyncio
    async def test_chunk_processor_validates_vertex_ai_anthropic_format(self):
        """Test that Vertex AI with streamRawPredict gets Anthropic validation."""
        mock_response = MagicMock()

        async def mock_aiter_bytes():
            # Anthropic format via Vertex AI, but incomplete
            yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123"}}\n\n'

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        mock_passthrough_handler = MagicMock()

        collected_chunks = []

        with patch("asyncio.create_task"):
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=mock_response,
                request_body={"model": "claude-3-sonnet@20240229"},
                litellm_logging_obj=mock_logging_obj,
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=datetime.now(),
                passthrough_success_handler_obj=mock_passthrough_handler,
                url_route="/vertex-ai/publishers/anthropic/models/claude-3-sonnet@20240229:streamRawPredict",
            ):
                collected_chunks.append(chunk)

        # Should have 2 chunks: message_start + incomplete stream error
        assert len(collected_chunks) == 2
        assert b"incomplete_stream_error" in collected_chunks[-1]
