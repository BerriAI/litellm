"""
Test that FakeAnthropicMessagesStreamIterator is wrapped with logging
when websearch_interception converts stream=True to stream=False.

Fixes: https://github.com/BerriAI/litellm/issues/23150
"""

import asyncio
import inspect
import os
import sys
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler


@pytest.mark.asyncio
async def test_fake_stream_wrapped_with_logging_handler():
    """
    When websearch_interception converts stream to non-stream and no agentic
    loop runs, the FakeAnthropicMessagesStreamIterator should be wrapped with
    BaseAnthropicMessagesStreamingIterator.async_sse_wrapper so that
    _handle_streaming_logging is called after all chunks are yielded.
    """
    handler = BaseLLMHTTPHandler()

    # Create a mock non-streaming response (dict) like AnthropicMessagesResponse
    mock_response = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    # Create a mock logging object that indicates websearch converted the stream
    mock_logging_obj = MagicMock()
    mock_logging_obj.model_call_details = {
        "websearch_interception_converted_stream": True,
    }
    mock_logging_obj.dynamic_success_callbacks = []

    # No callbacks that would trigger agentic loop
    with patch("litellm.callbacks", []):
        result = await handler._call_agentic_completion_hooks(
            response=mock_response,
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=mock_logging_obj,
            stream=True,
            custom_llm_provider="anthropic",
            kwargs={},
        )

    # The result should be an async generator (from async_sse_wrapper)
    # NOT a FakeAnthropicMessagesStreamIterator directly
    assert result is not None
    assert inspect.isasyncgen(result), (
        f"Expected async generator from async_sse_wrapper, got {type(result)}"
    )

    # Consume the stream and verify we get SSE-formatted chunks
    chunks = []
    async for chunk in result:
        chunks.append(chunk)

    # Should have multiple SSE events (message_start, content_block_start, etc.)
    assert len(chunks) > 0

    # Verify chunks are bytes (SSE format from async_sse_wrapper)
    for chunk in chunks:
        assert isinstance(chunk, bytes), f"Expected bytes chunk, got {type(chunk)}"


@pytest.mark.asyncio
async def test_fake_stream_logging_handler_called():
    """
    Verify that _handle_streaming_logging is actually called after the
    fake stream is fully consumed.
    """
    handler = BaseLLMHTTPHandler()

    mock_response = {
        "id": "msg_test456",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Test response"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 15, "output_tokens": 8},
    }

    mock_logging_obj = MagicMock()
    mock_logging_obj.model_call_details = {
        "websearch_interception_converted_stream": True,
    }
    mock_logging_obj.dynamic_success_callbacks = []

    with patch("litellm.callbacks", []):
        result = await handler._call_agentic_completion_hooks(
            response=mock_response,
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=mock_logging_obj,
            stream=True,
            custom_llm_provider="anthropic",
            kwargs={},
        )

    # Patch the _handle_streaming_logging to track if it's called
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator"
        ".BaseAnthropicMessagesStreamingIterator._handle_streaming_logging",
        new_callable=AsyncMock,
    ) as mock_logging:
        # Consume the stream
        async for _ in result:
            pass

        # Verify _handle_streaming_logging was called
        mock_logging.assert_called_once()


@pytest.mark.asyncio
async def test_no_websearch_conversion_returns_none():
    """
    When websearch_interception_converted_stream is False,
    _call_agentic_completion_hooks should return None (no fake stream needed).
    """
    handler = BaseLLMHTTPHandler()

    mock_response = {
        "id": "msg_test789",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Normal response"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    mock_logging_obj = MagicMock()
    mock_logging_obj.model_call_details = {}  # No websearch conversion
    mock_logging_obj.dynamic_success_callbacks = []

    with patch("litellm.callbacks", []):
        result = await handler._call_agentic_completion_hooks(
            response=mock_response,
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=mock_logging_obj,
            stream=False,
            custom_llm_provider="anthropic",
            kwargs={},
        )

    assert result is None
