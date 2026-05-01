"""
Test that FakeAnthropicMessagesStreamIterator is wrapped with logging
when websearch_interception converts stream=True to stream=False.

Fixes: https://github.com/BerriAI/litellm/issues/23150
"""

import inspect
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler


def _make_mock_response():
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _make_mock_logging_obj():
    mock_logging_obj = MagicMock()
    mock_logging_obj.model_call_details = {
        "websearch_interception_converted_stream": True,
    }
    mock_logging_obj.dynamic_success_callbacks = []
    return mock_logging_obj


@pytest.mark.asyncio
async def test_fake_stream_wrapped_with_logging_handler():
    """
    When websearch_interception converts stream to non-stream and no agentic
    loop runs, the FakeAnthropicMessagesStreamIterator should be wrapped with
    BaseAnthropicMessagesStreamingIterator.async_sse_wrapper so that
    _handle_streaming_logging is called after all chunks are yielded.
    """
    handler = BaseLLMHTTPHandler()

    # Patch _handle_streaming_logging to prevent background task failures
    # from the mock logging object (MagicMock is not async-compatible).
    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator"
            ".BaseAnthropicMessagesStreamingIterator._handle_streaming_logging",
            new_callable=AsyncMock,
        ),
        patch("litellm.callbacks", []),
    ):
        result = await handler._call_agentic_completion_hooks(
            response=_make_mock_response(),
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=_make_mock_logging_obj(),
            stream=True,
            custom_llm_provider="anthropic",
            kwargs={},
        )

    # The result should be an async generator (from async_sse_wrapper)
    assert result is not None
    assert inspect.isasyncgen(result), (
        f"Expected async generator from async_sse_wrapper, got {type(result)}"
    )

    # Consume the stream and verify we get SSE-formatted chunks
    chunks = []
    async for chunk in result:
        chunks.append(chunk)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, bytes), f"Expected bytes chunk, got {type(chunk)}"


@pytest.mark.asyncio
async def test_fake_stream_logging_handler_called():
    """
    Verify that _handle_streaming_logging is actually called after the
    fake stream is fully consumed.
    """
    handler = BaseLLMHTTPHandler()

    # Patch _handle_streaming_logging BEFORE creating the generator so the
    # mock is captured by async_sse_wrapper's closure.
    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator"
            ".BaseAnthropicMessagesStreamingIterator._handle_streaming_logging",
            new_callable=AsyncMock,
        ) as mock_logging,
        patch("litellm.callbacks", []),
    ):
        result = await handler._call_agentic_completion_hooks(
            response=_make_mock_response(),
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=_make_mock_logging_obj(),
            stream=True,
            custom_llm_provider="anthropic",
            kwargs={},
        )

        # Consume the stream fully so _handle_streaming_logging fires
        async for _ in result:
            pass

        mock_logging.assert_called_once()


@pytest.mark.asyncio
async def test_no_websearch_conversion_returns_none():
    """
    When websearch_interception_converted_stream is False,
    _call_agentic_completion_hooks should return None.
    """
    handler = BaseLLMHTTPHandler()

    mock_logging_obj = MagicMock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.dynamic_success_callbacks = []

    with patch("litellm.callbacks", []):
        result = await handler._call_agentic_completion_hooks(
            response=_make_mock_response(),
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
