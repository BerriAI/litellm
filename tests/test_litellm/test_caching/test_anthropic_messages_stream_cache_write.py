"""
Tests for streaming cache write in AgenticAnthropicStreamingIterator.

Verifies that when a stream completes and the response is rebuilt from SSE events,
the complete response is persisted to the LiteLLM cache.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.agentic_streaming_iterator import (
    AgenticAnthropicStreamingIterator,
)


def _make_sse_bytes() -> bytes:
    """Create valid SSE bytes representing a complete Anthropic response."""
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello!"},
            },
        ),
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
        ),
        (
            "message_stop",
            {"type": "message_stop"},
        ),
    ]
    lines = []
    for event_type, data in events:
        lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return "\n".join(lines).encode()


async def _make_async_iter(data: bytes):
    """Create an async iterator that yields the data in one chunk."""
    yield data


def _make_mock_logging_obj(
    should_store: bool = True,
    preset_cache_key: str = "test-cache-key",
):
    """Create a mock logging_obj with _llm_caching_handler attached."""
    caching_handler = MagicMock()
    caching_handler.request_kwargs = {
        "stream": True,
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    caching_handler.preset_cache_key = preset_cache_key
    caching_handler.original_function = MagicMock()
    caching_handler._should_store_result_in_cache = MagicMock(return_value=should_store)
    caching_handler.dual_cache = MagicMock()

    logging_obj = MagicMock()
    logging_obj._llm_caching_handler = caching_handler
    logging_obj.litellm_call_id = "test-call-id"
    return logging_obj


def _make_mock_http_handler():
    """Create a mock http_handler whose agentic hook returns None (no follow-up)."""
    handler = MagicMock()
    handler._call_agentic_completion_hooks = AsyncMock(return_value=None)
    return handler


def _create_iterator(
    sse_bytes: bytes,
    logging_obj=None,
    http_handler=None,
):
    """Helper to create an AgenticAnthropicStreamingIterator for testing."""
    if logging_obj is None:
        logging_obj = _make_mock_logging_obj()
    if http_handler is None:
        http_handler = _make_mock_http_handler()

    return AgenticAnthropicStreamingIterator(
        completion_stream=_make_async_iter(sse_bytes),
        http_handler=http_handler,
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hi"}],
        anthropic_messages_provider_config=MagicMock(),
        anthropic_messages_optional_request_params={},
        logging_obj=logging_obj,
        custom_llm_provider="anthropic",
        kwargs={},
    )


@pytest.mark.asyncio
async def test_persist_to_cache_called_on_stream_completion():
    """Should persist rebuilt response to cache when stream completes."""
    mock_cache = MagicMock()
    mock_cache.async_add_cache = AsyncMock(return_value=None)

    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj()
    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", mock_cache):
        # Consume the entire stream
        chunks = []
        async for chunk in iterator:
            chunks.append(chunk)

    # Verify we got the SSE bytes
    assert len(chunks) >= 1

    # Verify async_add_cache was called
    mock_cache.async_add_cache.assert_called_once()

    # Verify the cached response is the rebuilt dict as JSON
    call_args = mock_cache.async_add_cache.call_args
    cached_json = call_args[0][0]
    cached_response = json.loads(cached_json)
    assert cached_response["id"] == "msg_test_123"
    assert cached_response["model"] == "claude-sonnet-4-20250514"
    assert cached_response["role"] == "assistant"
    assert cached_response["stop_reason"] == "end_turn"
    assert cached_response["content"][0]["type"] == "text"
    assert cached_response["content"][0]["text"] == "Hello!"
    assert cached_response["usage"]["input_tokens"] == 10
    assert cached_response["usage"]["output_tokens"] == 5

    # Verify dual_cache was passed
    assert call_args[1]["dynamic_cache_object"] is not None

    # Verify cache_key was passed
    assert call_args[1]["cache_key"] == "test-cache-key"


@pytest.mark.asyncio
async def test_persist_to_cache_uses_request_cache_key_as_fallback():
    """Should use request_kwargs cache_key when preset_cache_key is None."""
    mock_cache = MagicMock()
    mock_cache.async_add_cache = AsyncMock(return_value=None)

    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj(preset_cache_key=None)
    # Set a cache_key in request_kwargs
    logging_obj._llm_caching_handler.request_kwargs["cache_key"] = "fallback-key"

    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", mock_cache):
        async for _ in iterator:
            pass

    mock_cache.async_add_cache.assert_called_once()
    call_kwargs = mock_cache.async_add_cache.call_args[1]
    assert call_kwargs["cache_key"] == "fallback-key"


@pytest.mark.asyncio
async def test_persist_to_cache_skipped_when_no_caching_handler():
    """Should not attempt cache write when logging_obj has no _llm_caching_handler."""
    mock_cache = MagicMock()
    mock_cache.async_add_cache = AsyncMock(return_value=None)

    sse_bytes = _make_sse_bytes()
    logging_obj = MagicMock()
    logging_obj._llm_caching_handler = None
    logging_obj.litellm_call_id = "test-call-id"

    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", mock_cache):
        async for _ in iterator:
            pass

    mock_cache.async_add_cache.assert_not_called()


@pytest.mark.asyncio
async def test_persist_to_cache_skipped_when_should_store_returns_false():
    """Should not cache when _should_store_result_in_cache returns False."""
    mock_cache = MagicMock()
    mock_cache.async_add_cache = AsyncMock(return_value=None)

    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj(should_store=False)

    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", mock_cache):
        async for _ in iterator:
            pass

    mock_cache.async_add_cache.assert_not_called()


@pytest.mark.asyncio
async def test_persist_to_cache_skipped_when_stream_not_in_request_kwargs():
    """Should not cache when request_kwargs.stream is not True."""
    mock_cache = MagicMock()
    mock_cache.async_add_cache = AsyncMock(return_value=None)

    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj()
    logging_obj._llm_caching_handler.request_kwargs["stream"] = False

    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", mock_cache):
        async for _ in iterator:
            pass

    mock_cache.async_add_cache.assert_not_called()


@pytest.mark.asyncio
async def test_persist_to_cache_skipped_when_litellm_cache_is_none():
    """Should not attempt cache write when litellm.cache is None."""
    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj()

    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", None):
        async for _ in iterator:
            pass

    # No assertion needed - just verify no exception was raised
    assert iterator._response_cached is False


@pytest.mark.asyncio
async def test_persist_to_cache_not_called_twice():
    """Should only persist to cache once even if called multiple times."""
    mock_cache = MagicMock()
    mock_cache.async_add_cache = AsyncMock(return_value=None)

    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj()
    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", mock_cache):
        async for _ in iterator:
            pass
        # Call persist again directly - should be a no-op
        rebuilt = (
            AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(
                [sse_bytes]
            )
        )
        iterator._persist_to_cache(rebuilt)

    # Should only have been called once
    mock_cache.async_add_cache.assert_called_once()


@pytest.mark.asyncio
async def test_persist_to_cache_removes_metadata_if_none():
    """Should remove metadata from request_kwargs if it's None."""
    mock_cache = MagicMock()
    mock_cache.async_add_cache = AsyncMock(return_value=None)

    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj()
    logging_obj._llm_caching_handler.request_kwargs["metadata"] = None

    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", mock_cache):
        async for _ in iterator:
            pass

    call_kwargs = mock_cache.async_add_cache.call_args[1]
    assert "metadata" not in call_kwargs


@pytest.mark.asyncio
async def test_persist_to_cache_removes_custom_llm_provider():
    """Should remove custom_llm_provider from request_kwargs."""
    mock_cache = MagicMock()
    mock_cache.async_add_cache = AsyncMock(return_value=None)

    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj()
    logging_obj._llm_caching_handler.request_kwargs["custom_llm_provider"] = "anthropic"

    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", mock_cache):
        async for _ in iterator:
            pass

    call_kwargs = mock_cache.async_add_cache.call_args[1]
    assert "custom_llm_provider" not in call_kwargs


@pytest.mark.asyncio
async def test_persist_to_cache_with_real_local_cache():
    """Should persist to a real in-memory cache and be retrievable."""
    from litellm.caching.caching import Cache

    real_cache = Cache(type="local")
    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj(preset_cache_key="real-cache-test-key")
    # Set dual_cache to None so real cache doesn't try to await a mock
    logging_obj._llm_caching_handler.dual_cache = None
    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", real_cache):
        # Consume the entire stream — triggers _persist_to_cache
        async for _ in iterator:
            pass

        # Allow the async cache write task to complete
        await asyncio.sleep(0.1)

        # Verify the response was actually stored in the real cache
        cached = await real_cache.async_get_cache(cache_key="real-cache-test-key")
        assert cached is not None

        # The cache may return a string or dict depending on the backend
        if isinstance(cached, str):
            cached_response = json.loads(cached)
        else:
            cached_response = cached
        assert cached_response["id"] == "msg_test_123"
        assert cached_response["content"][0]["text"] == "Hello!"
        assert cached_response["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_persist_to_cache_handles_exceptions_gracefully():
    """Should not raise when cache write setup fails."""
    sse_bytes = _make_sse_bytes()
    logging_obj = _make_mock_logging_obj()
    # Make _should_store_result_in_cache raise an exception
    logging_obj._llm_caching_handler._should_store_result_in_cache.side_effect = (
        RuntimeError("unexpected error")
    )

    iterator = _create_iterator(sse_bytes, logging_obj=logging_obj)

    with patch("litellm.cache", MagicMock()):
        # Should not raise
        async for _ in iterator:
            pass

    assert iterator._response_cached is False
