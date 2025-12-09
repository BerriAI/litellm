"""
Tests for async_post_call_streaming_iterator_hook fix.

Verifies that the hook:
1. Is an async generator (not a sync function)
2. Properly iterates through callback chain
3. Actually yields chunks from async generators
"""

import os
import sys
from typing import AsyncGenerator, Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging


class MockStreamingCallback(CustomLogger):
    """Test callback that tracks chunk processing."""

    def __init__(self, prefix: str = ""):
        super().__init__()
        self.prefix = prefix
        self.chunks_processed = 0

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[Any, None],
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """Transform chunks by tracking and optionally prefixing."""
        async for chunk in response:
            self.chunks_processed += 1
            # Optionally modify chunk content for testing
            if self.prefix and isinstance(chunk, dict):
                if "choices" in chunk:
                    for choice in chunk["choices"]:
                        if "delta" in choice and "content" in choice["delta"]:
                            choice["delta"]["content"] = (
                                f"[{self.prefix}]" + choice["delta"]["content"]
                            )
            yield chunk


async def mock_streaming_response() -> AsyncGenerator[dict, None]:
    """Simulate an LLM streaming response."""
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " "}}]},
        {"choices": [{"delta": {"content": "World"}}]},
        {"choices": [{"delta": {"content": "!"}}]},
    ]
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_streaming_hook_is_async_generator():
    """Verify that the hook is an async generator that yields chunks."""
    # Arrange
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    callback = MockStreamingCallback()

    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    request_data = {"model": "gpt-4", "messages": []}

    with patch.object(litellm, "callbacks", [callback]):
        # Act
        result = proxy_logging.async_post_call_streaming_iterator_hook(
            response=mock_streaming_response(),
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )

        # Assert - result should be an async generator
        assert hasattr(result, "__anext__"), "Result should be an async iterator"

        # Collect chunks
        collected_chunks = []
        async for chunk in result:
            collected_chunks.append(chunk)

        # Verify all chunks were yielded
        assert (
            len(collected_chunks) == 4
        ), f"Expected 4 chunks, got {len(collected_chunks)}"
        assert (
            callback.chunks_processed == 4
        ), "Callback should have processed 4 chunks"


@pytest.mark.asyncio
async def test_streaming_hook_chains_multiple_callbacks():
    """Verify that multiple callbacks are properly chained."""
    # Arrange
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    callback1 = MockStreamingCallback(prefix="CB1")
    callback2 = MockStreamingCallback(prefix="CB2")

    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    request_data = {"model": "gpt-4", "messages": []}

    with patch.object(litellm, "callbacks", [callback1, callback2]):
        # Act
        result = proxy_logging.async_post_call_streaming_iterator_hook(
            response=mock_streaming_response(),
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )

        # Collect chunks
        collected_chunks = []
        async for chunk in result:
            collected_chunks.append(chunk)

        # Assert - both callbacks should have processed all chunks
        assert callback1.chunks_processed == 4
        assert callback2.chunks_processed == 4

        # Verify chaining worked (CB2 wraps CB1's output)
        first_content = collected_chunks[0]["choices"][0]["delta"]["content"]
        assert "[CB2]" in first_content, "CB2 prefix should be present"
        assert "[CB1]" in first_content, "CB1 prefix should be present (wrapped by CB2)"


@pytest.mark.asyncio
async def test_streaming_hook_handles_empty_callbacks():
    """Verify that the hook works with no callbacks registered."""
    # Arrange
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())

    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    request_data = {"model": "gpt-4", "messages": []}

    with patch.object(litellm, "callbacks", []):
        # Act
        result = proxy_logging.async_post_call_streaming_iterator_hook(
            response=mock_streaming_response(),
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )

        # Collect chunks
        collected_chunks = []
        async for chunk in result:
            collected_chunks.append(chunk)

        # Assert - all chunks should pass through unchanged
        assert len(collected_chunks) == 4


@pytest.mark.asyncio
async def test_streaming_hook_propagates_callback_errors():
    """Verify that callback errors during iteration are properly propagated."""
    # Arrange
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())

    class FailingCallback(CustomLogger):
        async def async_post_call_streaming_iterator_hook(
            self,
            user_api_key_dict: UserAPIKeyAuth,
            response: AsyncGenerator[Any, None],
            request_data: dict,
        ) -> AsyncGenerator[Any, None]:
            raise RuntimeError("Callback failed!")
            yield  # Make it a generator

    failing_callback = FailingCallback()

    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    request_data = {"model": "gpt-4", "messages": []}

    with patch.object(litellm, "callbacks", [failing_callback]):
        # Act
        result = proxy_logging.async_post_call_streaming_iterator_hook(
            response=mock_streaming_response(),
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )

        # Assert - error should propagate when iterating
        with pytest.raises(RuntimeError, match="Callback failed!"):
            async for _ in result:
                pass
