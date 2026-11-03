"""
Tests for async_post_call_streaming_iterator_hook fix.

Verifies that the hook:
1. Is an async generator (not a sync function)
2. Properly iterates through callback chain
3. Actually yields chunks from async generators
"""

import logging
from typing import AsyncGenerator, Any
from unittest.mock import MagicMock, patch

import pytest


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
        request_data: dict[str, object],
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
        assert callback.chunks_processed == 4, "Callback should have processed 4 chunks"


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


class CleanupRecordingCallback(CustomLogger):
    """Iterator hook whose cleanup marks when it ran."""

    def __init__(self):
        super().__init__()
        self.cleaned_up = False

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[Any, None],
        request_data: dict[str, object],
    ) -> AsyncGenerator[Any, None]:
        try:
            async for chunk in response:
                yield chunk
        finally:
            self.cleaned_up = True


@pytest.mark.asyncio
async def test_closing_the_stream_runs_every_callback_cleanup_before_returning():
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    callbacks = [CleanupRecordingCallback(), CleanupRecordingCallback()]

    with patch.object(litellm, "callbacks", callbacks):
        ProxyLogging._callback_capabilities_cache.clear()
        stream = proxy_logging.async_post_call_streaming_iterator_hook(
            response=mock_streaming_response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
            request_data={"model": "gpt-4", "messages": []},
        )
        first = await stream.__anext__()
        await stream.aclose()
    ProxyLogging._callback_capabilities_cache.clear()

    assert first == {"choices": [{"delta": {"content": "Hello"}}]}
    assert [callback.cleaned_up for callback in callbacks] == [True, True]


class RaisingCleanupCallback(CustomLogger):
    """Iterator hook whose cleanup raises."""

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[Any, None],
        request_data: dict[str, object],
    ) -> AsyncGenerator[Any, None]:
        try:
            async for chunk in response:
                yield chunk
        finally:
            raise RuntimeError("cleanup failed")


@pytest.mark.asyncio
async def test_closing_the_stream_still_cleans_up_inner_callbacks_when_an_outer_cleanup_raises():
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    inner = CleanupRecordingCallback()

    with patch.object(litellm, "callbacks", [inner, RaisingCleanupCallback()]):
        ProxyLogging._callback_capabilities_cache.clear()
        stream = proxy_logging.async_post_call_streaming_iterator_hook(
            response=mock_streaming_response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
            request_data={"model": "gpt-4", "messages": []},
        )
        await stream.__anext__()
        await stream.aclose()
    ProxyLogging._callback_capabilities_cache.clear()

    assert inner.cleaned_up is True


class _PlainAsyncIterator:
    def __init__(self, response: AsyncGenerator[Any, None]) -> None:
        self._response = response

    def __aiter__(self) -> "_PlainAsyncIterator":
        return self

    async def __anext__(self) -> Any:
        return await self._response.__anext__()


class PlainIteratorCallback(CustomLogger):
    """Iterator hook that returns an async iterator with no aclose."""

    def async_post_call_streaming_iterator_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # a plain async iterator worked before aclose handling
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[Any, None],
        request_data: dict[str, object],
    ) -> _PlainAsyncIterator:
        return _PlainAsyncIterator(response)


@pytest.mark.asyncio
async def test_a_hook_returning_a_plain_async_iterator_streams_every_chunk():
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())

    with patch.object(litellm, "callbacks", [PlainIteratorCallback()]):
        ProxyLogging._callback_capabilities_cache.clear()
        received = [
            chunk
            async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
                response=mock_streaming_response(),
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                request_data={"model": "gpt-4", "messages": []},
            )
        ]
    ProxyLogging._callback_capabilities_cache.clear()

    assert [chunk async for chunk in mock_streaming_response()] == received


class _ClosableAsyncIterator(_PlainAsyncIterator):
    def __init__(self, response: AsyncGenerator[Any, None]) -> None:
        super().__init__(response)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _SyncClosableAsyncIterator(_PlainAsyncIterator):
    def __init__(self, response: AsyncGenerator[Any, None]) -> None:
        super().__init__(response)
        self.closed = False

    def aclose(self) -> None:
        self.closed = True


class ClosableIteratorCallback(CustomLogger):
    """Iterator hook that returns a non-generator async iterator with its own aclose."""

    def __init__(
        self,
        iterator_type: type[_ClosableAsyncIterator] | type[_SyncClosableAsyncIterator] = _ClosableAsyncIterator,
    ) -> None:
        super().__init__()
        self.iterator_type = iterator_type
        self.returned: tuple[_ClosableAsyncIterator | _SyncClosableAsyncIterator, ...] = ()

    def async_post_call_streaming_iterator_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # a custom async iterator worked before aclose handling
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[Any, None],
        request_data: dict[str, object],
    ) -> _ClosableAsyncIterator | _SyncClosableAsyncIterator:
        iterator = self.iterator_type(response)
        self.returned = (*self.returned, iterator)
        return iterator


@pytest.mark.asyncio
async def test_closing_the_stream_closes_a_hook_iterator_that_is_not_a_generator():
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    callback = ClosableIteratorCallback()

    with patch.object(litellm, "callbacks", [callback]):
        ProxyLogging._callback_capabilities_cache.clear()
        stream = proxy_logging.async_post_call_streaming_iterator_hook(
            response=mock_streaming_response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
            request_data={"model": "gpt-4", "messages": []},
        )
        await stream.__anext__()
        await stream.aclose()
    ProxyLogging._callback_capabilities_cache.clear()

    assert [iterator.closed for iterator in callback.returned] == [True]


@pytest.mark.asyncio
async def test_a_hook_iterator_with_a_synchronous_aclose_streams_everything_and_is_closed():
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    callback = ClosableIteratorCallback(iterator_type=_SyncClosableAsyncIterator)

    with patch.object(litellm, "callbacks", [callback]):
        ProxyLogging._callback_capabilities_cache.clear()
        received = [
            chunk
            async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
                response=mock_streaming_response(),
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                request_data={"model": "gpt-4", "messages": []},
            )
        ]
    ProxyLogging._callback_capabilities_cache.clear()

    assert [chunk async for chunk in mock_streaming_response()] == received
    assert [iterator.closed for iterator in callback.returned] == [True]


class _RaisingAcloseIterator(_ClosableAsyncIterator):
    """Non-generator async iterator whose asynchronous aclose raises."""

    def __init__(self, response: AsyncGenerator[Any, None], error: Exception) -> None:
        super().__init__(response)
        self.error = error

    async def aclose(self) -> None:
        self.closed = True
        raise self.error


class _SyncRaisingAcloseIterator(_SyncClosableAsyncIterator):
    """Non-generator async iterator whose synchronous aclose raises."""

    def __init__(self, response: AsyncGenerator[Any, None], error: Exception) -> None:
        super().__init__(response)
        self.error = error

    def aclose(self) -> None:
        self.closed = True
        raise self.error


class RaisingAcloseIteratorCallback(CustomLogger):
    """Iterator hook that returns a non-generator async iterator whose aclose raises."""

    def __init__(
        self,
        iterator_type: type[_RaisingAcloseIterator] | type[_SyncRaisingAcloseIterator] = _RaisingAcloseIterator,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.iterator_type = iterator_type
        self.error = error if error is not None else RuntimeError("cleanup failed")
        self.returned: tuple[_RaisingAcloseIterator | _SyncRaisingAcloseIterator, ...] = ()

    def async_post_call_streaming_iterator_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # a custom async iterator worked before aclose handling
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[Any, None],
        request_data: dict[str, object],
    ) -> _RaisingAcloseIterator | _SyncRaisingAcloseIterator:
        iterator = self.iterator_type(response, self.error)
        self.returned = (*self.returned, iterator)
        return iterator


@pytest.mark.asyncio
async def test_a_hook_iterator_whose_aclose_raises_still_finishes_the_stream(caplog: pytest.LogCaptureFixture) -> None:
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    callback = RaisingAcloseIteratorCallback()

    with patch.object(litellm, "callbacks", [callback]):
        ProxyLogging._callback_capabilities_cache.clear()
        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            received = [
                chunk
                async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
                    response=mock_streaming_response(),
                    user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                    request_data={"model": "gpt-4", "messages": []},
                )
            ]
    ProxyLogging._callback_capabilities_cache.clear()

    assert [chunk async for chunk in mock_streaming_response()] == received
    assert [iterator.closed for iterator in callback.returned] == [True]
    warnings_emitted = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "RaisingAcloseIteratorCallback" in record.getMessage()
    ]
    assert len(warnings_emitted) == 1
    assert "RuntimeError" in warnings_emitted[0]
    assert "cleanup failed" not in warnings_emitted[0]


@pytest.mark.asyncio
async def test_a_hook_iterator_whose_synchronous_aclose_raises_still_finishes_the_stream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    callback = RaisingAcloseIteratorCallback(
        iterator_type=_SyncRaisingAcloseIterator, error=ValueError("sync cleanup failed")
    )

    with patch.object(litellm, "callbacks", [callback]):
        ProxyLogging._callback_capabilities_cache.clear()
        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            received = [
                chunk
                async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
                    response=mock_streaming_response(),
                    user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                    request_data={"model": "gpt-4", "messages": []},
                )
            ]
    ProxyLogging._callback_capabilities_cache.clear()

    assert [chunk async for chunk in mock_streaming_response()] == received
    assert [iterator.closed for iterator in callback.returned] == [True]
    warnings_emitted = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "RaisingAcloseIteratorCallback" in record.getMessage()
    ]
    assert len(warnings_emitted) == 1
    assert "ValueError" in warnings_emitted[0]


@pytest.mark.asyncio
async def test_a_hook_iterator_with_a_clean_aclose_streams_everything_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    callback = ClosableIteratorCallback()

    with patch.object(litellm, "callbacks", [callback]):
        ProxyLogging._callback_capabilities_cache.clear()
        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            received = [
                chunk
                async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
                    response=mock_streaming_response(),
                    user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                    request_data={"model": "gpt-4", "messages": []},
                )
            ]
    ProxyLogging._callback_capabilities_cache.clear()

    assert [chunk async for chunk in mock_streaming_response()] == received
    assert [iterator.closed for iterator in callback.returned] == [True]
    assert not [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "ClosableIteratorCallback" in record.getMessage()
    ]
