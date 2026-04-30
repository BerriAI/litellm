"""Regression tests for LIT-2642 — interrupted streams must still flush usage."""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


def _make_streaming_response(chunks: List[bytes]):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.headers = httpx.Headers({"content-type": "application/vnd.amazon.eventstream"})
    mock.raise_for_status = MagicMock(return_value=None)

    async def _aiter_bytes():
        for chunk in chunks:
            yield chunk

    mock.aiter_bytes = _aiter_bytes
    mock.aclose = AsyncMock()
    return mock


def _make_logging_obj():
    mock = MagicMock()
    mock.async_flush_passthrough_collected_chunks = AsyncMock()
    return mock


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


@pytest.mark.asyncio
async def test_async_streaming_flushes_on_normal_completion():
    from litellm.passthrough.main import _async_streaming

    chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]
    mock_response = _make_streaming_response(chunks)

    async def response_coro():
        return mock_response

    mock_logging_obj = _make_logging_obj()
    provider_config = MagicMock()

    received = []
    async for chunk in _async_streaming(
        response=response_coro(),
        litellm_logging_obj=mock_logging_obj,
        provider_config=provider_config,
    ):
        received.append(chunk)

    assert received == chunks

    await asyncio.sleep(0)

    mock_logging_obj.async_flush_passthrough_collected_chunks.assert_called_once()
    call_kwargs = (
        mock_logging_obj.async_flush_passthrough_collected_chunks.call_args.kwargs
    )
    assert call_kwargs["raw_bytes"] == chunks
    assert call_kwargs["provider_config"] is provider_config


@pytest.mark.asyncio
async def test_async_streaming_flushes_on_client_disconnect():
    from litellm.passthrough.main import _async_streaming

    chunks = [
        b'{"chunk": 1, "outputTokens": 10}',
        b'{"chunk": 2, "outputTokens": 12}',
        b'{"chunk": 3, "outputTokens": 8}',
    ]
    mock_response = _make_streaming_response(chunks)

    async def response_coro():
        return mock_response

    mock_logging_obj = _make_logging_obj()
    provider_config = MagicMock()

    gen = _async_streaming(
        response=response_coro(),
        litellm_logging_obj=mock_logging_obj,
        provider_config=provider_config,
    )

    received = [await gen.__anext__()]
    await gen.aclose()

    assert received == [chunks[0]]

    await asyncio.sleep(0)

    mock_logging_obj.async_flush_passthrough_collected_chunks.assert_called_once()
    call_kwargs = (
        mock_logging_obj.async_flush_passthrough_collected_chunks.call_args.kwargs
    )
    assert call_kwargs["raw_bytes"] == [chunks[0]]


@pytest.mark.asyncio
async def test_async_streaming_does_not_flush_on_4xx():
    from litellm.passthrough.main import _async_streaming

    err_response = MagicMock(spec=httpx.Response)
    err_response.status_code = 429

    def _raise():
        raise httpx.HTTPStatusError(
            "429",
            request=httpx.Request("POST", "https://example.com"),
            response=httpx.Response(
                429, request=httpx.Request("POST", "https://example.com")
            ),
        )

    err_response.raise_for_status = _raise
    err_response.aclose = AsyncMock()

    async def response_coro():
        return err_response

    mock_logging_obj = _make_logging_obj()

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in _async_streaming(
            response=response_coro(),
            litellm_logging_obj=mock_logging_obj,
            provider_config=MagicMock(),
        ):
            pass

    mock_logging_obj.async_flush_passthrough_collected_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_async_streaming_flushes_on_upstream_exception_with_partial_data():
    from litellm.passthrough.main import _async_streaming

    partial_chunks = [b"partial-chunk-1", b"partial-chunk-2"]

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock(return_value=None)
    mock_response.aclose = AsyncMock()

    async def _aiter_bytes_then_raise():
        for c in partial_chunks:
            yield c
        raise httpx.ReadError("upstream disconnected")

    mock_response.aiter_bytes = _aiter_bytes_then_raise

    async def response_coro():
        return mock_response

    mock_logging_obj = _make_logging_obj()
    provider_config = MagicMock()

    received = []
    with pytest.raises(httpx.ReadError):
        async for chunk in _async_streaming(
            response=response_coro(),
            litellm_logging_obj=mock_logging_obj,
            provider_config=provider_config,
        ):
            received.append(chunk)

    assert received == partial_chunks

    await asyncio.sleep(0)

    mock_logging_obj.async_flush_passthrough_collected_chunks.assert_called_once()
    call_kwargs = (
        mock_logging_obj.async_flush_passthrough_collected_chunks.call_args.kwargs
    )
    assert call_kwargs["raw_bytes"] == partial_chunks


def test_sync_streaming_flushes_on_normal_completion():
    from litellm.passthrough.main import _sync_streaming

    chunks = [b"a", b"b", b"c"]

    mock_response = MagicMock(spec=httpx.Response)

    def _iter_bytes():
        yield from chunks

    mock_response.iter_bytes = _iter_bytes

    mock_logging_obj = MagicMock()
    mock_logging_obj.flush_passthrough_collected_chunks = MagicMock()
    provider_config = MagicMock()

    with patch("litellm.utils.executor", _ImmediateExecutor()):
        received = list(
            _sync_streaming(
                response=mock_response,
                litellm_logging_obj=mock_logging_obj,
                provider_config=provider_config,
            )
        )

    assert received == chunks
    mock_logging_obj.flush_passthrough_collected_chunks.assert_called_once()


def test_sync_streaming_flushes_on_early_close():
    from litellm.passthrough.main import _sync_streaming

    chunks = [b"first", b"second", b"third"]

    mock_response = MagicMock(spec=httpx.Response)

    def _iter_bytes():
        yield from chunks

    mock_response.iter_bytes = _iter_bytes

    mock_logging_obj = MagicMock()
    mock_logging_obj.flush_passthrough_collected_chunks = MagicMock()
    provider_config = MagicMock()

    with patch("litellm.utils.executor", _ImmediateExecutor()):
        gen = _sync_streaming(
            response=mock_response,
            litellm_logging_obj=mock_logging_obj,
            provider_config=provider_config,
        )

        first = next(gen)
        gen.close()

    assert first == chunks[0]
    mock_logging_obj.flush_passthrough_collected_chunks.assert_called_once()
    call_kwargs = mock_logging_obj.flush_passthrough_collected_chunks.call_args.kwargs
    assert call_kwargs["raw_bytes"] == [chunks[0]]
