import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

from litellm.llms.custom_httpx.httpx_stream_handler import HttpxStreamHandler


class MockHttpxResponse:
    """Mock httpx.Response for testing streaming iteration."""

    def __init__(self, chunks=None, exception_to_raise=None, exception_at_chunk=None):
        self.chunks = chunks or [b"chunk1", b"chunk2", b"chunk3"]
        self.exception_to_raise = exception_to_raise
        self.exception_at_chunk = exception_at_chunk
        self._closed = False

    async def aiter_bytes(self, chunk_size=None):
        for i, chunk in enumerate(self.chunks):
            if self.exception_to_raise and i == self.exception_at_chunk:
                raise self.exception_to_raise
            yield chunk

    def iter_bytes(self, chunk_size=None):
        for i, chunk in enumerate(self.chunks):
            if self.exception_to_raise and i == self.exception_at_chunk:
                raise self.exception_to_raise
            yield chunk

    async def aclose(self):
        self._closed = True

    def close(self):
        self._closed = True


@pytest.mark.asyncio
async def test_aiter_bytes_normal_flow():
    """should yield all chunks from the upstream response via aiter_bytes"""
    mock_response = MockHttpxResponse(chunks=[b"hello", b"world", b"test"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    chunks = []
    async for chunk in handler.aiter_bytes():
        chunks.append(chunk)

    assert chunks == [b"hello", b"world", b"test"]


def test_iter_bytes_normal_flow():
    """should yield all chunks from the upstream response via iter_bytes"""
    mock_response = MockHttpxResponse(chunks=[b"hello", b"world", b"test"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    chunks = []
    for chunk in handler.iter_bytes():
        chunks.append(chunk)

    assert chunks == [b"hello", b"world", b"test"]


@pytest.mark.asyncio
async def test_aiter_bytes_with_chunk_size():
    """should pass chunk_size through to the underlying response"""
    captured = {}

    class CapturingResponse(MockHttpxResponse):
        async def aiter_bytes(self, chunk_size=None):
            captured["chunk_size"] = chunk_size
            async for chunk in super().aiter_bytes(chunk_size=chunk_size):
                yield chunk

    mock_response = CapturingResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    async for _ in handler.aiter_bytes(chunk_size=8192):
        pass

    assert captured["chunk_size"] == 8192


@pytest.mark.asyncio
async def test_aiter_bytes_consumed_once():
    """should raise RuntimeError on second call to aiter_bytes"""
    mock_response = MockHttpxResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    async for _ in handler.aiter_bytes():
        pass

    with pytest.raises(RuntimeError, match="already been consumed"):
        async for _ in handler.aiter_bytes():
            pass


def test_iter_bytes_consumed_once():
    """should raise RuntimeError on second call to iter_bytes"""
    mock_response = MockHttpxResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    for _ in handler.iter_bytes():
        pass

    with pytest.raises(RuntimeError, match="already been consumed"):
        for _ in handler.iter_bytes():
            pass


@pytest.mark.asyncio
async def test_aiter_bytes_closes_response():
    """should call aclose on the response after async iteration completes"""
    mock_response = MockHttpxResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    async for _ in handler.aiter_bytes():
        pass

    assert mock_response._closed is True


def test_iter_bytes_closes_response():
    """should call close on the response after sync iteration completes"""
    mock_response = MockHttpxResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    for _ in handler.iter_bytes():
        pass

    assert mock_response._closed is True


@pytest.mark.asyncio
async def test_aiter_bytes_closes_response_on_error():
    """should call aclose on the response even when iteration raises an error"""
    mock_response = MockHttpxResponse(
        chunks=[b"chunk1", b"chunk2"],
        exception_to_raise=RuntimeError("upstream error"),
        exception_at_chunk=1,
    )
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    with pytest.raises(RuntimeError, match="upstream error"):
        async for _ in handler.aiter_bytes():
            pass

    assert mock_response._closed is True


def test_iter_bytes_closes_response_on_error():
    """should call close on the response even when iteration raises an error"""
    mock_response = MockHttpxResponse(
        chunks=[b"chunk1", b"chunk2"],
        exception_to_raise=RuntimeError("upstream error"),
        exception_at_chunk=1,
    )
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    with pytest.raises(RuntimeError, match="upstream error"):
        for _ in handler.iter_bytes():
            pass

    assert mock_response._closed is True


@pytest.mark.asyncio
async def test_async_cleanup_callback_invoked():
    """should invoke the async cleanup callback after aiter_bytes completes"""
    cleanup_called = False

    async def async_cleanup():
        nonlocal cleanup_called
        cleanup_called = True

    mock_response = MockHttpxResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response, cleanup=async_cleanup)  # type: ignore

    async for _ in handler.aiter_bytes():
        pass

    assert cleanup_called is True


def test_sync_cleanup_callback_invoked():
    """should invoke the sync cleanup callback after iter_bytes completes"""
    cleanup_called = False

    def sync_cleanup():
        nonlocal cleanup_called
        cleanup_called = True

    mock_response = MockHttpxResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response, cleanup=sync_cleanup)  # type: ignore

    for _ in handler.iter_bytes():
        pass

    assert cleanup_called is True


@pytest.mark.asyncio
async def test_hidden_params_default_empty():
    """should initialize _hidden_params as an empty dict"""
    mock_response = MockHttpxResponse()
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    assert handler._hidden_params == {}


@pytest.mark.asyncio
async def test_hidden_params_can_be_set():
    """should allow _hidden_params to be set for cost tracking metadata"""
    mock_response = MockHttpxResponse()
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    handler._hidden_params = {
        "model_id": "test-model",
        "response_cost": 0.001,
    }

    assert handler._hidden_params["model_id"] == "test-model"
    assert handler._hidden_params["response_cost"] == 0.001


@pytest.mark.asyncio
async def test_aclose_handles_response_close_error():
    """should not raise when the underlying response.aclose() fails"""

    class FailingCloseResponse(MockHttpxResponse):
        async def aclose(self):
            raise ConnectionError("connection already closed")

    mock_response = FailingCloseResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    # Should not raise
    await handler.aclose()


def test_close_handles_response_close_error():
    """should not raise when the underlying response.close() fails"""

    class FailingCloseResponse(MockHttpxResponse):
        def close(self):
            raise ConnectionError("connection already closed")

    mock_response = FailingCloseResponse(chunks=[b"data"])
    handler = HttpxStreamHandler(response=mock_response)  # type: ignore

    # Should not raise
    handler.close()
