import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import aiohttp.client_exceptions
import aiohttp.http_exceptions
import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.custom_httpx.aiohttp_transport import (
    AiohttpResponseStream,
    LiteLLMAiohttpTransport,
    map_aiohttp_exceptions,
)


class MockAiohttpResponse:
    """Mock aiohttp ClientResponse for testing"""

    def __init__(
        self,
        status=200,
        headers=None,
        content_chunks=None,
        exception_to_raise=None,
        exception_at_chunk=None,
    ):
        self.status = status
        self.headers = headers or {}
        self.content = MockContent(
            content_chunks, exception_to_raise, exception_at_chunk
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockContent:
    """Mock aiohttp response content for testing"""

    def __init__(self, chunks=None, exception_to_raise=None, exception_at_chunk=None):
        self.chunks = chunks or [b"chunk1", b"chunk2", b"chunk3"]
        self.exception_to_raise = exception_to_raise
        self.exception_at_chunk = exception_at_chunk or (len(self.chunks) - 1)
        self.chunk_index = 0

    async def iter_chunked(self, chunk_size):
        for i, chunk in enumerate(self.chunks):
            if self.exception_to_raise and i == self.exception_at_chunk:
                # Raise exception at specified chunk to simulate partial transfer
                raise self.exception_to_raise
            yield chunk


@pytest.mark.asyncio
async def test_aiohttp_response_stream_normal_flow():
    """Test normal flow of AiohttpResponseStream without exceptions"""
    mock_response = MockAiohttpResponse(content_chunks=[b"hello", b"world", b"test"])

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    chunks = []

    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [b"hello", b"world", b"test"]


@pytest.mark.asyncio
async def test_transfer_encoding_error_no_httpx_read_error():
    """Test that TransferEncodingError doesn't get converted to httpx.ReadError"""
    import logging

    # Create a TransferEncodingError wrapped in ClientPayloadError (like in real scenarios)
    transfer_error = aiohttp.http_exceptions.TransferEncodingError(
        message="400, message: Not enough data for satisfy transfer length header."
    )

    # Wrap it in ClientPayloadError as aiohttp does
    client_payload_error = aiohttp.ClientPayloadError(
        "Response payload is not completed"
    )
    client_payload_error.__cause__ = transfer_error

    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2", b"chunk3"],
        exception_to_raise=client_payload_error,
        exception_at_chunk=1,  # Error occurs at chunk 1
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    received_chunks = []

    # This should NOT raise httpx.ReadError or any other exception
    # It should handle the error gracefully and just return what was received
    async for chunk in stream:
        received_chunks.append(chunk)
    print(f"received_chunks: {received_chunks}")

    # Should have received the first chunk before the error
    assert received_chunks == [b"chunk1"]
    assert len(received_chunks) == 1


@pytest.mark.asyncio
async def test_client_payload_error_graceful_handling():
    """Test that ClientPayloadError is handled gracefully without stacktrace"""
    # Create a ClientPayloadError directly
    client_error = aiohttp.client_exceptions.ClientPayloadError(
        "Response payload is not completed"
    )

    mock_response = MockAiohttpResponse(
        content_chunks=[b"data1", b"data2", b"data3"],
        exception_to_raise=client_error,
        exception_at_chunk=2,  # Error occurs at chunk 2
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    received_chunks = []

    # This should handle the error gracefully without raising
    async for chunk in stream:
        received_chunks.append(chunk)

    # Should have received chunks before the error
    assert received_chunks == [b"data1", b"data2"]
    assert len(received_chunks) == 2


@pytest.mark.asyncio
async def test_unknown_aiohttp_exception_gets_mapped():
    """Test that unknown aiohttp exceptions still get mapped to httpx exceptions"""
    # Create an aiohttp exception that's not specifically handled
    # Using InvalidURL which should map to httpx.InvalidURL
    invalid_url_error = aiohttp.InvalidURL("Invalid URL format")

    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2"],
        exception_to_raise=invalid_url_error,
        exception_at_chunk=0,  # Error occurs immediately
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore

    # This should raise httpx.InvalidURL (mapped from aiohttp.InvalidURL)
    with pytest.raises(httpx.InvalidURL):
        async for chunk in stream:
            pass


@pytest.mark.asyncio
async def test_timeout_exception_gets_mapped():
    """Test that aiohttp timeout exceptions get mapped to httpx timeout exceptions"""
    # Create an aiohttp timeout exception
    timeout_error = aiohttp.ServerTimeoutError("Server timeout")

    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2"],
        exception_to_raise=timeout_error,
        exception_at_chunk=1,  # Error occurs at chunk 1
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    received_chunks = []

    # This should raise httpx.TimeoutException (mapped from aiohttp.ServerTimeoutError)
    with pytest.raises(httpx.TimeoutException):
        async for chunk in stream:
            received_chunks.append(chunk)

    # Should have received the first chunk before the error
    assert received_chunks == [b"chunk1"]
