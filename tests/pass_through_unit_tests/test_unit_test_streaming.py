import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import httpx
import pytest
import litellm
from typing import AsyncGenerator
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.types import EndpointType
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)


# Helper function to mock async iteration
async def aiter_mock(iterable):
    for item in iterable:
        yield item


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_type,url_route",
    [
        (
            EndpointType.VERTEX_AI,
            "v1/projects/pathrise-convert-1606954137718/locations/us-central1/publishers/google/models/gemini-1.0-pro:generateContent",
        ),
        (EndpointType.ANTHROPIC, "/v1/messages"),
    ],
)
async def test_chunk_processor_yields_raw_bytes(endpoint_type, url_route):
    """
    Test that the chunk_processor yields raw bytes

    This is CRITICAL for pass throughs streaming with Vertex AI and Anthropic
    """
    # Mock inputs
    response = AsyncMock(spec=httpx.Response)
    raw_chunks = [
        b'{"id": "1", "content": "Hello"}',
        b'{"id": "2", "content": "World"}',
        b'\n\ndata: {"id": "3"}',  # Testing different byte formats
    ]

    # Mock aiter_bytes to return an async generator
    async def mock_aiter_bytes():
        for chunk in raw_chunks:
            yield chunk

    response.aiter_bytes = mock_aiter_bytes

    request_body = {"key": "value"}
    litellm_logging_obj = MagicMock()
    start_time = datetime.now()
    passthrough_success_handler_obj = MagicMock()
    litellm_logging_obj.async_success_handler = AsyncMock()

    # Capture yielded chunks and perform detailed assertions
    received_chunks = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body=request_body,
        litellm_logging_obj=litellm_logging_obj,
        endpoint_type=endpoint_type,
        start_time=start_time,
        passthrough_success_handler_obj=passthrough_success_handler_obj,
        url_route=url_route,
    ):
        # Assert each chunk is bytes
        assert isinstance(chunk, bytes), f"Chunk should be bytes, got {type(chunk)}"
        # Assert no decoding/encoding occurred (chunk should be exactly as input)
        assert (
            chunk in raw_chunks
        ), f"Chunk {chunk} was modified during processing. For pass throughs streaming, chunks should be raw bytes"
        received_chunks.append(chunk)

    # Assert all chunks were processed
    assert len(received_chunks) == len(raw_chunks), "Not all chunks were processed"

    # collected chunks all together
    assert b"".join(received_chunks) == b"".join(
        raw_chunks
    ), "Collected chunks do not match raw chunks"


def test_convert_raw_bytes_to_str_lines():
    """
    Test that the _convert_raw_bytes_to_str_lines method correctly converts raw bytes to a list of strings
    """
    # Test case 1: Single chunk
    raw_bytes = [b'data: {"content": "Hello"}\n']
    result = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert result == ['data: {"content": "Hello"}']

    # Test case 2: Multiple chunks
    raw_bytes = [b'data: {"content": "Hello"}\n', b'data: {"content": "World"}\n']
    result = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert result == ['data: {"content": "Hello"}', 'data: {"content": "World"}']

    # Test case 3: Empty input
    raw_bytes = []
    result = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert result == []

    # Test case 4: Chunks with empty lines
    raw_bytes = [b'data: {"content": "Hello"}\n\n', b'\ndata: {"content": "World"}\n']
    result = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert result == ['data: {"content": "Hello"}', 'data: {"content": "World"}']
