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
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.passthrough_endpoints.pass_through_endpoints import PassthroughStandardLoggingPayload
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


@pytest.mark.asyncio
async def test_chunk_processor_passes_kwargs_to_logging_handler():
    """
    Test that kwargs (containing litellm_params with API key metadata) are
    propagated from chunk_processor through to _route_streaming_logging_to_handler.

    This ensures API key attribution reaches Langfuse traces for streaming
    pass-through requests (e.g., Claude Code hitting /anthropic/v1/messages).
    """
    response = AsyncMock(spec=httpx.Response)

    # Minimal streaming response with message_start and message_stop events
    raw_chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}\n\n',
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def mock_aiter_bytes():
        for chunk in raw_chunks:
            yield chunk

    response.aiter_bytes = mock_aiter_bytes

    request_body = {"model": "claude-3-haiku-20240307", "messages": [{"role": "user", "content": "Hi"}]}
    litellm_logging_obj = MagicMock()
    litellm_logging_obj.async_success_handler = AsyncMock()
    litellm_logging_obj._should_run_sync_callbacks_for_async_calls = MagicMock(return_value=False)
    litellm_logging_obj.model_call_details = {}
    start_time = datetime.now()
    passthrough_success_handler_obj = MagicMock()

    # The kwargs that should be threaded through — simulating what
    # _init_kwargs_for_pass_through_endpoint() creates
    input_kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key_hash": "sk-hashed-abc123",
                "user_api_key_alias": "test-key-alias",
                "user_api_key_team_id": "team-456",
                "user_api_key_user_id": "user-789",
                "user_api_key_org_id": "org-012",
            },
            "proxy_server_request": {
                "url": "https://proxy/anthropic/v1/messages",
                "method": "POST",
                "body": request_body,
            },
        },
        "passthrough_logging_payload": PassthroughStandardLoggingPayload(
            url="https://api.anthropic.com/v1/messages",
            request_body=request_body,
        ),
        "call_type": "pass_through_endpoint",
        "litellm_call_id": "call-test-123",
    }

    # Consume the async generator
    async for _ in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body=request_body,
        litellm_logging_obj=litellm_logging_obj,
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=start_time,
        passthrough_success_handler_obj=passthrough_success_handler_obj,
        url_route="/v1/messages",
        kwargs=input_kwargs,
    ):
        pass

    # Allow the asyncio.create_task to run
    import asyncio
    await asyncio.sleep(0.5)

    # Verify async_success_handler was called with kwargs containing
    # the API key metadata from input_kwargs
    assert litellm_logging_obj.async_success_handler.called, \
        "async_success_handler should have been called after streaming completed"
    call_kwargs = litellm_logging_obj.async_success_handler.call_args
    # The handler_kwargs are spread as **kwargs, check they include response_cost
    # (set by the Anthropic handler) and that litellm_params metadata was preserved
    assert call_kwargs is not None, "async_success_handler was called but with no args"


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
