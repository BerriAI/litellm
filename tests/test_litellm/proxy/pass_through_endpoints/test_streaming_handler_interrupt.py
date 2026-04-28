"""Regression tests for LIT-2642 — interrupted pass-through streams must still log usage."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _make_streaming_response(chunks):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200

    async def _aiter_bytes():
        for c in chunks:
            yield c

    mock.aiter_bytes = _aiter_bytes
    return mock


@pytest.mark.asyncio
async def test_chunk_processor_logs_on_normal_completion():
    chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]
    response = _make_streaming_response(chunks)

    mock_logging_obj = MagicMock()
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ) as mock_route:
        received = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        ):
            received.append(chunk)

        await asyncio.sleep(0)

    assert received == chunks
    mock_route.assert_called_once()
    call_kwargs = mock_route.call_args.kwargs
    assert call_kwargs["raw_bytes"] == chunks


@pytest.mark.asyncio
async def test_chunk_processor_logs_on_client_disconnect():
    chunks = [b"event-1", b"event-2", b"event-3"]
    response = _make_streaming_response(chunks)

    mock_logging_obj = MagicMock()
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ) as mock_route:
        gen = PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        )

        first = await gen.__anext__()
        await gen.aclose()

        await asyncio.sleep(0)

    assert first == chunks[0]
    mock_route.assert_called_once()
    call_kwargs = mock_route.call_args.kwargs
    assert call_kwargs["raw_bytes"] == [chunks[0]]


@pytest.mark.asyncio
async def test_chunk_processor_does_not_schedule_logging_when_no_chunks():
    response = _make_streaming_response([])

    mock_logging_obj = MagicMock()
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ) as mock_route:
        received = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        ):
            received.append(chunk)

    assert received == []
    mock_route.assert_not_called()
