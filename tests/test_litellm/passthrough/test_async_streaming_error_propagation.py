"""
Tests for error propagation in async passthrough streaming routes.

Verifies that streaming passthrough wrappers preserve the previous guarantees:
HTTP 4xx/5xx failures must raise instead of being silently forwarded as bytes,
and successful streaming responses should still yield chunks normally.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


def _make_mock_response(status_code: int, body: bytes, headers: dict = None):  # type: ignore[assignment]
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.headers = httpx.Headers(headers or {"content-type": "text/event-stream"})

    def _raise_for_status():
        if status_code >= 400:
            request = httpx.Request(
                "POST", "https://azure.example.com/openai/responses"
            )
            real_response = httpx.Response(
                status_code=status_code,
                content=body,
                request=request,
                headers=headers or {},
            )
            raise httpx.HTTPStatusError(
                message=f"{status_code} Error",
                request=request,
                response=real_response,
            )

    mock.raise_for_status = _raise_for_status

    async def _aiter_bytes():
        yield body

    mock.aiter_bytes = _aiter_bytes
    return mock


def _make_mock_logging_obj():
    mock = MagicMock()
    mock.async_flush_passthrough_collected_chunks = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_async_streaming_429_raises():
    """429 from upstream should raise HTTPStatusError, not yield error bytes."""
    from litellm.passthrough.main import AsyncPassthroughStreamingResponse
    
    error_body = json.dumps(
        {"error": {"code": "429", "message": "Rate limit exceeded."}}
    ).encode()
    mock_response = _make_mock_response(429, error_body)
    
    async def response_coro():
        return mock_response
    
    chunks = []
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        async for chunk in AsyncPassthroughStreamingResponse(
            response=response_coro(),
            litellm_logging_obj=_make_mock_logging_obj(),
            provider_config=MagicMock(),
        ):
            chunks.append(chunk)
    
    assert exc_info.value.response.status_code == 429
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_async_streaming_500_raises():
    """500 from upstream should also raise, not yield error bytes."""
    from litellm.passthrough.main import AsyncPassthroughStreamingResponse
    
    error_body = json.dumps(
        {"error": {"code": "500", "message": "Internal server error"}}
    ).encode()
    mock_response = _make_mock_response(500, error_body)
    
    async def response_coro():
        return mock_response
    
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        async for _ in AsyncPassthroughStreamingResponse(
            response=response_coro(),
            litellm_logging_obj=_make_mock_logging_obj(),
            provider_config=MagicMock(),
        ):
            pass
    
    assert exc_info.value.response.status_code == 500


@pytest.mark.asyncio
async def test_async_passthrough_wrapper_200_yields_chunks():
    """Successful 200 streaming responses should continue to work normally."""
    from litellm.passthrough.main import AsyncPassthroughStreamingResponse

    sse_data = b'data: {"type":"response.created"}\n\ndata: [DONE]\n\n'
    mock_response = _make_mock_response(200, sse_data)
    mock_logging_obj = _make_mock_logging_obj()

    async def response_coro():
        return mock_response

    async_stream = AsyncPassthroughStreamingResponse(
        response=response_coro(),
        litellm_logging_obj=mock_logging_obj,
        provider_config=MagicMock(),
    )

    chunks = []
    async for chunk in async_stream:
        chunks.append(chunk)

    await asyncio.sleep(0)

    assert len(chunks) == 1
    assert b"response.created" in chunks[0]
    mock_logging_obj.async_flush_passthrough_collected_chunks.assert_awaited_once()
