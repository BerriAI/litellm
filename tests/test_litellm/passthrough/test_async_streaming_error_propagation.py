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
async def test_async_passthrough_wrapper_429_raises_before_iteration():
    """429 from upstream should be raised before the wrapper is constructed."""
    error_body = json.dumps(
        {"error": {"code": "429", "message": "Rate limit exceeded."}}
    ).encode()
    mock_response = _make_mock_response(429, error_body)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        mock_response.raise_for_status()

    assert exc_info.value.response.status_code == 429


@pytest.mark.asyncio
async def test_async_passthrough_wrapper_500_raises_before_iteration():
    """500 from upstream should be raised before the wrapper is constructed."""
    error_body = json.dumps(
        {"error": {"code": "500", "message": "Internal server error"}}
    ).encode()
    mock_response = _make_mock_response(500, error_body)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        mock_response.raise_for_status()

    assert exc_info.value.response.status_code == 500


@pytest.mark.asyncio
async def test_async_passthrough_wrapper_200_yields_chunks():
    """Successful 200 streaming responses should continue to work normally."""
    from litellm.passthrough.main import _AsyncPassthroughStreamingResponse

    sse_data = b'data: {"type":"response.created"}\n\ndata: [DONE]\n\n'
    mock_response = _make_mock_response(200, sse_data)
    mock_logging_obj = _make_mock_logging_obj()
    async_stream = _AsyncPassthroughStreamingResponse(
        response=mock_response,
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


@pytest.mark.asyncio
async def test_async_passthrough_wrapper_closes_response_on_iteration_error():
    """Wrapper should close the upstream response if iteration raises."""
    from litellm.passthrough.main import _AsyncPassthroughStreamingResponse

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers({"content-type": "text/event-stream"})
    mock_response.aclose = AsyncMock()

    async def _failing_aiter_bytes():
        raise RuntimeError("stream failed")
        yield b""

    mock_response.aiter_bytes = _failing_aiter_bytes

    async_stream = _AsyncPassthroughStreamingResponse(
        response=mock_response,
        litellm_logging_obj=_make_mock_logging_obj(),
        provider_config=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        async for chunk in async_stream:
            _ = chunk

    mock_response.aclose.assert_awaited_once()
