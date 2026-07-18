"""
Tests for error propagation in _async_streaming passthrough routes.

Verifies that HTTP 4xx/5xx errors from upstream (e.g. Azure 429 rate limits)
raise exceptions instead of being silently forwarded as raw bytes under HTTP 200.

See: litellm/passthrough/main.py _async_streaming()
"""

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
    from litellm.passthrough.main import _async_streaming

    error_body = json.dumps(
        {"error": {"code": "429", "message": "Rate limit exceeded."}}
    ).encode()
    mock_response = _make_mock_response(429, error_body)

    async def response_coro():
        return mock_response

    chunks = []
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        async for chunk in _async_streaming(
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
    from litellm.passthrough.main import _async_streaming

    error_body = json.dumps(
        {"error": {"code": "500", "message": "Internal server error"}}
    ).encode()
    mock_response = _make_mock_response(500, error_body)

    async def response_coro():
        return mock_response

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        async for _ in _async_streaming(
            response=response_coro(),
            litellm_logging_obj=_make_mock_logging_obj(),
            provider_config=MagicMock(),
        ):
            pass

    assert exc_info.value.response.status_code == 500


@pytest.mark.asyncio
async def test_async_streaming_200_yields_chunks():
    """Successful 200 streaming responses should continue to work normally."""
    from litellm.passthrough.main import _async_streaming

    sse_data = b'data: {"type":"response.created"}\n\ndata: [DONE]\n\n'
    mock_response = _make_mock_response(200, sse_data)

    async def response_coro():
        return mock_response

    chunks = []
    async for chunk in _async_streaming(
        response=response_coro(),
        litellm_logging_obj=_make_mock_logging_obj(),
        provider_config=MagicMock(),
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert b"response.created" in chunks[0]
