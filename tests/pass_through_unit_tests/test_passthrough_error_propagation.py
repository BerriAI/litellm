"""
Regression tests: passthrough routes must propagate upstream 4xx/5xx errors
instead of silently forwarding them as HTTP 200 with an error JSON body.

Covers the routes used by production customers:
  - /vertex_ai/{endpoint}   (VertexAI Anthropic + VertexAI Gemini)
  - /anthropic/{endpoint}
  - /vllm/{endpoint} with router model (streaming async-generator path)

All tests are mock-only (no real network calls).
"""

import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from litellm.proxy._types import ProxyException

sys.path.insert(0, os.path.abspath("../.."))

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MODULE = "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"


class MockRequest:
    """Minimal FastAPI Request lookalike used across all tests."""

    def __init__(self, body: dict, method: str = "POST"):
        self._body = body
        self.method = method
        self.headers = {"content-type": "application/json"}
        self.query_params = {}
        self.url = "https://litellm-proxy/test"

    async def body(self) -> bytes:
        return json.dumps(self._body).encode()


@pytest.fixture
def user_api_key_dict():
    from litellm.proxy._types import UserAPIKeyAuth
    return UserAPIKeyAuth(api_key="test-key")


# ---------------------------------------------------------------------------
# pass_through_request 429 tests (covers Vertex AI, Anthropic, Gemini paths)
# ---------------------------------------------------------------------------


def _make_mock_streaming_response(status_code: int, body: bytes) -> MagicMock:
    """
    Build a mock that looks like an httpx.Response returned from send(stream=True).
    raise_for_status() raises HTTPStatusError for 4xx/5xx.
    """
    error_response = httpx.Response(
        status_code=status_code,
        content=body,
        request=httpx.Request("POST", "https://upstream.example.com/"),
    )
    mock = MagicMock()
    mock.status_code = status_code
    mock.headers = httpx.Headers({"content-type": "application/json"})

    def _raise():
        if status_code >= 400:
            raise httpx.HTTPStatusError(
                message=f"{status_code}",
                request=error_response.request,
                response=error_response,
            )

    mock.raise_for_status = _raise

    async def _aiter_bytes():
        yield body

    mock.aiter_bytes = _aiter_bytes
    return mock


@pytest.mark.asyncio
async def test_pass_through_request_streaming_429_raises_http_exception(user_api_key_dict):
    """
    When upstream returns 429 on a streaming request, pass_through_request must
    raise HTTPException(429), NOT return StreamingResponse(200) with error body.

    This is the path used by /vertex_ai/, /anthropic/, /gemini/ endpoints via
    create_pass_through_route → pass_through_request.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    body = json.dumps({"error": {"code": "429", "message": "Too many requests"}}).encode()
    mock_429 = _make_mock_streaming_response(429, body)

    mock_async_client = Mock()
    mock_async_client.build_request = Mock(return_value=MagicMock(spec=httpx.Request))
    mock_async_client.send = AsyncMock(return_value=mock_429)
    mock_client_obj = Mock()
    mock_client_obj.client = mock_async_client

    request = MockRequest({"stream": True, "contents": [{"parts": [{"text": "hello"}]}]})

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=mock_client_obj,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(return_value={"stream": True}),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new=AsyncMock(),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
        new=AsyncMock(),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await pass_through_request(
                request=request,  # type: ignore[arg-type]
                target="https://upstream.example.com/v1/generate",
                custom_headers={},
                user_api_key_dict=user_api_key_dict,
                stream=True,
            )

    assert exc_info.value.code == "429"


@pytest.mark.asyncio
async def test_pass_through_request_streaming_500_raises_http_exception(user_api_key_dict):
    """500 from upstream also must raise, not forward as 200."""
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    body = json.dumps({"error": {"code": "500", "message": "Internal error"}}).encode()
    mock_500 = _make_mock_streaming_response(500, body)

    mock_async_client = Mock()
    mock_async_client.build_request = Mock(return_value=MagicMock(spec=httpx.Request))
    mock_async_client.send = AsyncMock(return_value=mock_500)
    mock_client_obj = Mock()
    mock_client_obj.client = mock_async_client

    request = MockRequest({"stream": True})

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=mock_client_obj,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(return_value={"stream": True}),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new=AsyncMock(),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
        new=AsyncMock(),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await pass_through_request(
                request=request,  # type: ignore[arg-type]
                target="https://upstream.example.com/v1/generate",
                custom_headers={},
                user_api_key_dict=user_api_key_dict,
                stream=True,
            )

    assert exc_info.value.code == "500"


@pytest.mark.asyncio
async def test_pass_through_request_streaming_200_returns_streaming_response(user_api_key_dict):
    """Sanity: streaming 200 must still return StreamingResponse(200)."""
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    mock_200 = _make_mock_streaming_response(200, b'data: {"text":"hi"}\n\n')

    mock_async_client = Mock()
    mock_async_client.build_request = Mock(return_value=MagicMock(spec=httpx.Request))
    mock_async_client.send = AsyncMock(return_value=mock_200)
    mock_client_obj = Mock()
    mock_client_obj.client = mock_async_client

    request = MockRequest({"stream": True})

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=mock_client_obj,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        new=AsyncMock(return_value={"stream": True}),
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
        new=AsyncMock(),
    ):
        response = await pass_through_request(
            request=request,  # type: ignore[arg-type]
            target="https://upstream.example.com/v1/generate",
            custom_headers={},
            user_api_key_dict=user_api_key_dict,
            stream=True,
        )

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# vllm_proxy_route router model streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vllm_router_model_streaming_asyncgen(user_api_key_dict):
    """
    When allm_passthrough_route returns an AsyncGenerator (the normal _async_streaming
    path), vllm_proxy_route must wrap it in StreamingResponse(content=result) —
    NOT call result.aiter_bytes() which would AttributeError on an AsyncGenerator.
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        vllm_proxy_route,
    )

    sse_chunks = [b'data: {"text":"hello"}\n\n', b"data: [DONE]\n\n"]

    async def fake_async_gen():
        for chunk in sse_chunks:
            yield chunk

    mock_router = MagicMock()
    mock_router.allm_passthrough_route = AsyncMock(return_value=fake_async_gen())

    request = MockRequest({"model": "my-vllm-model", "prompt": "hi", "stream": True})

    with patch(
        "litellm.proxy.proxy_server.llm_router",
        mock_router,
    ), patch(
        f"{MODULE}.is_passthrough_request_using_router_model",
        return_value=True,
    ), patch(
        f"{MODULE}.is_passthrough_request_streaming",
        return_value=True,
    ), patch(
        f"{MODULE}.get_request_body",
        new=AsyncMock(return_value={"model": "my-vllm-model", "prompt": "hi", "stream": True}),
    ):
        response = await vllm_proxy_route(
            endpoint="v1/completions",
            request=request,  # type: ignore[arg-type]
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    assert b"".join(chunks) == b"".join(sse_chunks)


@pytest.mark.asyncio
async def test_vllm_router_model_streaming_httpx_response(user_api_key_dict):
    """
    When allm_passthrough_route returns an httpx.Response (non-generator path),
    vllm_proxy_route must use result.aiter_bytes() and forward the actual status code.
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        vllm_proxy_route,
    )

    req = httpx.Request("POST", "https://vllm.example.com/v1/completions")
    httpx_response = httpx.Response(
        status_code=200,
        content=b'data: {"text":"hello"}\n\ndata: [DONE]\n\n',
        headers={"content-type": "text/event-stream"},
        request=req,
    )

    mock_router = MagicMock()
    mock_router.allm_passthrough_route = AsyncMock(return_value=httpx_response)

    request = MockRequest({"model": "my-vllm-model", "prompt": "hi", "stream": True})

    with patch(
        "litellm.proxy.proxy_server.llm_router",
        mock_router,
    ), patch(
        f"{MODULE}.is_passthrough_request_using_router_model",
        return_value=True,
    ), patch(
        f"{MODULE}.is_passthrough_request_streaming",
        return_value=True,
    ), patch(
        f"{MODULE}.get_request_body",
        new=AsyncMock(return_value={"model": "my-vllm-model", "prompt": "hi", "stream": True}),
    ):
        response = await vllm_proxy_route(
            endpoint="v1/completions",
            request=request,  # type: ignore[arg-type]
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
