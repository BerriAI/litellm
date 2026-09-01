"""
Tests for rerank_endpoints/endpoints.py response headers.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response

import litellm.proxy.common_request_processing as common_request_processing_mod
import litellm.proxy.proxy_server as proxy_server_mod
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.rerank_endpoints.endpoints import rerank
from litellm.types.utils import RerankResponse

HIDDEN_PARAMS = {
    "model_id": "deployment-1",
    "api_base": "https://bedrock-agent-runtime.us-east-1.amazonaws.com",
    "response_cost": 0.002,
    "_response_ms": 1500.5,
    "litellm_overhead_time_ms": 12.5,
    "callback_duration_ms": 1.25,
    "timing_llm_api_ms": 1488.0,
    "timing_pre_processing_ms": 10.0,
    "timing_post_processing_ms": 2.5,
    "timing_message_copy_ms": 0.01,
}


def _build_request() -> Request:
    body = json.dumps({"model": "rerank-model", "query": "q", "documents": ["a", "b"]}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/rerank",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        },
        receive=receive,
    )


async def _call_rerank(hidden_params: dict = HIDDEN_PARAMS) -> Response:
    response = RerankResponse(id="rerank-1", results=[{"index": 0, "relevance_score": 0.9}])
    response._hidden_params = dict(hidden_params)

    fastapi_response = Response()
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
    proxy_logging_obj.update_request_status = AsyncMock()

    async def fake_add_litellm_data_to_request(**kwargs):
        return {**kwargs["data"], "litellm_call_id": "call-123"}

    async def fake_route_request(**kwargs):
        async def _call():
            return response

        return _call()

    with (
        patch.object(proxy_server_mod, "add_litellm_data_to_request", fake_add_litellm_data_to_request),  # test-quality-ok: the rerank route reads these proxy_server module globals; no injection seam on the FastAPI handler
        patch.object(proxy_server_mod, "route_request", fake_route_request),  # test-quality-ok: the rerank route reads these proxy_server module globals; no injection seam on the FastAPI handler
        patch.object(proxy_server_mod, "proxy_logging_obj", proxy_logging_obj),  # test-quality-ok: the rerank route reads these proxy_server module globals; no injection seam on the FastAPI handler
        patch.object(proxy_server_mod, "llm_router", MagicMock()),  # test-quality-ok: the rerank route reads these proxy_server module globals; no injection seam on the FastAPI handler
        patch.object(proxy_server_mod, "version", "1.2.3"),  # test-quality-ok: the rerank route reads these proxy_server module globals; no injection seam on the FastAPI handler
    ):
        await rerank(
            request=_build_request(),
            fastapi_response=fastapi_response,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        )

    return fastapi_response


@pytest.mark.asyncio
async def test_rerank_emits_latency_and_cost_headers():
    """/rerank must surface the same hidden_params-derived headers as /chat/completions."""
    fastapi_response = await _call_rerank()

    assert fastapi_response.headers["x-litellm-call-id"] == "call-123"
    assert fastapi_response.headers["x-litellm-response-duration-ms"] == "1500.5"
    assert fastapi_response.headers["x-litellm-overhead-duration-ms"] == "12.5"
    assert fastapi_response.headers["x-litellm-callback-duration-ms"] == "1.25"
    assert fastapi_response.headers["x-litellm-response-cost"] == "0.002"


@pytest.mark.asyncio
async def test_rerank_emits_detailed_timing_headers_when_enabled():
    """LITELLM_DETAILED_TIMING must also work on /rerank, not just /chat/completions."""
    with patch.object(common_request_processing_mod, "LITELLM_DETAILED_TIMING", True):  # test-quality-ok: LITELLM_DETAILED_TIMING is a module constant; toggling it is the behavior under test
        fastapi_response = await _call_rerank()

    assert fastapi_response.headers["x-litellm-timing-llm-api-ms"] == "1488.0"
    assert fastapi_response.headers["x-litellm-timing-pre-processing-ms"] == "10.0"
    assert fastapi_response.headers["x-litellm-timing-post-processing-ms"] == "2.5"
    assert fastapi_response.headers["x-litellm-timing-message-copy-ms"] == "0.01"


@pytest.mark.asyncio
async def test_rerank_emits_zero_response_cost_header():
    """A free deployment costs 0.0, which is a real cost and must not be dropped."""
    fastapi_response = await _call_rerank({**HIDDEN_PARAMS, "response_cost": 0.0})

    assert fastapi_response.headers["x-litellm-response-cost"] == "0.0"


@pytest.mark.asyncio
async def test_rerank_omits_detailed_timing_headers_when_disabled():
    with patch.object(common_request_processing_mod, "LITELLM_DETAILED_TIMING", False):  # test-quality-ok: LITELLM_DETAILED_TIMING is a module constant; toggling it is the behavior under test
        fastapi_response = await _call_rerank()

    assert "x-litellm-timing-llm-api-ms" not in fastapi_response.headers
