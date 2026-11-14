"""
Tests for rerank_endpoints/endpoints.py response headers.
"""

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient

import litellm.proxy.common_request_processing as common_request_processing_mod
import litellm.proxy.proxy_server as proxy_server_mod
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
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


def _build_request(headers: tuple[tuple[bytes, bytes], ...] = ()) -> Request:
    body = json.dumps({"model": "rerank-model", "query": "q", "documents": ["a", "b"]}).encode()
    messages: Final = iter(({"type": "http.request", "body": body, "more_body": False},))

    async def receive():
        message: Final = next(messages, None)
        if message is None:
            await asyncio.Event().wait()
        return message

    return Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/rerank",
            "headers": [(b"content-type", b"application/json"), *headers],
            "query_string": b"",
        },
        receive=receive,
    )


async def _call_rerank(
    hidden_params: dict = HIDDEN_PARAMS, general_settings: Mapping[str, object] = MappingProxyType({})
) -> Response:
    response = RerankResponse(id="rerank-1", results=[{"index": 0, "relevance_score": 0.9}])
    response._hidden_params = dict(hidden_params)

    fastapi_response = Response()
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
    proxy_logging_obj.update_request_status = AsyncMock()

    async def fake_add_litellm_data_to_request(**kwargs):
        return dict(kwargs["data"])

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
        patch.object(proxy_server_mod, "general_settings", general_settings),  # test-quality-ok: the rerank route reads these proxy_server module globals; no injection seam on the FastAPI handler
    ):
        await rerank(
            request=_build_request(headers=((b"x-litellm-call-id", b"call-123"),)),
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
async def test_connected_rerank_with_cancel_on_disconnect_still_returns_its_headers():
    fastapi_response = await asyncio.wait_for(_call_rerank(general_settings={"cancel_on_disconnect": True}), timeout=5)

    assert fastapi_response.headers["x-litellm-call-id"] == "call-123"
    assert fastapi_response.headers["x-litellm-response-cost"] == "0.002"
    assert fastapi_response.headers["x-litellm-model-id"] == "deployment-1"


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


async def _rerank_failure(
    failure: Exception,
    *,
    raised_before_routing: bool,
    monkeypatch: pytest.MonkeyPatch,
    headers: tuple[tuple[bytes, bytes], ...] = (),
) -> ProxyException:
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.pre_call_hook = AsyncMock(
        side_effect=failure if raised_before_routing else lambda **kwargs: kwargs["data"]
    )
    proxy_logging_obj.post_call_failure_hook = AsyncMock()

    async def fake_add_litellm_data_to_request(**kwargs: object) -> object:
        return kwargs["data"]

    async def failing_route_request(**kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(proxy_server_mod, "add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr(proxy_server_mod, "route_request", failing_route_request)
    monkeypatch.setattr(proxy_server_mod, "proxy_logging_obj", proxy_logging_obj)
    monkeypatch.setattr(proxy_server_mod, "llm_router", MagicMock())
    monkeypatch.setattr(proxy_server_mod, "version", "1.2.3")

    with pytest.raises(ProxyException) as raised:
        await rerank(
            request=_build_request(headers),
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        )
    return raised.value


@pytest.fixture
def propagating_proxy_logger() -> Iterator[None]:
    verbose_proxy_logger.propagate = True
    try:
        yield
    finally:
        verbose_proxy_logger.propagate = False


@pytest.mark.asyncio
async def test_failure_log_carries_the_callers_litellm_call_id(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, propagating_proxy_logger: None
) -> None:
    """LIT-7836: the /rerank error line must carry the same litellm_call_id the client
    sent, both in the rendered message and as a structured log record field."""
    call_id = "rerank-call-7836"
    failure = HTTPException(status_code=401, detail={"error": "invalid api key"})

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
        raised = await _rerank_failure(
            failure,
            raised_before_routing=False,
            monkeypatch=monkeypatch,
            headers=((b"x-litellm-call-id", call_id.encode()),),
        )

    assert raised.headers["x-litellm-call-id"] == call_id
    record = next(r for r in caplog.records if "Exception occured" in r.getMessage())
    assert record.litellm_call_id == call_id
    assert call_id in record.getMessage()


@pytest.mark.asyncio
async def test_a_model_the_router_cannot_serve_answers_an_openai_typed_error(monkeypatch: pytest.MonkeyPatch):
    """A bare HTTPException carries no type or param, so the tail used to ship the
    literal string "None" in both fields."""
    failure = HTTPException(status_code=404, detail={"error": "rerank: Invalid model name passed in model=rerank-model"})

    error = await _rerank_failure(failure, raised_before_routing=False, monkeypatch=monkeypatch)

    assert (error.type, error.param, error.code) == ("invalid_request_error", None, "404")


@pytest.mark.asyncio
async def test_a_rejection_raised_before_routing_keeps_its_own_status(monkeypatch: pytest.MonkeyPatch):
    """A ProxyException stores its status as the string ``code``, which the tail used to
    miss and rewrap as a 500 while keeping the 4xx type and param."""
    rejection = ProxyException(message="session_id is required", type="bad_request_error", param="session_id", code=400)

    error = await _rerank_failure(rejection, raised_before_routing=True, monkeypatch=monkeypatch)

    assert (error.type, error.param, error.code) == ("bad_request_error", "session_id", "400")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_on_disconnect", [True, "true"])
async def test_hang_up_with_cancel_on_disconnect_cancels_the_rerank_call_and_answers_499(
    monkeypatch: pytest.MonkeyPatch, cancel_on_disconnect: object
) -> None:
    body: Final = json.dumps({"model": "rerank-model", "query": "q", "documents": ["a", "b"]}).encode()
    messages: Final = iter(({"type": "http.request", "body": body, "more_body": False}, {"type": "http.disconnect"}))
    upstream_started: Final = asyncio.Event()
    upstream_cancelled: Final = asyncio.Event()

    async def receive() -> dict[str, object]:
        message: Final = next(messages)
        if message["type"] == "http.disconnect":
            await upstream_started.wait()
        return message

    async def held_upstream() -> None:
        upstream_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            upstream_cancelled.set()
            raise

    async def holding_route_request(**kwargs: object) -> object:
        return held_upstream()

    async def fake_add_litellm_data_to_request(**kwargs: object) -> object:
        return kwargs["data"]

    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
    proxy_logging_obj.post_call_failure_hook = AsyncMock()
    monkeypatch.setattr(proxy_server_mod, "general_settings", {"cancel_on_disconnect": cancel_on_disconnect})
    monkeypatch.setattr(proxy_server_mod, "add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr(proxy_server_mod, "route_request", holding_route_request)
    monkeypatch.setattr(proxy_server_mod, "proxy_logging_obj", proxy_logging_obj)
    monkeypatch.setattr(proxy_server_mod, "llm_router", MagicMock())
    monkeypatch.setattr(proxy_server_mod, "version", "1.2.3")
    request: Final = Request(
        scope={"type": "http", "method": "POST", "path": "/rerank", "headers": [], "query_string": b""},
        receive=receive,
    )

    with pytest.raises(ProxyException) as raised:
        await asyncio.wait_for(
            rerank(request=request, fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth(api_key="sk-test")),
            timeout=5,
        )

    assert raised.value.code == "499"
    assert upstream_cancelled.is_set()


def test_failed_rerank_spend_row_records_arerank(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    import litellm
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.proxy_server import ProxyConfig
    from litellm.proxy.rerank_endpoints.endpoints import router as rerank_router
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
    from litellm.proxy.utils import ProxyLogging

    provider: Final = respx_mock.post(url__startswith="https://rerank-provider.example").respond(
        400, json={"message": "invalid documents"}
    )
    cache: Final = UserApiKeyCache()
    spend_writer: Final = MagicMock(update_database=AsyncMock())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: spend_writer)])
    monkeypatch.setattr(proxy_server_mod, "general_settings", {})
    monkeypatch.setattr(
        proxy_server_mod,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "rerank-model",
                    "litellm_params": {
                        "model": "cohere/rerank-v3.5",
                        "api_base": "https://rerank-provider.example",
                        "api_key": "provider-key",
                    },
                }
            ],
            num_retries=0,
        ),
    )
    monkeypatch.setattr(proxy_server_mod, "proxy_config", ProxyConfig())
    monkeypatch.setattr(proxy_server_mod, "proxy_logging_obj", ProxyLogging(user_api_key_cache=cache))
    monkeypatch.setattr(proxy_server_mod, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server_mod, "user_model", None)
    monkeypatch.setattr(proxy_server_mod, "version", "1.2.3")
    app: Final = FastAPI()
    app.include_router(rerank_router)
    app.add_exception_handler(ProxyException, proxy_server_mod.openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-test")
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = client.post(
                "/v1/rerank", json={"model": "rerank-model", "query": "q", "documents": ["a", "b"]}
            )
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert (response.status_code, provider.call_count) == (400, 1), response.text
    db_call: Final = spend_writer.update_database.call_args.kwargs
    payload: Final = get_logging_payload(
        kwargs=db_call["kwargs"],
        response_obj=db_call["completion_response"],
        start_time=db_call["start_time"],
        end_time=db_call["end_time"],
    )
    assert (payload["status"], payload["call_type"]) == ("failure", "arerank")
