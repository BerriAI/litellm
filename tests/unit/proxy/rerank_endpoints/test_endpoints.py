"""
Tests for rerank_endpoints/endpoints.py response headers.
"""

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from queue import SimpleQueue
from typing import Final
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import httpx
import pytest
import respx
from fastapi import HTTPException, Request, Response

import litellm
import litellm.proxy.common_request_processing as common_request_processing_mod
import litellm.proxy.proxy_server as proxy_server_mod
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
from litellm.proxy.rerank_endpoints.endpoints import rerank
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import ProxyLogging
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


def _build_request(
    headers: tuple[tuple[bytes, bytes], ...] = (),
    documents: tuple[str, ...] = ("a", "b"),
    path: str = "/rerank",
    root_path: str = "",
    metadata: Mapping[str, object] | None = None,
) -> Request:
    body = json.dumps(
        {
            "model": "rerank-model",
            "query": "q",
            "documents": documents,
            **({"metadata": metadata} if metadata is not None else {}),
        }
    ).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": path,
            "root_path": root_path,
            "headers": [(b"content-type", b"application/json"), *headers],
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
@pytest.mark.parametrize(
    ("path", "root_path", "expected_path"),
    [
        ("/rerank", "", "/rerank"),
        ("/v1/rerank", "", "/v1/rerank"),
        ("/v2/rerank", "", "/v2/rerank"),
        ("/gateway/v2/rerank", "/gateway", "/v2/rerank"),
    ],
)
@pytest.mark.parametrize("partial_failure", [False, True])
async def test_decision_rerank_writes_one_log_with_gateway_endpoint_and_aggregate_usage(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    path: str,
    root_path: str,
    expected_path: str,
    partial_failure: bool,
) -> None:
    model: Final = "typesafe/jevrank-spend-unit-test"
    rate: Final = 0.004
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "typesafe",
            "mode": "evaluation",
            "rerank_strategy": "pointwise",
            "input_cost_per_token": rate,
            "output_cost_per_token": 0,
        },
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cache", None)
    llm_router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "rerank-model",
                "litellm_params": {"model": model, "api_key": "test-key"},
            }
        ],
        num_retries=0,
    )
    writer: Final = create_autospec(DBSpendUpdateWriter, instance=True)
    db_logger: Final = ProxyDBLogger(spend_writer=lambda: writer)
    proxy_logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    proxy_logging.db_spend_update_writer = writer
    monkeypatch.setattr(litellm, "callbacks", [db_logger])
    monkeypatch.setattr(proxy_server_mod, "llm_router", llm_router)
    monkeypatch.setattr(proxy_server_mod, "proxy_logging_obj", proxy_logging)
    monkeypatch.setattr(proxy_server_mod, "general_settings", {})
    monkeypatch.setattr(proxy_server_mod, "prisma_client", None)

    def score(request: httpx.Request) -> httpx.Response:
        passage: Final = json.loads(json.loads(request.content)["state"])["passage"]
        if passage == "failed":
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(
            200,
            json={
                "model": "jevrank-spend-unit-test",
                "answers": {"relevance": {"type": "noul", "noul": 0.8}},
                "usage": {
                    "input_tokens": 100 if passage == "first" else 80,
                    "output_tokens": 0,
                },
            },
        )

    upstream: Final = respx_mock.post("https://api.typesafe.ai/v1/systemone").mock(
        side_effect=score
    )
    request: Final = _build_request(
        headers=((b"x-litellm-call-id", b"rerank-spend"),),
        documents=("first", "failed" if partial_failure else "middle", "last"),
        path=path,
        root_path=root_path,
        metadata={
            "tags": ["search"],
            "spend_logs_metadata": {
                "customer": "retained",
                "gateway_endpoint": {"method": "DELETE", "path": "/client-forged"},
            },
        },
    )
    call: Final = rerank(
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", request_route=expected_path),
    )
    if partial_failure:
        with pytest.raises(ProxyException) as failure:
            await call
        assert failure.value.code == "503"
    else:
        await call
    await asyncio.sleep(0)
    GLOBAL_LOGGING_WORKER.start()
    await GLOBAL_LOGGING_WORKER.flush()

    assert upstream.call_count == (2 if partial_failure else 3)
    writer.update_database.assert_awaited_once()
    write: Final = writer.update_database.call_args.kwargs
    recorded_at: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row: Final = get_logging_payload(
        kwargs={**write["kwargs"], "response_cost": write["response_cost"]},
        response_obj=write["completion_response"],
        start_time=recorded_at,
        end_time=recorded_at,
    )
    tokens: Final = 100 if partial_failure else 260
    assert row["request_id"] == "rerank-spend"
    assert row["call_type"] == "arerank"
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (tokens, 0, tokens)
    assert row["spend"] == pytest.approx(tokens * rate)
    assert json.loads(row["metadata"])["spend_logs_metadata"] == {
        "customer": "retained",
        "gateway_endpoint": {"method": "POST", "path": expected_path},
    }
    assert json.loads(row["request_tags"]) == ["search"]


@pytest.mark.asyncio
@pytest.mark.parametrize("decision_model", [True, False])
async def test_rerank_keeps_the_selected_deployment_success_callback(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, decision_model: bool
) -> None:
    provider: Final = "typesafe" if decision_model else "cohere"
    model: Final = f"{provider}/rerank-callback-unit-test"
    events: Final[SimpleQueue[object]] = SimpleQueue()

    async def record_success(
        kwargs: Mapping[str, object],
        response: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        events.put(response)

    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": provider,
            "mode": "evaluation" if decision_model else "rerank",
            **({"rerank_strategy": "pointwise"} if decision_model else {}),
            "input_cost_per_token": 0.004,
            "output_cost_per_token": 0,
        },
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cache", None)
    llm_router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "rerank-model",
                "litellm_params": {
                    "model": model,
                    "api_key": "test-key",
                    "success_callback": [record_success],
                },
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(proxy_server_mod, "llm_router", llm_router)
    monkeypatch.setattr(
        proxy_server_mod,
        "proxy_logging_obj",
        ProxyLogging(user_api_key_cache=UserApiKeyCache()),
    )
    monkeypatch.setattr(proxy_server_mod, "general_settings", {})
    monkeypatch.setattr(proxy_server_mod, "prisma_client", None)
    if decision_model:
        respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
            json={
                "model": "rerank-callback-unit-test",
                "answers": {"relevance": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }
        )
    else:
        respx_mock.post("https://api.cohere.com/v2/rerank").respond(
            json={
                "id": "native-rerank",
                "results": [{"index": 0, "relevance_score": 0.8}],
            }
        )
    response: Final = await rerank(
        request=_build_request(),
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", request_route="/rerank"),
    )
    await asyncio.sleep(0)
    GLOBAL_LOGGING_WORKER.start()
    await GLOBAL_LOGGING_WORKER.flush()

    assert events.qsize() == 1
    assert events.get_nowait() is response
