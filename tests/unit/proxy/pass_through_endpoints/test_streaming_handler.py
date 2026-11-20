import asyncio
import json
import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import get_standard_logging_object_payload
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamContext,
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy._types import SpendLogsPayload
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import ClaudeSubscriptionUsage

MODEL = "gemini-stream-pricing-probe"
PROMPT_TOKENS = 1000
COMPLETION_TOKENS = 1000
GEMINI_INPUT_RATE = 1e-07
GEMINI_OUTPUT_RATE = 4e-07
VERTEX_INPUT_RATE = 1.5e-07
VERTEX_OUTPUT_RATE = 6e-07
GEMINI_COST = PROMPT_TOKENS * GEMINI_INPUT_RATE + COMPLETION_TOKENS * GEMINI_OUTPUT_RATE
VERTEX_COST = PROMPT_TOKENS * VERTEX_INPUT_RATE + COMPLETION_TOKENS * VERTEX_OUTPUT_RATE


@pytest.fixture(autouse=True)
def divergent_rate_cards(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setitem(
        litellm.model_cost,
        f"gemini/{MODEL}",
        {
            "input_cost_per_token": GEMINI_INPUT_RATE,
            "output_cost_per_token": GEMINI_OUTPUT_RATE,
            "litellm_provider": "gemini",
            "mode": "chat",
        },
    )
    monkeypatch.setitem(
        litellm.model_cost,
        f"vertex_ai/{MODEL}",
        {
            "input_cost_per_token": VERTEX_INPUT_RATE,
            "output_cost_per_token": VERTEX_OUTPUT_RATE,
            "litellm_provider": "vertex_ai",
            "mode": "chat",
        },
    )
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("/v1/messages", "/anthropic/v1/messages"))
@pytest.mark.parametrize("overage", (False, True, None))
async def test_oauth_stream_retains_billing_headers_through_reassembly_and_spend_logging(
    monkeypatch: pytest.MonkeyPatch, route: str, overage: bool | None
) -> None:
    model: Final = "claude-subscription-stream-test"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02, "litellm_provider": "anthropic", "mode": "chat"},
    )
    now: Final = datetime(2025, 1, 1)
    logger: Final = LiteLLMLoggingObj(
        model=model,
        messages=[],
        stream=True,
        call_type="anthropic_messages" if route == "/v1/messages" else "pass_through_endpoint",
        start_time=now,
        litellm_call_id="subscription",
        function_id="subscription",
    )
    logger.update_environment_variables(
        model=model,
        optional_params={},
        custom_llm_provider="anthropic",
        litellm_params={"litellm_metadata": {"used_client_oauth_token": True}},
    )
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_subscription",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10}},
        {"type": "message_stop"},
    )
    body: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
    headers: Final = (
        {}
        if overage is None
        else {
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-overage-in-use": "true" if overage else "false",
        }
    )
    observed: Final[asyncio.Future[tuple[float | None, ClaudeSubscriptionUsage | None, SpendLogsPayload]]] = (
        asyncio.get_running_loop().create_future()
    )

    async def log_stream(
        *,
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict[str, object],
        endpoint_type: EndpointType,
        start_time: datetime,
        raw_bytes: list[bytes],
        end_time: datetime,
    ) -> None:
        rebuilt, _ = PassThroughStreamingHandler._build_passthrough_logging_result(
            litellm_logging_obj=litellm_logging_obj,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            url_route=url_route,
            request_body=request_body,
            endpoint_type=endpoint_type,
            start_time=now,
            raw_bytes=raw_bytes,
            end_time=now,
            model=model,
        )
        cost: Final = litellm_logging_obj.response_cost_calculator(rebuilt)
        details: Final = {**litellm_logging_obj.model_call_details, "response_cost": cost}
        standard: Final = get_standard_logging_object_payload(
            kwargs=details,
            init_response_obj=rebuilt,
            start_time=now,
            end_time=now,
            logging_obj=litellm_logging_obj,
            status="success",
        )
        assert standard is not None
        spend: Final = get_logging_payload(
            kwargs={**details, "standard_logging_object": standard},
            response_obj=rebuilt,
            start_time=now,
            end_time=now,
        )
        observed.set_result((cost, standard["metadata"].get("claude_subscription_usage"), spend))

    upstream: Final = httpx.Response(200, headers=headers, stream=httpx.ByteStream(body))
    forwarded: Final = tuple(
        [
            chunk
            async for chunk in PassThroughStreamingHandler.chunk_processor(
                response=upstream,
                request_body={"model": model, "stream": True},
                litellm_logging_obj=logger,
                endpoint_type=EndpointType.ANTHROPIC,
                start_time=now,
                passthrough_success_handler_obj=PassThroughEndpointLogging(),
                url_route=route,
                route_streaming_logging=log_stream,
            )
        ]
    )
    await GLOBAL_LOGGING_WORKER.flush()
    cost, usage, spend = observed.result()
    expected_usage: Final = (
        None
        if overage is None
        else {
            "is_overage": overage,
            "status": "allowed",
            "overage_status": None,
        }
    )
    assert b"".join(forwarded) == body
    assert cost == pytest.approx(0.0 if overage is False else 100 * 0.01 + 10 * 0.02)
    assert usage == expected_usage
    assert json.loads(spend["metadata"])["claude_subscription_usage"] == expected_usage
    assert spend["spend"] == cost


def _chunks() -> list[str]:
    payload = {
        "candidates": [
            {
                "content": {"parts": [{"text": "hi"}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": PROMPT_TOKENS,
            "candidatesTokenCount": COMPLETION_TOKENS,
            "totalTokenCount": PROMPT_TOKENS + COMPLETION_TOKENS,
        },
        "modelVersion": MODEL,
    }
    return [f"data: {json.dumps(payload)}"]


def _logging_obj() -> LiteLLMLoggingObj:
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.model_call_details = {}
    logging_obj.optional_params = {}
    logging_obj.litellm_params = {}
    logging_obj.litellm_call_id = "test-call-id"
    return logging_obj


@pytest.mark.parametrize(
    "endpoint_type, expected_provider, expected_cost",
    [
        (EndpointType.GEMINI, "gemini", GEMINI_COST),
        (EndpointType.VERTEX_AI, "vertex_ai", VERTEX_COST),
    ],
)
def test_streaming_generate_content_bills_against_the_requested_provider(
    endpoint_type, expected_provider, expected_cost
):
    logging_obj = _logging_obj()

    _, kwargs = PassThroughStreamingHandler._build_passthrough_logging_result(
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route="/v1/generateContent",
        request_body={},
        endpoint_type=endpoint_type,
        start_time=datetime.now(),
        raw_bytes=[chunk.encode("utf-8") for chunk in _chunks()],
        end_time=datetime.now(),
        model=MODEL,
    )

    assert kwargs["response_cost"] == pytest.approx(expected_cost)
    assert logging_obj.model_call_details["custom_llm_provider"] == expected_provider


def test_vertex_generate_content_payload_prices_gemini_urls_at_gemini_rates():
    logging_obj = _logging_obj()

    result = VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route=f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent",
        request_body={},
        endpoint_type=EndpointType.VERTEX_AI,
        start_time=datetime.now(),
        all_chunks=_chunks(),
        model=MODEL,
        end_time=datetime.now(),
    )

    assert result["kwargs"]["response_cost"] == pytest.approx(GEMINI_COST)
    assert logging_obj.model_call_details["custom_llm_provider"] == "gemini"


def _interrupted_anthropic_stream(model: str, output_text: str) -> list[bytes]:
    def sse(event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_interrupted",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 29, "output_tokens": 2},
        },
    }
    block_start = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": output_text}}
    return [
        sse("message_start", message_start),
        sse("content_block_start", block_start),
        sse("content_block_delta", delta),
    ]


@pytest.mark.asyncio
async def test_interrupted_anthropic_stream_recovers_output_tokens_off_the_event_loop(caplog):
    from unittest.mock import AsyncMock

    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        uncounted_text,
        warm_tokenizer,
    )

    caplog.set_level(logging.WARNING, logger="LiteLLM")
    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
    model = "claude-fable-5"
    warm_tokenizer(model)
    logging_obj = _logging_obj()
    logging_obj.model_call_details = {"model": model, "stream": True}
    logging_obj.litellm_params = {}
    logging_obj.get_router_model_id.return_value = None
    logging_obj.dispatch_success_handlers = AsyncMock()

    _, took, lags = await timed_with_loop_lags(
        lambda: PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=PassThroughEndpointLogging(),
            url_route="/anthropic/v1/messages",
            request_body={"model": model, "stream": True},
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=datetime.now(),
            raw_bytes=_interrupted_anthropic_stream(model, uncounted_text()),
            end_time=datetime.now(),
            model=model,
        )
    )

    logging_obj.dispatch_success_handlers.assert_awaited_once()
    logged_usage = logging_obj.dispatch_success_handlers.await_args.kwargs["result"].usage
    assert logged_usage.completion_tokens > 100_000
    assert_loop_stayed_free(took, lags)


@pytest.mark.asyncio
async def test_failed_anthropic_stream_records_partial_usage_off_the_event_loop(caplog):
    from unittest.mock import AsyncMock

    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        uncounted_text,
        warm_tokenizer,
    )

    caplog.set_level(logging.WARNING, logger="LiteLLM")
    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
    model = "claude-fable-5"
    warm_tokenizer(model)
    logging_obj = _logging_obj()
    logging_obj.model_call_details = {"model": model, "stream": True}
    logging_obj.litellm_params = {}
    logging_obj.get_router_model_id.return_value = None
    logging_obj.attempt_scoped_copy.return_value = logging_obj
    logging_obj.dispatch_failure_handlers = AsyncMock()
    logging_obj.dispatch_attempt_failure_handlers = AsyncMock(
        side_effect=logging_obj.dispatch_failure_handlers
    )

    _, took, lags = await timed_with_loop_lags(
        lambda: PassThroughStreamingHandler.schedule_stream_failure_logging(
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            request_body={"model": model, "stream": True},
            raw_bytes=_interrupted_anthropic_stream(model, uncounted_text()),
            exception=RuntimeError("upstream closed the stream"),
        )
    )
    await GLOBAL_LOGGING_WORKER.flush()

    logging_obj.dispatch_failure_handlers.assert_awaited_once()
    partial_usage = logging_obj.record_partial_usage_for_failure.call_args.kwargs["usage"]
    assert partial_usage.completion_tokens > 100_000
    assert_loop_stayed_free(took, lags)


@pytest.mark.asyncio
async def test_interrupted_vertex_stream_bills_the_deployment_custom_price(monkeypatch: pytest.MonkeyPatch) -> None:
    deployment_id: Final = "vertex-custom-priced-deployment"
    custom_rates: Final = {"input_cost_per_token": 2e-09, "output_cost_per_token": 3e-09}
    monkeypatch.setitem(
        litellm.model_cost, deployment_id, {**custom_rates, "litellm_provider": "vertex_ai", "mode": "chat"}
    )
    logging_obj: Final = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "ping"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="interrupted-stream",
        function_id="interrupted-stream",
    )
    logging_obj.optional_params = {}
    logging_obj.litellm_params = {"metadata": {"model_info": {"id": deployment_id, **custom_rates}}}
    url_route: Final = (
        f"https://aiplatform.googleapis.com/v1/projects/p/locations/us"
        f"/publishers/google/models/{MODEL}:streamGenerateContent?alt=sse"
    )

    await PassThroughStreamingHandler.schedule_stream_failure_logging(
        litellm_logging_obj=logging_obj,
        endpoint_type=EndpointType.VERTEX_AI,
        request_body={},
        raw_bytes=[chunk.encode("utf-8") for chunk in _chunks()],
        exception=RuntimeError("upstream closed the stream"),
        stream_context=PassThroughStreamContext(PassThroughEndpointLogging(), url_route, datetime.now()),
    )
    recorded: Final = logging_obj.model_call_details["response_cost"]
    await GLOBAL_LOGGING_WORKER.flush()

    assert recorded == pytest.approx(PROMPT_TOKENS * 2e-09 + COMPLETION_TOKENS * 3e-09)
