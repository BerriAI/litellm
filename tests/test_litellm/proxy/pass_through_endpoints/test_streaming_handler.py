import json
from collections.abc import Iterator
from datetime import datetime
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

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
async def test_interrupted_anthropic_stream_recovers_output_tokens_off_the_event_loop():
    from unittest.mock import AsyncMock

    from tests.large_text import text
    from tests.test_litellm.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

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
            raw_bytes=_interrupted_anthropic_stream(model, text * 100),
            end_time=datetime.now(),
            model=model,
        )
    )

    logging_obj.dispatch_success_handlers.assert_awaited_once()
    logged_usage = logging_obj.dispatch_success_handlers.await_args.kwargs["result"].usage
    assert logged_usage.completion_tokens > 100_000
    assert_loop_stayed_free(took, lags)


@pytest.mark.asyncio
async def test_failed_anthropic_stream_records_partial_usage_off_the_event_loop():
    from unittest.mock import AsyncMock

    from tests.large_text import text
    from tests.test_litellm.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    model = "claude-fable-5"
    warm_tokenizer(model)
    logging_obj = _logging_obj()
    logging_obj.model_call_details = {"model": model, "stream": True}
    logging_obj.litellm_params = {}
    logging_obj.get_router_model_id.return_value = None
    logging_obj.dispatch_failure_handlers = AsyncMock()

    _, took, lags = await timed_with_loop_lags(
        lambda: PassThroughStreamingHandler.schedule_stream_failure_logging(
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            request_body={"model": model, "stream": True},
            raw_bytes=_interrupted_anthropic_stream(model, text * 100),
            exception=RuntimeError("upstream closed the stream"),
        )
    )
    await GLOBAL_LOGGING_WORKER.flush()

    logging_obj.dispatch_failure_handlers.assert_awaited_once()
    partial_usage = logging_obj.record_partial_usage_for_failure.call_args.kwargs["usage"]
    assert partial_usage.completion_tokens > 100_000
    assert_loop_stayed_free(took, lags)
