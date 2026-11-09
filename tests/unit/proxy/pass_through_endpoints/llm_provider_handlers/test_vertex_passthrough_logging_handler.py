import asyncio
import json
from collections.abc import Iterator
from datetime import datetime
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import EmbeddingResponse, ModelResponse

_PREDICT_ROUTE = "/v1/projects/p/locations/us-central1/publishers/google/models/text-embedding-004:predict"
_INTERACTIONS_ROUTE = "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/interactions"


def _handle(url_route: str, payload: object) -> PassThroughEndpointLoggingTypedDict:
    logging_obj = Logging(
        model="unknown",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.optional_params = {}
    response = httpx.Response(200, json=payload)
    return VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route=url_route,
        result=response.text,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        cache_hit=False,
        request_body={"model": "gemini-omni-flash-preview"},
    )


def test_predict_response_with_text_embeddings_is_logged_as_an_embedding_response():
    result = _handle(
        _PREDICT_ROUTE,
        {
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 3}}},
                {"embeddings": {"values": [0.3, 0.4], "statistics": {"token_count": 4}}},
            ],
            "metadata": {"billableCharacterCount": 9},
        },
    )

    response = result["result"]
    assert isinstance(response, EmbeddingResponse)
    assert response.data == [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
    ]
    assert response.usage.prompt_tokens == 7
    assert result["kwargs"]["model"] == "text-embedding-004"
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"


@pytest.mark.parametrize("payload", [["not", "an", "object"], "text", 7, 1.5, True])
def test_predict_response_that_is_not_a_json_object_is_rejected_without_echoing_it(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _handle(_PREDICT_ROUTE, payload)

    assert "input_value" not in str(exc_info.value)


def test_interactions_usage_object_is_read_into_prompt_and_completion_tokens():
    result = _handle(
        _INTERACTIONS_ROUTE,
        {
            "id": "interactions/abc",
            "model": "gemini-omni-flash-preview",
            "usage": {
                "total_tokens": 41,
                "total_input_tokens": 12,
                "input_tokens_by_modality": [{"modality": "text", "tokens": 12}],
                "total_output_tokens": 9,
                "output_tokens_by_modality": [{"modality": "text", "tokens": 9}],
                "total_thought_tokens": 20,
            },
        },
    )

    response = result["result"]
    assert isinstance(response, ModelResponse)
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 29
    assert response.usage.completion_tokens_details.text_tokens == 9
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"


_UPLIFT_MODEL: Final = "gemini-regional-uplift-probe"
_UPLIFT_INPUT_RATE: Final = 1e-06
_UPLIFT_OUTPUT_RATE: Final = 4e-06
_UPLIFT_MULTIPLIER: Final = 1.1


@pytest.fixture
def regional_uplift_rate_card(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    rates: Final = {
        "input_cost_per_token": _UPLIFT_INPUT_RATE,
        "output_cost_per_token": _UPLIFT_OUTPUT_RATE,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, _UPLIFT_MODEL, rates)
    monkeypatch.setattr(litellm, "vertex_language_models", {*litellm.vertex_language_models, _UPLIFT_MODEL})
    monkeypatch.setitem(
        litellm.model_cost,
        f"vertex_ai/{_UPLIFT_MODEL}",
        {**rates, "regional_endpoint_uplift_multiplier": _UPLIFT_MULTIPLIER},
    )
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _generate_content_payload() -> dict[str, object]:
    return {
        "candidates": [{"content": {"parts": [{"text": "pong"}], "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 1, "totalTokenCount": 8},
        "modelVersion": _UPLIFT_MODEL,
    }


def _cost_the_logger_records_for(
    url_route: str,
    *,
    stream: bool,
    reply: dict[str, object],
    stream_events: list[dict[str, object]],
    model_info: dict[str, object] | None = None,
    route_provider: str | None = None,
    base_model: object = None,
) -> float:
    logging_obj: Final = Logging(
        model="unknown",
        messages=[{"role": "user", "content": "ping"}],
        stream=stream,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(
        litellm_params={"metadata": {"model_info": model_info}} if model_info is not None else {}, optional_params={}
    )
    if route_provider is not None:
        logging_obj.model_call_details["custom_llm_provider"] = route_provider
    if base_model is not None:
        logging_obj.model_call_details["litellm_params"] = {"base_model": base_model}
    if stream:
        asyncio.run(
            PassThroughStreamingHandler.route_streaming_logging_to_handler(
                litellm_logging_obj=logging_obj,
                passthrough_success_handler_obj=PassThroughEndpointLogging(),
                url_route=url_route,
                request_body={},
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=datetime(2026, 1, 1),
                raw_bytes=[f"data: {json.dumps(event)}\n\n".encode() for event in stream_events],
                end_time=datetime(2026, 1, 1),
            )
        )
        return logging_obj.model_call_details["response_cost"]
    response: Final = httpx.Response(200, json=reply)
    result: Final = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route=url_route,
        result=response.text,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        cache_hit=False,
        request_body={},
    )
    logging_obj._process_hidden_params_and_response_cost(
        result["result"], datetime(2026, 1, 1), datetime(2026, 1, 1), build_logging_payload=False
    )
    return logging_obj.model_call_details["response_cost"]


def _cost_the_logger_records(location: str, *, stream: bool, model_info: dict[str, object] | None = None) -> float:
    method: Final = "streamGenerateContent" if stream else "generateContent"
    url_route: Final = (
        f"https://aiplatform.googleapis.com/v1/projects/p/locations/{location}"
        f"/publishers/google/models/{_UPLIFT_MODEL}:{method}"
    )
    return _cost_the_logger_records_for(
        url_route,
        stream=stream,
        reply=_generate_content_payload(),
        stream_events=[_generate_content_payload()],
        model_info=model_info,
    )


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
@pytest.mark.parametrize("location, multiplier", [("global", 1.0), ("us", _UPLIFT_MULTIPLIER)])
def test_generate_content_spend_follows_the_vertex_location(
    regional_uplift_rate_card: None, stream: bool, location: str, multiplier: float
) -> None:
    expected: Final = (7 * _UPLIFT_INPUT_RATE + 1 * _UPLIFT_OUTPUT_RATE) * multiplier

    assert _cost_the_logger_records(location, stream=stream) == pytest.approx(expected), (location, stream)


@pytest.mark.parametrize(
    "gemini_api_base", ["https://generativelanguage.googleapis.com", "https://llm-egress.internal.example/google"]
)
@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
def test_gemini_route_spend_has_no_vertex_uplift_whatever_host_serves_it(
    regional_uplift_rate_card: None, monkeypatch: pytest.MonkeyPatch, gemini_api_base: str, stream: bool
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        f"gemini/{_UPLIFT_MODEL}",
        {
            "input_cost_per_token": _UPLIFT_INPUT_RATE,
            "output_cost_per_token": _UPLIFT_OUTPUT_RATE,
            "litellm_provider": "gemini",
            "mode": "chat",
        },
    )
    method: Final = "streamGenerateContent?alt=sse" if stream else "generateContent"

    recorded: Final = _cost_the_logger_records_for(
        f"{gemini_api_base}/v1beta/models/{_UPLIFT_MODEL}:{method}",
        stream=stream,
        reply=_generate_content_payload(),
        stream_events=[_generate_content_payload()],
        route_provider="gemini",
    )

    assert recorded == pytest.approx(7 * _UPLIFT_INPUT_RATE + 1 * _UPLIFT_OUTPUT_RATE), (gemini_api_base, stream)


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
def test_a_vertex_host_that_only_ends_with_the_gemini_host_name_keeps_the_vertex_uplift(
    regional_uplift_rate_card: None, stream: bool
) -> None:
    method: Final = "streamGenerateContent" if stream else "generateContent"

    recorded: Final = _cost_the_logger_records_for(
        "https://vertex-generativelanguage.googleapis.com/v1/projects/p/locations/us/publishers/google/models/"
        f"{_UPLIFT_MODEL}:{method}",
        stream=stream,
        reply=_generate_content_payload(),
        stream_events=[_generate_content_payload()],
    )

    assert recorded == pytest.approx((7 * _UPLIFT_INPUT_RATE + 1 * _UPLIFT_OUTPUT_RATE) * _UPLIFT_MULTIPLIER), stream


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
def test_a_subdomain_of_the_gemini_host_bills_the_gemini_price(
    regional_uplift_rate_card: None, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        f"gemini/{_UPLIFT_MODEL}",
        {
            "input_cost_per_token": _UPLIFT_INPUT_RATE,
            "output_cost_per_token": _UPLIFT_OUTPUT_RATE,
            "litellm_provider": "gemini",
            "mode": "chat",
        },
    )
    method: Final = "streamGenerateContent?alt=sse" if stream else "generateContent"

    recorded: Final = _cost_the_logger_records_for(
        f"https://eu.generativelanguage.googleapis.com/v1beta/models/{_UPLIFT_MODEL}:{method}",
        stream=stream,
        reply=_generate_content_payload(),
        stream_events=[_generate_content_payload()],
    )

    assert recorded == pytest.approx(7 * _UPLIFT_INPUT_RATE + 1 * _UPLIFT_OUTPUT_RATE), stream


@pytest.mark.parametrize("value", [123, ["gemini-2.5-flash-lite"], {"model": "gemini-2.5-flash-lite"}])
@pytest.mark.parametrize("carrier", ["root", "model_info"])
@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
def test_a_non_string_base_model_bills_the_list_price(
    regional_uplift_rate_card: None, stream: bool, carrier: str, value: object
) -> None:
    method: Final = "streamGenerateContent" if stream else "generateContent"

    recorded: Final = _cost_the_logger_records_for(
        f"https://aiplatform.googleapis.com/v1/projects/p/locations/us/publishers/google/models/{_UPLIFT_MODEL}:{method}",
        stream=stream,
        reply=_generate_content_payload(),
        stream_events=[_generate_content_payload()],
        model_info={"base_model": value} if carrier == "model_info" else None,
        base_model=value if carrier == "root" else None,
    )

    assert recorded == pytest.approx((7 * _UPLIFT_INPUT_RATE + 1 * _UPLIFT_OUTPUT_RATE) * _UPLIFT_MULTIPLIER), stream


_CUSTOM_PRICED_DEPLOYMENT: Final = "vertex-custom-priced-deployment"
_CUSTOM_INPUT_RATE: Final = 2e-09
_CUSTOM_OUTPUT_RATE: Final = 3e-09


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
@pytest.mark.parametrize("location", ["global", "us"])
def test_generate_content_spend_uses_the_deployment_custom_price(
    regional_uplift_rate_card: None, monkeypatch: pytest.MonkeyPatch, stream: bool, location: str
) -> None:
    custom_rates: Final = {"input_cost_per_token": _CUSTOM_INPUT_RATE, "output_cost_per_token": _CUSTOM_OUTPUT_RATE}
    monkeypatch.setitem(
        litellm.model_cost, _CUSTOM_PRICED_DEPLOYMENT, {**custom_rates, "litellm_provider": "vertex_ai", "mode": "chat"}
    )
    expected: Final = 7 * _CUSTOM_INPUT_RATE + 1 * _CUSTOM_OUTPUT_RATE

    recorded: Final = _cost_the_logger_records(
        location, stream=stream, model_info={"id": _CUSTOM_PRICED_DEPLOYMENT, **custom_rates}
    )

    assert recorded == pytest.approx(expected), (location, stream)


_CLAUDE_UPLIFT_MODEL: Final = "claude-regional-uplift-probe"
_CLAUDE_INPUT_RATE: Final = 3e-06
_CLAUDE_OUTPUT_RATE: Final = 1.5e-05


@pytest.fixture
def claude_regional_uplift_rate_card(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    rates: Final = {
        "input_cost_per_token": _CLAUDE_INPUT_RATE,
        "output_cost_per_token": _CLAUDE_OUTPUT_RATE,
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, _CLAUDE_UPLIFT_MODEL, {**rates, "litellm_provider": "anthropic"})
    monkeypatch.setattr(litellm, "anthropic_models", {*litellm.anthropic_models, _CLAUDE_UPLIFT_MODEL})
    monkeypatch.setitem(
        litellm.model_cost,
        f"vertex_ai/{_CLAUDE_UPLIFT_MODEL}",
        {
            **rates,
            "litellm_provider": "vertex_ai-anthropic_models",
            "regional_endpoint_uplift_multiplier": _UPLIFT_MULTIPLIER,
        },
    )
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _claude_message() -> dict[str, object]:
    return {
        "id": "msg_probe",
        "type": "message",
        "role": "assistant",
        "model": _CLAUDE_UPLIFT_MODEL,
        "content": [{"type": "text", "text": "pong"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 1},
    }


def _claude_stream_events() -> list[dict[str, object]]:
    message_start: Final = {**_claude_message(), "content": [], "usage": {"input_tokens": 7, "output_tokens": 0}}
    return [
        {"type": "message_start", "message": message_start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "pong"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
@pytest.mark.parametrize("location, multiplier", [("global", 1.0), ("us-east5", _UPLIFT_MULTIPLIER)])
def test_claude_raw_predict_spend_follows_the_vertex_location(
    claude_regional_uplift_rate_card: None, stream: bool, location: str, multiplier: float
) -> None:
    method: Final = "streamRawPredict" if stream else "rawPredict"
    url_route: Final = (
        f"https://aiplatform.googleapis.com/v1/projects/p/locations/{location}"
        f"/publishers/anthropic/models/{_CLAUDE_UPLIFT_MODEL}:{method}"
    )
    expected: Final = (7 * _CLAUDE_INPUT_RATE + 1 * _CLAUDE_OUTPUT_RATE) * multiplier

    recorded: Final = _cost_the_logger_records_for(
        url_route, stream=stream, reply=_claude_message(), stream_events=_claude_stream_events()
    )

    assert recorded == pytest.approx(expected), (location, stream)


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
def test_claude_raw_predict_spend_uses_the_deployment_custom_price(
    claude_regional_uplift_rate_card: None, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    custom_rates: Final = {"input_cost_per_token": _CUSTOM_INPUT_RATE, "output_cost_per_token": _CUSTOM_OUTPUT_RATE}
    monkeypatch.setitem(
        litellm.model_cost, _CUSTOM_PRICED_DEPLOYMENT, {**custom_rates, "litellm_provider": "vertex_ai", "mode": "chat"}
    )
    method: Final = "streamRawPredict" if stream else "rawPredict"
    url_route: Final = (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
        f"/publishers/anthropic/models/{_CLAUDE_UPLIFT_MODEL}:{method}"
    )

    recorded: Final = _cost_the_logger_records_for(
        url_route,
        stream=stream,
        reply=_claude_message(),
        stream_events=_claude_stream_events(),
        model_info={"id": _CUSTOM_PRICED_DEPLOYMENT, **custom_rates},
    )

    assert recorded == pytest.approx(7 * _CUSTOM_INPUT_RATE + 1 * _CUSTOM_OUTPUT_RATE), stream


def test_a_claude_stream_through_a_configured_vertex_gateway_with_a_short_path_bills_the_vertex_price(
    claude_regional_uplift_rate_card: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        f"vertex_ai/{_CLAUDE_UPLIFT_MODEL}",
        {
            "input_cost_per_token": _CLAUDE_INPUT_RATE,
            "output_cost_per_token": _CLAUDE_OUTPUT_RATE,
            "litellm_provider": "vertex_ai-anthropic_models",
            "mode": "chat",
        },
    )
    litellm.get_model_info.cache_clear()

    recorded: Final = _cost_the_logger_records_for(
        f"https://vertex-gateway.internal.example/vxc/models/{_CLAUDE_UPLIFT_MODEL}:streamRawPredict",
        stream=True,
        reply=_claude_message(),
        stream_events=_claude_stream_events(),
    )

    assert recorded == pytest.approx(7 * _CLAUDE_INPUT_RATE + 1 * _CLAUDE_OUTPUT_RATE)


def test_build_complete_streaming_response_assembles_stream_generate_content_sse_chunks():
    logging_obj: Final = Logging(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={}, model="gemini-2.5-flash")
    first_chunk: Final = {"candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}, "index": 0}]}
    last_chunk: Final = {
        "candidates": [
            {"content": {"parts": [{"text": " there!"}], "role": "model"}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8, "totalTokenCount": 18},
    }

    response: Final = VertexPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=[f"data: {json.dumps(first_chunk)}", f"data: {json.dumps(last_chunk)}"],
        litellm_logging_obj=logging_obj,
        model="gemini-2.5-flash",
        url_route="/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent",
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Hello there!"
    assert response.choices[0].finish_reason == "stop"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (10, 8, 18)
