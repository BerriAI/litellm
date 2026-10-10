import json
from datetime import datetime
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
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


def test_generate_content_cost_uses_the_deployments_custom_pricing(monkeypatch: pytest.MonkeyPatch):
    """A routed generateContent call (streamed calls are logged through this
    path) is billed at the price set on the deployment it was routed to, not
    the cost map's shared price for the backend model."""
    import litellm
    from litellm.types.utils import Usage

    deployment_id: Final = "vertex-gemini-with-custom-price"
    custom_price: Final = {"input_cost_per_token": 1.5e-6, "output_cost_per_token": 7.5e-6}
    monkeypatch.setitem(
        litellm.model_cost,
        deployment_id,
        {**custom_price, "litellm_provider": "vertex_ai-language-models", "mode": "chat"},
    )
    logging_obj: Final = Logging(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(
        litellm_params={"metadata": {"model_info": {"id": deployment_id, **custom_price}}},
        optional_params={},
        model="gemini-2.5-flash",
    )
    response: Final = ModelResponse(
        model="gemini-2.5-flash",
        usage=Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000),
    )

    kwargs: Final = VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
        litellm_model_response=response,
        model="gemini-2.5-flash",
        kwargs={},
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai",
        vertex_location=None,
    )

    assert kwargs["response_cost"] == pytest.approx(1.5 + 7.5)


def test_assembled_stream_is_priced_at_the_deployments_custom_pricing(monkeypatch: pytest.MonkeyPatch):
    """The logger prices a streamed generateContent call from the response the
    handler assembles; that response must not arrive carrying a cost priced
    by model name alone, which the logger would reuse as already calculated."""
    import litellm

    deployment_id: Final = "vertex-gemini-with-custom-price"
    custom_price: Final = {"input_cost_per_token": 1.5e-6, "output_cost_per_token": 7.5e-6}
    monkeypatch.setitem(
        litellm.model_cost,
        deployment_id,
        {**custom_price, "litellm_provider": "vertex_ai-language-models", "mode": "chat"},
    )
    logging_obj: Final = Logging(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(
        litellm_params={"litellm_metadata": {"model_info": {"id": deployment_id, **custom_price}}},
        optional_params={},
        model="gemini-2.5-flash",
        custom_llm_provider="vertex_ai",
    )
    last_chunk: Final = {
        "candidates": [{"content": {"parts": [{"text": "Hi."}], "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {
            "promptTokenCount": 1_000_000,
            "candidatesTokenCount": 1_000_000,
            "totalTokenCount": 2_000_000,
        },
        "modelVersion": "gemini-2.5-flash",
    }

    response: Final = VertexPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=[f"data: {json.dumps(last_chunk)}"],
        litellm_logging_obj=logging_obj,
        model="gemini-2.5-flash",
        url_route="/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-flash:streamGenerateContent",
    )

    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(1.5 + 7.5)
