import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
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

MODEL = "gemini-3.1-flash-image"

# gemini/ rate card: 2.5e-07 in, 1.5e-06 out. vertex_ai/ rate card is exactly 2x that.
GEMINI_COST = 1000 * 2.5e-07 + 1000 * 1.5e-06
VERTEX_COST = 2 * GEMINI_COST


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
            "promptTokenCount": 1000,
            "candidatesTokenCount": 1000,
            "totalTokenCount": 2000,
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
    """A streamed gemini/* request must not be priced off the vertex_ai/ rate card."""
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
    """The AI Studio host resolves to `gemini`, so the cost must follow it, not the vertex_ai default."""
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
