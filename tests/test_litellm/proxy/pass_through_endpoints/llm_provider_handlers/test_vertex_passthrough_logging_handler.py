import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.types.utils import ModelResponse, Usage

MODEL = "gemini-3.1-flash-image"
PROMPT_TOKENS = 1000
COMPLETION_TOKENS = 500


def _expected_cost(prefix: str) -> float:
    entry = litellm.model_cost[f"{prefix}/{MODEL}"]
    return PROMPT_TOKENS * entry["input_cost_per_token"] + COMPLETION_TOKENS * entry["output_cost_per_token"]


def _model_response() -> ModelResponse:
    response = ModelResponse(model=MODEL)
    response.usage = Usage(
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
    )
    return response


def _logging_obj() -> LiteLLMLoggingObj:
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.litellm_call_id = "call-id"
    logging_obj.model_call_details = {}
    logging_obj.optional_params = {}
    return logging_obj


def test_rate_cards_differ_so_the_assertions_below_are_meaningful():
    assert _expected_cost("gemini") != _expected_cost("vertex_ai")


@pytest.mark.parametrize("provider", ["gemini", "vertex_ai"])
def test_generate_content_payload_prices_with_the_provider_it_is_given(provider):
    """
    The helper accepts custom_llm_provider and every call site resolves it, but it used
    to hardcode "vertex_ai" in the completion_cost call and apply the argument only to
    the log label. A gemini request was therefore labelled gemini and billed at Vertex
    rates.
    """
    logging_obj = _logging_obj()

    kwargs = VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
        litellm_model_response=_model_response(),
        model=MODEL,
        kwargs={},
        start_time=datetime.now(),
        end_time=datetime.now(),
        logging_obj=logging_obj,
        custom_llm_provider=provider,
    )

    assert kwargs["response_cost"] == pytest.approx(_expected_cost(provider))
    assert kwargs["custom_llm_provider"] == provider
    assert logging_obj.model_call_details["custom_llm_provider"] == provider


def _sse_chunks() -> list[str]:
    return [
        "data: "
        + json.dumps(
            {
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
                "responseId": "r1",
            }
        )
    ]


def _collect(url_route: str, custom_llm_provider: str | None) -> dict:
    return VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
        litellm_logging_obj=_logging_obj(),
        passthrough_success_handler_obj=MagicMock(),
        url_route=url_route,
        request_body={},
        endpoint_type=MagicMock(),
        start_time=datetime.now(),
        all_chunks=_sse_chunks(),
        model=MODEL,
        end_time=datetime.now(),
        custom_llm_provider=custom_llm_provider,
    )["kwargs"]


def test_collected_chunks_use_an_explicitly_passed_provider():
    """Native google_genai streams know their provider but send a hostname-less url_route."""
    assert _collect("/v1/generateContent", "gemini")["response_cost"] == pytest.approx(_expected_cost("gemini"))


def test_collected_chunks_fall_back_to_sniffing_the_url_when_no_provider_is_passed():
    """Pass-through callers only know the upstream URL, so hostname sniffing must still work."""
    gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/x:streamGenerateContent"
    assert _collect(gemini_url, None)["response_cost"] == pytest.approx(_expected_cost("gemini"))


def test_collected_chunks_still_price_vertex_at_vertex_rates():
    vertex_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/x:streamGenerateContent"
    assert _collect(vertex_url, None)["response_cost"] == pytest.approx(_expected_cost("vertex_ai"))
