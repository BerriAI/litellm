from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, cast

import httpx

import litellm
import litellm.cost_calculator as cost_calculator
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.llms.vertex_ai.ocr.deepseek_transformation import VertexAIDeepSeekOCRConfig

MODEL: Final = "vertex_ai/deepseek-ai/deepseek-ocr-maas"
PROMPT_TOKENS: Final = 281
COMPLETION_TOKENS: Final = 6


def _response() -> OCRResponse:
    transform_response: Final = cast(  # cast-ok: the legacy provider method has untyped variadic parameters
        Callable[..., OCRResponse], VertexAIDeepSeekOCRConfig().transform_ocr_response
    )
    return transform_response(
        model="deepseek-ai/deepseek-ocr-maas",
        raw_response=httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"content": "invoice 123"}}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            },
        ),
        logging_obj=None,
    )


def test_deepseek_ocr_preserves_billable_token_usage() -> None:
    response: Final = _response()

    assert response.usage_info is not None
    assert response.usage_info.prompt_tokens == PROMPT_TOKENS
    assert response.usage_info.completion_tokens == COMPLETION_TOKENS
    assert response.usage_info.total_tokens == PROMPT_TOKENS + COMPLETION_TOKENS
    assert response.usage_info.pages_processed is None


def test_ocr_usage_maps_to_standard_logging() -> None:
    normalize_usage: Final = cast(  # cast-ok: the legacy logging helper has an untyped dictionary signature
        Callable[..., dict[str, object]], StandardLoggingPayloadSetup.get_usage_as_dict
    )
    token_usage: Final = normalize_usage(response_obj=_response().model_dump())
    page_usage: Final = normalize_usage(
        response_obj={
            "usage_info": {
                "pages_processed": 5,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
        }
    )

    assert token_usage["prompt_tokens"] == PROMPT_TOKENS
    assert token_usage["completion_tokens"] == COMPLETION_TOKENS
    assert token_usage["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS
    assert page_usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "pages_processed": 5,
    }


def test_deepseek_ocr_cost_uses_provider_token_rates(local_model_cost_map: object) -> None:
    model_info: Final = litellm.get_model_info(model=MODEL, custom_llm_provider="vertex_ai")
    response: Final = _response()
    calculate_cost: Final = cast(  # cast-ok: completion_cost has legacy untyped optional parameters
        Callable[..., float], cost_calculator.completion_cost
    )

    cost: Final = calculate_cost(
        completion_response=response,
        model=MODEL,
        custom_llm_provider="vertex_ai",
        call_type="ocr",
    )

    input_cost_per_token: Final = model_info["input_cost_per_token"]
    output_cost_per_token: Final = model_info["output_cost_per_token"]
    assert input_cost_per_token is not None
    assert output_cost_per_token is not None
    expected: Final = PROMPT_TOKENS * input_cost_per_token + COMPLETION_TOKENS * output_cost_per_token
    assert model_info.get("ocr_cost_per_page") is None
    assert math.isclose(cost, expected, rel_tol=1e-12)
    assert cost > 0
