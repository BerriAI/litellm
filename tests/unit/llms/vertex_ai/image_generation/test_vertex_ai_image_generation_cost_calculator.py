import os
from datetime import datetime
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.vertex_ai.gemini.cost_calculator import cost_per_web_search_request
from litellm.llms.vertex_ai.image_edit.vertex_gemini_transformation import (
    VertexAIGeminiImageEditConfig,
)
from litellm.llms.vertex_ai.image_generation.cost_calculator import (
    cost_calculator as vertex_image_generation_cost_calculator,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    ModelInfo,
    PromptTokensDetailsWrapper,
    Usage,
)


@pytest.mark.parametrize("call_type", ("image_edit", "aimage_edit"))
@pytest.mark.parametrize("has_image, has_usage", ((True, True), (False, True), (True, False)))
def test_vertex_image_edit_cost_uses_input_tokens_and_retains_image_fallback(
    call_type: str,
    has_image: bool,
    has_usage: bool,
) -> None:
    model: Final = "gemini-3.1-flash-image"
    model_info: Final = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")
    response: Final = VertexAIGeminiImageEditConfig().transform_image_edit_response(
        model=model,
        raw_response=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP" if has_image else "IMAGE_RECITATION",
                        **(
                            {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1n"}}]}}
                            if has_image
                            else {}
                        ),
                    }
                ],
                **(
                    {
                        "usageMetadata": {
                            "promptTokenCount": 1131,
                            "candidatesTokenCount": 100 if has_image else 0,
                            "totalTokenCount": 1231 if has_image else 1131,
                            "promptTokensDetails": [
                                {"modality": "TEXT", "tokenCount": 7},
                                {"modality": "IMAGE", "tokenCount": 1124},
                            ],
                        }
                    }
                    if has_usage
                    else {}
                ),
            },
        ),
        logging_obj=Logging(
            model=model,
            messages=[],
            stream=False,
            call_type=call_type,
            start_time=datetime(2025, 1, 1),
            litellm_call_id="edit",
            function_id="edit",
        ),
    )
    cost: Final = litellm.completion_cost(
        model=model,
        custom_llm_provider="vertex_ai",
        completion_response=response,
        call_type=call_type,
    )
    expected: Final = (
        7 * (model_info["input_cost_per_token"] or 0)
        + 1124 * (model_info.get("input_cost_per_image_token") or model_info["input_cost_per_token"] or 0)
        + (100 if has_image else 0) * (model_info.get("output_cost_per_image_token") or 0)
        if has_usage
        else model_info["output_cost_per_image"]
    )
    assert expected is not None and expected > 0
    assert cost == pytest.approx(expected)
    assert vertex_image_generation_cost_calculator(model=model, image_response=response) == pytest.approx(expected)


def _image_response_with_web_search(web_search_requests):
    usage = ImageUsage(
        input_tokens=20,
        input_tokens_details=ImageUsageInputTokensDetails(
            text_tokens=20,
            image_tokens=0,
        ),
        output_tokens=1120,
        total_tokens=1140,
    )
    if web_search_requests is not None:
        usage.web_search_requests = web_search_requests
    return ImageResponse(data=[ImageObject(b64_json="img1")], usage=usage)


def test_vertex_image_generation_cost_adds_web_search_grounding(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")

    grounded = vertex_image_generation_cost_calculator(
        model=model,
        image_response=_image_response_with_web_search(3),
    )
    ungrounded = vertex_image_generation_cost_calculator(
        model=model,
        image_response=_image_response_with_web_search(None),
    )

    expected_web_search_cost = cost_per_web_search_request(
        usage=Usage(
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=3)
        ),
        model_info=model_info,
    )
    assert expected_web_search_cost > 0
    assert round(grounded - ungrounded, 10) == round(expected_web_search_cost, 10)


def test_vertex_image_generation_cost_no_web_search_when_absent(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini-3-pro-image-preview"

    cost_zero = vertex_image_generation_cost_calculator(
        model=model,
        image_response=_image_response_with_web_search(0),
    )
    cost_none = vertex_image_generation_cost_calculator(
        model=model,
        image_response=_image_response_with_web_search(None),
    )

    assert cost_zero == cost_none


@pytest.mark.parametrize("priority_image_rate", [None, 8e-5])
def test_vertex_image_generation_cost_prices_token_details_by_service_tier(priority_image_rate: float | None) -> None:
    model: Final = "gemini-scripted-image-tokens"
    model_info: Final[ModelInfo] = {
        "key": model,
        "max_tokens": 1000,
        "max_input_tokens": 1000,
        "max_output_tokens": 1000,
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 1e-5,
        "output_cost_per_image_token": 4e-5,
        "input_cost_per_token_priority": 4e-6,
        "output_cost_per_token_priority": 2e-5,
        "output_cost_per_image_token_priority": priority_image_rate,
        "litellm_provider": "vertex_ai",
        "mode": "image_generation",
        "supported_openai_params": None,
    }
    usage: Final = ImageUsage.model_validate(
        {
            "input_tokens": 12,
            "input_tokens_details": {"text_tokens": 12, "image_tokens": 0},
            "output_tokens": 1340,
            "total_tokens": 1352,
            "completion_tokens_details": {
                "text_tokens": 10,
                "image_tokens": 1290,
                "reasoning_tokens": 40,
            },
            "output_tokens_details": {
                "text_tokens": 10,
                "image_tokens": 1290,
                "reasoning_tokens": 40,
            },
        }
    )
    image_response: Final = ImageResponse(
        data=[ImageObject(b64_json="synthetic-image")],
        usage=usage,
    )
    standard_cost: Final = vertex_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
        model_info=model_info,
        vertex_location="global",
    )
    priority_cost: Final = vertex_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
        model_info=model_info,
        vertex_location="global",
        service_tier="priority",
    )

    expected_standard_cost: Final = 12 * 2e-6 + (10 + 40) * 1e-5 + 1290 * 4e-5
    expected_priority_cost: Final = (
        12 * 4e-6 + (10 + 40) * 2e-5 + 1290 * (4e-5 if priority_image_rate is None else priority_image_rate)
    )
    assert standard_cost == pytest.approx(expected_standard_cost)
    assert priority_cost == pytest.approx(expected_priority_cost)


def test_vertex_image_generation_without_usage_keeps_flat_image_price() -> None:
    cost: Final = vertex_image_generation_cost_calculator(
        model="gemini-scripted-image-tokens",
        image_response=ImageResponse(data=[ImageObject(b64_json="image")]),
        model_info={"output_cost_per_image": 0.07},
    )
    assert cost == pytest.approx(0.07)
