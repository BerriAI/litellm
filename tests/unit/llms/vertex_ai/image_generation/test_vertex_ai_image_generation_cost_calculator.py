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
