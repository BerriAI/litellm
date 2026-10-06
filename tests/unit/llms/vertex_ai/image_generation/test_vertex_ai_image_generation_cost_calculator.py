import os
from typing import Final

import litellm
import pytest
from litellm.llms.vertex_ai.gemini.cost_calculator import cost_per_web_search_request
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


def test_vertex_image_generation_cost_prices_token_details_by_service_tier() -> None:
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
    expected_priority_cost: Final = 12 * 4e-6 + (10 + 40) * 2e-5 + 1290 * 4e-5
    assert standard_cost == pytest.approx(expected_standard_cost)
    assert priority_cost == pytest.approx(expected_priority_cost)
