import os

import pytest

import litellm
from litellm.llms.gemini.cost_calculator import cost_per_web_search_request
from litellm.llms.gemini.image_edit.cost_calculator import (
    cost_calculator as gemini_image_edit_cost_calculator,
)
from litellm.llms.gemini.image_generation.cost_calculator import (
    cost_calculator as gemini_image_generation_cost_calculator,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    PromptTokensDetailsWrapper,
    Usage,
)


def _make_usage(web_search_requests: int) -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            web_search_requests=web_search_requests,
        ),
    )


def test_per_query_billing():
    """web_search_billing_unit=per_query charges per search query."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.014,
        },
    }
    cost = cost_per_web_search_request(usage=_make_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.014 * 3)


def test_per_prompt_billing():
    """web_search_billing_unit=per_prompt (default) clamps to 1."""
    model_info = {
        "key": "gemini/gemini-2.5-flash",
        "search_context_cost_per_query": {
            "search_context_size_medium": 0.035,
        },
    }
    cost = cost_per_web_search_request(usage=_make_usage(3), model_info=model_info)
    assert cost == pytest.approx(0.035 * 1)


def test_default_billing_unit_is_per_prompt():
    """Without web_search_billing_unit, defaults to per_prompt (clamp to 1)."""
    model_info = {"key": "gemini/gemini-2.0-flash"}
    cost = cost_per_web_search_request(usage=_make_usage(2), model_info=model_info)
    assert cost == pytest.approx(0.035 * 1)


def test_zero_requests():
    """Zero web search requests should return zero cost."""
    model_info = {
        "key": "gemini/gemini-3-flash-preview",
        "web_search_billing_unit": "per_query",
    }
    cost = cost_per_web_search_request(usage=_make_usage(0), model_info=model_info)
    assert cost == 0.0


def test_no_usage_details():
    """Missing prompt_tokens_details should return zero cost."""
    model_info = {"key": "gemini/gemini-3-flash-preview"}
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    cost = cost_per_web_search_request(usage=usage, model_info=model_info)
    assert cost == 0.0


def test_gemini_image_edit_cost_prefers_token_usage_metadata():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    input_text_tokens = 20
    input_image_tokens = 1120
    output_image_tokens = 1120
    prompt_tokens = input_text_tokens + input_image_tokens
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")],
        usage=ImageUsage(
            input_tokens=prompt_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=input_image_tokens,
            ),
            output_tokens=output_image_tokens,
            total_tokens=prompt_tokens + output_image_tokens,
        ),
    )

    cost = gemini_image_edit_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = (
        prompt_tokens * model_info["input_cost_per_token"]
        + output_image_tokens * model_info["output_cost_per_image_token"]
    )
    flat_image_cost = (
        len(image_response.data or []) * model_info["output_cost_per_image"]
    )
    assert round(cost, 10) == round(expected_cost, 10)
    assert cost != flat_image_cost


def test_gemini_image_edit_cost_uses_output_token_details():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    input_text_tokens = 20
    output_text_tokens = 213
    output_image_tokens = 1120
    output_tokens = output_text_tokens + output_image_tokens
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1")],
        usage=ImageUsage(
            input_tokens=input_text_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=0,
            ),
            output_tokens=output_tokens,
            total_tokens=input_text_tokens + output_tokens,
            prompt_tokens=input_text_tokens,
            completion_tokens=output_tokens,
            prompt_tokens_details={
                "text_tokens": input_text_tokens,
                "image_tokens": 0,
            },
            completion_tokens_details={
                "text_tokens": output_text_tokens,
                "image_tokens": output_image_tokens,
            },
            output_tokens_details={
                "text_tokens": output_text_tokens,
                "image_tokens": output_image_tokens,
            },
        ),
    )

    cost = gemini_image_edit_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = (
        input_text_tokens * model_info["input_cost_per_token"]
        + output_text_tokens * model_info["output_cost_per_token"]
        + output_image_tokens * model_info["output_cost_per_image_token"]
    )
    all_output_as_image_cost = (
        input_text_tokens * model_info["input_cost_per_token"]
        + (output_text_tokens + output_image_tokens)
        * model_info["output_cost_per_image_token"]
    )
    assert round(cost, 10) == round(expected_cost, 10)
    assert cost != all_output_as_image_cost


def test_gemini_image_generation_cost_uses_output_token_details():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")

    input_text_tokens = 20
    output_text_tokens = 213
    output_image_tokens = 1120
    output_tokens = output_text_tokens + output_image_tokens
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1")],
        usage=ImageUsage(
            input_tokens=input_text_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=0,
            ),
            output_tokens=output_tokens,
            total_tokens=input_text_tokens + output_tokens,
            prompt_tokens=input_text_tokens,
            completion_tokens=output_tokens,
            prompt_tokens_details={
                "text_tokens": input_text_tokens,
                "image_tokens": 0,
            },
            completion_tokens_details={
                "text_tokens": output_text_tokens,
                "image_tokens": output_image_tokens,
            },
            output_tokens_details={
                "text_tokens": output_text_tokens,
                "image_tokens": output_image_tokens,
            },
        ),
    )

    cost = gemini_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = (
        input_text_tokens * model_info["input_cost_per_token"]
        + output_text_tokens * model_info["output_cost_per_token"]
        + output_image_tokens * model_info["output_cost_per_image_token"]
    )
    all_output_as_image_cost = (
        input_text_tokens * model_info["input_cost_per_token"]
        + (output_text_tokens + output_image_tokens)
        * model_info["output_cost_per_image_token"]
    )
    assert round(cost, 10) == round(expected_cost, 10)
    assert cost != all_output_as_image_cost


def test_gemini_image_edit_cost_falls_back_to_flat_image_pricing():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model = "gemini/gemini-3-pro-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")]
    )

    cost = gemini_image_edit_cost_calculator(
        model=model,
        image_response=image_response,
    )

    assert cost == len(image_response.data or []) * model_info["output_cost_per_image"]
