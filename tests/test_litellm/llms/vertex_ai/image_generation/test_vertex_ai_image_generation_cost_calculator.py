import os

import litellm
from litellm.llms.vertex_ai.gemini.cost_calculator import cost_per_web_search_request
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


def test_vertex_image_generation_cost_adds_web_search_grounding():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
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


def test_vertex_image_generation_cost_no_web_search_when_absent():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
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
