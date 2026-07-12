"""
Google AI Image Generation Cost Calculator
"""

from typing import Any

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
    calculate_image_response_web_search_cost,
)
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    Google AI Image Generation Cost Calculator
    """
    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="gemini",
    )

    if not isinstance(image_response, ImageResponse):
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")

    web_search_cost = calculate_image_response_web_search_cost(
        image_response=image_response,
        custom_llm_provider="gemini",
        model_info=_model_info,
    )

    token_based_cost = calculate_image_response_cost_from_usage(
        model=model,
        image_response=image_response,
        custom_llm_provider="gemini",
    )
    if token_based_cost is not None:
        return token_based_cost + web_search_cost

    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images + web_search_cost
