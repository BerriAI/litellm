"""
Vertex AI Image Generation Cost Calculator
"""

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
)
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: ImageResponse,
) -> float:
    """
    Vertex AI Image Generation Cost Calculator
    """
    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="vertex_ai",
    )

    token_based_cost = calculate_image_response_cost_from_usage(
        model=model,
        image_response=image_response,
        custom_llm_provider="vertex_ai",
    )
    if token_based_cost is not None:
        return token_based_cost

    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = 0
    if image_response.data:
        num_images = len(image_response.data)
    return output_cost_per_image * num_images
