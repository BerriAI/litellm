"""
Vertex AI Image Edit Cost Calculator
"""

from typing import Any

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    Vertex AI image edit cost calculator.

    Mirrors image generation pricing: charge per returned image based on
    model metadata (`output_cost_per_image`).
    """
    model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="vertex_ai",
    )

    output_cost_per_image: float = model_info.get("output_cost_per_image") or 0.0

    if not isinstance(image_response, ImageResponse):
        raise ValueError(
            f"image_response must be of type ImageResponse got type={type(image_response)}"
        )

    num_images = len(image_response.data or [])
    return output_cost_per_image * num_images
