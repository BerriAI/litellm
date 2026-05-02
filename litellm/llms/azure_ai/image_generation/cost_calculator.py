from typing import Any

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
)
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    Cost calculator for Azure AI image generation models.

    Azure AI supports both flat per-image pricing and token-based image pricing.
    Prefer usage-based calculation when the response provides token usage, then
    fall back to per-image pricing for models that only expose flat image cost.
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(
            f"image_response must be of type ImageResponse got type={type(image_response)}"
        )

    usage_based_cost = calculate_image_response_cost_from_usage(
        model=model,
        image_response=image_response,
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
    )
    if usage_based_cost is not None:
        return usage_based_cost

    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
    )
    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images
