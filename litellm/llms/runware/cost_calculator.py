from typing import Any

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    Runware image generation cost calculator.

    Tries to use Runware's reported cost from the response metadata (via includeCost),
    then falls back to model_info pricing.
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(
            f"image_response must be of type ImageResponse got type={type(image_response)}"
        )

    # Try to use Runware's reported cost from response
    hidden_params = getattr(image_response, "_hidden_params", None)
    if hidden_params and "runware_cost" in hidden_params:
        runware_cost = hidden_params["runware_cost"]
        if runware_cost is not None:
            return float(runware_cost)

    # Fall back to model_info pricing
    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.RUNWARE.value,
    )
    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = 0
    if image_response.data:
        num_images = len(image_response.data)
    return output_cost_per_image * num_images
