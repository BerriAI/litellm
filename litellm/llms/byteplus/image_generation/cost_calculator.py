from typing import Any

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    BytePlus (seedream) image generation cost calculator.

    Cost is per generated image (output_cost_per_image from the model cost map)
    multiplied by the number of images returned.
    """
    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.BYTEPLUS.value,
    )
    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0

    if not isinstance(image_response, ImageResponse):
        raise ValueError(
            f"image_response must be of type ImageResponse got type={type(image_response)}"
        )

    num_images: int = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images
