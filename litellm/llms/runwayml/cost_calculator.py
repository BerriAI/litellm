from typing import Any, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import resolve_image_model_info
from litellm.types.utils import ImageResponse, ModelInfo


def cost_calculator(
    model: str,
    image_response: Any,
    model_info: ModelInfo | None = None,
) -> float:
    """
    RunwayML image generation cost calculator.

    RunwayML charges per image generated, not per pixel.
    Pricing is stored in model_prices_and_context_window.json with output_cost_per_image.
    """
    _model_info: Final = resolve_image_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.RUNWAYML.value,
        model_info=model_info,
    )
    output_cost_per_image: Final[float] = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = 0
    if isinstance(image_response, ImageResponse):
        if image_response.data:
            num_images = len(image_response.data)
        return output_cost_per_image * num_images
    else:
        raise ValueError(f"image_response must be of type ImageResponse, got type={type(image_response)}")
