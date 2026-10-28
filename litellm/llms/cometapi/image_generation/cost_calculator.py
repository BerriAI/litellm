from typing import Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import image_cost, resolve_image_model_info
from litellm.types.utils import ImageResponse, ModelInfo


def cost_calculator(
    model: str,
    image_response: object,
    model_info: ModelInfo | None = None,
) -> float:
    """
    CometAPI image generation cost calculator
    """
    _model_info: Final = resolve_image_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.COMETAPI.value,
        model_info=model_info,
    )
    num_images: int = 0
    if isinstance(image_response, ImageResponse):
        if image_response.data:
            num_images = len(image_response.data)
        return image_cost(_model_info, num_images) or 0.0
    else:
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")
