from typing import Any, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
)
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
    size: str | None = None,
    n: int | None = None,
) -> float:
    """
    Azure AI image generation cost calculator
    """
    _model_info: Final = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
    )

    if isinstance(image_response, ImageResponse):
        token_based_cost: Final = calculate_image_response_cost_from_usage(
            model=model,
            image_response=image_response,
            custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
        )
        if token_based_cost is not None:
            return token_based_cost

        num_images: Final = n if n is not None else len(image_response.data or ())
        output_cost_per_image: Final[float] = _model_info.get("output_cost_per_image") or 0.0
        if output_cost_per_image:
            return output_cost_per_image * num_images

        model_cost: Final = litellm.model_cost[_model_info["key"]]
        input_cost_per_pixel: Final[float] = model_cost.get("input_cost_per_pixel") or 0.0
        if input_cost_per_pixel:
            from litellm.cost_calculator import default_image_cost_calculator

            cost_model: Final = (
                model if model.startswith(f"{litellm.LlmProviders.AZURE_AI.value}/") else f"azure_ai/{model}"
            )
            return default_image_cost_calculator(
                model=cost_model,
                custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
                size=size or image_response.size,
                n=num_images,
            )
        return 0.0

    raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")
