from typing import Optional

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: ImageResponse,
    size: Optional[str] = None,
    optional_params: Optional[dict] = None,
) -> float:
    """
    Bedrock image generation cost calculator

    Handles both Stability 1 and Stability 3 models
    """
    if litellm.AmazonStability3Config()._is_stability_3_model(model=model):
        pass
    else:
        # Stability 1 models
        optional_params = optional_params or {}

        # see model_prices_and_context_window.json for details on how steps is used
        # Reference pricing by steps for stability 1: https://aws.amazon.com/bedrock/pricing/
        _steps = optional_params.get("steps", 50)
        steps = "max-steps" if _steps > 50 else "50-steps"

        # size is stored in model_prices_and_context_window.json as 1024-x-1024
        # current size has 1024x1024
        size = size or "1024-x-1024"
        model = f"{size}/{steps}/{model}"

    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="bedrock",
    )

    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = 0
    if image_response.data:
        num_images = len(image_response.data)
    return output_cost_per_image * num_images
