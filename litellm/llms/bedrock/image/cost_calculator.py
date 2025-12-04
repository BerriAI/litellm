from typing import Optional

from litellm.llms.bedrock.image.image_handler import BedrockImageGeneration
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
    config_class = BedrockImageGeneration.get_config_class(model=model)
    return config_class.cost_calculator(
        model=model,
        image_response=image_response,
        size=size,
        optional_params=optional_params,
    )
