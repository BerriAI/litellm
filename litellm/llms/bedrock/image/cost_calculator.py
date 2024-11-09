from typing import Optional

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    size: Optional[str],
    image_response: ImageResponse,
) -> float:
    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="bedrock",
    )

    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = len(image_response.data)

    return output_cost_per_image * num_images
