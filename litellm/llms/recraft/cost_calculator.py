import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: ImageResponse,
) -> float:
    """
    Recraft image generation cost calculator
    """
    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.RECRAFT.value,
    )
    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = 0
    if image_response.data:
        num_images = len(image_response.data)
    return output_cost_per_image * num_images
