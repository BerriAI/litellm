import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(model: str, image_response: ImageResponse) -> float:
    model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.TOGETHER_AI.value,
    )
    output_cost_per_image = model_info.get("output_cost_per_image") or 0.0
    num_images = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images
