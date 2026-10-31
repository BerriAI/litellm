from typing import Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import resolve_image_model_info
from litellm.types.utils import ImageResponse, ModelInfo


def cost_calculator(model: str, image_response: ImageResponse, model_info: ModelInfo | None = None) -> float:
    prices: Final = resolve_image_model_info(
        model=model, custom_llm_provider=litellm.LlmProviders.DASHSCOPE.value, model_info=model_info
    )
    output_cost_per_image: Final = prices.get("output_cost_per_image")
    if output_cost_per_image is None:
        raise ValueError(f"No output_cost_per_image found for dashscope/{model}")
    input_images: Final = image_response.hidden_params.get("billable_input_image_count")
    input_image_cost: Final = (prices.get("input_cost_per_image") or 0.0) * (
        input_images if isinstance(input_images, int) else 0
    )
    return output_cost_per_image * len(image_response.data or ()) + input_image_cost
