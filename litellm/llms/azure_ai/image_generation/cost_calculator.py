from typing import Any

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    Cost calculator for Azure AI image generation models.

    Azure AI supports both flat per-image pricing and token-based image pricing.
    Prefer usage-based calculation when the response provides token usage, then
    fall back to per-image pricing for models that only expose flat image cost.
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(
            f"image_response must be of type ImageResponse got type={type(image_response)}"
        )

    usage_based_cost = _calculate_cost_from_usage(
        model=model,
        image_response=image_response,
    )
    if usage_based_cost is not None:
        return usage_based_cost

    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
    )
    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images


def _calculate_cost_from_usage(
    model: str,
    image_response: ImageResponse,
) -> float | None:
    usage = image_response.usage
    if usage is None:
        return None

    prompt_tokens = usage.input_tokens
    completion_tokens = usage.output_tokens
    total_tokens = usage.total_tokens
    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return None
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None

    model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
    )
    input_cost_per_token = model_info.get("input_cost_per_token") or 0.0
    output_cost_per_image_token = (
        model_info.get("output_cost_per_image_token")
        or model_info.get("output_cost_per_token")
        or 0.0
    )
    return (float(prompt_tokens) * input_cost_per_token) + (
        float(completion_tokens) * output_cost_per_image_token
    )
