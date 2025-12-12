"""
Cost calculator for OpenAI image generation models (gpt-image-1, gpt-image-1-mini)

These models use token-based pricing instead of pixel-based pricing like DALL-E.
"""

from typing import Optional

import litellm
from litellm import verbose_logger
from litellm.types.utils import ImageResponse


def _is_gpt_image_model(model: str) -> bool:
    """Check if model is a gpt-image model that uses token-based pricing"""
    model_lower = model.lower()
    return "gpt-image-1" in model_lower or "gpt-image-1-mini" in model_lower


def cost_calculator(
    model: str,
    image_response: ImageResponse,
    custom_llm_provider: Optional[str] = None,
) -> float:
    """
    Calculate cost for OpenAI gpt-image-1 and gpt-image-1-mini models.

    These models use token-based pricing:
    - Text input tokens (from prompt)
    - Image input tokens (from input images for editing)
    - Image output tokens (for generated images)

    Args:
        model: The model name (e.g., "gpt-image-1", "gpt-image-1-mini")
        image_response: The ImageResponse containing usage data
        custom_llm_provider: Optional provider name

    Returns:
        float: Total cost in USD

    Pricing (as of 2025):
        gpt-image-1:
            - Text Input: $5.00/1M tokens
            - Image Input: $10.00/1M tokens
            - Image Output: $40.00/1M tokens
            - Cached Text: $1.25/1M tokens
            - Cached Image: $2.50/1M tokens

        gpt-image-1-mini:
            - Text Input: $2.00/1M tokens
            - Image Input: $2.50/1M tokens
            - Image Output: $8.00/1M tokens
            - Cached Text: $0.20/1M tokens
            - Cached Image: $0.25/1M tokens
    """
    # Get usage from response
    usage = getattr(image_response, "usage", None)

    if usage is None:
        verbose_logger.debug(
            f"No usage data available for {model}, cannot calculate token-based cost"
        )
        return 0.0

    # Get token counts
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    # Get token details for input breakdown
    input_tokens_details = getattr(usage, "input_tokens_details", None)
    text_tokens = 0
    image_tokens = 0

    if input_tokens_details:
        text_tokens = getattr(input_tokens_details, "text_tokens", 0) or 0
        image_tokens = getattr(input_tokens_details, "image_tokens", 0) or 0
    else:
        # If no details, assume all input tokens are text tokens
        text_tokens = input_tokens

    # Get pricing from model_cost dict (includes all fields unlike get_model_info)
    model_cost = litellm.model_cost

    # Try different model name variations
    base_model = model.split("/")[-1] if "/" in model else model
    provider_prefix = custom_llm_provider or "openai"
    model_names_to_try = [
        model,  # as-is
        f"{provider_prefix}/{base_model}",  # with provider prefix
        base_model,  # just the model name
    ]

    cost_info = None
    for model_name in model_names_to_try:
        if model_name in model_cost:
            cost_info = model_cost[model_name]
            break

    if cost_info is None:
        verbose_logger.debug(
            f"Could not find pricing for model {model}. Tried: {model_names_to_try}"
        )
        return 0.0

    # Get cost rates from model_cost dict
    input_cost_per_token = cost_info.get("input_cost_per_token", 0) or 0
    input_cost_per_image_token = cost_info.get("input_cost_per_image_token", 0) or 0
    output_cost_per_image_token = cost_info.get("output_cost_per_image_token", 0) or 0

    # If no image token cost is defined, fall back to regular token cost
    if input_cost_per_image_token == 0:
        input_cost_per_image_token = input_cost_per_token
    if output_cost_per_image_token == 0:
        output_cost_per_image_token = cost_info.get("output_cost_per_token", 0) or 0

    # Calculate costs
    text_input_cost = text_tokens * input_cost_per_token
    image_input_cost = image_tokens * input_cost_per_image_token
    output_cost = output_tokens * output_cost_per_image_token

    total_cost = text_input_cost + image_input_cost + output_cost

    verbose_logger.debug(
        f"OpenAI gpt-image cost calculation for {model}: "
        f"text_tokens={text_tokens} (${text_input_cost:.6f}), "
        f"image_tokens={image_tokens} (${image_input_cost:.6f}), "
        f"output_tokens={output_tokens} (${output_cost:.6f}), "
        f"total=${total_cost:.6f}"
    )

    return total_cost
