"""
Cost calculator for OpenAI image generation models (gpt-image-1, gpt-image-1-mini)

These models use token-based pricing instead of pixel-based pricing like DALL-E.
"""

from typing import Optional

from litellm import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import ImageResponse, Usage


def cost_calculator(
    model: str,
    image_response: ImageResponse,
    custom_llm_provider: Optional[str] = None,
) -> float:
    """
    Calculate cost for OpenAI gpt-image-1 and gpt-image-1-mini models.

    Uses the same usage format as Responses API, so we reuse the helper
    to transform to chat completion format and use generic_cost_per_token.

    Args:
        model: The model name (e.g., "gpt-image-1", "gpt-image-1-mini")
        image_response: The ImageResponse containing usage data
        custom_llm_provider: Optional provider name

    Returns:
        float: Total cost in USD
    """
    usage = getattr(image_response, "usage", None)

    if usage is None:
        verbose_logger.debug(
            f"No usage data available for {model}, cannot calculate token-based cost"
        )
        return 0.0

    # If usage is already a Usage object with completion_tokens_details set,
    # use it directly (it was already transformed in convert_to_image_response)
    if isinstance(usage, Usage) and usage.completion_tokens_details is not None:
        chat_usage = usage
    else:
        # Transform ImageUsage to Usage using the existing helper
        # ImageUsage has the same format as ResponseAPIUsage
        from litellm.responses.utils import ResponseAPILoggingUtils

        chat_usage = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

    # Use generic_cost_per_token for cost calculation
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=chat_usage,
        custom_llm_provider=custom_llm_provider or "openai",
    )

    total_cost = prompt_cost + completion_cost

    verbose_logger.debug(
        f"OpenAI gpt-image cost calculation for {model}: "
        f"prompt_cost=${prompt_cost:.6f}, completion_cost=${completion_cost:.6f}, "
        f"total=${total_cost:.6f}"
    )

    return total_cost
