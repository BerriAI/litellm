"""
Cost calculator for OpenAI image generation models (gpt-image family)

These models use token-based pricing instead of pixel-based pricing like DALL-E.
"""

from typing import Optional

from litellm import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
    generic_cost_per_token,
)
from litellm.types.utils import ImageResponse, Usage


def cost_calculator(
    model: str,
    image_response: ImageResponse,
    custom_llm_provider: Optional[str] = None,
) -> float:
    """Calculate cost for OpenAI gpt-image models (token-based pricing)."""
    usage = getattr(image_response, "usage", None)
    if usage is None:
        verbose_logger.debug(
            f"No usage data available for {model}, cannot calculate token-based cost"
        )
        return 0.0

    provider = custom_llm_provider or "openai"

    # A chat Usage with an explicit output breakdown: cost via generic_cost_per_token.
    if isinstance(usage, Usage) and usage.completion_tokens_details is not None:
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model, usage=usage, custom_llm_provider=provider
        )
        return prompt_cost + completion_cost

    # ImageUsage / ResponseAPIUsage: reuse the shared helper (same path as
    # azure_ai/gemini/vertex_ai). It prices generated output tokens at
    # output_cost_per_image_token, classifying them as image tokens when the provider
    # does not itemize output and splitting text/image when it does.
    if getattr(usage, "input_tokens", None) is not None:
        token_based_cost = calculate_image_response_cost_from_usage(
            model=model, image_response=image_response, custom_llm_provider=provider
        )
        if token_based_cost is not None:
            return token_based_cost

    # Fallback: a Usage with no output breakdown that the image helper can't read —
    # cost via generic_cost_per_token (text rate) instead of returning 0.0.
    if isinstance(usage, Usage):
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model, usage=usage, custom_llm_provider=provider
        )
        return prompt_cost + completion_cost

    return 0.0
