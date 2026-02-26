"""
Google AI Image Generation Cost Calculator
"""

from typing import Any, Optional

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ImageResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


def _calculate_token_based_cost(model: str, image_response: ImageResponse) -> Optional[float]:
    """
    Calculate token-based image generation cost when usage metadata is available.

    Falls back to None when usage metadata is missing/incomplete.
    """
    usage = image_response.usage
    if usage is None:
        return None

    prompt_tokens = usage.input_tokens
    completion_tokens = usage.output_tokens
    total_tokens = usage.total_tokens

    if (
        prompt_tokens is None
        or completion_tokens is None
        or total_tokens is None
    ):
        return None
    # ImageResponse may carry a default zeroed usage object even when provider
    # usage metadata is absent. Treat this as missing usage and fall back.
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None

    input_tokens_details = getattr(usage, "input_tokens_details", None)
    prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
    if input_tokens_details is not None:
        prompt_tokens_details = PromptTokensDetailsWrapper(
            text_tokens=getattr(input_tokens_details, "text_tokens", None),
            image_tokens=getattr(input_tokens_details, "image_tokens", None),
            cached_tokens=0,
        )

    normalized_usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            text_tokens=0,
            image_tokens=completion_tokens,
            reasoning_tokens=0,
            audio_tokens=0,
        ),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=normalized_usage,
        custom_llm_provider="gemini",
    )
    return prompt_cost + completion_cost


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    Vertex AI Image Generation Cost Calculator
    """
    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="gemini",
    )

    if isinstance(image_response, ImageResponse):
        token_based_cost = _calculate_token_based_cost(
            model=model, image_response=image_response
        )
        if token_based_cost is not None:
            return token_based_cost

    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = 0
    if isinstance(image_response, ImageResponse):
        if image_response.data:
            num_images = len(image_response.data)
        return output_cost_per_image * num_images
    else:
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")
