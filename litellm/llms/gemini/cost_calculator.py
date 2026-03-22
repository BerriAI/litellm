"""
This file is used to calculate the cost of the Gemini API.

Handles the context caching for Gemini API.
"""

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def cost_per_token(
    model: str, usage: "Usage", service_tier: Optional[str] = None
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Follows the same logic as Anthropic's cost per token calculation.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

    return generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="gemini",
        service_tier=service_tier,
    )


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculates the cost per web search request for a given model, prompt tokens, and completion tokens.

    Reads the per-request cost from model_info["web_search_cost_per_request"] so that
    different model families (e.g. Gemini 2.x at $35/1K vs Gemini 3.x at $14/1K) are
    billed correctly.  Falls back to $35/1K when the field is absent for backward
    compatibility.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # Read per-request cost from model_info; fall back to legacy $35/1K default
    _cost_per_request = (
        model_info.get("web_search_cost_per_request") if model_info else None
    )
    if _cost_per_request is None:
        _cost_per_request = 35e-3

    number_of_web_search_requests = 0
    # Get number of web search requests
    if (
        usage is not None
        and usage.prompt_tokens_details is not None
        and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
        and hasattr(usage.prompt_tokens_details, "web_search_requests")
        and usage.prompt_tokens_details.web_search_requests is not None
    ):
        number_of_web_search_requests = usage.prompt_tokens_details.web_search_requests

    # Calculate total cost
    total_cost = _cost_per_request * number_of_web_search_requests

    return total_cost
