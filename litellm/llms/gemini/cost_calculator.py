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

    Uses custom pricing from model_info if available (via `web_search_cost_per_request` key),
    otherwise falls back to the default of $0.014 (Google's Gemini 3 family rate).
    The previous hardcoded default of $0.035 is obsolete.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # cost per web search request
    # Check for custom override in model_info first
    custom_cost = model_info.get("web_search_cost_per_request")
    if custom_cost is not None and isinstance(custom_cost, (int, float)) and custom_cost > 0:
        cost_per_web_search_request = custom_cost
    else:
        # Default Gemini 3 family rate ($0.014 per request)
        cost_per_web_search_request = 14e-3

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
    else:
        number_of_web_search_requests = 0

    # Calculate total cost
    total_cost = cost_per_web_search_request * number_of_web_search_requests

    return total_cost
