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

    Uses ``search_context_cost_per_query`` from ``model_info`` when available
    (keyed by ``search_context_size_medium`` as the default tier).  Falls back
    to the legacy $0.035 hardcode for models that haven't been updated yet.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # Resolve per-request cost from model_info, fallback to legacy default
    _DEFAULT_COST = 35e-3
    search_costs = model_info.get("search_context_cost_per_query") or {}
    _cost = search_costs.get("search_context_size_medium", _DEFAULT_COST)

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
    total_cost = _cost * number_of_web_search_requests

    return total_cost
