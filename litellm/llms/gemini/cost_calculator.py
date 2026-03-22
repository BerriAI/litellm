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
    Calculates the cost of web search (grounding with Google Search).

    Billing mode is determined by ``web_search_billing_unit`` in model_info:
    - ``"per_query"``: charged per individual search query (Gemini 3.x).
    - ``"per_prompt"`` (default): charged per grounded prompt (Gemini 2.x),
      regardless of how many queries were executed internally.

    Reads the per-request cost from ``search_context_cost_per_query`` in
    ``model_info`` when available, falling back to $0.035 for models not
    yet updated in the pricing JSON.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    _DEFAULT_COST = 35e-3
    search_costs = model_info.get("search_context_cost_per_query") or {}
    _cost = search_costs.get("search_context_size_medium", _DEFAULT_COST)

    number_of_web_search_requests = 0
    if (
        usage is not None
        and usage.prompt_tokens_details is not None
        and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
        and hasattr(usage.prompt_tokens_details, "web_search_requests")
        and usage.prompt_tokens_details.web_search_requests is not None
    ):
        number_of_web_search_requests = usage.prompt_tokens_details.web_search_requests

    # per_prompt billing: clamp to 1 (flat fee per grounded API call)
    billing_mode = model_info.get("web_search_billing_unit", "per_prompt")
    if number_of_web_search_requests > 0 and billing_mode == "per_prompt":
        number_of_web_search_requests = 1

    return _cost * number_of_web_search_requests
