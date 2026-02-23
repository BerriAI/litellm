"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Optional, Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage
import litellm


def cost_per_token(model: str, usage: "Usage") -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="anthropic"
    )

    # Apply provider_specific_entry multipliers for geo/speed routing
    try:
        model_info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
        provider_specific_entry: dict = model_info.get("provider_specific_entry") or {}

        multiplier = 1.0
        if (
            hasattr(usage, "inference_geo")
            and usage.inference_geo
            and usage.inference_geo.lower() not in ["global", "not_available"]
        ):
            multiplier *= provider_specific_entry.get(
                usage.inference_geo.lower(), 1.0
            )
        if hasattr(usage, "speed") and usage.speed == "fast":
            multiplier *= provider_specific_entry.get("fast", 1.0)

        if multiplier != 1.0:
            prompt_cost *= multiplier
            completion_cost *= multiplier
    except Exception:
        pass

    return prompt_cost, completion_cost


def get_cost_for_anthropic_web_search(
    model_info: Optional["ModelInfo"] = None,
    usage: Optional["Usage"] = None,
) -> float:
    """
    Get the cost of using a web search tool for Anthropic.
    """
    from litellm.types.utils import SearchContextCostPerQuery

    ## Check if web search requests are in the usage object
    if model_info is None:
        return 0.0

    if (
        usage is None
        or usage.server_tool_use is None
        or usage.server_tool_use.web_search_requests is None
    ):
        return 0.0

    ## Get the cost per web search request
    search_context_pricing: SearchContextCostPerQuery = (
        model_info.get("search_context_cost_per_query") or SearchContextCostPerQuery()
    )
    cost_per_web_search_request = search_context_pricing.get(
        "search_context_size_medium", 0.0
    )
    if cost_per_web_search_request is None or cost_per_web_search_request == 0.0:
        return 0.0

    ## Calculate the total cost
    total_cost = cost_per_web_search_request * usage.server_tool_use.web_search_requests
    return total_cost
