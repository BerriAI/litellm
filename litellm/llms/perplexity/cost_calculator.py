"""
Helper util for handling perplexity-specific cost calculation
- e.g.: citation tokens, search queries
"""

from typing import Tuple, Union

from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing perplexity-specific usage information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider="perplexity")

    def _safe_float_cast(value: Union[str, int, float, None, object], default: float = 0.0) -> float:
        """Safely cast a value to float with proper type handling for mypy."""
        if value is None:
            return default
        try:
            return float(value)  # type: ignore
        except (ValueError, TypeError):
            return default

    ## CALCULATE INPUT COST
    input_cost_per_token = _safe_float_cast(model_info.get("input_cost_per_token"))
    prompt_cost: float = (usage.prompt_tokens or 0) * input_cost_per_token

    ## ADD CITATION TOKENS COST (if present)
    citation_tokens = getattr(usage, "citation_tokens", 0) or 0
    citation_cost_value = model_info.get("citation_cost_per_token")
    if citation_tokens > 0 and citation_cost_value is not None:
        citation_cost_per_token = _safe_float_cast(citation_cost_value)
        prompt_cost += citation_tokens * citation_cost_per_token

    ## CALCULATE OUTPUT COST
    output_cost_per_token = _safe_float_cast(model_info.get("output_cost_per_token"))
    completion_cost: float = (usage.completion_tokens or 0) * output_cost_per_token

    ## ADD REASONING TOKENS COST (if present)
    reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
    # Also check completion_tokens_details if reasoning_tokens is not directly available
    if reasoning_tokens == 0 and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
    
    reasoning_cost_value = model_info.get("output_cost_per_reasoning_token")
    if reasoning_tokens > 0 and reasoning_cost_value is not None:
        reasoning_cost_per_token = _safe_float_cast(reasoning_cost_value)
        completion_cost += reasoning_tokens * reasoning_cost_per_token

    ## ADD SEARCH QUERIES COST (if present)
    num_search_queries = 0
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        num_search_queries = getattr(usage.prompt_tokens_details, "web_search_requests", 0) or 0
    
    # Check both possible keys for search cost (legacy and current)
    search_cost_value = model_info.get("search_queries_cost_per_query") or model_info.get("search_context_cost_per_query")
    if num_search_queries > 0 and search_cost_value is not None:
        # Handle both dict and float formats
        if isinstance(search_cost_value, dict):
            # Use the "low" size as default - tests expect 0.005 / 1000
            search_cost_per_query = _safe_float_cast(search_cost_value.get("search_context_size_low", 0)) / 1000
        else:
            search_cost_per_query = _safe_float_cast(search_cost_value)
        search_cost = num_search_queries * search_cost_per_query
        # Add search cost to completion cost (similar to how other providers handle it)
        completion_cost += search_cost

    return prompt_cost, completion_cost 