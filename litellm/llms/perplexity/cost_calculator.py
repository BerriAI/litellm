"""
Helper util for handling perplexity-specific cost calculation
- e.g.: citation tokens, search queries
"""

from typing import Tuple

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

    ## CALCULATE INPUT COST
    prompt_cost: float = usage.prompt_tokens * model_info["input_cost_per_token"]

    ## ADD CITATION TOKENS COST (if present)
    citation_tokens = getattr(usage, "citation_tokens", 0) or 0
    if citation_tokens > 0 and model_info.get("citation_cost_per_token"):
        prompt_cost += citation_tokens * model_info["citation_cost_per_token"]

    ## CALCULATE OUTPUT COST
    completion_cost: float = usage.completion_tokens * model_info["output_cost_per_token"]

    ## ADD REASONING TOKENS COST (if present)
    reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
    # Also check completion_tokens_details if reasoning_tokens is not directly available
    if reasoning_tokens == 0 and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
    
    if reasoning_tokens > 0 and model_info.get("output_cost_per_reasoning_token"):
        completion_cost += reasoning_tokens * model_info["output_cost_per_reasoning_token"]

    ## ADD SEARCH QUERIES COST (if present)
    num_search_queries = 0
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        num_search_queries = getattr(usage.prompt_tokens_details, "web_search_requests", 0) or 0
    
    if num_search_queries > 0 and model_info.get("search_queries_cost_per_1000"):
        search_cost = (num_search_queries / 1000) * model_info["search_queries_cost_per_1000"]
        # Add search cost to completion cost (similar to how other providers handle it)
        completion_cost += search_cost

    return prompt_cost, completion_cost 