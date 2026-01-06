"""
Cost calculation for search providers.
"""
from typing import Optional, Tuple

from litellm.utils import get_model_info


def search_provider_cost_per_query(
    model: str,
    custom_llm_provider: Optional[str] = None,
    number_of_queries: int = 1,
    optional_params: Optional[dict] = None,
) -> Tuple[float, float]:
    """
    Calculate cost for search-only providers.
    
    Returns (input_cost, output_cost) where input_cost = queries * cost_per_query
    Supports tiered pricing based on max_results parameter.
    
    Args:
        model: Model name (e.g., "exa_ai/search", "tavily/search")
        custom_llm_provider: Provider name (e.g., "exa_ai", "tavily")
        number_of_queries: Number of search queries performed (default: 1)
        optional_params: Optional parameters including max_results for tiered pricing
        
    Returns:
        Tuple of (input_cost, output_cost) where output_cost is always 0.0
    """
    model_info = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    
    # Check for tiered pricing (e.g., Exa AI based on max_results)
    tiered_pricing = model_info.get("tiered_pricing")
    if tiered_pricing and isinstance(tiered_pricing, list):
        max_results = (optional_params or {}).get("max_results", 10)  # default 10 results
        cost_per_query = 0.0
        
        for tier in tiered_pricing:
            range_min, range_max = tier["max_results_range"]
            if range_min <= max_results <= range_max:
                cost_per_query = tier["input_cost_per_query"]
                break
        else:
            # Fallback to highest tier if out of range
            cost_per_query = tiered_pricing[-1]["input_cost_per_query"]
    else:
        # Simple flat rate
        cost_per_query = float(model_info.get("input_cost_per_query") or 0.0)
    
    total_cost = number_of_queries * cost_per_query
    return (total_cost, 0.0)  # (input_cost, output_cost)

