"""
Cost calculator for Dashscope Chat models. 

Handles tiered pricing and prompt caching scenarios.
"""

from typing import List, Optional, Tuple, Union

from litellm.types.utils import Usage
from litellm.utils import get_model_info


def _calculate_tiered_cost(
    tokens: int, 
    tiered_pricing: List[dict], 
    cost_key: str,
    fallback_cost_key: Optional[str] = None
) -> float:
    """
    Calculate cost using tiered pricing structure.
    
    Args:
        tokens: Number of tokens to calculate cost for
        tiered_pricing: List of tier dictionaries with range and cost_per_token
        cost_key: Key to look for in tier dict (e.g., "input_cost_per_token")
        fallback_cost_key: Fallback key if primary key not found
    
    Returns:
        Total cost for the tokens
    """
    if not tiered_pricing or tokens <= 0:
        return 0.0
    
    total_cost = 0.0
    tokens_processed = 0
    
    for tier in tiered_pricing:
        if tokens_processed >= tokens:
            break
            
        # Get the range for this tier
        tier_range = tier.get("range", [])
        if len(tier_range) != 2:
            continue
            
        range_start, range_end = tier_range
        
        # Skip if all our tokens are before this tier starts
        if tokens <= range_start:
            break
            
        # Calculate how many tokens fall in this tier
        tier_start_for_calculation = max(range_start, tokens_processed)
        tier_end_for_calculation = min(range_end, tokens)
        
        if tier_end_for_calculation > tier_start_for_calculation:
            tokens_in_tier = tier_end_for_calculation - tier_start_for_calculation
            
            # Get cost per token for this tier
            cost_per_token = tier.get(cost_key)
            if cost_per_token is None and fallback_cost_key:
                cost_per_token = tier.get(fallback_cost_key)
                
            if cost_per_token is not None:
                total_cost += tokens_in_tier * cost_per_token
                
            tokens_processed = tier_end_for_calculation
    
    return total_cost


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given Dashscope model.

    Handles:
    - Tiered pricing based on input token ranges
    - Caching discounts (cache_read_input_token_cost)
    - Reasoning tokens (output_cost_per_reasoning_token)
    - Standard input/output pricing for non-tiered models

    Args:
        model: Model name without provider prefix
        usage: LiteLLM Usage block

    Returns:
        Tuple[float, float] - (prompt_cost_in_usd, completion_cost_in_usd)
    """
    # Get model info
    model_info = get_model_info(model=model, custom_llm_provider="dashscope")
    
    prompt_cost = 0.0
    completion_cost = 0.0
    
    # Check if model uses tiered pricing
    tiered_pricing = model_info.get("tiered_pricing")
    
    if tiered_pricing and isinstance(tiered_pricing, list):
        # TIERED PRICING MODEL
        
        # Calculate input cost with tiering
        text_tokens = usage.prompt_tokens
        cached_tokens = 0
        
        if usage.prompt_tokens_details and hasattr(usage.prompt_tokens_details, "cached_tokens"):
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
            text_tokens = text_tokens - cached_tokens
        
        # Calculate cost for regular input tokens
        prompt_cost = _calculate_tiered_cost(
            tokens=text_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="input_cost_per_token"
        )
        
        # Add cached token cost (usually discounted)
        if cached_tokens > 0:
            cache_cost = _calculate_tiered_cost(
                tokens=cached_tokens,
                tiered_pricing=tiered_pricing,
                cost_key="cache_read_input_token_cost"
            )
            prompt_cost += cache_cost
        
        # Calculate output cost with tiering
        completion_tokens = usage.completion_tokens or 0
        reasoning_tokens = 0
        
        # Check for reasoning tokens
        if (hasattr(usage, "completion_tokens_details") and 
            usage.completion_tokens_details and
            hasattr(usage.completion_tokens_details, "reasoning_tokens")):
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
            # Regular completion tokens = total - reasoning tokens
            completion_tokens = completion_tokens - reasoning_tokens
        
        # Calculate cost for regular output tokens
        completion_cost = _calculate_tiered_cost(
            tokens=completion_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="output_cost_per_token"
        )
        
        # Add reasoning token cost (usually higher)
        if reasoning_tokens > 0:
            reasoning_cost = _calculate_tiered_cost(
                tokens=reasoning_tokens,
                tiered_pricing=tiered_pricing,
                cost_key="output_cost_per_reasoning_token",
                fallback_cost_key="output_cost_per_token"  # Fallback to regular output cost
            )
            completion_cost += reasoning_cost
            
    else:
        # STANDARD FLAT PRICING MODEL
        
        # Input cost
        input_cost_per_token = model_info.get("input_cost_per_token", 0.0)
        text_tokens = usage.prompt_tokens
        cached_tokens = 0
        
        if usage.prompt_tokens_details and hasattr(usage.prompt_tokens_details, "cached_tokens"):
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
            text_tokens = text_tokens - cached_tokens
        
        prompt_cost = text_tokens * input_cost_per_token
        
        # Add cached token cost
        if cached_tokens > 0:
            cache_read_cost = model_info.get("cache_read_input_token_cost", input_cost_per_token) or input_cost_per_token
            prompt_cost += cached_tokens * cache_read_cost
        
        # Output cost
        output_cost_per_token = model_info.get("output_cost_per_token", 0.0)
        completion_tokens = usage.completion_tokens or 0
        reasoning_tokens = 0
        
        # Check for reasoning tokens
        if (hasattr(usage, "completion_tokens_details") and 
            usage.completion_tokens_details and
            hasattr(usage.completion_tokens_details, "reasoning_tokens")):
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
            completion_tokens = completion_tokens - reasoning_tokens
        
        completion_cost = completion_tokens * output_cost_per_token
        
        # Add reasoning token cost
        if reasoning_tokens > 0:
            reasoning_cost_per_token = model_info.get("output_cost_per_reasoning_token", output_cost_per_token) or output_cost_per_token
            completion_cost += reasoning_tokens * reasoning_cost_per_token

    return prompt_cost, completion_cost
