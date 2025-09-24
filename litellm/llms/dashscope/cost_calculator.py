"""
Cost calculator for Dashscope Chat models. 

Handles tiered pricing and prompt caching scenarios.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


@dataclass
class TokenBreakdown:
    """Token breakdown for cost calculation."""
    text_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int


def _extract_token_breakdown(usage: Usage) -> TokenBreakdown:
    """Extract token counts from usage, handling cached and reasoning tokens."""
    cached_tokens = 0
    if usage.prompt_tokens_details and hasattr(usage.prompt_tokens_details, "cached_tokens"):
        cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
    
    text_tokens = usage.prompt_tokens - cached_tokens
    
    reasoning_tokens = 0
    if (hasattr(usage, "completion_tokens_details") and 
        usage.completion_tokens_details and
        hasattr(usage.completion_tokens_details, "reasoning_tokens")):
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
    
    completion_tokens = (usage.completion_tokens or 0) - reasoning_tokens
    
    return TokenBreakdown(text_tokens, cached_tokens, completion_tokens, reasoning_tokens)


def _calculate_tiered_cost(
    tokens: int, 
    tiered_pricing: List[dict], 
    cost_key: str,
    fallback_cost_key: Optional[str] = None
) -> float:
    """Calculate cost using tiered pricing structure.
    
    Finds the appropriate tier based on token count and applies that tier's rate to all tokens.
    """
    if not tiered_pricing or tokens <= 0:
        return 0.0
    
    # Find the appropriate tier for the token count
    for tier in tiered_pricing:
        tier_range = tier.get("range", [])
        if len(tier_range) != 2:
            continue
            
        range_start, range_end = tier_range
        
        # Check if tokens fall within this tier's range
        if range_start <= tokens <= range_end:
            cost_per_token = tier.get(cost_key) or tier.get(fallback_cost_key, 0)
            return tokens * cost_per_token
    
    # If no tier matches, use the last tier (highest tier)
    if tiered_pricing:
        last_tier = tiered_pricing[-1]
        cost_per_token = last_tier.get(cost_key) or last_tier.get(fallback_cost_key, 0)
        return tokens * cost_per_token
    
    return 0.0


def _calculate_flat_cost(tokens: int, cost_per_token: float) -> float:
    """Calculate cost using flat pricing."""
    return tokens * cost_per_token


def _calculate_prompt_cost(breakdown: TokenBreakdown, model_info: ModelInfo, tiered_pricing: Optional[List[dict]]) -> float:
    """Calculate total prompt cost including cached tokens."""
    if tiered_pricing:
        text_cost = _calculate_tiered_cost(
            tokens=breakdown.text_tokens, 
            tiered_pricing=tiered_pricing, 
            cost_key="input_cost_per_token"
        )
        cache_cost = _calculate_tiered_cost(
            tokens=breakdown.cached_tokens, 
            tiered_pricing=tiered_pricing, 
            cost_key="cache_read_input_token_cost"
        )
        return text_cost + cache_cost
    
    input_cost = model_info.get("input_cost_per_token", 0.0)
    cache_cost = model_info.get("cache_read_input_token_cost", input_cost) or input_cost
    
    return (_calculate_flat_cost(tokens=breakdown.text_tokens, cost_per_token=input_cost) + 
            _calculate_flat_cost(tokens=breakdown.cached_tokens, cost_per_token=cache_cost))


def _calculate_completion_cost(breakdown: TokenBreakdown, model_info: ModelInfo, tiered_pricing: Optional[List[dict]]) -> float:
    """Calculate total completion cost including reasoning tokens."""
    if tiered_pricing:
        completion_cost = _calculate_tiered_cost(
            tokens=breakdown.completion_tokens, 
            tiered_pricing=tiered_pricing, 
            cost_key="output_cost_per_token"
        )
        reasoning_cost = _calculate_tiered_cost(
            tokens=breakdown.reasoning_tokens, 
            tiered_pricing=tiered_pricing, 
            cost_key="output_cost_per_reasoning_token",
            fallback_cost_key="output_cost_per_token"
        )
        return completion_cost + reasoning_cost
    
    output_cost = model_info.get("output_cost_per_token", 0.0)
    reasoning_cost = model_info.get("output_cost_per_reasoning_token", output_cost) or output_cost
    
    return (_calculate_flat_cost(tokens=breakdown.completion_tokens, cost_per_token=output_cost) + 
            _calculate_flat_cost(tokens=breakdown.reasoning_tokens, cost_per_token=reasoning_cost))


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculate cost per token for Dashscope models.
    
    Supports both tiered and flat pricing with cached and reasoning tokens.
    
    Args:
        model: Model name without provider prefix
        usage: LiteLLM Usage block
        
    Returns:
        Tuple[float, float] - (prompt_cost_in_usd, completion_cost_in_usd)
    """
    model_info = get_model_info(model=model, custom_llm_provider="dashscope")
    breakdown = _extract_token_breakdown(usage)
    tiered_pricing = model_info.get("tiered_pricing") if isinstance(model_info.get("tiered_pricing"), list) else None
    
    prompt_cost = _calculate_prompt_cost(
        breakdown=breakdown, 
        model_info=model_info, 
        tiered_pricing=tiered_pricing
    )
    completion_cost = _calculate_completion_cost(
        breakdown=breakdown, 
        model_info=model_info, 
        tiered_pricing=tiered_pricing
    )
    
    return prompt_cost, completion_cost
