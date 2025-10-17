"""
Helper util for handling XAI-specific cost calculation
- Uses the generic cost calculator which already handles tiered pricing correctly
- Handles XAI-specific reasoning token billing (billed as part of completion tokens)
"""

from typing import Tuple

from litellm.types.utils import Usage
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given XAI model, prompt tokens, and completion tokens.
    Uses the generic cost calculator for all pricing logic, with XAI-specific reasoning token handling.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing XAI-specific usage information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    # XAI-specific completion cost calculation
    # For XAI models, completion is billed as (visible completion tokens + reasoning tokens)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning_tokens = 0
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = int(getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0)

    total_completion_tokens = completion_tokens + reasoning_tokens
    
    modified_usage = Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=usage.total_tokens,
        prompt_tokens_details=usage.prompt_tokens_details,
        completion_tokens_details=None 
    )
    
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=modified_usage,
        custom_llm_provider="xai"
    )

    return prompt_cost, completion_cost
