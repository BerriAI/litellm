"""
Helper util for handling XAI-specific cost calculation
- e.g.: reasoning tokens for grok models
"""

from typing import Tuple, Union

from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given XAI model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing XAI-specific usage information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider="xai")

    def _safe_float_cast(
        value: Union[str, int, float, None, object], default: float = 0.0
    ) -> float:
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

    ## CALCULATE OUTPUT COST
    output_cost_per_token = _safe_float_cast(model_info.get("output_cost_per_token"))

    # For XAI models, completion is billed as (visible completion tokens + reasoning tokens)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning_tokens = 0
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = int(
            getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        )

    completion_cost = (completion_tokens + reasoning_tokens) * output_cost_per_token

    return prompt_cost, completion_cost
