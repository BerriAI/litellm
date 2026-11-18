"""
Cost calculator for DeepSeek Chat models.

Handles prompt caching scenario.
"""

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import branch_cost
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Follows the same logic as Anthropic's cost per token calculation.
    """
    return branch_cost(model, "deepseek", usage, get_model_info(model=model, custom_llm_provider="deepseek"))
