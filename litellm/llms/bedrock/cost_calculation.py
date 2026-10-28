"""
Helper util for handling bedrock-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING

from litellm.litellm_core_utils.llm_cost_calc.utils import cost_per_text

if TYPE_CHECKING:
    from litellm.types.utils import Usage


def cost_per_token(model: str, usage: "Usage", service_tier: str | None = None) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Follows the same logic as Anthropic's cost per token calculation.
    """
    return cost_per_text(model=model, usage=usage, custom_llm_provider="bedrock", service_tier=service_tier)
