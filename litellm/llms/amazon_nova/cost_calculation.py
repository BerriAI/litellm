"""
Helper util for handling amazon nova cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

if TYPE_CHECKING:
    from litellm.types.utils import Usage


def cost_per_token(model: str, usage: "Usage") -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.
    Follows the same logic as Anthropic's cost per token calculation.
    """
    return generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="amazon_nova"
    )