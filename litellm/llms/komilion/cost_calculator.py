"""
Cost calculator for Komilion models.

Komilion dynamically routes requests, so exact per-token costs depend on which
model is selected at runtime. The costs listed in the model pricing JSON
represent typical averages for each tier.
"""

from typing import Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    return generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="komilion"
    )
