"""
Helper util for handling bedrock-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Final

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import price_model
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

if TYPE_CHECKING:
    from litellm.types.utils import Usage


def cost_per_token(model: str, usage: "Usage", service_tier: str | None = None) -> tuple[float, float]:
    """
    Prompt and completion cost of ``usage`` on a Bedrock model, with the geo uplift its row carries.
    """
    priced: Final = price_model(model=model, provider="bedrock", usage=usage, service_tier=service_tier)
    if priced is not None:
        return priced
    return generic_cost_per_token(model=model, usage=usage, custom_llm_provider="bedrock", service_tier=service_tier)
