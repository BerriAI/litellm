"""
Helper util for handling bedrock-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import price_model
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    generic_cost_per_token,
    get_provider_specific_geo_multiplier,
)

if TYPE_CHECKING:
    from litellm.types.utils import Usage


def cost_per_token(model: str, usage: "Usage", service_tier: str | None = None) -> tuple[float, float]:
    """
    Prompt and completion cost of ``usage`` on a Bedrock model, with the geo uplift its row carries.
    """
    priced: Final = price_model(model=model, provider="bedrock", usage=usage, service_tier=service_tier)
    if priced is not None:
        return priced
    return _python_cost_per_token(model, usage, service_tier)


def _python_cost_per_token(model: str, usage: "Usage", service_tier: str | None) -> tuple[float, float]:
    """Main's pricing, for a call the Rust core leaves to Python. An unreadable geo value raises, as on main."""
    model_info: Final = litellm.get_model_info(model=model, custom_llm_provider="bedrock")
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="bedrock",
        service_tier=service_tier,
        model_info=model_info,
    )
    geo_multiplier: Final = get_provider_specific_geo_multiplier(model_info=model_info, usage=usage)
    return prompt_cost * geo_multiplier, completion_cost * geo_multiplier
