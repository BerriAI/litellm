"""
Helper util for handling bedrock-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    get_provider_specific_geo_multiplier,
    price_token_usage,
)

if TYPE_CHECKING:
    from litellm.types.utils import Usage


def cost_per_token(model: str, usage: "Usage", service_tier: str | None = None) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Follows the same logic as Anthropic's cost per token calculation.
    """
    model_info: Final = litellm.get_model_info(model=model, custom_llm_provider="bedrock")
    return price_token_usage(
        model=model,
        usage=usage,
        custom_llm_provider="bedrock",
        service_tier=service_tier,
        resolved_model_info=model_info,
        current_time=None,
        factors=(get_provider_specific_geo_multiplier(model_info=model_info, usage=usage),),
    )
