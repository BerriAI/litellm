"""
Helper util for handling bedrock-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    generic_cost_per_token,
    get_provider_specific_geo_multiplier,
)

if TYPE_CHECKING:
    from litellm.types.utils import Usage


def cost_per_token(model: str, usage: "Usage", service_tier: str | None = None) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Follows the same logic as Anthropic's cost per token calculation.
    """
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
