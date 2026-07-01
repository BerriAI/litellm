"""
For calculating cost of fireworks ai serverless inference models.
"""

from typing import Tuple

from litellm.constants import (
    FIREWORKS_AI_4_B,
    FIREWORKS_AI_16_B,
    FIREWORKS_AI_56_B_MOE,
    FIREWORKS_AI_176_B_MOE,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


def get_base_model_for_pricing(model_name: str) -> str:
    """
    Resolves a Fireworks model name to its pricing category based on parameter count.

    Returns:
    - str: model pricing category if mapped else received model name
    """
    import re

    model_name = model_name.lower()

    moe_match = re.search(r"(\d+)x(\d+)b", model_name)
    if moe_match:
        total_billion = int(moe_match.group(1)) * int(moe_match.group(2))
        if total_billion <= FIREWORKS_AI_56_B_MOE:
            return "fireworks-ai-moe-up-to-56b"
        elif total_billion <= FIREWORKS_AI_176_B_MOE:
            return "fireworks-ai-56b-to-176b"

    re_params_match = re.search(r"(\d+)b", model_name)
    if re_params_match is not None:
        params_billion = float(re_params_match.group(1))

        if params_billion <= FIREWORKS_AI_4_B:
            return "fireworks-ai-up-to-4b"
        elif params_billion <= FIREWORKS_AI_16_B:
            return "fireworks-ai-4.1b-to-16b"
        elif params_billion > FIREWORKS_AI_16_B:
            return "fireworks-ai-above-16b"

    return "fireworks-ai-default"


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing prompt caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    try:
        return generic_cost_per_token(model=model, usage=usage, custom_llm_provider="fireworks_ai")
    except Exception as e:
        from litellm._logging import verbose_logger

        verbose_logger.debug(
            "fireworks_ai cost_per_token: model=%s not found, falling back to param-size pricing. Error: %s",
            model,
            str(e),
        )
        base_model = get_base_model_for_pricing(model_name=model)
        return generic_cost_per_token(model=base_model, usage=usage, custom_llm_provider="fireworks_ai")
