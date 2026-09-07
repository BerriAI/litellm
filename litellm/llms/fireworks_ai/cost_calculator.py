"""
For calculating cost of fireworks ai serverless inference models.
"""

import math
from datetime import datetime
from typing import Final

from litellm.constants import (
    FIREWORKS_AI_4_B,
    FIREWORKS_AI_16_B,
    FIREWORKS_AI_56_B_MOE,
    FIREWORKS_AI_176_B_MOE,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import TokenRates, apply_off_peak_pricing
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info

NO_CACHE_READ_RATE: Final = float("nan")


# Extract the number of billion parameters from the model name
# only used for together_computer LLMs
def get_base_model_for_pricing(model_name: str) -> str:
    """
    Helper function for calculating together ai pricing.

    Returns:
    - str: model pricing category if mapped else received model name
    """
    import re

    model_name = model_name.lower()

    # Check for MoE models in the form <number>x<number>b
    moe_match: Final = re.search(r"(\d+)x(\d+)b", model_name)
    if moe_match:
        total_billion: Final = int(moe_match.group(1)) * int(moe_match.group(2))
        if total_billion <= FIREWORKS_AI_56_B_MOE:
            return "fireworks-ai-moe-up-to-56b"
        elif total_billion <= FIREWORKS_AI_176_B_MOE:
            return "fireworks-ai-56b-to-176b"

    # Check for standard models in the form <number>b
    re_params_match: Final = re.search(r"(\d+)b", model_name)
    if re_params_match is not None:
        params_match: Final = str(re_params_match.group(1))
        params_billion: Final = float(params_match)

        # Determine the category based on the number of parameters
        if params_billion <= FIREWORKS_AI_4_B:
            return "fireworks-ai-up-to-4b"
        elif params_billion <= FIREWORKS_AI_16_B:
            return "fireworks-ai-4.1b-to-16b"
        elif params_billion > FIREWORKS_AI_16_B:
            return "fireworks-ai-above-16b"

    # If no matches, return the original model_name
    return "fireworks-ai-default"


def _resolve_model_info(model: str) -> ModelInfo:
    try:
        return get_model_info(model=model, custom_llm_provider="fireworks_ai")
    except Exception:
        base_model: Final = get_base_model_for_pricing(model_name=model)
        return get_model_info(model=base_model, custom_llm_provider="fireworks_ai")


def cost_per_token(model: str, usage: Usage, current_time: datetime | None = None) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens,
    swapping in the model's off_peak_pricing rates while one of its windows is open.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information
        - current_time: the moment the request is billed at; defaults to now, UTC

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    model_info: Final = _resolve_model_info(model)
    standard_cache_read_rate: Final = model_info.get("cache_read_input_token_cost")
    rates: Final = apply_off_peak_pricing(
        model_info,
        current_time,
        TokenRates(
            input_rate=model_info["input_cost_per_token"] or 0.0,
            output_rate=model_info["output_cost_per_token"] or 0.0,
            cache_read_rate=standard_cache_read_rate if standard_cache_read_rate is not None else NO_CACHE_READ_RATE,
            cache_creation_rate=0.0,
            reasoning_rate=None,
        ),
    )
    cache_read_rate: Final[float] = rates.input_rate if math.isnan(rates.cache_read_rate) else rates.cache_read_rate

    prompt_tokens_details: Final = usage.prompt_tokens_details
    cached_tokens: Final[int] = (
        prompt_tokens_details.cached_tokens
        if prompt_tokens_details is not None and prompt_tokens_details.cached_tokens is not None
        else 0
    )
    non_cached_prompt_tokens: Final[int] = max(usage.prompt_tokens - cached_tokens, 0)
    prompt_cost: Final[float] = non_cached_prompt_tokens * rates.input_rate + cached_tokens * cache_read_rate
    completion_cost: Final[float] = usage.completion_tokens * rates.output_rate

    return prompt_cost, completion_cost
