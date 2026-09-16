"""
For calculating cost of fireworks ai serverless inference models.
"""

from datetime import datetime
from typing import (
    Final,
    cast,  # noqa: TID251  # the fallback entry is a dict copy of a ReadOnly TypedDict; no cast-free way to retype it
)

from litellm.constants import (
    FIREWORKS_AI_4_B,
    FIREWORKS_AI_16_B,
    FIREWORKS_AI_56_B_MOE,
    FIREWORKS_AI_176_B_MOE,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


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


def _with_cache_read_fallback(model_info: ModelInfo) -> ModelInfo:
    """Most fireworks_ai price-map entries publish no cache-read rate though the provider bills
    cached reads at the input rate; the shared map is never mutated, so a copy carries the fallback."""
    input_rate: Final = model_info.get("input_cost_per_token")
    if model_info.get("cache_read_input_token_cost") is not None or input_rate is None:
        return model_info
    off_peak: Final = model_info.get("off_peak_pricing")
    if off_peak is None or "cache_read_input_token_cost" in off_peak:
        return cast(ModelInfo, {**model_info, "cache_read_input_token_cost": input_rate})
    return cast(
        ModelInfo,
        {
            **model_info,
            "cache_read_input_token_cost": input_rate,
            "off_peak_pricing": {
                **off_peak,
                "cache_read_input_token_cost": off_peak.get("input_cost_per_token", input_rate),
            },
        },
    )


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
    model_info: Final = _with_cache_read_fallback(_resolve_model_info(model))
    return generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="fireworks_ai",
        model_info=model_info,
        current_time=current_time,
    )
