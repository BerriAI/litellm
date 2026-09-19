"""
Fireworks AI serverless cache-read pricing defaults.
"""

from typing import (
    Final,
    cast,  # noqa: TID251  # the derived entry is a dict copy of a ReadOnly TypedDict; no cast-free way to retype it
)

from litellm.constants import FIREWORKS_AI_DEFAULT_CACHE_READ_RATE_RATIO
from litellm.types.utils import ModelInfo


def with_default_cache_read_rate(model_info: ModelInfo) -> ModelInfo:
    """Entries without a cache-read rate get the documented discount off the input rate; the shared map is
    never mutated, so a copy carries it."""
    input_rate: Final = model_info.get("input_cost_per_token")
    if model_info.get("cache_read_input_token_cost") is not None or input_rate is None:
        return model_info
    cache_read_rate: Final = input_rate * FIREWORKS_AI_DEFAULT_CACHE_READ_RATE_RATIO
    off_peak: Final = model_info.get("off_peak_pricing")
    if off_peak is None or "cache_read_input_token_cost" in off_peak:
        return cast(ModelInfo, {**model_info, "cache_read_input_token_cost": cache_read_rate})
    return cast(
        ModelInfo,
        {
            **model_info,
            "cache_read_input_token_cost": cache_read_rate,
            "off_peak_pricing": {
                **off_peak,
                "cache_read_input_token_cost": (
                    off_peak["input_cost_per_token"] * FIREWORKS_AI_DEFAULT_CACHE_READ_RATE_RATIO
                    if "input_cost_per_token" in off_peak
                    else cache_read_rate
                ),
            },
        },
    )
