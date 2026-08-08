import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_token_base_cost,
)
from litellm.types.utils import Usage


@pytest.mark.parametrize(
    ("tier_field", "tier_value", "rate_index"),
    [
        (
            "output_cost_per_token_above_32k_tokens",
            18e-6,
            1,
        ),
        (
            "cache_read_input_token_cost_above_32k_tokens",
            4e-6,
            4,
        ),
    ],
)
def test_standalone_threshold_field_selects_tier(
    tier_field: str,
    tier_value: float,
    rate_index: int,
) -> None:
    """Output and cache fields can independently define a tier."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 5e-7,
        tier_field: tier_value,
    }
    usage = Usage(
        prompt_tokens=40_000,
        completion_tokens=1_000,
        total_tokens=41_000,
    )

    rates = _get_token_base_cost(
        model_info=model_info,
        usage=usage,
    )

    assert rates[rate_index] == pytest.approx(tier_value)


def test_cost_types_select_thresholds_independently() -> None:
    """An inactive input tier must not suppress an active output tier."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "input_cost_per_token_above_128k_tokens": 9e-6,
        "output_cost_per_token_above_32k_tokens": 18e-6,
    }
    usage = Usage(
        prompt_tokens=40_000,
        completion_tokens=1_000,
        total_tokens=41_000,
    )

    rates = _get_token_base_cost(
        model_info=model_info,
        usage=usage,
    )

    assert rates[0] == pytest.approx(1e-6)
    assert rates[1] == pytest.approx(18e-6)
