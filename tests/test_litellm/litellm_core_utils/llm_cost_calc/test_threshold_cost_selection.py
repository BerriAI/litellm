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


@pytest.mark.parametrize(
    ("tier_field", "tier_value", "rate_index"),
    [
        ("output_cost_per_token_above_200k_tokens_priority", 1.5e-6, 1),
        ("cache_read_input_token_cost_above_200k_tokens_priority", 4e-7, 4),
    ],
)
def test_tier_qualified_threshold_field_selects_tier_for_matching_service_tier(
    tier_field: str, tier_value: float, rate_index: int
) -> None:
    """A tier-qualified threshold key with no standard sibling must apply when the request's
    service tier matches."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 5e-7,
        tier_field: tier_value,
    }
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
    )

    rates = _get_token_base_cost(
        model_info=model_info,
        usage=usage,
        service_tier="priority",
    )

    assert rates[rate_index] == pytest.approx(tier_value)


def test_tier_qualified_threshold_is_ignored_under_the_default_tier() -> None:
    """service_tier=None must keep billing the flat rate: the priority key is inactive."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 5e-7,
        "output_cost_per_token_above_200k_tokens_priority": 1.5e-6,
    }
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
    )

    rates = _get_token_base_cost(model_info=model_info, usage=usage)

    assert rates[1] == pytest.approx(2e-6)


def test_tier_qualified_threshold_is_ignored_under_a_different_service_tier() -> None:
    """A priority-qualified key must not bill a flex request."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 5e-7,
        "output_cost_per_token_above_200k_tokens_priority": 1.5e-6,
    }
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
    )

    rates = _get_token_base_cost(
        model_info=model_info,
        usage=usage,
        service_tier="flex",
    )

    assert rates[1] == pytest.approx(2e-6)
