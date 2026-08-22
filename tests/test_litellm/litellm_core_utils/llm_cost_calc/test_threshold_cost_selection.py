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
    "model_info",
    [
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "output_cost_per_token_above_200k_tokens": 1e-6,
            "output_cost_per_token_above_200k_tokens_priority": 1.5e-6,
        },
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "output_cost_per_token_above_200k_tokens_priority": 1.5e-6,
            "output_cost_per_token_above_200k_tokens": 1e-6,
        },
    ],
    ids=["standard-first", "priority-first"],
)
def test_equal_threshold_matching_tier_wins_for_output(model_info):
    """At an equal threshold the matching service-tier rate must win regardless of the
    order the standard and tier-qualified keys appear in model_info."""
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
    )

    rates = _get_token_base_cost(model_info=model_info, usage=usage, service_tier="priority")

    assert rates[1] == pytest.approx(1.5e-6)


@pytest.mark.parametrize(
    "model_info",
    [
        {
            "input_cost_per_token": 1e-6,
            "cache_read_input_token_cost": 2e-7,
            "cache_read_input_token_cost_above_200k_tokens": 3e-7,
            "cache_read_input_token_cost_above_200k_tokens_priority": 4e-7,
        },
        {
            "input_cost_per_token": 1e-6,
            "cache_read_input_token_cost": 2e-7,
            "cache_read_input_token_cost_above_200k_tokens_priority": 4e-7,
            "cache_read_input_token_cost_above_200k_tokens": 3e-7,
        },
    ],
    ids=["standard-first", "priority-first"],
)
def test_equal_threshold_matching_tier_wins_for_cache_read(model_info):
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
    )

    rates = _get_token_base_cost(model_info=model_info, usage=usage, service_tier="priority")

    assert rates[4] == pytest.approx(4e-7)


def test_equal_threshold_default_tier_keeps_standard():
    """service_tier=None must ignore the tier-qualified key and keep the standard rate."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "output_cost_per_token_above_200k_tokens": 1e-6,
        "output_cost_per_token_above_200k_tokens_priority": 1.5e-6,
    }
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
    )

    rates = _get_token_base_cost(model_info=model_info, usage=usage)

    assert rates[1] == pytest.approx(1e-6)


def test_equal_threshold_wrong_tier_keeps_standard():
    """A priority-qualified key must not win under a different service tier."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "output_cost_per_token_above_200k_tokens": 1e-6,
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

    assert rates[1] == pytest.approx(1e-6)


def test_unequal_threshold_highest_standard_still_wins():
    """A higher standard threshold governs over a lower matching-tier threshold."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "output_cost_per_token_above_200k_tokens": 1e-6,
        "output_cost_per_token_above_128k_tokens_priority": 1.5e-6,
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

    assert rates[1] == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("tier_field", "tier_value", "rate_index"),
    [
        ("output_cost_per_token_above_200k_tokens_priority", 1.5e-6, 1),
        ("cache_read_input_token_cost_above_200k_tokens_priority", 4e-7, 4),
    ],
)
def test_fast_service_tier_uses_priority_qualified_threshold(tier_field, tier_value, rate_index):
    """fast aliases to priority pricing, so a priority-qualified threshold with no standard
    sibling must apply to a fast request too."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 1e-6,
        "cache_read_input_token_cost": 2e-7,
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
        service_tier="fast",
    )

    assert rates[rate_index] == pytest.approx(tier_value)


@pytest.mark.parametrize(
    ("service_tier", "expected"),
    [
        ("priority", 1.5e-6),
        ("flex", 1e-6),
        (None, 1e-6),
    ],
    ids=["priority-guard", "flex-guard", "default-guard"],
)
def test_fast_alias_guard_tiers_for_priority_only_output_threshold(service_tier, expected):
    """priority must keep working under the alias; flex and default must not pick up the
    priority rate."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 1e-6,
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
        service_tier=service_tier,
    )

    assert rates[1] == pytest.approx(expected)


@pytest.mark.parametrize("service_tier", ["priority", "fast"], ids=["priority", "fast"])
def test_fast_builtin_standard_plus_priority_threshold_resolves_priority(service_tier):
    """The builtin shape (standard sibling present) keeps resolving the priority variant via
    the existing standard-key alias lookup."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 1e-6,
        "output_cost_per_token_above_200k_tokens": 1e-6,
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
        service_tier=service_tier,
    )

    assert rates[1] == pytest.approx(1.5e-6)


@pytest.mark.parametrize(
    "model_info",
    [
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 1e-6,
            "output_cost_per_token_above_200k_tokens": 1e-6,
            "output_cost_per_token_above_200k_tokens_priority": 1.5e-6,
        },
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 1e-6,
            "output_cost_per_token_above_200k_tokens_priority": 1.5e-6,
            "output_cost_per_token_above_200k_tokens": 1e-6,
        },
    ],
    ids=["standard-first", "priority-first"],
)
def test_fast_equal_threshold_matching_tier_wins_both_orders(model_info):
    """The Phase D equal-threshold tie-break must hold under the fast alias, independent of
    insertion order."""
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
    )

    rates = _get_token_base_cost(
        model_info=model_info,
        usage=usage,
        service_tier="fast",
    )

    assert rates[1] == pytest.approx(1.5e-6)


def test_fast_service_tier_uppercase_uses_priority_qualified_threshold():
    """The scanner must be as case-insensitive as the lookup path."""
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 1e-6,
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
        service_tier="FAST",
    )

    assert rates[1] == pytest.approx(1.5e-6)


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
