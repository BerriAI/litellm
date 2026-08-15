import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import (
    calculate_tiered_cost,
    tier_rate,
)


def test_calculate_tiered_cost_honors_explicit_zero_primary_rate():
    tiers = [
        {
            "range": [0, 100000],
            "input_cost_per_token": 1e-6,
            "cache_read_input_token_cost": 0.0,
        }
    ]

    cost = calculate_tiered_cost(
        tokens=10000,
        tiered_pricing=tiers,
        cost_key="cache_read_input_token_cost",
        fallback_cost_key="input_cost_per_token",
    )

    assert cost == 0.0


def test_calculate_tiered_cost_honors_explicit_zero_primary_rate_in_overflow_tier():
    tiers = [
        {
            "range": [0, 100000],
            "input_cost_per_token": 1e-6,
            "cache_read_input_token_cost": 0.0,
        }
    ]

    cost = calculate_tiered_cost(
        tokens=150000,
        tiered_pricing=tiers,
        cost_key="cache_read_input_token_cost",
        fallback_cost_key="input_cost_per_token",
    )

    assert cost == 0.0


def test_calculate_tiered_cost_falls_back_when_primary_rate_is_missing():
    tiers = [{"range": [0, 100000], "input_cost_per_token": 1e-6}]

    cost = calculate_tiered_cost(
        tokens=10000,
        tiered_pricing=tiers,
        cost_key="cache_read_input_token_cost",
        fallback_cost_key="input_cost_per_token",
    )

    assert cost == 10000 * 1e-6


def test_tier_rate_honors_explicit_zero_primary_rate():
    tier = {"cache_read_input_token_cost": 0.0, "input_cost_per_token": 1e-6}

    assert tier_rate(tier, "cache_read_input_token_cost", "input_cost_per_token") == 0.0


def test_tier_rate_falls_back_when_primary_rate_is_missing():
    tier = {"input_cost_per_token": 1e-6}

    assert tier_rate(tier, "cache_read_input_token_cost", "input_cost_per_token") == 1e-6
