
import math
from datetime import datetime, timezone

import pytest


import litellm
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import OffPeakPricing, PromptTokensDetailsWrapper, Usage

MODEL = "accounts/fireworks/models/glm-5p2"
INPUT_COST = 1.4e-06
# Read the cached rate from the price map so this test tracks the shipped value
# (glm-5p2 is $0.14/1M) instead of hardcoding a number that breaks when it changes.
CACHE_READ_COST = litellm.get_model_info(model=MODEL, custom_llm_provider="fireworks_ai")["cache_read_input_token_cost"]
OUTPUT_COST = 4.4e-06


def _usage(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )


def test_cached_prompt_tokens_billed_at_cache_read_rate():
    prompt_tokens = 7036
    cached_tokens = 7020
    completion_tokens = 8

    prompt_cost, completion_cost = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, cached_tokens, completion_tokens)
    )

    expected_prompt_cost = (prompt_tokens - cached_tokens) * INPUT_COST + cached_tokens * CACHE_READ_COST
    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(completion_tokens * OUTPUT_COST)

    full_rate_cost = prompt_tokens * INPUT_COST
    assert prompt_cost < full_rate_cost


def test_warm_call_cheaper_than_cold_call():
    prompt_tokens = 7036
    completion_tokens = 8

    cold_prompt_cost, _ = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 16, completion_tokens)
    )
    warm_prompt_cost, _ = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 7020, completion_tokens)
    )

    assert warm_prompt_cost < cold_prompt_cost


def test_no_cached_tokens_matches_full_input_rate():
    prompt_tokens = 100
    completion_tokens = 10

    prompt_cost, completion_cost = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 0, completion_tokens)
    )

    assert prompt_cost == pytest.approx(prompt_tokens * INPUT_COST)
    assert completion_cost == pytest.approx(completion_tokens * OUTPUT_COST)


OFF_PEAK_MODEL = "accounts/fireworks/models/off-peak-test"
OFF_PEAK_WINDOW = "14:00-00:00"
INSIDE_WINDOW = datetime(2026, 9, 3, 17, 25, tzinfo=timezone.utc)
OUTSIDE_WINDOW = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)
STANDARD_INPUT_COST = 1.5e-07
STANDARD_OUTPUT_COST = 6e-07
STANDARD_CACHE_READ_COST = 1.5e-08


def _register_off_peak_model(off_peak_pricing: OffPeakPricing, cache_read_cost: float | None = STANDARD_CACHE_READ_COST) -> None:
    litellm.model_cost[f"fireworks_ai/{OFF_PEAK_MODEL}"] = {
        "litellm_provider": "fireworks_ai",
        "mode": "chat",
        "input_cost_per_token": STANDARD_INPUT_COST,
        "output_cost_per_token": STANDARD_OUTPUT_COST,
        "off_peak_pricing": off_peak_pricing,
        **({} if cache_read_cost is None else {"cache_read_input_token_cost": cache_read_cost}),
    }


def test_off_peak_window_swaps_in_the_off_peak_rates():
    """
    Regression (LIT-6874): a deployment configured with off_peak_pricing kept billing the
    standard fireworks_ai rates inside its window, while the same block on a deepseek
    deployment billed the off-peak rates.
    """
    _register_off_peak_model(
        {
            "hours_utc": OFF_PEAK_WINDOW,
            "input_cost_per_token": 1e-08,
            "output_cost_per_token": 2e-08,
            "cache_read_input_token_cost": 1e-09,
        }
    )
    usage = _usage(prompt_tokens=1000, cached_tokens=300, completion_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model=OFF_PEAK_MODEL, usage=usage, current_time=INSIDE_WINDOW)

    assert math.isclose(prompt_cost, (700 * 1e-08) + (300 * 1e-09), rel_tol=1e-10)
    assert math.isclose(completion_cost, 200 * 2e-08, rel_tol=1e-10)

    peak_prompt_cost, peak_completion_cost = cost_per_token(
        model=OFF_PEAK_MODEL, usage=usage, current_time=OUTSIDE_WINDOW
    )

    assert math.isclose(peak_prompt_cost, (700 * STANDARD_INPUT_COST) + (300 * STANDARD_CACHE_READ_COST), rel_tol=1e-10)
    assert math.isclose(peak_completion_cost, 200 * STANDARD_OUTPUT_COST, rel_tol=1e-10)


def test_off_peak_rates_left_unset_keep_the_standard_rates():
    """A block that only overrides the input rate leaves output and cache reads on the standard rates."""
    _register_off_peak_model({"hours_utc": OFF_PEAK_WINDOW, "input_cost_per_token": 1e-08})
    usage = _usage(prompt_tokens=1000, cached_tokens=300, completion_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model=OFF_PEAK_MODEL, usage=usage, current_time=INSIDE_WINDOW)

    assert math.isclose(prompt_cost, (700 * 1e-08) + (300 * STANDARD_CACHE_READ_COST), rel_tol=1e-10)
    assert math.isclose(completion_cost, 200 * STANDARD_OUTPUT_COST, rel_tol=1e-10)


def test_off_peak_window_bills_cached_tokens_at_the_off_peak_input_rate_without_a_cache_read_rate():
    """Most fireworks_ai price-map entries carry no cache_read_input_token_cost, so cached tokens
    fall back to the input rate, and inside the window that has to be the off-peak one."""
    _register_off_peak_model(
        {"hours_utc": OFF_PEAK_WINDOW, "input_cost_per_token": 1e-08, "output_cost_per_token": 2e-08},
        cache_read_cost=None,
    )
    usage = _usage(prompt_tokens=1000, cached_tokens=300, completion_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model=OFF_PEAK_MODEL, usage=usage, current_time=INSIDE_WINDOW)

    assert math.isclose(prompt_cost, 1000 * 1e-08, rel_tol=1e-10)
    assert math.isclose(completion_cost, 200 * 2e-08, rel_tol=1e-10)

    peak_prompt_cost, _ = cost_per_token(model=OFF_PEAK_MODEL, usage=usage, current_time=OUTSIDE_WINDOW)

    assert math.isclose(peak_prompt_cost, 1000 * STANDARD_INPUT_COST, rel_tol=1e-10)


def test_off_peak_defaults_to_the_current_time():
    """The proxy's cost dispatch passes no clock, so an all-day window has to apply on the
    default current time."""
    _register_off_peak_model({"hours_utc": "00:00-00:00", "input_cost_per_token": 1e-08, "output_cost_per_token": 2e-08})
    usage = _usage(prompt_tokens=1000, cached_tokens=0, completion_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model=OFF_PEAK_MODEL, usage=usage)

    assert math.isclose(prompt_cost, 1000 * 1e-08, rel_tol=1e-10)
    assert math.isclose(completion_cost, 200 * 2e-08, rel_tol=1e-10)
