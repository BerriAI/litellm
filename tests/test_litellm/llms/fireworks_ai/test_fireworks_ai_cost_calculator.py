import math
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_prompt_caching_savings,
    generic_cost_per_token,
    get_token_type_cost_breakdown,
)
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    OffPeakPricing,
    PromptTokensDetailsWrapper,
    Usage,
)

MODEL = "accounts/fireworks/models/glm-5p2"
INPUT_COST = 1.4e-06
CACHE_READ_COST = litellm.get_model_info(model=MODEL, custom_llm_provider="fireworks_ai")["cache_read_input_token_cost"]
OUTPUT_COST = 4.4e-06


def _usage(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )


def test_warm_call_cheaper_than_cold_call():
    prompt_tokens = 7036
    completion_tokens = 8

    cold_prompt_cost, _ = cost_per_token(model=MODEL, usage=_usage(prompt_tokens, 16, completion_tokens))
    warm_prompt_cost, _ = cost_per_token(model=MODEL, usage=_usage(prompt_tokens, 7020, completion_tokens))

    assert warm_prompt_cost < cold_prompt_cost


OFF_PEAK_MODEL = "accounts/fireworks/models/off-peak-test"
OFF_PEAK_WINDOW = "14:00-00:00"
INSIDE_WINDOW = datetime(2026, 9, 3, 17, 25, tzinfo=timezone.utc)
OUTSIDE_WINDOW = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)
STANDARD_INPUT_COST = 1.5e-07
STANDARD_OUTPUT_COST = 6e-07
STANDARD_CACHE_READ_COST = 1.5e-08


def _register_off_peak_model(
    off_peak_pricing: OffPeakPricing,
    cache_read_cost: float | None = STANDARD_CACHE_READ_COST,
    model: str = OFF_PEAK_MODEL,
) -> None:
    litellm.model_cost = {  # test-quality-ok: conftest restores litellm.model_cost after each test
        **litellm.model_cost,
        f"fireworks_ai/{model}": {
            "litellm_provider": "fireworks_ai",
            "mode": "chat",
            "input_cost_per_token": STANDARD_INPUT_COST,
            "output_cost_per_token": STANDARD_OUTPUT_COST,
            "off_peak_pricing": off_peak_pricing,
            **({} if cache_read_cost is None else {"cache_read_input_token_cost": cache_read_cost}),
        },
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


def test_off_peak_window_bills_cached_tokens_at_the_discounted_off_peak_input_rate_without_a_cache_read_rate():
    """Entries without a cache-read rate use Fireworks' documented 50% cached-token discount."""
    _register_off_peak_model(
        {"hours_utc": OFF_PEAK_WINDOW, "input_cost_per_token": 1e-08, "output_cost_per_token": 2e-08},
        cache_read_cost=None,
    )
    usage = _usage(prompt_tokens=1000, cached_tokens=300, completion_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model=OFF_PEAK_MODEL, usage=usage, current_time=INSIDE_WINDOW)

    assert math.isclose(prompt_cost, (700 * 1e-08) + (300 * 1e-08 * 0.5), rel_tol=1e-10)
    assert math.isclose(completion_cost, 200 * 2e-08, rel_tol=1e-10)

    peak_prompt_cost, _ = cost_per_token(model=OFF_PEAK_MODEL, usage=usage, current_time=OUTSIDE_WINDOW)

    assert math.isclose(
        peak_prompt_cost,
        (700 * STANDARD_INPUT_COST) + (300 * STANDARD_INPUT_COST * 0.5),
        rel_tol=1e-10,
    )

    no_input_rate_model = "accounts/fireworks/models/off-peak-no-input-rate-test"
    _register_off_peak_model(
        {"hours_utc": OFF_PEAK_WINDOW, "output_cost_per_token": 2e-08},
        cache_read_cost=None,
        model=no_input_rate_model,
    )

    standard_cache_prompt_cost, _ = cost_per_token(model=no_input_rate_model, usage=usage, current_time=INSIDE_WINDOW)

    assert math.isclose(
        standard_cache_prompt_cost,
        (700 * STANDARD_INPUT_COST) + (300 * STANDARD_INPUT_COST * 0.5),
        rel_tol=1e-10,
    )


def test_an_entry_without_a_cache_read_rate_bills_cached_tokens_at_the_documented_default_discount():
    """Fireworks documents a default 50% cached-token discount for serverless models:
    https://docs.fireworks.ai/guides/prompt-caching, accessed 2026-09-19."""
    model = "accounts/fireworks/models/default-cache-read-test"
    litellm.model_cost = {  # test-quality-ok: conftest restores litellm.model_cost after each test
        **litellm.model_cost,
        f"fireworks_ai/{model}": {
            "litellm_provider": "fireworks_ai",
            "mode": "chat",
            "input_cost_per_token": INPUT_COST,
            "output_cost_per_token": OUTPUT_COST,
        },
    }
    usage = _usage(prompt_tokens=1000, cached_tokens=300, completion_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    assert math.isclose(prompt_cost, (700 * INPUT_COST) + (300 * INPUT_COST * 0.5), rel_tol=1e-10)
    assert prompt_cost < 1000 * INPUT_COST
    assert math.isclose(completion_cost, 200 * OUTPUT_COST, rel_tol=1e-10)


def test_fireworks_cache_read_rates_match_breakdown_and_caching_savings():
    model = "accounts/fireworks/models/breakdown-cache-read-test"
    litellm.model_cost = {  # test-quality-ok: the save/restore conftest returns litellm.model_cost to the original object after each test, so replacing the map for this entry leaks nothing
        **litellm.model_cost,
        f"fireworks_ai/{model}": {
            "litellm_provider": "fireworks_ai",
            "mode": "chat",
            "input_cost_per_token": INPUT_COST,
            "output_cost_per_token": OUTPUT_COST,
        },
    }
    usage = _usage(prompt_tokens=1000, cached_tokens=300, completion_tokens=200)

    breakdown = get_token_type_cost_breakdown(
        model=model,
        custom_llm_provider="fireworks_ai",
        usage=usage,
    )
    prompt_cost, _ = cost_per_token(model=model, usage=usage)
    savings = calculate_prompt_caching_savings(
        model_info=litellm.get_model_info(model=model, custom_llm_provider="fireworks_ai"),
        usage=usage,
        custom_llm_provider="fireworks_ai",
    )

    assert math.isclose(breakdown.cache_read_cost, 300 * INPUT_COST * 0.5, rel_tol=1e-10)
    assert math.isclose(breakdown.rates.cache_read_input_token_cost, INPUT_COST * 0.5, rel_tol=1e-10)
    assert math.isclose(
        (700 * breakdown.rates.input_cost_per_token) + breakdown.cache_read_cost, prompt_cost, rel_tol=1e-10
    )
    assert math.isclose(savings, 300 * INPUT_COST * 0.5, rel_tol=1e-10)


def test_generic_cost_per_token_applies_fireworks_cache_read_default_with_or_without_model_info():
    model = "accounts/fireworks/models/generic-cache-read-test"
    litellm.model_cost = {  # test-quality-ok: conftest restores litellm.model_cost after each test
        **litellm.model_cost,
        f"fireworks_ai/{model}": {
            "litellm_provider": "fireworks_ai",
            "mode": "chat",
            "input_cost_per_token": INPUT_COST,
            "output_cost_per_token": OUTPUT_COST,
        },
    }
    usage = _usage(prompt_tokens=1000, cached_tokens=300, completion_tokens=200)
    expected_prompt_cost = (700 * INPUT_COST) + (300 * INPUT_COST * 0.5)

    implicit_model_info_cost, _ = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="fireworks_ai",
    )
    explicit_model_info_cost, _ = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="fireworks_ai",
        model_info=litellm.get_model_info(model=model, custom_llm_provider="fireworks_ai"),
    )

    assert math.isclose(implicit_model_info_cost, expected_prompt_cost, rel_tol=1e-10)
    assert math.isclose(explicit_model_info_cost, expected_prompt_cost, rel_tol=1e-10)


def test_off_peak_defaults_to_the_current_time():
    """The proxy's cost dispatch passes no clock, so an all-day window has to apply on the
    default current time."""
    _register_off_peak_model(
        {"hours_utc": "00:00-00:00", "input_cost_per_token": 1e-08, "output_cost_per_token": 2e-08}
    )
    usage = _usage(prompt_tokens=1000, cached_tokens=0, completion_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model=OFF_PEAK_MODEL, usage=usage)

    assert math.isclose(prompt_cost, 1000 * 1e-08, rel_tol=1e-10)
    assert math.isclose(completion_cost, 200 * 2e-08, rel_tol=1e-10)


COMPONENT_MODEL = "accounts/fireworks/models/cost-components-test"
COMPONENT_INPUT_COST = 1e-06
COMPONENT_OUTPUT_COST = 2e-06
COMPONENT_CACHE_READ_COST = 1e-07
COMPONENT_CACHE_CREATION_COST = 3e-06
COMPONENT_REASONING_COST = 4e-06
COMPONENT_AUDIO_IN_COST = 5e-06
COMPONENT_AUDIO_OUT_COST = 6e-06


def test_cache_write_reasoning_and_audio_tokens_are_billed_at_their_component_rates():
    litellm.model_cost = {  # test-quality-ok: the save/restore conftest returns litellm.model_cost to the original object after each test, so replacing the map for this entry leaks nothing
        **litellm.model_cost,
        f"fireworks_ai/{COMPONENT_MODEL}": {
            "litellm_provider": "fireworks_ai",
            "mode": "chat",
            "input_cost_per_token": COMPONENT_INPUT_COST,
            "output_cost_per_token": COMPONENT_OUTPUT_COST,
            "cache_read_input_token_cost": COMPONENT_CACHE_READ_COST,
            "cache_creation_input_token_cost": COMPONENT_CACHE_CREATION_COST,
            "output_cost_per_reasoning_token": COMPONENT_REASONING_COST,
            "input_cost_per_audio_token": COMPONENT_AUDIO_IN_COST,
            "output_cost_per_audio_token": COMPONENT_AUDIO_OUT_COST,
        },
    }
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=300,
            cache_creation_tokens=200,
            audio_tokens=100,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=200,
            audio_tokens=50,
        ),
    )

    prompt_cost, completion_cost = cost_per_token(model=COMPONENT_MODEL, usage=usage)

    expected_prompt_cost = (
        400 * COMPONENT_INPUT_COST
        + 300 * COMPONENT_CACHE_READ_COST
        + 200 * COMPONENT_CACHE_CREATION_COST
        + 100 * COMPONENT_AUDIO_IN_COST
    )
    expected_completion_cost = (
        250 * COMPONENT_OUTPUT_COST + 200 * COMPONENT_REASONING_COST + 50 * COMPONENT_AUDIO_OUT_COST
    )
    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(expected_completion_cost)


def test_an_entry_without_an_input_rate_gets_no_cache_read_fallback():
    litellm.model_cost = {  # test-quality-ok: the save/restore conftest returns litellm.model_cost to the original object after each test, so replacing the map for this entry leaks nothing
        **litellm.model_cost,  # pyright: ignore[reportUnknownMemberType]  # the SDK types model_cost as dict[Unknown, Unknown]
        "fireworks_ai/accounts/fireworks/models/no-input-rate-test": {
            "litellm_provider": "fireworks_ai",
            "mode": "chat",
            "output_cost_per_token": 2e-06,
        },
    }
    usage: Final = _usage(prompt_tokens=1000, cached_tokens=300, completion_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model="accounts/fireworks/models/no-input-rate-test", usage=usage)

    assert prompt_cost == 0
    assert completion_cost == 200 * 2e-06
