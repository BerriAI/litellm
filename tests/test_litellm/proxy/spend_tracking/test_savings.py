import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.proxy.spend_tracking.savings import (
    compute_autorouter_savings,
    compute_savings_spend,
)
from litellm.types.utils import Usage


def _anthropic_costs(model: str) -> tuple[float, float]:
    info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
    input_cost = info["input_cost_per_token"] or 0.0
    cache_read_cost = info.get("cache_read_input_token_cost") or input_cost
    return input_cost, cache_read_cost


def _cached_usage_object() -> dict:
    """A cache-heavy Anthropic request, shaped as the spend log records it.

    `prompt_tokens` is the inclusive total: 3 uncached text tokens plus 500 read
    from cache plus 12304 written to cache.
    """
    return {
        "prompt_tokens": 12807,
        "completion_tokens": 500,
        "total_tokens": 13307,
        "prompt_tokens_details": {"cached_tokens": 500, "cache_creation_tokens": 12304, "text_tokens": 3},
        "cache_creation_input_tokens": 12304,
        "cache_read_input_tokens": 500,
    }


def _cost_on(model: str, usage_object: dict) -> float:
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model, usage=Usage(**usage_object), custom_llm_provider="anthropic"
    )
    return prompt_cost + completion_cost


def _flat_rates(model: str) -> tuple[float, float, float]:
    info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
    input_cost = info["input_cost_per_token"] or 0.0
    return (
        input_cost,
        info["output_cost_per_token"] or 0.0,
        info.get("cache_creation_input_token_cost") or input_cost,
    )


def test_compression_savings_priced_at_input_rate():
    input_cost, _ = _anthropic_costs("claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=4389,
        cache_read_input_tokens=0,
    )
    assert result.compression == pytest.approx(4389 * input_cost)
    assert result.compression > 0
    assert result.prompt_caching == 0.0


def test_prompt_caching_savings_priced_at_input_minus_cache_read():
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    # A model that supports prompt caching must charge less to read from cache;
    # otherwise this test is asserting nothing.
    assert cache_read_cost < input_cost
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        cache_read_input_tokens=8200,
    )
    assert result.prompt_caching == pytest.approx(8200 * (input_cost - cache_read_cost))
    assert result.prompt_caching > 0
    assert result.compression == 0.0


def test_unknown_model_fails_open_to_zero():
    result = compute_savings_spend(
        model="totally-made-up-model-xyz",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        cache_read_input_tokens=1000,
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_missing_model_fails_open_to_zero():
    result = compute_savings_spend(
        model=None,
        custom_llm_provider=None,
        compression_saved_tokens=1000,
        cache_read_input_tokens=1000,
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_negative_token_counts_clamp_to_zero():
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=-500,
        cache_read_input_tokens=-500,
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_autorouter_savings_does_not_double_charge_cache_tokens():
    """`prompt_tokens` already includes cache-read and cache-creation tokens.

    Charging those tokens again at the full input rate, or subtracting a separate
    cache-write penalty on top of them, prices the same tokens twice. Both arms go
    through litellm's cost engine on the identical usage, so each token is priced
    exactly once, in its own dimension.
    """
    usage_object = _cached_usage_object()
    result = compute_autorouter_savings(
        baseline_model="claude-opus-5",
        selected_model="claude-haiku-4-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        usage=Usage(**usage_object),
    )

    expected = _cost_on("claude-opus-5", usage_object) - _cost_on("claude-haiku-4-5", usage_object)
    assert result == pytest.approx(expected)
    assert result > 0

    # The double-counting formula this replaced: every prompt token (cache reads
    # and cache writes included) charged at the flat input rate on both sides,
    # minus a cache-write penalty already accounted for inside the selected arm.
    base_in, base_out, base_write = _flat_rates("claude-opus-5")
    sel_in, sel_out, sel_write = _flat_rates("claude-haiku-4-5")
    prompt_tokens = usage_object["prompt_tokens"]
    completion_tokens = usage_object["completion_tokens"]
    double_counted = max(
        (prompt_tokens * base_in + completion_tokens * base_out)
        - (prompt_tokens * sel_in + completion_tokens * sel_out)
        - usage_object["cache_creation_input_tokens"] * sel_write,
        0.0,
    )
    assert result != pytest.approx(double_counted)


def test_autorouter_savings_charges_cache_reads_at_the_cache_read_rate():
    """A request served almost entirely from cache is cheap on both models, so the
    routed saving must be far smaller than the same token count would suggest at
    full input price."""
    usage_object = {
        "prompt_tokens": 10_000,
        "completion_tokens": 0,
        "total_tokens": 10_000,
        "prompt_tokens_details": {"cached_tokens": 10_000, "text_tokens": 0},
        "cache_read_input_tokens": 10_000,
    }
    result = compute_autorouter_savings(
        baseline_model="claude-opus-5",
        selected_model="claude-haiku-4-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        usage=Usage(**usage_object),
    )

    base_read = litellm.get_model_info("claude-opus-5", "anthropic")["cache_read_input_token_cost"]
    sel_read = litellm.get_model_info("claude-haiku-4-5", "anthropic")["cache_read_input_token_cost"]
    assert result == pytest.approx(10_000 * (base_read - sel_read))

    base_in = litellm.get_model_info("claude-opus-5", "anthropic")["input_cost_per_token"]
    sel_in = litellm.get_model_info("claude-haiku-4-5", "anthropic")["input_cost_per_token"]
    assert result < 10_000 * (base_in - sel_in)


def test_autorouter_savings_zero_when_model_unchanged():
    result = compute_autorouter_savings(
        baseline_model="claude-opus-5",
        selected_model="claude-opus-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        usage=Usage(**_cached_usage_object()),
    )
    assert result == 0.0


def test_autorouter_savings_floored_at_zero_on_escalation():
    # Routing UP to a pricier model must never show as negative savings.
    result = compute_autorouter_savings(
        baseline_model="claude-haiku-4-5",
        selected_model="claude-opus-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        usage=Usage(**_cached_usage_object()),
    )
    assert result == 0.0


def test_autorouter_savings_unknown_baseline_fails_open_to_zero():
    result = compute_autorouter_savings(
        baseline_model="totally-made-up-model-xyz",
        selected_model="claude-haiku-4-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        usage=Usage(**_cached_usage_object()),
    )
    assert result == 0.0


def test_autorouter_savings_zero_without_baseline():
    # No configured/produced baseline -> the driver contributes nothing.
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        cache_read_input_tokens=0,
        baseline_model=None,
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter == 0.0


def test_compute_savings_spend_includes_autorouter_driver():
    usage_object = _cached_usage_object()
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        cache_read_input_tokens=0,
        baseline_model="claude-opus-5",
        baseline_provider="anthropic",
        usage_object=usage_object,
    )
    expected = _cost_on("claude-opus-5", usage_object) - _cost_on("claude-haiku-4-5", usage_object)
    assert result.autorouter == pytest.approx(expected)
    assert result.autorouter > 0


def test_compute_savings_spend_without_usage_object_keeps_other_drivers():
    """A row with no recorded usage still prices compression and caching; only the
    counterfactual driver needs the usage breakdown."""
    input_cost, _ = _anthropic_costs("claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        cache_read_input_tokens=0,
        baseline_model="claude-opus-5",
        usage_object=None,
    )
    assert result.compression == pytest.approx(1000 * input_cost)
    assert result.autorouter == 0.0
