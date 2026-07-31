import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

import litellm
from litellm.proxy.spend_tracking.savings import (
    compute_autorouter_savings,
    compute_savings_spend,
)


def _anthropic_costs(model: str) -> tuple[float, float]:
    info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
    input_cost = info["input_cost_per_token"] or 0.0
    cache_read_cost = info.get("cache_read_input_token_cost") or input_cost
    return input_cost, cache_read_cost


def _rates(model: str) -> tuple[float, float, float]:
    info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
    input_cost = info["input_cost_per_token"] or 0.0
    output_cost = info["output_cost_per_token"] or 0.0
    cache_write_cost = info.get("cache_creation_input_token_cost") or input_cost
    return input_cost, output_cost, cache_write_cost


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


def test_autorouter_savings_prices_completion_at_output_rate():
    # Completion tokens must be priced at each model's OUTPUT rate, not its input
    # rate. Output is several times input on flagship models, so pricing
    # completions at the input rate materially understates the routed savings.
    base_in, base_out, _ = _rates("claude-opus-5")
    sel_in, sel_out, sel_cache_write = _rates("claude-haiku-4-5")
    assert base_out > base_in  # otherwise this test asserts nothing
    prompt_tokens, completion_tokens, cache_creation = 1000, 500, 200

    result = compute_autorouter_savings(
        baseline_model="claude-opus-5",
        selected_model="claude-haiku-4-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_creation_input_tokens=cache_creation,
    )

    baseline_cost = prompt_tokens * base_in + completion_tokens * base_out
    selected_cost = prompt_tokens * sel_in + completion_tokens * sel_out
    penalty = cache_creation * sel_cache_write
    assert result == pytest.approx(baseline_cost - selected_cost - penalty)

    # A mutant that priced completions at the input rate would land here instead.
    wrong = (prompt_tokens * base_in + completion_tokens * base_in) - (
        prompt_tokens * sel_in + completion_tokens * sel_in
    ) - cache_creation * base_in
    assert result != pytest.approx(wrong)


def test_autorouter_cache_write_penalty_uses_selected_model_rate():
    # The switch penalty is a cache-creation charge on the deployment actually
    # written to (the selected model), priced at its cache-creation rate, not the
    # baseline's input rate.
    base_in, base_out, _ = _rates("claude-opus-5")
    sel_in, sel_out, sel_cache_write = _rates("claude-haiku-4-5")
    prompt_tokens, completion_tokens, cache_creation = 0, 0, 1000

    result = compute_autorouter_savings(
        baseline_model="claude-opus-5",
        selected_model="claude-haiku-4-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_creation_input_tokens=cache_creation,
    )
    # With no prompt/completion tokens, savings is purely the negative penalty,
    # floored at zero, so a pure-penalty request never reads as savings.
    assert result == 0.0

    # Confirm the penalty magnitude uses the SELECTED model's cache-write rate by
    # giving enough token delta to stay positive, then isolating the penalty.
    prompt_tokens = 100_000
    with_penalty = compute_autorouter_savings(
        "claude-opus-5", "claude-haiku-4-5", "anthropic", "anthropic",
        prompt_tokens, 0, cache_creation,
    )
    without_penalty = compute_autorouter_savings(
        "claude-opus-5", "claude-haiku-4-5", "anthropic", "anthropic",
        prompt_tokens, 0, 0,
    )
    assert without_penalty - with_penalty == pytest.approx(cache_creation * sel_cache_write)


def test_autorouter_savings_zero_when_model_unchanged():
    result = compute_autorouter_savings(
        baseline_model="claude-opus-5",
        selected_model="claude-opus-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_creation_input_tokens=0,
    )
    assert result == 0.0


def test_autorouter_savings_floored_at_zero_on_escalation():
    # Routing UP to a pricier model must never show as negative savings.
    result = compute_autorouter_savings(
        baseline_model="claude-haiku-4-5",
        selected_model="claude-opus-5",
        baseline_provider="anthropic",
        selected_provider="anthropic",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_creation_input_tokens=0,
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
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert result.autorouter == 0.0


def test_compute_savings_spend_includes_autorouter_driver():
    base_in, base_out, _ = _rates("claude-opus-5")
    sel_in, sel_out, _ = _rates("claude-haiku-4-5")
    prompt_tokens, completion_tokens = 2000, 800

    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        cache_read_input_tokens=0,
        baseline_model="claude-opus-5",
        baseline_provider="anthropic",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_creation_input_tokens=0,
    )
    expected = (prompt_tokens * base_in + completion_tokens * base_out) - (
        prompt_tokens * sel_in + completion_tokens * sel_out
    )
    assert result.autorouter == pytest.approx(expected)
    assert result.autorouter > 0
