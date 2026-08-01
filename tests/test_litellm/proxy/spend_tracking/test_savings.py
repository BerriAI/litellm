import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.proxy.spend_tracking.savings import (
    _baseline_usage,
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


def _usage(fresh: int, cached: int, written: int, out: int) -> Usage:
    """Usage as the spend log records it; `prompt_tokens` is the inclusive total."""
    return Usage(
        prompt_tokens=fresh + cached + written,
        completion_tokens=out,
        total_tokens=fresh + cached + written + out,
        prompt_tokens_details={"cached_tokens": cached, "cache_creation_tokens": written, "text_tokens": fresh},
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=written,
    )


def _savings(baseline: str, selected: str, usage: Usage) -> float:
    return compute_autorouter_savings(
        baseline_model=baseline,
        selected_model=selected,
        selected_provider="anthropic",
        usage=usage,
    )


def test_switching_models_mid_conversation_charges_the_cold_cache_write():
    """Staying on one model writes the cache once and reads it thereafter. Switching
    leaves the new model cold, so it pays to write the whole prompt again; when that
    charge outweighs the cheaper rates the route lost money and must report a loss.

    Pricing the baseline as if it too re-wrote the cache credits a charge it never
    paid, which is how a losing switch used to read as the largest saving on the page.
    """
    usage = _usage(fresh=3, cached=500, written=12304, out=500)
    result = _savings("claude-sonnet-5", "claude-haiku-4-5", usage)

    sonnet = litellm.get_model_info("claude-sonnet-5", "anthropic")
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    warm_baseline = (
        3 * sonnet["input_cost_per_token"]
        + 12804 * sonnet["cache_read_input_token_cost"]
        + 500 * sonnet["output_cost_per_token"]
    )
    actually_paid = (
        3 * haiku["input_cost_per_token"]
        + 500 * haiku["cache_read_input_token_cost"]
        + 12304 * haiku["cache_creation_input_token_cost"]
        + 500 * haiku["output_cost_per_token"]
    )
    assert result == pytest.approx(warm_baseline - actually_paid)
    assert result < 0, "a cache-thrashing switch must report a loss, not a saving"

    phantom = 12304 * sonnet["cache_creation_input_token_cost"]
    assert result != pytest.approx(warm_baseline + phantom - actually_paid)


def test_a_cold_switch_never_beats_turning_caching_off():
    """Switching to a cold model makes it write the whole prompt again. That write is a
    real cost of switching, so the same traffic must look worse than if caching were off
    entirely.

    The baseline is priced as a warm cache even though this request read nothing: a
    switch reads nothing precisely because the new model's cache is empty, and staying
    on one model would have had the prompt cached already. Gating the warm baseline on
    a read charged the baseline a write it would never repeat, which made a cold switch
    report a larger saving than no caching at all.
    """
    cold_switch = _savings("anthropic/claude-opus-5", "claude-haiku-4-5", _usage(0, 0, 20_000, 1_000))
    caching_off = _savings("anthropic/claude-opus-5", "claude-haiku-4-5", _usage(20_000, 0, 0, 1_000))

    assert cold_switch < caching_off

    opus = litellm.get_model_info("claude-opus-5", "anthropic")
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    warm_baseline = 20_000 * opus["cache_read_input_token_cost"] + 1_000 * opus["output_cost_per_token"]
    actually_paid = 20_000 * haiku["cache_creation_input_token_cost"] + 1_000 * haiku["output_cost_per_token"]
    assert cold_switch == pytest.approx(warm_baseline - actually_paid)


def test_moving_one_token_between_cache_buckets_does_not_move_the_answer():
    """A continuing conversation writes a few new tokens and reads the rest. Treating the
    presence of a write as the signal for a switch made that ordinary increment flip the
    result, so a request reading 19,999 and writing 1 landed somewhere entirely different
    from one reading 20,000 and writing none.
    """
    reads_nothing = _savings("anthropic/claude-opus-5", "claude-haiku-4-5", _usage(0, 0, 20_000, 1_000))
    reads_one = _savings("anthropic/claude-opus-5", "claude-haiku-4-5", _usage(0, 1, 19_999, 1_000))
    assert reads_one == pytest.approx(reads_nothing, abs=1e-4)


def test_multimodal_prompts_are_priced_on_the_baseline_too():
    """The baseline is this same request met by a warm cache, so every field it was
    priced on has to survive. Rebuilding the details from the cache buckets alone
    dropped the image and audio counts, which priced the baseline as a text-only
    request that never ran and shrank the reported saving on multimodal traffic.
    """
    details = {"cached_tokens": 0, "cache_creation_tokens": 16_000, "text_tokens": 0, "image_tokens": 4_000}
    with_images = Usage(
        prompt_tokens=20_000,
        completion_tokens=1_000,
        total_tokens=21_000,
        prompt_tokens_details=details,
    )
    baseline = _baseline_usage(with_images)

    assert baseline.prompt_tokens_details.image_tokens == 4_000, "image tokens must survive into the baseline"

    opus = litellm.get_model_info("claude-opus-5", "anthropic")
    priced, _ = generic_cost_per_token(model="claude-opus-5", usage=baseline, custom_llm_provider="anthropic")
    text_only = 20_000 * opus["cache_read_input_token_cost"]
    assert priced > text_only, "dropping the image tokens undercharges the baseline and hides the saving"


def test_the_baseline_is_never_charged_a_cache_write():
    """Carrying the details through must not carry the 5m/1h creation breakdown with
    them. `generic_cost_per_token` charges a creation cost whenever that breakdown is
    present, even against a zeroed creation count, which would put the phantom write
    back on the baseline for every long-cache request.
    """
    long_cache = Usage(
        prompt_tokens=20_000,
        completion_tokens=1_000,
        total_tokens=21_000,
        prompt_tokens_details={
            "cached_tokens": 0,
            "cache_creation_tokens": 20_000,
            "text_tokens": 0,
            "cache_creation_token_details": {"ephemeral_1h_input_tokens": 20_000},
        },
    )
    baseline = _baseline_usage(long_cache)

    opus = litellm.get_model_info("claude-opus-5", "anthropic")
    priced, _ = generic_cost_per_token(model="claude-opus-5", usage=baseline, custom_llm_provider="anthropic")
    assert priced == pytest.approx(20_000 * opus["cache_read_input_token_cost"]), (
        "the baseline reads a warm cache; it never pays to create one"
    )


def test_uncached_request_is_the_plain_rate_difference():
    usage = _usage(fresh=2000, cached=0, written=0, out=500)
    sonnet = litellm.get_model_info("claude-sonnet-5", "anthropic")
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    assert _savings("claude-sonnet-5", "claude-haiku-4-5", usage) == pytest.approx(
        2000 * (sonnet["input_cost_per_token"] - haiku["input_cost_per_token"])
        + 500 * (sonnet["output_cost_per_token"] - haiku["output_cost_per_token"])
    )


def test_escalation_reports_its_real_cost():
    """Routing up to a pricier model is a real cost; hiding it behind a zero floor
    would let the dashboard only ever move in one direction."""
    usage = _usage(fresh=2000, cached=0, written=0, out=500)
    assert _savings("claude-haiku-4-5", "claude-sonnet-5", usage) < 0


def test_autorouter_savings_zero_when_model_unchanged():
    assert _savings("claude-opus-5", "claude-opus-5", _usage(3, 500, 12304, 500)) == 0.0


def test_autorouter_savings_unknown_baseline_fails_open_to_zero():
    assert _savings("totally-made-up-model-xyz", "claude-haiku-4-5", _usage(3, 500, 12304, 500)) == 0.0


def test_autorouter_savings_zero_without_baseline():
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        cache_read_input_tokens=0,
        baseline_model=None,
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter == 0.0


def test_compute_savings_spend_carries_a_losing_switch_through():
    """The signed value must survive into SavingsSpend; clamping it here would put the
    dashboard back to only ever showing gains."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        cache_read_input_tokens=0,
        baseline_model="claude-sonnet-5",
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter < 0


def test_malformed_usage_object_does_not_fail_the_spend_write():
    """The daily spend write must survive an unusable usage_object; losing one row's
    savings is recoverable, losing the row is not."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        cache_read_input_tokens=0,
        baseline_model="claude-opus-5",
        usage_object={"prompt_tokens": ["not", "a", "number"]},
    )
    assert result.autorouter == 0.0
    assert result.compression > 0


def test_model_without_cache_read_pricing_yields_no_caching_savings():
    """A model with no discounted cache-read rate cannot have saved anything by
    reading from cache, so the driver must report zero rather than the full input rate."""
    model = "azure/gpt-3.5-turbo"
    assert litellm.get_model_info(model=model).get("cache_read_input_token_cost") is None
    result = compute_savings_spend(
        model=model,
        custom_llm_provider="azure",
        compression_saved_tokens=0,
        cache_read_input_tokens=5000,
    )
    assert result.prompt_caching == 0.0


def test_the_same_deployment_spelled_two_ways_is_not_a_switch():
    """The spend log records a normalized model name while the baseline arrives as the
    operator wrote it in config. Comparing the raw strings makes a request that never
    changed model look like a switch, and prices one deployment against itself."""
    # Must be a cached request: the baseline arm is priced against a warm cache and the
    # selected arm against what was actually paid, so treating one deployment as two
    # charges it a cold-cache write it never took, inventing a loss on a request that
    # never changed model. An uncached request prices identically either way and would
    # make this assertion vacuous.
    usage = _usage(fresh=3, cached=500, written=12304, out=500)
    assert _savings("anthropic/claude-opus-5", "claude-opus-5", usage) == 0.0
    assert _savings("claude-opus-5", "anthropic/claude-opus-5", usage) == 0.0


def test_baseline_is_priced_under_its_own_provider():
    """Two providers can serve the same bare model name at different rates, so dropping
    the provider prices the baseline against a vendor the operator never named. Here it
    decides whether routing reads as a saving or a loss."""
    usage = Usage(prompt_tokens=100_000, completion_tokens=10_000, total_tokens=110_000)
    azure = compute_autorouter_savings(
        baseline_model="azure_ai/deepseek-r1",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=usage,
    )
    deepseek = compute_autorouter_savings(
        baseline_model="deepseek/deepseek-r1",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=usage,
    )
    assert azure != pytest.approx(deepseek)
    assert azure > 0 > deepseek


def test_unresolvable_baseline_fails_open_to_zero():
    usage = _usage(fresh=2000, cached=0, written=0, out=500)
    assert _savings("no-such-provider-xyz/no-such-model", "claude-haiku-4-5", usage) == 0.0
