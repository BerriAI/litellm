

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.proxy.spend_tracking.savings import (
    _baseline_usage,
    compute_autorouter_savings,
    compute_savings_spend,
)
from litellm.router import Router
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
        usage_object={"cache_read_input_tokens": 8200},
    )
    assert result.prompt_caching == pytest.approx(8200 * (input_cost - cache_read_cost))
    assert result.prompt_caching > 0
    assert result.compression == 0.0


def _net_caching_savings_against_biller(usage_object: dict, model: str = "claude-sonnet-5") -> float:
    """True net caching savings, priced by the real cost calculator.

    Bills the request as it happened, then bills the same token total with nothing
    cached, and returns the difference. Deriving the expectation from
    ``generic_cost_per_token`` rather than restating the formula is what makes these
    tests able to fail: a wrong formula in savings.py cannot also be wrong here.
    """
    prompt_tokens = usage_object["prompt_tokens"]
    uncached = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage_object["completion_tokens"],
        "total_tokens": prompt_tokens + usage_object["completion_tokens"],
        "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 0, "text_tokens": prompt_tokens},
    }
    return _cost_on(model, uncached) - _cost_on(model, usage_object)


def _caching_usage(read: int, written: int, text: int = 10, out: int = 100) -> dict:
    prompt_tokens = text + read + written
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": out,
        "total_tokens": prompt_tokens + out,
        "prompt_tokens_details": {
            "cached_tokens": read,
            "cache_creation_tokens": written,
            "text_tokens": text,
        },
        "cache_creation_input_tokens": written,
        "cache_read_input_tokens": read,
    }


def test_prompt_caching_savings_nets_out_the_cache_write_premium():
    """A cache-writing request is only credited the read discount minus the write premium."""
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    _, _, cache_write_cost = _flat_rates("claude-sonnet-5")
    # Anthropic charges a premium to write; without it this test asserts nothing.
    assert cache_write_cost > input_cost
    usage_object = _caching_usage(read=20000, written=500)
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object=usage_object,
    )
    assert result.prompt_caching == pytest.approx(_net_caching_savings_against_biller(usage_object))
    # Strictly less than the gross read discount, which is what shipped before.
    assert result.prompt_caching < 20000 * (input_cost - cache_read_cost)
    assert result.prompt_caching > 0


def test_prompt_caching_savings_go_negative_on_a_write_only_request():
    """A cold turn that writes cache and gets no hits genuinely cost more than not caching."""
    usage_object = _caching_usage(read=0, written=20000)
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object=usage_object,
    )
    true_savings = _net_caching_savings_against_biller(usage_object)
    assert true_savings < 0
    assert result.prompt_caching == pytest.approx(true_savings)
    assert result.prompt_caching < 0


def test_prompt_caching_savings_negative_when_writes_outweigh_reads():
    """The wrong-sign case: a few hits against a big write bill is still a net loss."""
    usage_object = _caching_usage(read=1000, written=20000)
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object=usage_object,
    )
    true_savings = _net_caching_savings_against_biller(usage_object)
    assert true_savings < 0
    assert result.prompt_caching == pytest.approx(true_savings)
    # The gross formula reported this as a saving; the sign itself is the regression.
    assert result.prompt_caching < 0


def test_read_only_request_is_unchanged_by_the_write_premium():
    """No cache writes means nothing to net out, so the read discount stands alone."""
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object=_caching_usage(read=20000, written=0),
    )
    assert result.prompt_caching == pytest.approx(20000 * (input_cost - cache_read_cost))


def test_openai_style_cache_write_tokens_are_netted_out():
    """Providers reporting writes under prompt_tokens_details are netted the same way."""
    _, _, cache_write_cost = _flat_rates("claude-sonnet-5")
    input_cost, _ = _anthropic_costs("claude-sonnet-5")
    with_top_level = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object={"cache_read_input_tokens": 5000, "cache_creation_input_tokens": 800},
    )
    nested_only = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object={
            "prompt_tokens_details": {"cached_tokens": 5000, "cache_write_tokens": 800},
        },
    )
    assert nested_only.prompt_caching == pytest.approx(with_top_level.prompt_caching)
    assert nested_only.prompt_caching == pytest.approx(
        5000 * (input_cost - _anthropic_costs("claude-sonnet-5")[1]) - 800 * (cache_write_cost - input_cost)
    )


def test_model_without_a_cache_write_price_takes_no_premium():
    """An absent write price must mean zero premium, never a bonus.

    ``_get_cost_per_unit`` in the cost calculator defaults a missing price to 0.0. Were
    that default copied here the premium would be ``0 - input_cost``, and a model with no
    write pricing would report cache writes as free money. This is the common case: most
    of the pricing map publishes a cache-read price and no cache-write price.
    """
    model = "amazon.nova-2-lite-v1:0"
    info = litellm.get_model_info(model=model)
    input_cost = info["input_cost_per_token"]
    cache_read_cost = info["cache_read_input_token_cost"]
    assert info.get("cache_creation_input_token_cost") is None, (
        "fixture drifted: this test needs a model that publishes no cache-write price"
    )

    result = compute_savings_spend(
        model=model,
        custom_llm_provider=None,
        compression_saved_tokens=0,
        usage_object=_caching_usage(read=5000, written=5000),
    )
    assert result.prompt_caching == pytest.approx(5000 * (input_cost - cache_read_cost))
    assert result.prompt_caching > 0


def test_zero_cache_write_price_is_read_as_unpublished():
    """A ``0.0`` write price means "no separate price", not "writes are free".

    ``deepseek-chat`` carries an explicit zero in the pricing map. Taken literally the
    premium would be ``0 - input_cost``, paying out a saving of ``writes * input_cost``
    on traffic that cached nothing. No provider gives cache writes away, so a falsy
    price falls open to the input cost like an absent one does.
    """
    info = litellm.get_model_info(model="deepseek-chat", custom_llm_provider="deepseek")
    assert info.get("cache_creation_input_token_cost") == 0.0, (
        "fixture drifted: this test exists because deepseek-chat publishes a literal 0.0 write price"
    )

    result = compute_savings_spend(
        model="deepseek-chat",
        custom_llm_provider="deepseek",
        compression_saved_tokens=0,
        usage_object=_caching_usage(read=0, written=10000),
    )
    assert result.prompt_caching == pytest.approx(0.0)


def test_zero_cache_read_price_stays_literal():
    """The read leg must NOT copy the write leg's falsy fall-open.

    The two zeros mean opposite things. A free cache *write* is unpublished pricing, so
    it falls open to input. A free cache *read* is real and is the largest discount
    available -- 15 models charge for input and serve reads for nothing. Falling that
    open to the input cost would zero out their savings entirely.
    """
    model = "gemini-robotics-er-1.5-preview"
    info = litellm.get_model_info(model=model)
    input_cost = info["input_cost_per_token"]
    assert info.get("cache_read_input_token_cost") == 0.0 and input_cost > 0, (
        "fixture drifted: this test needs a model with paid input and free cache reads"
    )

    result = compute_savings_spend(
        model=model,
        custom_llm_provider=None,
        compression_saved_tokens=0,
        usage_object=_caching_usage(read=10000, written=0),
    )
    # free reads => the whole input rate is saved, not zero
    assert result.prompt_caching == pytest.approx(10000 * input_cost)


def test_sub_input_cache_write_price_is_an_extra_saving():
    """A few models price writes below input; there the premium is a real credit.

    Clamping the premium at zero would silently undercount these, so the subtraction
    stays signed. ``azure/eu/gpt-4o-2024-11-20`` ships a write price at ~0.5x input.
    """
    model = "azure/eu/gpt-4o-2024-11-20"
    info = litellm.get_model_info(model=model)
    input_cost = info["input_cost_per_token"]
    cheap_write = info["cache_creation_input_token_cost"]
    assert 0 < cheap_write < input_cost, "fixture drifted: this test needs a model pricing cache writes below input"
    # no published read price, so the read leg mirrors input and contributes nothing;
    # the whole result is the negative premium, i.e. a credit.
    assert info.get("cache_read_input_token_cost") is None

    result = compute_savings_spend(
        model=model,
        custom_llm_provider=None,
        compression_saved_tokens=0,
        usage_object=_caching_usage(read=1000, written=4000),
    )
    assert result.prompt_caching == pytest.approx(4000 * (input_cost - cheap_write))
    assert result.prompt_caching > 0


def test_negative_cache_write_count_clamps_to_zero():
    """A malformed negative write count must not be read as a saving."""
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object={"cache_read_input_tokens": 1000, "cache_creation_input_tokens": -5000},
    )
    assert result.prompt_caching == pytest.approx(1000 * (input_cost - cache_read_cost))


def test_unknown_model_fails_open_to_zero():
    result = compute_savings_spend(
        model="totally-made-up-model-xyz",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        usage_object={"cache_read_input_tokens": 1000},
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_missing_model_fails_open_to_zero():
    result = compute_savings_spend(
        model=None,
        custom_llm_provider=None,
        compression_saved_tokens=1000,
        usage_object={"cache_read_input_tokens": 1000},
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_negative_token_counts_clamp_to_zero():
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=-500,
        usage_object={"cache_read_input_tokens": -500},
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


def _savings(baseline: str, selected: str, usage: Usage, continuing: bool = True) -> float:
    """Savings for a request, defaulting to a conversation already underway.

    `continuing=True` is the mid-conversation case, where the baseline had the prompt
    cached and this request's write is what the switch cost. `continuing=False` is a
    conversation's first turn, where nothing was cached for any model.
    """
    return compute_autorouter_savings(
        baseline_model=baseline,
        selected_model=selected,
        selected_provider="anthropic",
        usage=usage,
        conversation_continuing=continuing,
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
    baseline = _baseline_usage(with_images, conversation_continuing=True)

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
    baseline = _baseline_usage(long_cache, conversation_continuing=True)

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
        routing_decision=None,
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter == 0.0


def test_compute_savings_spend_carries_a_losing_switch_through(monkeypatch):
    """The signed value must survive into SavingsSpend; clamping it here would put the
    dashboard back to only ever showing gains."""
    monkeypatch.setattr(litellm, "autorouter_savings_baseline_model", "claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision={"conversation_continuing": True},
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter < 0


def test_the_driver_is_off_until_a_baseline_is_configured():
    """No configured counterfactual means there is nothing to measure against, so the
    driver reports zero rather than inventing a model the operator never named."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        routing_decision={"conversation_continuing": True},
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter == 0.0
    assert result.compression > 0, "the other drivers keep working"


def test_malformed_usage_object_does_not_fail_the_spend_write():
    """The daily spend write must survive an unusable usage_object; losing one row's
    savings is recoverable, losing the row is not."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        routing_decision={"conversation_continuing": True},
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
        usage_object={"cache_read_input_tokens": 5000},
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


def test_a_first_turn_is_the_rate_difference_not_a_switch_penalty():
    """Nothing was cached anywhere on a conversation's first turn, so the baseline would
    have paid the same cache write. Charging it to the selected arm alone reported a
    fraction of the real saving; on this shape roughly 4% of it.
    """
    usage = _usage(fresh=0, cached=0, written=20_000, out=1_000)
    first_turn = _savings("anthropic/claude-opus-5", "claude-haiku-4-5", usage, continuing=False)

    opus = litellm.get_model_info("claude-opus-5", "anthropic")
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    both_write = (20_000 * opus["cache_creation_input_token_cost"] + 1_000 * opus["output_cost_per_token"]) - (
        20_000 * haiku["cache_creation_input_token_cost"] + 1_000 * haiku["output_cost_per_token"]
    )
    assert first_turn == pytest.approx(both_write)

    mid_conversation = _savings("anthropic/claude-opus-5", "claude-haiku-4-5", usage)
    assert first_turn > mid_conversation * 10, "a first turn must not be priced as a switch"


def test_a_first_turn_that_saves_money_never_reports_a_loss():
    """The write premium is fixed by prompt size while the saving grows with completion
    length, so charging the write to a first turn made short answers over a large cached
    prompt read as losses on requests that genuinely saved. That is the shape most likely
    to be on the dashboard, and the sign has to be right.
    """
    short_answer = _usage(fresh=0, cached=0, written=20_000, out=200)
    assert _savings("anthropic/claude-opus-5", "claude-haiku-4-5", short_answer, continuing=False) > 0
    assert _savings("anthropic/claude-opus-5", "claude-haiku-4-5", short_answer) < 0


def test_an_undetermined_conversation_shape_stays_conservative():
    """The default must charge the write. A caller that cannot be read, or a surface the
    router never classified, has said nothing about whether the baseline was warm, and a
    savings figure must not inflate on a guess.
    """
    usage = _usage(fresh=0, cached=0, written=20_000, out=1_000)
    defaulted = compute_autorouter_savings(
        baseline_model="anthropic/claude-opus-5",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=usage,
    )
    assert defaulted == pytest.approx(_savings("anthropic/claude-opus-5", "claude-haiku-4-5", usage))
    assert defaulted < _savings("anthropic/claude-opus-5", "claude-haiku-4-5", usage, continuing=False)


def test_a_continuing_turn_on_the_same_model_writes_its_growth_on_both_arms():
    """A conversation that grew by a few tokens writes those on whatever model serves
    it, and they are new to every model, so the baseline would have written them too.
    Moving them into the baseline's read bucket forgives it a write it really owes and
    shrinks the reported saving on ordinary steady-state traffic.
    """
    usage = _usage(fresh=0, cached=19_900, written=100, out=1_000)
    opus = litellm.get_model_info("claude-opus-5", "anthropic")
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")

    def cost(info: dict) -> float:
        return (
            19_900 * info["cache_read_input_token_cost"]
            + 100 * info["cache_creation_input_token_cost"]
            + 1_000 * info["output_cost_per_token"]
        )

    both_write_the_growth = cost(opus) - cost(haiku)
    assert _savings("anthropic/claude-opus-5", "claude-haiku-4-5", usage) == pytest.approx(both_write_the_growth)


def test_a_switch_onto_a_partly_cached_model_still_pays_for_the_write():
    """A model holding a small prefix of this prompt still has to write the rest, and
    that write is the switch's cost. Keying the same-model case off reading *anything*
    rather than reading *most of it* would hand this request the full rate gap and
    inflate the saving by an order of magnitude.
    """
    mostly_written = _usage(fresh=0, cached=500, written=19_500, out=1_000)
    reported = _savings("anthropic/claude-opus-5", "claude-haiku-4-5", mostly_written)

    opus = litellm.get_model_info("claude-opus-5", "anthropic")
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    if_treated_as_same_model = (
        500 * opus["cache_read_input_token_cost"]
        + 19_500 * opus["cache_creation_input_token_cost"]
        + 1_000 * opus["output_cost_per_token"]
    ) - (
        500 * haiku["cache_read_input_token_cost"]
        + 19_500 * haiku["cache_creation_input_token_cost"]
        + 1_000 * haiku["output_cost_per_token"]
    )
    assert reported < if_treated_as_same_model / 10, "a mostly-cold switch must not be priced as a continuation"


def test_a_baseline_that_prices_caching_implicitly_still_pays_for_its_prompt():
    """OpenAI, Azure and Gemini entries carry no `cache_creation_input_token_cost`,
    because those providers cache implicitly and charge nothing to write. Leaving this
    request's written tokens in the creation bucket priced them at the 0.0 the cost
    resolver falls back to, so the baseline carried a 20k prompt for free and a first
    turn that saved money reported a loss. Those tokens are plain input on such a model.
    """
    first_turn = _usage(fresh=0, cached=0, written=20_000, out=1_000)
    reported = compute_autorouter_savings(
        baseline_model="gpt-5",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=first_turn,
        conversation_continuing=False,
    )

    gpt5 = litellm.get_model_info("gpt-5", "openai")
    assert gpt5.get("cache_creation_input_token_cost") is None, "pick a baseline with no cache-write rate"
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    baseline_pays_input = 20_000 * gpt5["input_cost_per_token"] + 1_000 * gpt5["output_cost_per_token"]
    actually_paid = 20_000 * haiku["cache_creation_input_token_cost"] + 1_000 * haiku["output_cost_per_token"]
    assert reported == pytest.approx(baseline_pays_input - actually_paid)
    assert reported > 0, "routing a cold first turn onto a cheaper model is a saving, not a loss"


def test_a_baseline_with_no_cache_read_rate_is_charged_its_input_rate():
    """The same hole on the other bucket. A baseline whose entry has no
    `cache_read_input_token_cost` reads for 0.0, so a continuing turn priced the whole
    prompt at nothing and every switch away from it reported a loss.
    """
    continuing = _usage(fresh=0, cached=0, written=20_000, out=1_000)
    reported = compute_autorouter_savings(
        baseline_model="xai/grok-4",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=continuing,
        conversation_continuing=True,
    )

    grok = litellm.get_model_info("grok-4", "xai")
    assert grok.get("cache_read_input_token_cost") is None, "pick a baseline with no cache-read rate"
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    baseline_pays_input = 20_000 * grok["input_cost_per_token"] + 1_000 * grok["output_cost_per_token"]
    actually_paid = 20_000 * haiku["cache_creation_input_token_cost"] + 1_000 * haiku["output_cost_per_token"]
    assert reported == pytest.approx(baseline_pays_input - actually_paid)


def _breakdown(input_cost: float, output_cost: float = 0.0, **extra: object) -> dict:
    """A `cost_breakdown` as the cost calculator records it on the spend log."""
    return {"input_cost": input_cost, "output_cost": output_cost, **extra}


def test_the_served_arm_is_read_from_the_record_not_repriced():
    """What the request cost on the model that served it is not a counterfactual; the
    cost calculator already billed it and wrote the number down. Recomputing it restates
    every pricing dimension the biller applied and drops the ones it forgets, so the
    driver disagrees with the `spend` column beside it.

    Pinned with a negotiated rate no public map lookup can produce, so re-pricing from
    the model name cannot land on this number. Tool spend and margin are recorded too and
    must stay out: the baseline cannot be priced with them, so charging them to the
    served arm alone would read as the router losing money on every tool call.
    """
    usage = _usage(fresh=20_000, cached=0, written=0, out=1_000)
    negotiated_input, negotiated_output = 0.0123, 0.0456

    reported = compute_autorouter_savings(
        baseline_model="anthropic/claude-opus-5",
        selected_model="gpt-5.5",
        selected_provider="openai",
        usage=usage,
        conversation_continuing=False,
        cost_breakdown=_breakdown(
            negotiated_input,
            negotiated_output,
            tool_usage_cost=5.0,
            margin_total_amount=2.0,
            total_cost=negotiated_input + negotiated_output + 7.0,
        ),
    )

    opus = litellm.get_model_info("claude-opus-5", "anthropic")
    public = 20_000 * opus["input_cost_per_token"] + 1_000 * opus["output_cost_per_token"]
    assert reported == pytest.approx(public - (negotiated_input + negotiated_output))


@pytest.mark.parametrize(
    "basis, expected_multiplier",
    [
        pytest.param({"service_tier": "priority"}, 2.0, id="priority tier doubles the baseline"),
        pytest.param({"data_residency": "eu"}, 1.1, id="eu residency uplifts the baseline"),
        pytest.param({}, 1.0, id="no basis recorded prices at standard"),
        pytest.param(None, 1.0, id="row predating the field prices at standard"),
        pytest.param({"service_tier": True, "data_residency": 17}, 1.0, id="a non-string basis is dropped"),
    ],
)
def test_the_baseline_is_priced_on_the_basis_the_request_was_billed_at(basis, expected_multiplier):
    """A request billed at a priority tier, or through a regional host, would have been
    billed the same way on the single model an operator ran instead of the router, so the
    counterfactual carries that basis too. Dropping it prices the two arms from different
    books; neither multiplier cancels out of the difference, because both are per-model.

    The served model has no tiered rates and no uplift of its own, so only the baseline
    can move: a fix that forwards the basis to the served arm alone leaves these numbers
    unchanged. The non-string case guards the JSON round trip, where `.lower()` inside
    the pricer would raise and be swallowed into a silent $0.00 for the whole row.
    """
    gpt = litellm.get_model_info("gpt-5.5", "openai")
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    assert gpt.get("input_cost_per_token_priority") == 2 * gpt["input_cost_per_token"]
    assert gpt.get("regional_processing_uplift_multiplier_eu") == 1.1
    assert haiku.get("input_cost_per_token_priority") is None, "served model must not move with the basis"
    assert haiku.get("regional_processing_uplift_multiplier_eu") is None

    usage = _usage(fresh=20_000, cached=0, written=0, out=1_000)
    served = 20_000 * haiku["input_cost_per_token"] + 1_000 * haiku["output_cost_per_token"]

    reported = compute_autorouter_savings(
        baseline_model="openai/gpt-5.5",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=usage,
        conversation_continuing=False,
        cost_breakdown=None if basis is None else _breakdown(served, **basis),
    )

    baseline = 20_000 * gpt["input_cost_per_token"] + 1_000 * gpt["output_cost_per_token"]
    assert reported == pytest.approx(expected_multiplier * baseline - served)


def test_the_baseline_is_priced_on_the_vertex_location_the_request_was_billed_at(monkeypatch):
    """A request served from a regional Vertex endpoint was billed with the
    regional-endpoint uplift, so the counterfactual single-model operator would
    have paid it too. The served model carries no uplift field, so only the
    baseline moves with the recorded location."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    gemini = litellm.get_model_info("gemini-3.5-flash", "vertex_ai")
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    assert gemini.get("regional_endpoint_uplift_multiplier") == 1.1
    assert haiku.get("regional_endpoint_uplift_multiplier") is None, "served model must not move with the basis"

    usage = _usage(fresh=20_000, cached=0, written=0, out=1_000)
    served = 20_000 * haiku["input_cost_per_token"] + 1_000 * haiku["output_cost_per_token"]
    baseline = 20_000 * gemini["input_cost_per_token"] + 1_000 * gemini["output_cost_per_token"]

    regional = compute_autorouter_savings(
        baseline_model="vertex_ai/gemini-3.5-flash",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=usage,
        conversation_continuing=False,
        cost_breakdown=_breakdown(served, vertex_location="us-east5"),
    )
    global_endpoint = compute_autorouter_savings(
        baseline_model="vertex_ai/gemini-3.5-flash",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=usage,
        conversation_continuing=False,
        cost_breakdown=_breakdown(served, vertex_location="global"),
    )

    assert regional == pytest.approx(1.1 * baseline - served)
    assert global_endpoint == pytest.approx(baseline - served)


def test_a_baseline_recorded_on_the_decision_turns_the_driver_on():
    """An operator who configures nothing still sees the driver work."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision={"conversation_continuing": True, "savings_baseline_model": "anthropic/claude-opus-5"},
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter != 0.0


def test_the_configured_baseline_overrides_the_recorded_one(monkeypatch):
    """The recorded baseline and its deployment id are both ignored under the setting."""
    monkeypatch.setattr(litellm, "autorouter_savings_baseline_model", "claude-sonnet-5")
    with_override = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision={
            "conversation_continuing": True,
            "savings_baseline_model": "anthropic/claude-opus-5",
            "savings_baseline_deployment_id": "some-deployment-id",
        },
        usage_object=_cached_usage_object(),
    )
    against_sonnet = compute_autorouter_savings(
        baseline_model="claude-sonnet-5",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=Usage(**_cached_usage_object()),
    )
    against_opus = compute_autorouter_savings(
        baseline_model="anthropic/claude-opus-5",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=Usage(**_cached_usage_object()),
    )
    assert against_sonnet != against_opus, "the test needs baselines that price apart"
    assert with_override.autorouter == against_sonnet


def test_a_non_string_recorded_baseline_is_ignored():
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision={"conversation_continuing": True, "savings_baseline_model": ["anthropic/claude-opus-5"]},
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter == 0.0


def test_prompt_caching_prices_at_the_deployment_rate_not_the_public_one():
    """A deployment's negotiated cache rates are what it really pays.

    Pricing the write premium off the public map instead reports a loss ~3x the real
    one here, which is the whole point of resolving deployment pricing first.
    """
    router = Router(
        model_list=[
            {
                "model_name": "cheap-sonnet",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "input_cost_per_token": 1e-06,
                    "cache_creation_input_token_cost": 1.25e-06,
                    "cache_read_input_token_cost": 1e-07,
                },
            },
        ]
    )
    deployment_id = router.get_model_list(model_name="cheap-sonnet")[0]["model_info"]["id"]

    result = compute_savings_spend(
        model="claude-sonnet-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object=_caching_usage(read=1000, written=20000),
        model_id=deployment_id,
        llm_router=lambda: router,
    )
    at_deployment_rates = 1000 * (1e-06 - 1e-07) - 20000 * (1.25e-06 - 1e-06)
    assert result.prompt_caching == pytest.approx(at_deployment_rates)

    at_public_rates = compute_savings_spend(
        model="claude-sonnet-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        usage_object=_caching_usage(read=1000, written=20000),
    )
    assert result.prompt_caching > at_public_rates.prompt_caching


def test_a_recorded_baseline_deployment_prices_at_its_configured_rate():
    """A hardest-tier deployment with a negotiated rate is what the traffic would
    really have cost; pricing its model publicly misstates the saving."""
    router = Router(
        model_list=[
            {
                "model_name": "top",
                "litellm_params": {
                    "model": "anthropic/claude-opus-5",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
        ]
    )
    deployment_id = router.get_model_list(model_name="top")[0]["model_info"]["id"]
    decision = {
        "conversation_continuing": True,
        "savings_baseline_model": "anthropic/claude-opus-5",
        "savings_baseline_deployment_id": deployment_id,
    }
    with_deployment_rate = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision=decision,
        usage_object=_cached_usage_object(),
        llm_router=lambda: router,
    )
    at_public_rate = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision={k: v for k, v in decision.items() if k != "savings_baseline_deployment_id"},
        usage_object=_cached_usage_object(),
        llm_router=lambda: router,
    )
    assert with_deployment_rate.autorouter > at_public_rate.autorouter


def _routed_decision() -> dict:
    return {"savings_baseline_model": "anthropic/claude-opus-5", "conversation_continuing": True}


def test_recorded_savings_win_over_recomputation():
    """The figure the logging path stamped is the one the rollup keeps, so the
    per-request record and the daily rollup cannot disagree."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision=_routed_decision(),
        usage_object=_cached_usage_object(),
        recorded_autorouter_savings=0.5,
    )
    assert result.autorouter == 0.5


def test_recorded_savings_survive_an_unusable_usage_object():
    """A recorded figure was computed when the usage still parsed; a later row whose
    usage_object no longer does must keep the number, not zero it."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision=_routed_decision(),
        usage_object={"prompt_tokens": ["not", "a", "number"]},
        recorded_autorouter_savings=0.25,
    )
    assert result.autorouter == 0.25


def test_a_boolean_is_not_a_recorded_savings_figure():
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision=None,
        usage_object=_cached_usage_object(),
        recorded_autorouter_savings=True,
    )
    assert result.autorouter == 0.0


def test_rows_written_before_the_field_shipped_recompute():
    """No recorded figure means the row predates the logging-path stamp; the writer
    recomputes exactly what the one shared helper would have recorded."""
    from litellm.proxy.spend_tracking.savings import autorouter_savings_for_request

    recomputed = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        routing_decision=_routed_decision(),
        usage_object=_cached_usage_object(),
    )
    direct = autorouter_savings_for_request(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        routing_decision=_routed_decision(),
        usage_object=_cached_usage_object(),
    )
    assert direct is not None and direct != 0.0
    assert recomputed.autorouter == direct


def test_driver_off_is_none_not_zero_for_the_request_helper():
    """None and 0.0 are different facts on the logging payload: absence means the
    request was never auto-routed, zero is a real figure for a routed request."""
    from litellm.proxy.spend_tracking.savings import autorouter_savings_for_request

    assert (
        autorouter_savings_for_request(
            model="claude-haiku-4-5",
            custom_llm_provider="anthropic",
            routing_decision=None,
            usage_object=_cached_usage_object(),
        )
        is None
    )
    assert (
        autorouter_savings_for_request(
            model="claude-haiku-4-5",
            custom_llm_provider="anthropic",
            routing_decision={"conversation_continuing": True},
            usage_object=_cached_usage_object(),
        )
        is None
    )


def test_logging_payload_never_stamps_internal_calls():
    """Shadow eval and classifier sub-calls carry a real routing decision but are not
    requests the caller made; a stamped figure would report savings for traffic no
    user sent, which the spend writer deliberately zeroes."""
    from litellm.proxy.spend_tracking.savings import autorouter_savings_for_logging_payload

    routed_metadata = {"routing_decision": _routed_decision()}
    stamped = autorouter_savings_for_logging_payload(
        request_metadata=routed_metadata,
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        model_id=None,
        usage_object=_cached_usage_object(),
        cost_breakdown=None,
    )
    assert stamped is not None and stamped != 0.0

    internal = autorouter_savings_for_logging_payload(
        request_metadata={**routed_metadata, "internal_call_origin": "shadow_eval_shadow"},
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        model_id=None,
        usage_object=_cached_usage_object(),
        cost_breakdown=None,
    )
    assert internal is None
