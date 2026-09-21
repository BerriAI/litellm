from typing import Final, Literal

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.anthropic.cost_calculation import cost_per_token as anthropic_cost_per_token
from litellm.proxy.spend_tracking.savings import (
    _baseline_usage,
    _resolve_model,
    compute_autorouter_savings,
    compute_savings_spend,
    marks_gateway_injection,
)
from litellm.router import Router
from litellm.types.utils import Usage

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


@pytest.mark.parametrize("modifier", [{"speed": "fast"}, {"inference_geo": "us"}])
@pytest.mark.parametrize("continuing", [False, True])
def test_baseline_preserves_anthropic_pricing_fields(modifier: dict[str, str], continuing: bool) -> None:
    usage: Final = _usage(1000, 0, 1000, 100).model_copy(update=modifier)
    expected: Final = (_usage(1000, 1000, 0, 100) if continuing else usage).model_copy(update=modifier)
    normalized: Final = _baseline_usage(expected)
    cache_fields: Final = {"prompt_tokens_details", "cache_read_input_tokens", "cache_creation_input_tokens"}
    assert normalized.model_dump(exclude=cache_fields) == usage.model_dump(exclude=cache_fields)
    assert usage.prompt_tokens_details.cached_tokens == 0
    selected_cost: Final = 0.013
    assert compute_autorouter_savings(
        "claude-opus-5", "claude-sonnet-5", "anthropic", usage, baseline_usage=expected,
        cost_breakdown={"input_cost": 0.01, "output_cost": 0.003},
    ) == pytest.approx(sum(anthropic_cost_per_token("claude-opus-5", expected)) - selected_cost)


def test_anthropic_baseline_keeps_negotiated_prices_with_provider_multiplier() -> None:
    info: Final = {
        **litellm.get_model_info("claude-opus-5", "anthropic"),
        "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6, "cache_read_input_token_cost": 3e-7,
    }
    usage: Final = _usage(1000, 1000, 0, 100).model_copy(update={"speed": "fast"})
    assert compute_autorouter_savings(
        "claude-opus-5", "claude-sonnet-5", "anthropic", usage, baseline_info=info, baseline_usage=usage,
        cost_breakdown={"input_cost": 0.01, "output_cost": 0.003},
    ) == pytest.approx(0.0015 * 2 - 0.013)


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
        gateway_injected_cache=True,
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
        gateway_injected_cache=True,
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


@pytest.mark.parametrize(
    "model,provider,prompt,reads,writes_5m,writes_1h,tier,region,location",
    [
        pytest.param("claude-sonnet-4-5", "anthropic", 50000, 20000, 20000, 0, None, None, None, id="5m"),
        pytest.param("claude-sonnet-4-5", "anthropic", 50000, 20000, 0, 20000, None, None, None, id="1h"),
        pytest.param("claude-sonnet-4-5", "anthropic", 50000, 20000, 12000, 8000, None, None, None, id="mixed-ttl"),
        pytest.param("claude-sonnet-4-5", "anthropic", 199999, 80000, 20000, 0, None, None, None, id="below-200k"),
        pytest.param("claude-sonnet-4-5", "anthropic", 200000, 80000, 20000, 0, None, None, None, id="exactly-200k"),
        pytest.param("claude-sonnet-4-5", "anthropic", 200001, 80000, 20000, 0, None, None, None, id="above-200k"),
        pytest.param("claude-sonnet-4-5", "anthropic", 250000, 80000, 12000, 8000, None, None, None, id="ttl-and-200k"),
        pytest.param(
            "claude-sonnet-4-5", "anthropic", 250000, 80000, 0, 20000, "priority", None, None, id="absent-tier"
        ),
        pytest.param("gpt-5.5", "openai", 100000, 80000, 0, 0, "priority", None, None, id="priority"),
        pytest.param("gpt-5.5", "openai", 100000, 80000, 0, 0, "flex", None, None, id="flex"),
        pytest.param("gpt-5.5", "openai", 100000, 80000, 0, 0, "batch", None, None, id="batch-resolver-fallback"),
        pytest.param("gpt-5.5", "openai", 300000, 80000, 0, 0, "flex", None, None, id="flex-and-272k"),
        pytest.param("gpt-5.5", "openai", 100000, 80000, 0, 0, "priority", "eu", None, id="priority-and-eu"),
        pytest.param("gemini-2.5-pro", "vertex_ai", 250000, 80000, 20000, 0, None, None, None, id="variant-only-write"),
        pytest.param("gemini-3.5-flash", "vertex_ai", 100000, 80000, 0, 0, None, None, "us-east5", id="vertex-region"),
        pytest.param("gpt-5.5", "openai", 300000, 0, 0, 0, "priority", "eu", None, id="no-cache"),
    ],
)
def test_caching_savings_agree_with_biller_on_the_request_pricing_basis(
    model: str,
    provider: str,
    prompt: int,
    reads: int,
    writes_5m: int,
    writes_1h: int,
    tier: str | None,
    region: str | None,
    location: str | None,
) -> None:
    pricing: Final = litellm.get_model_info(model=model, custom_llm_provider=provider)
    usage: Final = Usage(
        prompt_tokens=prompt,
        completion_tokens=100,
        total_tokens=prompt + 100,
        prompt_tokens_details={
            "cached_tokens": reads,
            "cache_creation_tokens": writes_5m + writes_1h,
            "text_tokens": prompt - reads - writes_5m - writes_1h,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": writes_5m,
                "ephemeral_1h_input_tokens": writes_1h,
            },
        },
    )
    uncached: Final = Usage(
        prompt_tokens=prompt,
        completion_tokens=100,
        total_tokens=prompt + 100,
        prompt_tokens_details={"text_tokens": prompt, "cached_tokens": 0, "cache_creation_tokens": 0},
    )
    costs: Final = tuple(
        sum(
            generic_cost_per_token(
                model=model,
                usage=arm,
                custom_llm_provider=provider,
                model_info=pricing,
                service_tier=tier,
                data_residency=region,
                vertex_location=location,
            )
        )
        for arm in (uncached, usage)
    )
    expected: Final = costs[0] - costs[1]
    for attributed in (False, True):
        result: Final = compute_savings_spend(
            model=model,
            custom_llm_provider=provider,
            compression_saved_tokens=4389,
            gateway_injected_cache=attributed,
            usage_object=usage.model_dump(),
            cost_breakdown={"service_tier": tier, "data_residency": region, "vertex_location": location},
            billed_at="2026-09-07T12:00:00+00:00",
        )
        assert result.prompt_caching == pytest.approx(expected)
        assert result.gateway_injected_caching == pytest.approx(expected if attributed else 0.0)
        assert result.compression == pytest.approx(4389 * (pricing["input_cost_per_token"] or 0.0))
        assert result.autorouter == 0.0
    if reads + writes_5m + writes_1h == 0:
        assert expected == 0.0


def test_negative_ttl_counts_do_not_become_cache_write_credits() -> None:
    results: Final = tuple(
        compute_savings_spend(
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
            compression_saved_tokens=0,
            gateway_injected_cache=True,
            usage_object={
                "prompt_tokens": 6000,
                "completion_tokens": 100,
                "prompt_tokens_details": {
                    "text_tokens": 1000,
                    "cache_creation_tokens": 5000,
                    "cache_creation_token_details": {
                        "ephemeral_5m_input_tokens": short_count,
                        "ephemeral_1h_input_tokens": 5000,
                    },
                },
            },
        )
        for short_count in (-5000, 0)
    )
    assert results[0] == results[1]
    assert results[0].prompt_caching < 0


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
        gateway_injected_cache=True,
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
        gateway_injected_cache=True,
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
        gateway_injected_cache=True,
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
        gateway_injected_cache=True,
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
        gateway_injected_cache=True,
        usage_object={"cache_read_input_tokens": 5000, "cache_creation_input_tokens": 800},
    )
    nested_only = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
        usage_object={
            "prompt_tokens_details": {"cached_tokens": 5000, "cache_write_tokens": 800},
        },
    )
    assert nested_only.prompt_caching == pytest.approx(with_top_level.prompt_caching)
    assert nested_only.prompt_caching == pytest.approx(
        5000 * (input_cost - _anthropic_costs("claude-sonnet-5")[1]) - 800 * (cache_write_cost - input_cost)
    )


def test_negative_cache_write_count_clamps_to_zero():
    """A malformed negative write count must not be read as a saving."""
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
        usage_object={"cache_read_input_tokens": 1000, "cache_creation_input_tokens": -5000},
    )
    assert result.prompt_caching == pytest.approx(1000 * (input_cost - cache_read_cost))


def test_unknown_model_fails_open_to_zero():
    result = compute_savings_spend(
        model="totally-made-up-model-xyz",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        gateway_injected_cache=True,
        usage_object={"cache_read_input_tokens": 1000},
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_missing_model_fails_open_to_zero():
    result = compute_savings_spend(
        model=None,
        custom_llm_provider=None,
        compression_saved_tokens=1000,
        gateway_injected_cache=True,
        usage_object={"cache_read_input_tokens": 1000},
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_negative_token_counts_clamp_to_zero():
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=-500,
        gateway_injected_cache=True,
        usage_object={"cache_read_input_tokens": -500},
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def _usage(fresh: int, cached: int, written: int, out: int, *, hour: bool = False, image: int = 0) -> Usage:
    """Usage as the spend log records it; `prompt_tokens` is the inclusive total."""
    return Usage(
        prompt_tokens=fresh + cached + written,
        completion_tokens=out,
        total_tokens=fresh + cached + written + out,
        prompt_tokens_details={
            "cached_tokens": cached, "cache_creation_tokens": written, "text_tokens": fresh - image, "image_tokens": image,
            "cache_creation_token_details": {"ephemeral_1h_input_tokens": written} if hour else None,
        },
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=written,
    )


def _savings(baseline: str, selected: str, usage: Usage, baseline_usage: Usage | None = None) -> float | None:
    return compute_autorouter_savings(
        baseline_model=baseline,
        selected_model=selected,
        selected_provider="anthropic",
        usage=usage,
        baseline_usage=baseline_usage,
    )


@pytest.mark.parametrize("baseline, selected, actual, modeled, loses_money", [
    pytest.param("claude-sonnet-5", "claude-haiku-4-5", _usage(3, 500, 12304, 500),
                 _usage(3, 12804, 0, 500), True, id="warm-baseline-cold-route"),
    pytest.param("claude-opus-5", "claude-opus-5", _usage(0, 0, 20000, 1000),
                 _usage(0, 20000, 0, 1000), True, id="same-model-cold-route"),
    pytest.param("claude-opus-5", "claude-sonnet-5", _usage(0, 19000, 1000, 1000),
                 _usage(0, 19500, 500, 1000), False, id="partly-cached-growth"),
    pytest.param("claude-opus-5", "claude-sonnet-5", _usage(0, 0, 100000, 1000, hour=True),
                 _usage(0, 0, 100000, 1000, hour=True), False, id="expired-one-hour"),
    pytest.param("claude-opus-5", "claude-sonnet-5", _usage(0, 0, 100000, 1000, hour=True),
                 _usage(0, 100000, 0, 1000), True, id="invented-one-hour-hit"),
    pytest.param("claude-opus-5", "claude-sonnet-5", _usage(4000, 0, 16000, 1000, hour=True, image=4000),
                 _usage(4000, 0, 16000, 1000, hour=True, image=4000), False, id="image-and-one-hour-write"),
])
def test_supplied_baseline_usage_is_priced_independently(
    baseline: str, selected: str, actual: Usage, modeled: Usage, loses_money: bool,
) -> None:
    result: Final = _savings(baseline, selected, actual, modeled)
    expected: Final = sum(generic_cost_per_token(model=baseline, usage=modeled, custom_llm_provider="anthropic")) - sum(
        generic_cost_per_token(model=selected, usage=actual, custom_llm_provider="anthropic")
    )
    assert result == pytest.approx(expected)
    assert result is not None and (result < 0) is loses_money
    assert _baseline_usage(modeled).prompt_tokens_details == modeled.prompt_tokens_details


@pytest.mark.parametrize("modifier, multiplier", [({}, 1.0), ({"inference_geo": "us"}, 1.1), ({"speed": "fast"}, 2.0)])
@pytest.mark.parametrize("negotiated", [False, True])
@pytest.mark.parametrize("provenance", [None, "modeled", "observed_initial"])
def test_observed_initial_uses_provider_billing_and_effective_rates(
    modifier: dict[str, str], multiplier: float, negotiated: bool,
    provenance: Literal["modeled", "observed_initial"] | None,
) -> None:
    usage: Final = _usage(1000, 2000, 3000, 100).model_copy(update=modifier)
    info: Final = litellm.get_model_info("claude-opus-5", "anthropic").copy()
    if negotiated:
        info["input_cost_per_token"] = 1e-6
        info["output_cost_per_token"] = 2e-6
        info["cache_read_input_token_cost"] = 3e-7
        info["cache_creation_input_token_cost"] = 4e-6
    billed: Final = anthropic_cost_per_token("claude-opus-5", usage, model_info=info)
    if negotiated:
        assert sum(billed) == pytest.approx(0.0138 * multiplier)
    assert compute_autorouter_savings(
        "anthropic/claude-opus-5", "claude-opus-5", "anthropic", usage,
        selected_info=info, baseline_info=info, baseline_usage=usage,
        baseline_deployment_id="same", selected_deployment_id="same",
        cost_breakdown={"input_cost": billed[0], "output_cost": billed[1]},
        baseline_provenance=provenance,
    ) == 0.0


@pytest.mark.parametrize("model, deployment, known, delta", [
    ("claude-sonnet-5", "same", "observed", 0.0),
    ("claude-opus-5", "other", "observed", 0.0),
    ("claude-opus-5", "", "observed", 0.0),
    ("claude-opus-5", "same", "missing", 0.0),
    ("claude-opus-5", "same", "different", 0.0),
    ("claude-opus-5", "same", "observed", 0.01),
    ("claude-opus-5", "same", "prices", 0.0),
    ("claude-opus-5", "same", "unbilled", 0.0),
])
def test_initial_provenance_cannot_override_mismatched_evidence(
    model: str, deployment: str, known: Literal["observed", "missing", "different", "prices", "unbilled"], delta: float,
) -> None:
    usage: Final = _usage(1000, 0, 1000, 100)
    billed: Final = anthropic_cost_per_token("claude-opus-5", usage)
    info: Final = litellm.get_model_info(model, "anthropic").copy()
    if known == "prices":
        info["cache_read_input_token_cost"] = 0.001  # No reads here: equal charge alone cannot establish equal rates.
    assert compute_autorouter_savings(
        "claude-opus-5", model, "anthropic", usage,
        baseline_usage=(None if known == "missing" else _usage(1000, 1000, 0, 100) if known == "different" else usage),
        selected_info=info,
        baseline_provenance="observed_initial",
        baseline_deployment_id="same", selected_deployment_id=deployment,
        cost_breakdown=None if known == "unbilled" else {"input_cost": billed[0] + delta, "output_cost": billed[1]},
    ) is None


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
    usage: Final = _usage(3, 500, 12304, 500)
    assert _savings("claude-opus-5", "claude-opus-5", usage, usage) == 0.0


def test_autorouter_savings_unknown_baseline_remains_unknown():
    assert _savings("totally-made-up-model-xyz", "claude-haiku-4-5", _usage(3, 500, 12304, 500)) is None


def test_autorouter_savings_zero_without_baseline():
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
        routing_decision=None,
        usage_object=_cached_usage_object(),
    )
    assert result.autorouter == 0.0


def test_compute_savings_spend_carries_a_recorded_losing_switch_through():
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
        routing_decision={"conversation_continuing": True, "savings_baseline_model": "anthropic/claude-sonnet-5"},
        usage_object=_cached_usage_object(),
        recorded_autorouter_savings=-0.01,
    )
    assert result.autorouter < 0


def test_the_driver_is_off_until_a_baseline_is_configured():
    """No configured counterfactual means there is nothing to measure against, so the
    driver reports zero rather than inventing a model the operator never named."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        gateway_injected_cache=True,
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
        gateway_injected_cache=True,
        routing_decision={"conversation_continuing": True},
        usage_object={"prompt_tokens": ["not", "a", "number"]},
    )
    assert result.autorouter == 0.0
    assert result.compression > 0


def test_equal_modeled_usage_is_zero_under_equivalent_model_names() -> None:
    usage: Final = _usage(3, 500, 12304, 500)
    assert _savings("anthropic/claude-opus-5", "claude-opus-5", usage, usage) == 0.0
    assert _savings("claude-opus-5", "anthropic/claude-opus-5", usage, usage) == 0.0


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


def test_unresolvable_baseline_remains_unknown():
    usage = _usage(fresh=2000, cached=0, written=0, out=500)
    assert _savings("no-such-provider-xyz/no-such-model", "claude-haiku-4-5", usage) is None


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
        baseline_usage=first_turn,
    )

    gpt5 = litellm.get_model_info("gpt-5", "openai")
    assert gpt5.get("cache_creation_input_token_cost") is None, "pick a baseline with no cache-write rate"
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    baseline_pays_input = 20_000 * gpt5["input_cost_per_token"] + 1_000 * gpt5["output_cost_per_token"]
    actually_paid = 20_000 * haiku["cache_creation_input_token_cost"] + 1_000 * haiku["output_cost_per_token"]
    assert reported == pytest.approx(baseline_pays_input - actually_paid)
    assert reported > 0, "routing a cold first turn onto a cheaper model is a saving, not a loss"


def _priced_chat_model_without_cache_read_rate() -> tuple[str, str, str]:
    """A chat model the bundled map prices per token for input and output but not for cache
    reads, derived from the map itself: a hardcoded pick goes stale the moment the registry
    prices that model's cache reads, which is exactly how this test's premise last broke.
    Candidates go through the savings module's own resolver, so the pick is one the code
    under test can actually price."""
    for key in sorted(litellm.model_cost):
        entry = litellm.model_cost[key]
        provider = entry.get("litellm_provider")
        if not isinstance(provider, str) or not key.startswith(f"{provider}/"):
            continue
        if entry.get("mode") != "chat" or entry.get("cache_read_input_token_cost") is not None:
            continue
        if not entry.get("input_cost_per_token") or not entry.get("output_cost_per_token"):
            continue
        if _resolve_model(key, None) is None:
            continue
        priced = compute_autorouter_savings(
            baseline_model=key,
            selected_model="claude-haiku-4-5",
            selected_provider="anthropic",
            usage=_usage(fresh=1_000, cached=0, written=0, out=100),
            conversation_continuing=True,
        )
        if priced is None or priced == 0.0:
            continue
        return key, key.removeprefix(f"{provider}/"), provider
    raise AssertionError("the bundled map has no per-token chat model without a cache-read rate")


def test_a_baseline_with_no_cache_read_rate_is_charged_its_input_rate():
    """The same hole on the other bucket. A baseline whose entry has no
    `cache_read_input_token_cost` reads for 0.0, so a continuing turn priced the whole
    prompt at nothing and every switch away from it reported a loss.
    """
    baseline_key, baseline_name, baseline_provider = _priced_chat_model_without_cache_read_rate()
    continuing = _usage(fresh=0, cached=0, written=20_000, out=1_000)
    reported = compute_autorouter_savings(
        baseline_model=baseline_key,
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=continuing,
        baseline_usage=_usage(0, 20000, 0, 1000),
    )

    baseline = litellm.get_model_info(baseline_name, baseline_provider)
    assert baseline.get("cache_read_input_token_cost") is None, "pick a baseline with no cache-read rate"
    haiku = litellm.get_model_info("claude-haiku-4-5", "anthropic")
    baseline_pays_input = 20_000 * baseline["input_cost_per_token"] + 1_000 * baseline["output_cost_per_token"]
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
        pytest.param({"service_tier": "priority"}, 2.5, id="priority tier uplifts the baseline"),
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
    assert gpt.get("input_cost_per_token_priority") == pytest.approx(2.5 * gpt["input_cost_per_token"])
    assert gpt.get("output_cost_per_token_priority") == pytest.approx(2.5 * gpt["output_cost_per_token"])
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
        gateway_injected_cache=True,
        routing_decision={"conversation_continuing": True, "savings_baseline_model": "anthropic/claude-opus-5"},
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
    )
    assert result.autorouter != 0.0


def test_a_leftover_configured_baseline_does_not_override_the_recorded_one(monkeypatch):
    """The proxy config loader setattrs unknown litellm_settings keys, so a stale
    autorouter_savings_baseline_model key must stay inert."""
    monkeypatch.setattr(litellm, "autorouter_savings_baseline_model", "claude-sonnet-5", raising=False)
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
        routing_decision={"conversation_continuing": True, "savings_baseline_model": "anthropic/claude-opus-5"},
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
    )
    against_opus = compute_autorouter_savings(
        baseline_model="anthropic/claude-opus-5",
        selected_model="claude-haiku-4-5",
        selected_provider="anthropic",
        usage=_usage(12807, 0, 0, 500),
    )
    assert result.autorouter == against_opus


def test_a_non_string_recorded_baseline_is_ignored():
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
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
        gateway_injected_cache=True,
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
        gateway_injected_cache=True,
        usage_object=_caching_usage(read=1000, written=20000),
    )
    assert result.prompt_caching > at_public_rates.prompt_caching


@pytest.mark.parametrize(
    "baseline_id, selected_id, selected_multiplier, billed_input, classifier_cost, expected",
    [
        ("baseline", "selected", 0.1, None, 0.0, 0.0135),
        ("baseline", "selected", 2.0, None, 0.0, -0.015),
        ("baseline", "selected", 1.0, None, 0.0, 0.0),
        ("baseline", "selected", 0.1, 0.004, 0.001, 0.01),
        ("baseline", "baseline", 0.1, 0.004, 0.001, 0.01),
        (None, "selected", 0.1, None, 0.0, 0.006),
        ("baseline", None, 0.1, None, 0.0, 0.0075),
        (None, None, 0.1, None, 0.0, 0.0),
        ("", "selected", 0.1, None, 0.0, 0.006),
        ("baseline", "", 0.1, None, 0.0, 0.0075),
    ],
)
def test_autorouter_savings_distinguishes_priced_deployments(
    baseline_id: str | None,
    selected_id: str | None,
    selected_multiplier: float,
    billed_input: float | None,
    classifier_cost: float,
    expected: float,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "anthropic/claude-opus-5",
                    "api_key": "test-key",
                    "input_cost_per_token": 1e-5 * multiplier,
                    "output_cost_per_token": 5e-5 * multiplier,
                },
                "model_info": {"id": name},
            }
            for name, multiplier in (("baseline", 1.0), ("selected", selected_multiplier))
        ]
    )
    result: Final = compute_savings_spend(
        model="claude-opus-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=False,
        model_id=selected_id,
        llm_router=lambda: router,
        routing_decision={
            "savings_baseline_model": "anthropic/claude-opus-5",
            "savings_baseline_deployment_id": baseline_id,
            "conversation_continuing": False,
            "classifier_cost": classifier_cost,
        },
        usage_object={"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
        cost_breakdown=None if billed_input is None else {"input_cost": billed_input, "output_cost": 0.0},
    )
    assert result.autorouter == pytest.approx(expected)


@pytest.mark.parametrize("selected_model", ["azure/contract-deployment", "contract-deployment"])
def test_autorouter_savings_recognizes_one_deployment_under_its_base_model(selected_model: str) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "contract",
                "litellm_params": {
                    "model": "azure/contract-deployment",
                    "api_key": "test-key",
                    "api_base": "https://example.openai.azure.com",
                    "input_cost_per_token": 0.0001,
                    "output_cost_per_token": 0.0002,
                    "cache_read_input_token_cost": 0.00001,
                },
                "model_info": {"id": "contract", "base_model": "azure/gpt-5.5"},
            }
        ]
    )
    result: Final = compute_savings_spend(
        model=selected_model,
        custom_llm_provider="azure",
        compression_saved_tokens=0,
        gateway_injected_cache=False,
        model_id="contract",
        llm_router=lambda: router,
        routing_decision={
            "savings_baseline_model": "azure/gpt-5.5",
            "savings_baseline_deployment_id": "contract",
            "conversation_continuing": True,
        },
        usage_object={
            "prompt_tokens": 21000,
            "completion_tokens": 100,
            "total_tokens": 21100,
            "prompt_tokens_details": {"text_tokens": 1000, "cached_tokens": 0, "cache_creation_tokens": 20000},
        },
        cost_breakdown={"input_cost": 2.1, "output_cost": 0.02},
    )
    assert result.autorouter == 0.0


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
        gateway_injected_cache=True,
        routing_decision=decision,
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
        llm_router=lambda: router,
    )
    at_public_rate = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
        routing_decision={k: v for k, v in decision.items() if k != "savings_baseline_deployment_id"},
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
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
        gateway_injected_cache=False,
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
        gateway_injected_cache=False,
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
        gateway_injected_cache=False,
        routing_decision=None,
        usage_object=_cached_usage_object(),
        recorded_autorouter_savings=True,
    )
    assert result.autorouter == 0.0


@pytest.mark.parametrize("continuing", [False, True])
def test_legacy_cache_rows_without_an_estimate_do_not_invent_a_new_figure(continuing: bool) -> None:
    from litellm.proxy.spend_tracking.savings import autorouter_savings_for_request

    recomputed = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=False,
        routing_decision={**_routed_decision(), "conversation_continuing": continuing},
        usage_object=_cached_usage_object(),
    )
    direct = autorouter_savings_for_request(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        routing_decision={**_routed_decision(), "conversation_continuing": continuing},
        usage_object=_cached_usage_object(),
    )
    assert direct is None
    assert recomputed.autorouter == 0.0


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
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
        cost_breakdown=None,
    )
    assert stamped is not None and stamped != 0.0

    internal = autorouter_savings_for_logging_payload(
        request_metadata={**routed_metadata, "internal_call_origin": "shadow_eval_shadow"},
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        model_id=None,
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
        cost_breakdown=None,
    )
    assert internal is None


def test_savings_are_net_of_a_priced_classifier():
    """The classifier call is part of what routing cost, so the per-request figure
    deducts it; a charge big enough to outweigh the model saving goes negative,
    since the figure is signed on purpose (GH #38816)."""
    from litellm.proxy.spend_tracking.savings import autorouter_savings_for_request

    gross = autorouter_savings_for_request(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        routing_decision=_routed_decision(),
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
    )
    net = autorouter_savings_for_request(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        routing_decision={**_routed_decision(), "classifier_cost": 0.005},
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
    )
    assert gross is not None and net == pytest.approx(gross - 0.005)


@pytest.mark.parametrize("classifier_cost", [0.0, "bogus", True])
def test_an_unpriced_classifier_deducts_nothing(classifier_cost: object):
    from litellm.proxy.spend_tracking.savings import autorouter_savings_for_request

    gross = autorouter_savings_for_request(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        routing_decision=_routed_decision(),
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
    )
    with_cost_field = autorouter_savings_for_request(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        routing_decision={**_routed_decision(), "classifier_cost": classifier_cost},
        usage_object=_usage(12807, 0, 0, 500).model_dump(),
    )
    assert with_cost_field == gross


def test_recorded_savings_are_already_net_and_not_deducted_again():
    """The deduction lives at the figure's computation owner, so a stamped figure is
    net by construction; the recorded-wins path must not subtract a second time."""
    result = compute_savings_spend(
        model="claude-haiku-4-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=False,
        routing_decision={**_routed_decision(), "classifier_cost": 0.005},
        usage_object=_cached_usage_object(),
        recorded_autorouter_savings=0.5,
    )
    assert result.autorouter == 0.5


def test_caching_savings_require_a_gateway_injected_breakpoint():
    """The same cached usage is attributed to the gateway only when it added a breakpoint.

    Client-sent cache_control and implicit provider caching (OpenAI, Gemini) produce
    cache reads the gateway had no hand in. Those still count as caching savings the
    customer really got, so the total is unchanged, but nothing about them is the
    gateway's doing and the attributed figure has to stay empty.
    """
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    credited = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
        usage_object=_caching_usage(read=8200, written=0),
    )
    expected = 8200 * (input_cost - cache_read_cost)
    assert credited.prompt_caching == pytest.approx(expected)
    assert credited.gateway_injected_caching == pytest.approx(expected)
    unattributed = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=False,
        usage_object=_caching_usage(read=8200, written=0),
    )
    assert unattributed.prompt_caching == pytest.approx(expected)
    assert unattributed.gateway_injected_caching == 0.0


def test_unattributed_write_only_request_still_reports_its_loss_in_the_total():
    """A write-only request really did cost more than not caching, whoever asked for it.

    The attributed figure drops it because the gateway added no breakpoint, and dropping a
    negative is why the attributed number can sit above the total rather than below it.
    """
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=False,
        usage_object=_caching_usage(read=0, written=20000),
    )
    assert result.prompt_caching < 0
    assert result.gateway_injected_caching == 0.0
    assert result.gateway_injected_caching > result.prompt_caching


def test_injected_request_keeps_its_negative_net():
    """A gateway-injected write-heavy request still reports its real loss."""
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        gateway_injected_cache=True,
        usage_object=_caching_usage(read=0, written=20000),
    )
    assert result.prompt_caching < 0


def test_attribution_does_not_touch_compression_or_autorouter_legs():
    input_cost, _ = _anthropic_costs("claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=4389,
        gateway_injected_cache=False,
        usage_object=_caching_usage(read=8200, written=0),
    )
    assert result.compression == pytest.approx(4389 * input_cost)
    assert result.prompt_caching > 0
    assert result.gateway_injected_caching == 0.0


def test_marks_gateway_injection_credits_only_the_deployment_that_was_injected():
    """Every retry, failover and fallback of a request shares one metadata bucket and one
    litellm_call_id, so the deployment is what tells those legs apart. A marker naming a
    sibling has to read here as no injection; that is what keeps the credit on the leg
    that earned it without any seam having to strip it. Anything that is not this row's
    own deployment, the missing key included, is fail-closed."""
    assert marks_gateway_injection(None, "dep-a") is False
    assert marks_gateway_injection({}, "dep-a") is False
    assert marks_gateway_injection({"litellm_gateway_injected_cache": "dep-a"}, "dep-a") is True
    assert marks_gateway_injection({"litellm_gateway_injected_cache": "dep-a"}, "dep-b") is False
    assert marks_gateway_injection({"litellm_gateway_injected_cache": "dep-a"}, None) is False
    # injected before a deployment was chosen, so it is in the payload every leg sends
    assert marks_gateway_injection({"litellm_gateway_injected_cache": ""}, "dep-a") is True
    assert marks_gateway_injection({"litellm_gateway_injected_cache": ""}, None) is True
    assert marks_gateway_injection({"litellm_call_id": "c1"}, "dep-a") is False


@pytest.mark.parametrize("classifier", [0.0, 0.02])
def test_observed_baseline_keeps_both_costs_and_classifier_overhead(classifier: float) -> None:
    from litellm.proxy.spend_tracking.savings import BaselineCostSnapshot, price_baseline_comparison

    snapshot: Final = BaselineCostSnapshot(
        model="baseline", provider="anthropic", prices=None,
        actual_spend=0.17, classifier_cost=classifier,
    )
    restored: Final = BaselineCostSnapshot.model_validate_json(snapshot.model_dump_json())
    result: Final = price_baseline_comparison(restored, Usage(prompt_tokens=100, completion_tokens=10), "observed_identical")
    assert result is not None
    assert result.baseline == snapshot.actual_spend
    assert result.actual == snapshot.actual_spend + classifier
    assert result.savings == pytest.approx(-classifier)
    assert price_baseline_comparison(restored, None, None) is None


def test_modeled_baseline_uses_recorded_prices_and_preserves_other_actual_charges() -> None:
    from litellm.proxy.spend_tracking.savings import BaselineCostSnapshot, price_baseline_comparison

    snapshot: Final = BaselineCostSnapshot(
        model="claude-opus-5", provider="anthropic",
        prices={
            **litellm.get_model_info("claude-opus-5", custom_llm_provider="anthropic"),
            "input_cost_per_token": 0.001, "output_cost_per_token": 0.002,
        },
        actual_token_cost=0.2, actual_spend=0.23, classifier_cost=0.01,
    )
    usage: Final = Usage(prompt_tokens=100, completion_tokens=10)
    result: Final = price_baseline_comparison(snapshot, usage, "modeled")
    assert result is not None
    assert result.actual == pytest.approx(0.24)
    assert result.baseline == pytest.approx(0.12 + 0.03)
    assert result.savings == pytest.approx(-0.09)
    assert price_baseline_comparison(snapshot.model_copy(update={"prices": None}), usage, "modeled") is None
    assert marks_gateway_injection({"litellm_gateway_injected_cache": True}, "dep-a") is False
