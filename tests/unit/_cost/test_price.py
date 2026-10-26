from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm._cost import billing_context, evaluate, price, require_priced, token_usage
from litellm._internal_context import pinned_billing_time
from litellm.exceptions import ModelNotMappedError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge import _native
from litellm.types.utils import ModelResponse, Usage


@pytest.mark.parametrize(
    ("tier_prices", "expected"),
    (
        ({"output_cost_per_token": 4.0}, (10.0, 20.0)),
        ({"input_cost_per_token": None, "output_cost_per_token": 4.0}, None),
        ({"input_cost_per_token": 0.0, "output_cost_per_token": 4.0}, (0.0, 40.0)),
        ({"input_cost_per_token": 3.0}, (30.0, 20.0)),
        ({"input_cost_per_token": 3.0, "output_cost_per_token": None}, None),
        ({"input_cost_per_token": 3.0, "output_cost_per_token": 0.0}, (30.0, 0.0)),
    ),
    ids=("input-absent", "input-null", "input-zero", "output-absent", "output-null", "output-zero"),
)
def test_tier_declared_null_is_unpriced_and_explicit_zero_remains_free(
    monkeypatch: pytest.MonkeyPatch, tier_prices: Mapping[str, object], expected: tuple[float, float] | None
) -> None:
    model: Final = "openai/m5-tier-presence"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
            "tiered_pricing": [{"range": [0, 100], **tier_prices}],
        },
    )
    response: Final = ModelResponse(model=model, usage=Usage(prompt_tokens=10, completion_tokens=10))
    if expected is None:
        with pytest.raises(ModelNotMappedError):
            litellm.cost_per_token(model, 10, 10, custom_llm_provider="openai")
        with pytest.raises(ModelNotMappedError):
            litellm.completion_cost(response, model=model, custom_llm_provider="openai")
    else:
        assert litellm.cost_per_token(model, 10, 10, custom_llm_provider="openai") == pytest.approx(expected)
        assert litellm.completion_cost(response, model=model, custom_llm_provider="openai") == pytest.approx(
            sum(expected)
        )


@pytest.mark.parametrize(
    ("rates", "usage", "expected"),
    (
        (
            {
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 2.0,
                "tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 3.0, "output_cost_per_token": 4.0}],
            },
            Usage(prompt_tokens=10, completion_tokens=10),
            (30.0, 40.0),
        ),
        (
            {
                "cache_read_input_token_cost": 9.0,
                "tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1.0, "output_cost_per_token": 2.0}],
            },
            Usage(prompt_tokens=100, completion_tokens=10, prompt_tokens_details={"cached_tokens": 40}),
            (100.0, 20.0),
        ),
        (
            {
                "cache_creation_input_token_cost_above_1hr": 9.0,
                "tiered_pricing": [
                    {
                        "range": [0, 100],
                        "input_cost_per_token": 1.0,
                        "output_cost_per_token": 2.0,
                        "cache_creation_input_token_cost": 3.0,
                    }
                ],
            },
            Usage(
                prompt_tokens=100,
                completion_tokens=10,
                prompt_tokens_details={
                    "cache_creation_tokens": 20,
                    "cache_creation_token_details": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 10},
                },
            ),
            (140.0, 20.0),
        ),
    ),
    ids=("flat-and-tier", "tier-cache-read-fallback", "tier-hourly-cache-write-fallback"),
)
def test_public_token_pricing_preserves_tier_local_fallbacks(
    monkeypatch: pytest.MonkeyPatch, rates: Mapping[str, object], usage: Usage, expected: tuple[float, float]
) -> None:
    model: Final = "openai/m5-tier-fallback"
    monkeypatch.setitem(litellm.model_cost, model, {"litellm_provider": "openai", "mode": "chat", **rates})
    assert litellm.cost_per_token(
        model, usage.prompt_tokens, usage.completion_tokens, custom_llm_provider="openai", usage_object=usage
    ) == pytest.approx(expected)
    assert litellm.completion_cost(
        ModelResponse(model=model, usage=usage), model=model, custom_llm_provider="openai"
    ) == pytest.approx(sum(expected))


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize(
    ("kind", "rates", "expected"),
    (
        ("unknown", None, None),
        ("metadata", {}, None),
        ("free", {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}, 0.0),
        ("paid", {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}, 50.0),
    ),
)
def test_unknown_free_and_priced_public_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    kind: str,
    rates: Mapping[str, float] | None,
    expected: float | None,
) -> None:
    model: Final = f"openai/m5-{kind}"
    if rates is not None:
        monkeypatch.setitem(litellm.model_cost, model, {"litellm_provider": "openai", "mode": "chat", **rates})
        info: Final = litellm.get_model_info(model, custom_llm_provider="openai")
        assert info["input_cost_per_token"] == rates.get("input_cost_per_token")
        assert info["output_cost_per_token"] == rates.get("output_cost_per_token")

    usage: Final = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    response: Final = ModelResponse(model=model, usage=usage)
    logging: Final = Logging(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=stream,
        call_type="completion",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="m5-price-surfaces",
        function_id="m5-price-surfaces",
    )
    logging.update_environment_variables(litellm_params={}, optional_params={}, custom_llm_provider="openai")
    if expected is None:
        with pytest.raises(ModelNotMappedError):
            litellm.cost_per_token(model, 10, 20, custom_llm_provider="openai", usage_object=usage)
        with pytest.raises(ModelNotMappedError):
            litellm.completion_cost(response, model=model, custom_llm_provider="openai")
        assert logging.response_cost_calculator(response) is None
        assert logging.model_call_details.get("response_cost_failure_debug_information") is not None
    else:
        split: Final = litellm.cost_per_token(model, 10, 20, custom_llm_provider="openai", usage_object=usage)
        assert split == pytest.approx((expected / 5, expected * 4 / 5))
        assert litellm.completion_cost(response, model=model, custom_llm_provider="openai") == pytest.approx(expected)
        assert logging.response_cost_calculator(response) == pytest.approx(expected)


def test_public_pricing_reads_direct_row_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "openai/m5-live-update"
    row: Final = {"litellm_provider": "openai", "input_cost_per_token": 1.0, "output_cost_per_token": 2.0}
    monkeypatch.setitem(litellm.model_cost, model, row)
    assert litellm.cost_per_token(model, 10, 20, custom_llm_provider="openai") == pytest.approx((10.0, 40.0))
    monkeypatch.setitem(row, "input_cost_per_token", 3.0)
    assert litellm.cost_per_token(model, 10, 20, custom_llm_provider="openai") == pytest.approx((30.0, 40.0))


def test_invalid_price_is_not_authorized_by_permissive_model_info(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "openai/m5-invalid-row"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "openai",
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
            "input_cost_per_audio_token": "invalid",
        },
    )
    assert litellm.get_model_info(model, custom_llm_provider="openai")["input_cost_per_token"] == 1.0
    with pytest.raises(ModelNotMappedError, match="input_cost_per_audio_token"):
        litellm.cost_per_token(model, 10, 20, custom_llm_provider="openai")


def test_custom_cost_uses_ephemeral_row_and_native_cache_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "openai/m5-custom-cost"
    row: Final = {"litellm_provider": "openai", "input_cost_per_token": 10.0, "output_cost_per_token": 20.0}
    monkeypatch.setitem(litellm.model_cost, model, row)
    usage: Final = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details={"cached_tokens": 40, "cache_creation_tokens": 20},
    )
    assert litellm.cost_per_token(
        model,
        100,
        10,
        usage_object=usage,
        custom_cost_per_token={"input_cost_per_token": 1.0, "output_cost_per_token": 2.0},
    ) == pytest.approx((100.0, 20.0))
    assert litellm.model_cost[model] is row
    assert row["input_cost_per_token"] == 10.0
    assert litellm.cost_per_token(model, 100, 10, custom_llm_provider="openai") == pytest.approx((1000.0, 200.0))


@pytest.mark.parametrize("tier", ("priority", "fast", " FAST "))
def test_billing_context_normalizes_priority_alias(tier: str) -> None:
    row: Final = {"litellm_provider": "openai", "input_cost_per_token": 1.0, "input_cost_per_token_priority": 3.0}
    usage: Final = Usage(prompt_tokens=10, completion_tokens=0)
    result: Final = require_priced(
        price(row, token_usage(usage, row), billing_context(datetime(2026, 1, 1), tier)),
        "priority",
    )
    assert (result.input, result.output) == (30.0, 0.0)


def test_adapter_preserves_missing_meter_and_native_fallback_result() -> None:
    when: Final = billing_context(datetime(2026, 1, 1))
    missing: Final = price({"litellm_provider": "openai"}, _native.UsageInput({"input_cost_per_token": 1}), when)
    assert isinstance(missing, _native.Unpriced)
    assert tuple(missing.missing) == ("input_cost_per_token",)
    fallback: Final = require_priced(
        price(
            {"litellm_provider": "openai", "input_cost_per_token": 2.0},
            _native.UsageInput({"cache_read_input_token_cost": 10}),
            when,
        ),
        "cache fallback",
    )
    assert fallback.input == 20.0
    assert fallback.lines[0].fallback_from is not None


@pytest.mark.parametrize("factor", (1.0, 1.1))
@pytest.mark.parametrize(
    ("prompt", "completion", "cache", "tier", "expected", "rates"),
    (
        (10, 10, 0, None, (10, 20), (1, 2)),
        (128000, 128000, 0, None, (128000, 256000), (1, 2)),
        (128001, 10, 0, None, (384003, 20), (3, 2)),
        (10, 128001, 0, None, (10, 512004), (1, 4)),
        (128001, 128001, 0, None, (384003, 512004), (3, 4)),
        (100, 10, 80, None, (100, 20), (1, 2)),
        (100, 10, 0, "priority", (100, 20), (1, 2)),
    ),
)
def test_vertex_directional_route_uses_real_aggregate_quantities_and_native_projected_rates(
    monkeypatch: pytest.MonkeyPatch,
    factor: float,
    prompt: int,
    completion: int,
    cache: int,
    tier: str | None,
    expected: tuple[float, float],
    rates: tuple[float, float],
) -> None:
    model: Final = "vertex_ai/gemini-2-directional-test"
    row: Final = {
        "litellm_provider": "vertex_ai",
        "mode": "chat",
        "input_cost_per_token": 1.0,
        "output_cost_per_token": 2.0,
        "input_cost_per_token_above_128k_tokens": 3.0,
        "output_cost_per_token_above_128k_tokens": 4.0,
        "cache_read_input_token_cost": 0.1,
        "input_cost_per_token_priority": 5.0,
        "output_cost_per_token_priority": 6.0,
        "regional_endpoint_uplift_multiplier": factor,
        "off_peak_pricing": {"hours_utc": "00:00-06:00", "input_cost_per_token": 0.1, "output_cost_per_token": 0.2},
    }
    monkeypatch.setitem(litellm.model_cost, model, row)
    shaped: Final = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")
    usage: Final = Usage(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details={"cached_tokens": cache}
    )
    when: Final = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
    with pinned_billing_time(when):
        actual: Final = litellm.cost_per_token(
            model,
            prompt,
            completion,
            custom_llm_provider="vertex_ai",
            usage_object=usage,
            service_tier=tier,
            vertex_location="us-east5",
        )
    assert actual == pytest.approx((expected[0] * factor, expected[1] * factor))
    for source in (row, shaped):
        evaluated: Final = evaluate(
            source,
            _native.UsageInput(
                {"input_cost_per_token": prompt, "output_cost_per_token": completion}, prompt_tokens=prompt
            ),
            billing_context(when, tier),
            project=("input_cost_per_token", "output_cost_per_token"),
            factors=(factor,),
            directional_token_threshold=128000,
        )
        assert isinstance(evaluated, _native.PriceEvaluation)
        priced: Final = require_priced(evaluated.pricing, model)
        assert (priced.input, priced.output) == pytest.approx(actual)
        assert tuple(selected.unit_price for selected in evaluated.rates) == pytest.approx(
            (rates[0] * factor, rates[1] * factor)
        )
