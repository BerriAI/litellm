import logging
import random
from datetime import datetime, timezone

import pytest

import litellm
from litellm.cost_calculator import batch_cost_calculator
from litellm.litellm_core_utils.llm_cost_calc.typed_token_cost import typed_cost_per_token
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _cost_map_billed_rates,
    generic_cost_per_token,
    get_batch_cost_rates,
    get_billed_token_rates,
    get_provider_specific_geo_multiplier,
    get_provider_specific_speed_multiplier,
)
from litellm.llms.anthropic.cost_calculation import cost_per_token as anthropic_cost_per_token
from litellm.llms.openai.cost_calculation import cost_per_token as openai_cost_per_token
from litellm.types.cost_map_configs import AnthropicModelOnAnthropicProvider, OpenAIModelOnOpenAIProvider
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)

_MONDAY_NOON_UTC = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
_CONFIGS = {"openai": OpenAIModelOnOpenAIProvider, "anthropic": AnthropicModelOnAnthropicProvider}
_LONG_TTL_KEY = {
    "openai": "cache_creation_input_token_cost_above_12hr",
    "anthropic": "cache_creation_input_token_cost_above_1hr",
}
_SERVICE_TIERS = {
    "openai": (None, "flex", "priority", "fast", "balanced", "ultrafast", "auto", "default", "PRIORITY"),
    "anthropic": (None, "priority", "fast", "auto", "standard", "standard_only"),
}
_TIER_RATE_KEYS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "output_cost_per_reasoning_token",
)
_OFF_PEAK_SCHEDULES = (
    {"hours_utc": "00:00-00:00"},
    {"hours_utc": ["13:00-14:00"]},
    {"windows": [{"hours_utc": "11:00-13:00", "weekdays": ["mon"]}]},
    {"windows": [{"hours_utc": ["11:00-13:00"], "weekdays": [2, "sat"]}], "weekday_timezone": "Asia/Shanghai"},
)
_CATALOG_ROWS = tuple(
    (name.removeprefix(f"{row['litellm_provider']}/"), row["litellm_provider"])
    for name, row in litellm.model_cost.items()
    if row.get("litellm_provider") in _CONFIGS
)


def _price(rng):
    return rng.choice([0.0, round(rng.uniform(1e-7, 1e-4), 9)])


def _some(rng, keys, chance):
    return {key: _price(rng) for key in keys if rng.random() < chance}


def _price_fields(provider):
    return [
        name
        for name, field in _CONFIGS[provider].model_fields.items()
        if "cost" in name and "float" in str(field.annotation)
    ]


def _random_row(rng, provider):
    row = _some(rng, _price_fields(provider), rng.choice([0.1, 0.5, 0.9]))
    if rng.random() < 0.8:
        row["input_cost_per_token"] = _price(rng)
    if rng.random() < 0.3:
        row["tiered_pricing"] = [
            {"range": tier_range, **_some(rng, (*_TIER_RATE_KEYS, _LONG_TTL_KEY[provider]), 0.6)}
            for tier_range in ([0, 50_000], [50_000, 200_000])
        ]
    if rng.random() < 0.4:
        row["off_peak_pricing"] = {**rng.choice(_OFF_PEAK_SCHEDULES), **_some(rng, _TIER_RATE_KEYS, 0.6)}
    if provider == "openai":
        for key in (
            "regional_processing_uplift_multiplier_eu",
            "regional_processing_uplift_multiplier_us",
            "fedramp_uplift_multiplier",
        ):
            if rng.random() < 0.5:
                row[key] = 1.1
    elif rng.random() < 0.5:
        row["provider_specific_entry"] = {
            key: value for key, value in (("us", 1.1), ("fast", 6.0)) if rng.random() < 0.7
        }
    return row


def _random_usage(rng, provider):
    prompt_tokens = rng.choice(
        [0, 1, 900, 49_999, 50_000, 50_001, 100_000, 100_001, 199_999, 250_000, 272_000, 272_001, 400_000]
    )
    completion_tokens = rng.choice([0, 700, 5000])
    prompt_details = {}
    if rng.random() < 0.5:
        prompt_details["cached_tokens"] = rng.choice([0, 100, 400])
    if rng.random() < 0.3:
        prompt_details["audio_tokens"] = rng.choice([50, 200])
    if rng.random() < 0.3:
        prompt_details["image_tokens"] = rng.choice([50, 200])
    if rng.random() < 0.2:
        prompt_details["text_tokens"] = rng.choice([100, 500])
    for key in ("video_tokens", "character_count", "image_count", "audio_length_seconds", "query_count"):
        if rng.random() < 0.15:
            prompt_details[key] = rng.choice([3, 40])
    if rng.random() < 0.2:
        prompt_details["cached_tokens_details"] = {
            key: 20 for key in ("audio_tokens", "image_tokens", "text_tokens") if rng.random() < 0.6
        }
    if rng.random() < 0.4:
        prompt_details["cache_creation_tokens"] = 300
        if rng.random() < 0.7:
            prompt_details["cache_creation_token_details"] = CacheCreationTokenDetails(
                **{
                    key: rng.choice([0, 60, 100])
                    for key in (
                        "ephemeral_5m_input_tokens",
                        "ephemeral_30m_input_tokens",
                        "ephemeral_1h_input_tokens",
                        "ephemeral_12h_input_tokens",
                    )
                    if rng.random() < 0.6
                }
            )
    completion_details = {}
    if rng.random() < 0.4:
        completion_details["reasoning_tokens"] = rng.choice([100, 300])
    for key in ("audio_tokens", "image_tokens", "text_tokens", "video_tokens"):
        if rng.random() < 0.2:
            completion_details[key] = 100
    anthropic_fields = {}
    if provider == "anthropic" and rng.random() < 0.3:
        anthropic_fields["inference_geo"] = rng.choice(["us", "global", "not_available", "US"])
    if provider == "anthropic" and rng.random() < 0.3:
        anthropic_fields["speed"] = rng.choice(["fast", "standard"])
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(**prompt_details) if prompt_details else None,
        completion_tokens_details=CompletionTokensDetailsWrapper(**completion_details) if completion_details else None,
        **anthropic_fields,
    )


def _provider_multiplier(provider, row, usage):
    if provider != "anthropic":
        return 1.0
    return get_provider_specific_speed_multiplier(model_info=row, usage=usage) * get_provider_specific_geo_multiplier(
        model_info=row, usage=usage
    )


def _typed_pricing(provider, row, usage, service_tier, data_residency):
    config = _CONFIGS[provider].model_validate(row)
    if provider == "openai":
        return config.token_pricing(service_tier, data_residency)
    return config.token_pricing(service_tier, usage.get("inference_geo"), usage.get("speed"))


def _random_cases(provider, count):
    rng = random.Random(f"typed-token-cost-{provider}")
    return [
        (
            _random_row(rng, provider),
            _random_usage(rng, provider),
            rng.choice(_SERVICE_TIERS[provider]),
            rng.choice([None, "eu", "us", "fedramp", "EU"]) if provider == "openai" else None,
        )
        for _ in range(count)
    ]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_typed_prices_bill_what_the_generic_calculator_bills(provider):
    for row, usage, service_tier, data_residency in _random_cases(provider, 2500):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="typed-pricing-test",
            usage=usage,
            custom_llm_provider=provider,
            service_tier=service_tier,
            data_residency=data_residency,
            model_info=row,
            current_time=_MONDAY_NOON_UTC,
        )
        multiplier = _provider_multiplier(provider, row, usage)

        typed = typed_cost_per_token(
            _typed_pricing(provider, row, usage, service_tier, data_residency), usage, _MONDAY_NOON_UTC
        )

        assert typed == pytest.approx((prompt_cost * multiplier, completion_cost * multiplier), rel=1e-9), (
            row,
            usage,
            service_tier,
            data_residency,
        )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_reported_rates_are_the_rates_the_generic_calculator_reports(provider):
    for row, usage, service_tier, data_residency in _random_cases(provider, 600):
        generic = _cost_map_billed_rates(
            model_info=row,
            usage=usage,
            custom_llm_provider=provider,
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=None,
            current_time=_MONDAY_NOON_UTC,
        )

        typed = get_billed_token_rates(
            model="typed-pricing-test",
            custom_llm_provider=provider,
            usage=usage,
            service_tier=service_tier,
            data_residency=data_residency,
            current_time=_MONDAY_NOON_UTC,
            model_info=row,
        )

        assert typed is not None
        assert typed.cache_read_input_image_token_cost == generic.cache_read_input_image_token_cost
        for field in (
            "input_cost_per_token",
            "output_cost_per_token",
            "cache_read_input_token_cost",
            "cache_read_input_audio_token_cost",
            "cache_creation_input_token_cost",
            "output_cost_per_reasoning_token",
        ):
            assert getattr(typed, field) == pytest.approx(getattr(generic, field), rel=1e-9), (field, row, usage)
        assert dict(typed.cache_creation_input_token_cost_by_ttl) == pytest.approx(
            dict(generic.cache_creation_input_token_cost_by_ttl), rel=1e-9
        ), (row, usage, service_tier)


def _catalog_usages():
    for prompt_tokens in (1_000, 150_000, 300_000):
        totals = {"prompt_tokens": prompt_tokens, "completion_tokens": 2_000, "total_tokens": prompt_tokens + 2_000}
        yield Usage(
            **totals,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=400,
                audio_tokens=50,
                image_tokens=50,
                cache_creation_tokens=300,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=100, ephemeral_12h_input_tokens=100
                ),
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=500, audio_tokens=100, image_tokens=100
            ),
        )
        yield Usage(**totals)


@pytest.mark.parametrize(("model", "provider"), _CATALOG_ROWS)
def test_every_catalog_row_bills_what_the_generic_calculator_bills(model, provider):
    row = litellm.get_model_info(model=model, custom_llm_provider=provider)
    for usage in _catalog_usages():
        for service_tier in _SERVICE_TIERS[provider]:
            prompt_cost, completion_cost = generic_cost_per_token(
                model=model, usage=usage, custom_llm_provider=provider, service_tier=service_tier, data_residency="eu"
            )
            typed = (
                openai_cost_per_token(model=model, usage=usage, service_tier=service_tier, data_residency="eu")
                if provider == "openai"
                else anthropic_cost_per_token(model=model, usage=usage, service_tier=service_tier)
            )
            assert typed == pytest.approx((prompt_cost, completion_cost), rel=1e-9), (service_tier, usage)

            batch = _CONFIGS[provider].model_validate(row).batch_prices(usage.prompt_tokens)
            generic_batch = get_batch_cost_rates(row, usage, provider)
            assert (batch.input, batch.output, batch.cache_read, batch.cache_write, batch.cache_read_image) == (
                generic_batch.input,
                generic_batch.output,
                generic_batch.cache_read,
                generic_batch.cache_creation,
                generic_batch.cache_read_image,
            )
            assert {
                ttl: rate
                for ttl, rate in (("1hr", batch.cache_write_1h), ("12hr", batch.cache_write_12h))
                if rate is not None
            } == dict(generic_batch.cache_creation_rates_by_ttl)


_BASE_RATES = {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
_ONE_HOUR_WRITE = Usage(
    prompt_tokens=1_000,
    completion_tokens=100,
    prompt_tokens_details=PromptTokensDetailsWrapper(
        cache_creation_tokens=400,
        cache_creation_token_details=CacheCreationTokenDetails(ephemeral_1h_input_tokens=400),
    ),
)
_TWELVE_HOUR_WRITE = Usage(
    prompt_tokens=1_000,
    completion_tokens=100,
    prompt_tokens_details=PromptTokensDetailsWrapper(
        cache_creation_tokens=400,
        cache_creation_token_details=CacheCreationTokenDetails(ephemeral_12h_input_tokens=400),
    ),
)
_LONG_PROMPT = Usage(prompt_tokens=250_000, completion_tokens=100)


@pytest.mark.parametrize(
    ("cost_per_token", "usage", "unbilled_field"),
    [
        (openai_cost_per_token, _ONE_HOUR_WRITE, "cache_creation_input_token_cost_above_1hr"),
        (anthropic_cost_per_token, _TWELVE_HOUR_WRITE, "cache_creation_input_token_cost_above_12hr"),
        (openai_cost_per_token, _LONG_PROMPT, "input_cost_per_token_above_200k_tokens"),
        (anthropic_cost_per_token, _LONG_PROMPT, "input_cost_per_token_above_200k_tokens"),
        (anthropic_cost_per_token, _LONG_PROMPT, "input_cost_per_token_flex"),
    ],
)
def test_a_price_outside_the_provider_schema_is_not_billed_and_is_reported(
    cost_per_token, usage, unbilled_field, caplog
):
    model = f"schema-test-{unbilled_field}"
    baseline = cost_per_token(model=model, usage=usage, service_tier="flex", model_info=_BASE_RATES)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        cost = cost_per_token(
            model=model, usage=usage, service_tier="flex", model_info={**_BASE_RATES, unbilled_field: 1.0}
        )

    assert cost == baseline
    assert unbilled_field in caplog.text


@pytest.mark.parametrize(
    ("cost_per_token", "usage", "billed_field"),
    [
        (openai_cost_per_token, _TWELVE_HOUR_WRITE, "cache_creation_input_token_cost_above_12hr"),
        (anthropic_cost_per_token, _ONE_HOUR_WRITE, "cache_creation_input_token_cost_above_1hr"),
    ],
)
def test_a_long_ttl_cache_write_bills_at_the_provider_long_ttl_rate(cost_per_token, usage, billed_field):
    prompt_cost, _ = cost_per_token(model="schema-test", usage=usage, model_info={**_BASE_RATES, billed_field: 5e-6})

    assert prompt_cost == pytest.approx(600 * 1e-6 + 400 * 5e-6)


def test_a_deployment_price_override_bills_an_openai_request_at_the_deployment_rate(monkeypatch):
    monkeypatch.setitem(litellm.model_cost, "typed-override-model", {**_BASE_RATES, "litellm_provider": "openai"})

    prompt_cost, completion_cost = litellm.cost_per_token(
        model="typed-override-model",
        custom_llm_provider="openai",
        usage_object=Usage(prompt_tokens=1_000, completion_tokens=100),
        requested_model_info_override={"input_cost_per_token": 7e-6, "output_cost_per_token": 9e-6},
    )

    assert (prompt_cost, completion_cost) == pytest.approx((1_000 * 7e-6, 100 * 9e-6))


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_a_batch_bills_the_long_context_batch_rate_only_above_the_provider_threshold(monkeypatch, provider):
    threshold = {"openai": 272_000, "anthropic": 100_000}[provider]
    monkeypatch.setitem(
        litellm.model_cost,
        "typed-batch-model",
        {
            **_BASE_RATES,
            "input_cost_per_token_batches": 5e-7,
            "output_cost_per_token_batches": 1e-6,
            f"input_cost_per_token_above_{threshold // 1000}k_tokens_batches": 8e-7,
            "input_cost_per_token_above_50k_tokens_batches": 1.0,
            "litellm_provider": provider,
        },
    )

    def batch_cost(prompt_tokens):
        return batch_cost_calculator(
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=100),
            model="typed-batch-model",
            custom_llm_provider=provider,
        )

    assert batch_cost(threshold) == pytest.approx((threshold * 5e-7, 100 * 1e-6))
    assert batch_cost(threshold + 1) == pytest.approx(((threshold + 1) * 8e-7, 100 * 1e-6))
