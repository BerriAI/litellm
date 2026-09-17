from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.proxy.common_utils.prompt_cache_pricing import price_cache_tokens
from litellm.types.management_endpoints.prompt_cache_prediction import CacheTokenBuckets


def _tiered_rate(entry: Mapping[str, float | None], field: str, total: int) -> float:
    above_rate: Final = entry.get(f"{field}_above_200k_tokens") if total > 200_000 else None
    if above_rate is not None:
        return above_rate
    return entry.get(field) or 0.0


def _expected_cache_cost(model: str, tokens: CacheTokenBuckets) -> float:
    key: Final = litellm.get_model_info(model=model, custom_llm_provider="anthropic")["key"]
    entry: Final = litellm.model_cost[key]
    total: Final = tokens.total_tokens
    one_hour_rate: Final = _tiered_rate(entry, "cache_creation_input_token_cost_above_1hr", total)
    return (
        tokens.uncached_input_tokens * _tiered_rate(entry, "input_cost_per_token", total)
        + tokens.cache_read_input_tokens * _tiered_rate(entry, "cache_read_input_token_cost", total)
        + tokens.cache_creation_5m_input_tokens * _tiered_rate(entry, "cache_creation_input_token_cost", total)
        + tokens.cache_creation_1h_input_tokens * one_hour_rate
    )


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-5", "anthropic/claude-sonnet-4-6"])
def test_prices_all_cache_buckets_at_total_context_tier(model: str) -> None:
    tokens: Final = CacheTokenBuckets(
        uncached_input_tokens=100_000,
        cache_read_input_tokens=50_000,
        cache_creation_5m_input_tokens=20_000,
        cache_creation_1h_input_tokens=40_000,
    )
    assert price_cache_tokens(model, "unconfigured-deployment", tokens) == pytest.approx(
        _expected_cache_cost(model, tokens)
    )


@pytest.mark.parametrize("total", [200_000, 200_001])
def test_long_context_tier_starts_above_threshold(total: int) -> None:
    model: Final = "anthropic/claude-sonnet-4-5"
    tokens: Final = CacheTokenBuckets(
        uncached_input_tokens=total - 100_000,
        cache_creation_1h_input_tokens=10_000,
        cache_read_input_tokens=90_000,
    )
    actual: Final = price_cache_tokens(model, "unconfigured-deployment", tokens)
    assert actual == pytest.approx(_expected_cache_cost(model, tokens))


def test_deployment_tariff_wins_without_proxy_discounts_or_margins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", litellm.model_cost.copy())
    litellm.Router(
        model_list=[
            {
                "model_name": "cache-pricing-test",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-6",
                    "api_key": "test-only",
                    "input_cost_per_token": 0.00001,
                    "output_cost_per_token": 0.00002,
                    "cache_read_input_token_cost": 0.000001,
                    "cache_creation_input_token_cost": 0.0000125,
                    "cache_creation_input_token_cost_above_1hr": 0.00002,
                },
                "model_info": {"id": "cache-pricing-test-a"},
            }
        ]
    )
    monkeypatch.setattr(litellm, "cost_discount_config", {"anthropic": 0.5})
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": {"percentage": 0.3, "fixed_amount": 1.0}})
    tokens: Final = CacheTokenBuckets(
        uncached_input_tokens=3_000,
        cache_read_input_tokens=4_000,
        cache_creation_5m_input_tokens=1_000,
        cache_creation_1h_input_tokens=2_000,
    )
    assert price_cache_tokens("anthropic/claude-sonnet-4-6", "cache-pricing-test-a", tokens) == pytest.approx(0.0865)


@pytest.mark.parametrize("rate", [None, -1.0, float("nan"), float("inf"), "0.00001", True])
def test_unknown_for_absent_or_invalid_active_cache_rate(monkeypatch: pytest.MonkeyPatch, rate: object) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "cache-pricing-invalid",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "input_cost_per_token": 0.00001,
            "output_cost_per_token": 0.00002,
            "cache_creation_input_token_cost_above_1hr": rate,
        },
    )
    tokens: Final = CacheTokenBuckets(cache_creation_1h_input_tokens=4_000)
    assert price_cache_tokens("anthropic/claude-sonnet-4-6", "cache-pricing-invalid", tokens) is None


def test_missing_input_price_is_unknown_even_when_get_model_info_defaults_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(litellm.model_cost, "cache-pricing-missing", {"litellm_provider": "anthropic", "mode": "chat"})
    tokens: Final = CacheTokenBuckets(uncached_input_tokens=4_000)
    assert price_cache_tokens("cache-pricing-missing", "unconfigured-deployment", tokens) is None


def test_explicit_free_pricing_is_not_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "cache-pricing-free",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "cache_read_input_token_cost": 0.0,
            "cache_creation_input_token_cost": 0.0,
            "cache_creation_input_token_cost_above_1hr": 0.0,
        },
    )
    tokens: Final = CacheTokenBuckets(uncached_input_tokens=100, cache_read_input_tokens=5_000)
    assert price_cache_tokens("anthropic/claude-sonnet-4-6", "cache-pricing-free", tokens) == 0.0
