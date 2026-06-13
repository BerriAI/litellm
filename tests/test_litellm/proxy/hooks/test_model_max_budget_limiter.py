import pytest

from litellm.caching.caching import DualCache
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)

KEY_HASH = "hashed-key-abc123"


@pytest.fixture
def limiter():
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())


@pytest.mark.asyncio
async def test_get_current_period_spend_returns_cached_value(limiter):
    cache_key = f"virtual_key_spend:{KEY_HASH}:gpt-4o:1d"
    await limiter.dual_cache.async_set_cache(key=cache_key, value=0.23)

    result = await limiter.get_current_period_spend(
        user_api_key_hash=KEY_HASH,
        model_max_budget={"gpt-4o": {"budget_limit": 0.50, "time_period": "1d"}},
    )

    assert result["gpt-4o"]["current_spend"] == 0.23


@pytest.mark.asyncio
async def test_get_current_period_spend_returns_zero_when_no_cache_entry(limiter):
    result = await limiter.get_current_period_spend(
        user_api_key_hash=KEY_HASH,
        model_max_budget={"gpt-4o": {"budget_limit": 0.50, "time_period": "1d"}},
    )

    assert result["gpt-4o"]["current_spend"] == 0.0


@pytest.mark.asyncio
async def test_get_current_period_spend_omits_model_without_time_period(limiter):
    result = await limiter.get_current_period_spend(
        user_api_key_hash=KEY_HASH,
        model_max_budget={"gpt-4o": {"budget_limit": 0.50}},
    )

    assert "gpt-4o" not in result


@pytest.mark.asyncio
async def test_get_current_period_spend_includes_budget_config_fields(limiter):
    result = await limiter.get_current_period_spend(
        user_api_key_hash=KEY_HASH,
        model_max_budget={
            "gpt-4o": {"budget_limit": 1.00, "time_period": "7d"},
            "gpt-4o-mini": {"budget_limit": 5.00, "time_period": "30d"},
        },
    )

    assert result["gpt-4o"]["budget_limit"] == 1.00
    assert result["gpt-4o"]["time_period"] == "7d"
    assert result["gpt-4o-mini"]["budget_limit"] == 5.00
    assert result["gpt-4o-mini"]["time_period"] == "30d"


@pytest.mark.asyncio
async def test_get_current_period_spend_provider_prefix_fallback(limiter):
    """openai/gpt-4o in model_max_budget should still find a cache entry keyed by the full name."""
    cache_key = f"virtual_key_spend:{KEY_HASH}:openai/gpt-4o:1d"
    await limiter.dual_cache.async_set_cache(key=cache_key, value=0.77)

    result = await limiter.get_current_period_spend(
        user_api_key_hash=KEY_HASH,
        model_max_budget={"openai/gpt-4o": {"budget_limit": 2.00, "time_period": "1d"}},
    )

    assert result["openai/gpt-4o"]["current_spend"] == 0.77
