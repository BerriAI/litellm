from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)


@pytest.mark.asyncio
async def test_virtual_key_model_budget_rejects_at_exact_limit():
    """Per-model virtual-key budget must reject when spend equals max_budget."""
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    limiter._get_virtual_key_spend_for_model = AsyncMock(return_value=10.0)

    user_api_key_dict = UserAPIKeyAuth(
        token="test-token",
        model_max_budget={"gpt-4o": {"budget_limit": 10.0, "time_period": "1d"}},
    )

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await limiter.is_key_within_model_budget(
            user_api_key_dict=user_api_key_dict, model="gpt-4o"
        )
    assert exc_info.value.current_cost == 10.0
    assert exc_info.value.max_budget == 10.0


@pytest.mark.asyncio
async def test_virtual_key_model_budget_admits_below_limit():
    """Per-model virtual-key budget must admit when spend is below max_budget."""
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    limiter._get_virtual_key_spend_for_model = AsyncMock(return_value=9.99)

    user_api_key_dict = UserAPIKeyAuth(
        token="test-token",
        model_max_budget={"gpt-4o": {"budget_limit": 10.0, "time_period": "1d"}},
    )

    assert (
        await limiter.is_key_within_model_budget(
            user_api_key_dict=user_api_key_dict, model="gpt-4o"
        )
        is True
    )


@pytest.mark.asyncio
async def test_end_user_model_budget_rejects_at_exact_limit():
    """Per-model end-user budget must reject when spend equals max_budget."""
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    limiter._get_end_user_spend_for_model = AsyncMock(return_value=10.0)

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await limiter.is_end_user_within_model_budget(
            end_user_id="customer-1",
            end_user_model_max_budget={
                "gpt-4o": {"budget_limit": 10.0, "time_period": "1d"}
            },
            model="gpt-4o",
        )
    assert exc_info.value.current_cost == 10.0
    assert exc_info.value.max_budget == 10.0


@pytest.mark.asyncio
async def test_end_user_model_budget_admits_below_limit():
    """Per-model end-user budget must admit when spend is below max_budget."""
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    limiter._get_end_user_spend_for_model = AsyncMock(return_value=9.99)

    assert (
        await limiter.is_end_user_within_model_budget(
            end_user_id="customer-1",
            end_user_model_max_budget={
                "gpt-4o": {"budget_limit": 10.0, "time_period": "1d"}
            },
            model="gpt-4o",
        )
        is True
    )
