"""
Unit tests for multi-budget-window enforcement on users.
"""

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.models.team import BudgetLimitEntry
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import _user_multi_budget_check


def _make_user(**kwargs) -> LiteLLM_UserTable:
    defaults = dict(
        user_id="test-user-123",
        spend=0.0,
        max_budget=None,
        budget_limits=[],
    )
    defaults.update(kwargs)
    return LiteLLM_UserTable(**defaults)


def test_user_model_accepts_budget_limits():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 50.0, "reset_at": None},
        ]
    )
    assert user.budget_limits is not None
    assert len(user.budget_limits) == 2


@pytest.mark.asyncio
async def test_user_no_budget_limits_passes():
    user = _make_user(budget_limits=[])
    await _user_multi_budget_check(user_object=user)


@pytest.mark.asyncio
async def test_user_none_budget_limits_passes():
    user = _make_user(budget_limits=None)
    await _user_multi_budget_check(user_object=user)


@pytest.mark.asyncio
async def test_user_none_object_passes():
    await _user_multi_budget_check(user_object=None)


@pytest.mark.asyncio
async def test_user_under_budget_passes():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 10.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 100.0, "reset_at": None},
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ):
        await _user_multi_budget_check(user_object=user)


@pytest.mark.asyncio
async def test_user_over_hourly_window_raises():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 100.0, "reset_at": None},
        ]
    )
    spend_by_window = [6.0, 6.0]
    call_count = 0

    async def fake_get_spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(user_object=user)

    err = exc_info.value
    assert err.status_code == 429
    assert "1hr" in str(err)
    assert "User=" in str(err)


@pytest.mark.asyncio
async def test_user_over_daily_window_raises():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 50.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 5.0, "reset_at": None},
        ]
    )
    spend_by_window = [1.0, 10.0]
    call_count = 0

    async def fake_get_spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(user_object=user)

    err = exc_info.value
    assert err.status_code == 429
    assert "1d" in str(err)


@pytest.mark.asyncio
async def test_user_budget_limit_entry_objects_coerced():
    user = _make_user(
        budget_limits=[
            BudgetLimitEntry(budget_duration="1hr", max_budget=10.0),
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ):
        await _user_multi_budget_check(user_object=user)
