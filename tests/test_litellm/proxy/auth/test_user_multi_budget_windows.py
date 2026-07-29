"""
Unit tests for multi-budget-window enforcement on users.
"""

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.models.team import BudgetLimitEntry, LiteLLM_TeamTable
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


def _spend_by_counter_key(spend_map: dict[str, float]):
    async def fake_get_spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        return spend_map.get(counter_key, 0.0)

    return fake_get_spend


@pytest.mark.asyncio
async def test_three_windows_all_under_budget():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 50.0, "reset_at": None},
            {"budget_duration": "1mo", "max_budget": 500.0, "reset_at": None},
        ]
    )
    spend_map = {
        "spend:user:test-user-123:window:1hr": 4.0,
        "spend:user:test-user-123:window:1d": 40.0,
        "spend:user:test-user-123:window:1mo": 400.0,
    }
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        side_effect=_spend_by_counter_key(spend_map),
    ):
        await _user_multi_budget_check(user_object=user)


@pytest.mark.asyncio
async def test_three_windows_hourly_under_daily_over():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 50.0, "reset_at": None},
            {"budget_duration": "1mo", "max_budget": 500.0, "reset_at": None},
        ]
    )
    spend_map = {
        "spend:user:test-user-123:window:1hr": 2.0,
        "spend:user:test-user-123:window:1d": 55.0,
        "spend:user:test-user-123:window:1mo": 100.0,
    }
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        side_effect=_spend_by_counter_key(spend_map),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(user_object=user)

    assert "1d" in str(exc_info.value)
    assert "1hr" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_three_windows_only_monthly_over():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 50.0, "reset_at": None},
            {"budget_duration": "1mo", "max_budget": 500.0, "reset_at": None},
        ]
    )
    spend_map = {
        "spend:user:test-user-123:window:1hr": 1.0,
        "spend:user:test-user-123:window:1d": 10.0,
        "spend:user:test-user-123:window:1mo": 600.0,
    }
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        side_effect=_spend_by_counter_key(spend_map),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(user_object=user)

    assert "1mo" in str(exc_info.value)


@pytest.mark.asyncio
async def test_three_windows_hourly_over_stops_early():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 50.0, "reset_at": None},
            {"budget_duration": "1mo", "max_budget": 500.0, "reset_at": None},
        ]
    )
    spend_map = {
        "spend:user:test-user-123:window:1hr": 10.0,
        "spend:user:test-user-123:window:1d": 200.0,
        "spend:user:test-user-123:window:1mo": 999.0,
    }
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        side_effect=_spend_by_counter_key(spend_map),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(user_object=user)

    assert "1hr" in str(exc_info.value)


@pytest.mark.asyncio
async def test_spend_at_exact_limit_raises():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 50.0, "reset_at": None},
        ]
    )
    spend_map = {
        "spend:user:test-user-123:window:1hr": 5.0,
        "spend:user:test-user-123:window:1d": 10.0,
    }
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        side_effect=_spend_by_counter_key(spend_map),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(user_object=user)

    assert "1hr" in str(exc_info.value)


def _make_team(team_id: str = "team-abc") -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(team_id=team_id)


@pytest.mark.asyncio
async def test_team_key_enforces_user_windows_by_default():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=10.0,
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await _user_multi_budget_check(
                user_object=user,
                team_object=_make_team(),
            )


@pytest.mark.asyncio
async def test_team_key_skips_user_windows_when_setting_enabled():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=10.0,
    ):
        await _user_multi_budget_check(
            user_object=user,
            team_object=_make_team(),
            general_settings={"skip_user_budget_on_team_key": True},
        )


@pytest.mark.asyncio
async def test_no_team_enforces_user_windows_regardless_of_setting():
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=10.0,
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await _user_multi_budget_check(
                user_object=user,
                team_object=None,
                general_settings={"skip_user_budget_on_team_key": True},
            )
