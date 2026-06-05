"""
Unit tests for multi-budget-window enforcement on API keys.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

import litellm
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _coerce_budget_limit_window_for_check,
    _user_multi_budget_check,
    _virtual_key_multi_budget_check,
    common_checks,
)


def _make_valid_token(**kwargs) -> UserAPIKeyAuth:
    defaults = dict(
        token="sk-test-token",
        key_name="test",
        spend=0.0,
        max_budget=None,
        budget_limits=[],
    )
    defaults.update(kwargs)
    return UserAPIKeyAuth(**defaults)


@pytest.mark.asyncio
async def test_no_budget_limits_passes():
    """Keys with empty budget_limits should pass without raising."""
    token = _make_valid_token(budget_limits=[])
    # Should not raise
    await _virtual_key_multi_budget_check(valid_token=token)


@pytest.mark.asyncio
async def test_under_budget_passes():
    """Key with spend under all windows should pass."""
    token = _make_valid_token(
        budget_limits=[
            {"budget_duration": "24h", "max_budget": 10.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 100.0, "reset_at": None},
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,  # well under both windows
    ):
        await _virtual_key_multi_budget_check(valid_token=token)


@pytest.mark.asyncio
async def test_over_first_window_raises():
    """Key exceeding the first (daily) window should raise BudgetExceededError."""
    token = _make_valid_token(
        budget_limits=[
            {"budget_duration": "24h", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 100.0, "reset_at": None},
        ]
    )

    spend_by_window = [6.0, 6.0]  # over daily, under monthly

    call_count = 0

    async def fake_get_spend(counter_key, fallback_spend):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _virtual_key_multi_budget_check(valid_token=token)

    err = exc_info.value
    assert err.status_code == 429
    assert "24h" in str(err)
    assert "Key over" in str(err)


@pytest.mark.asyncio
async def test_over_second_window_raises():
    """Key exceeding only the monthly window should raise BudgetExceededError referencing 30d."""
    token = _make_valid_token(
        budget_limits=[
            {"budget_duration": "24h", "max_budget": 50.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 5.0, "reset_at": None},
        ]
    )

    spend_by_window = [1.0, 10.0]  # under daily, over monthly

    call_count = 0

    async def fake_get_spend(counter_key, fallback_spend):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _virtual_key_multi_budget_check(valid_token=token)

    err = exc_info.value
    assert err.status_code == 429
    assert "30d" in str(err)


@pytest.mark.asyncio
async def test_budget_limit_entry_objects_coerced():
    """BudgetLimitEntry Pydantic objects (not dicts) must be handled without KeyError.

    While budget_limits is normally serialized as List[dict], the auth check must
    tolerate BudgetLimitEntry objects in case they arrive without prior serialization.
    """
    from litellm.proxy._types import BudgetLimitEntry

    token = _make_valid_token(budget_limits=[])
    # Bypass Pydantic validation to simulate BudgetLimitEntry objects reaching the check
    object.__setattr__(
        token,
        "budget_limits",
        [BudgetLimitEntry(budget_duration="24h", max_budget=10.0)],
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ):
        # Should not raise TypeError / KeyError — model_dump() coerces the object
        await _virtual_key_multi_budget_check(valid_token=token)


@pytest.mark.asyncio
async def test_user_budget_limits_under_budget_passes():
    """User budget_limits should use spend:user:{user_id}:window:{duration} counters."""
    user = LiteLLM_UserTable(
        user_id="user-budget-window",
        budget_limits=[
            {"budget_duration": "1d", "max_budget": 10.0, "reset_at": None},
        ],
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ) as mock_get_spend:
        await _user_multi_budget_check(user_object=user)

    mock_get_spend.assert_awaited_once_with(
        counter_key="spend:user:user-budget-window:window:1d",
        fallback_spend=0.0,
    )


@pytest.mark.asyncio
async def test_user_budget_limits_over_window_raises():
    """A user exceeding any configured budget window should be blocked."""
    user = LiteLLM_UserTable(
        user_id="user-budget-window",
        budget_limits=[
            {"budget_duration": "1d", "max_budget": 5.0, "reset_at": None},
        ],
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=6.0,
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(user_object=user)

    err = exc_info.value
    assert err.status_code == 429
    assert "User=user-budget-window" in str(err)
    assert "1d" in str(err)


@pytest.mark.parametrize("empty_duration", ["", None])
def test_coerce_skips_window_with_empty_budget_duration(empty_duration):
    """
    Regression: the auth check and the budget-reservation path must agree on
    what counts as a missing budget_duration. The reservation path rejects with
    `if not budget_duration`, so an empty-string duration must be rejected here
    too — otherwise the auth check builds a counter key ending in `:window:` that
    never accumulates spend, silently disabling the window instead of enforcing it.
    """
    assert (
        _coerce_budget_limit_window_for_check(
            window={"budget_duration": empty_duration, "max_budget": 10.0}
        )
        is None
    )


@pytest.mark.asyncio
async def test_empty_budget_duration_window_is_skipped_not_queried():
    """A window with an empty-string duration must be skipped entirely, never
    turned into a malformed `spend:key:...:window:` counter lookup."""
    token = _make_valid_token(
        budget_limits=[{"budget_duration": "", "max_budget": 1.0, "reset_at": None}]
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
    ) as mock_get_spend:
        await _virtual_key_multi_budget_check(valid_token=token)

    mock_get_spend.assert_not_awaited()


def _make_team(team_id: str = "team-xyz") -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=team_id,
        models=["gpt-3.5-turbo"],
        blocked=False,
        spend=0.0,
        max_budget=None,
    )


async def _invoke_common_checks(*, team_object, user_object):
    """Minimal common_checks call exercising just the user-budget block."""
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }
    valid_token = UserAPIKeyAuth(
        token="sk-test-token",
        models=["gpt-3.5-turbo"],
        spend=0.0,
        max_budget=None,
        budget_limits=[],
    )
    return await common_checks(
        request_body=request_body,
        team_object=team_object,
        user_object=user_object,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route="/chat/completions",
        llm_router=None,
        proxy_logging_obj=MagicMock(),
        valid_token=valid_token,
        request=MagicMock(spec=Request),
    )


@pytest.mark.asyncio
async def test_common_checks_skips_user_multi_budget_for_team_keys():
    """
    Regression: when a request goes through a team key, the user's personal
    budget-window check must be skipped — team budgets already apply, and the
    personal user windows would otherwise also block the request (and pin
    spend onto the user's window counter from a team-scoped call path).
    """
    user = LiteLLM_UserTable(
        user_id="user-with-windows",
        budget_limits=[
            {"budget_duration": "1d", "max_budget": 1.0, "reset_at": None},
        ],
    )
    team = _make_team()

    with patch(
        "litellm.proxy.auth.auth_checks._user_multi_budget_check",
        new_callable=AsyncMock,
    ) as user_check:
        await _invoke_common_checks(team_object=team, user_object=user)

    user_check.assert_not_awaited()


@pytest.mark.asyncio
async def test_common_checks_runs_user_multi_budget_for_personal_keys():
    """Companion to the team-key skip test: personal keys must still run it."""
    user = LiteLLM_UserTable(
        user_id="user-with-windows",
        budget_limits=[
            {"budget_duration": "1d", "max_budget": 1.0, "reset_at": None},
        ],
    )

    with patch(
        "litellm.proxy.auth.auth_checks._user_multi_budget_check",
        new_callable=AsyncMock,
    ) as user_check:
        await _invoke_common_checks(team_object=None, user_object=user)

    user_check.assert_awaited_once_with(user_object=user)
