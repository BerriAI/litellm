"""
Unit tests for multi-budget-window enforcement on API keys and teams.
"""

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _team_multi_budget_check,
    _virtual_key_multi_budget_check,
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


# ---------------------------------------------------------------------------
# Team multi-budget-window enforcement
# ---------------------------------------------------------------------------


def _make_team(**kwargs) -> LiteLLM_TeamTable:
    defaults = dict(team_id="team-1", budget_limits=[])
    defaults.update(kwargs)
    return LiteLLM_TeamTable(**defaults)


@pytest.mark.asyncio
async def test_team_no_budget_limits_passes():
    """Teams with empty budget_limits should pass without touching spend."""
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
    ) as mock_spend:
        await _team_multi_budget_check(team_object=_make_team(budget_limits=[]))
    mock_spend.assert_not_called()


@pytest.mark.asyncio
async def test_team_none_object_passes():
    """A None team_object must short-circuit without raising."""
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
    ) as mock_spend:
        await _team_multi_budget_check(team_object=None)
    mock_spend.assert_not_called()


@pytest.mark.asyncio
async def test_team_under_budget_passes():
    """Team spend under all windows should pass."""
    team = _make_team(
        budget_limits=[
            {"budget_duration": "24h", "max_budget": 10.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 100.0, "reset_at": None},
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ):
        await _team_multi_budget_check(team_object=team)


@pytest.mark.asyncio
async def test_team_over_first_window_raises_with_team_scoped_counter_key():
    """Team exceeding the daily window raises and reads the team-scoped counter.

    Pins both the >= comparison and the counter_key shape
    (spend:team:{team_id}:window:{budget_duration}); a mutation to either
    would let team budgets leak or read the wrong counter.
    """
    team = _make_team(
        team_id="team-abc",
        budget_limits=[
            {"budget_duration": "24h", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 100.0, "reset_at": None},
        ],
    )

    seen_keys = []
    spend_by_window = [6.0, 6.0]  # over daily, under monthly

    async def fake_get_spend(counter_key, fallback_spend):
        seen_keys.append(counter_key)
        return spend_by_window[len(seen_keys) - 1]

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _team_multi_budget_check(team_object=team)

    err = exc_info.value
    assert err.status_code == 429
    assert "24h" in str(err)
    assert "team-abc" in str(err)
    assert seen_keys[0] == "spend:team:team-abc:window:24h"


@pytest.mark.asyncio
async def test_team_over_second_window_raises():
    """Team exceeding only the monthly window raises referencing 30d."""
    team = _make_team(
        team_id="team-xyz",
        budget_limits=[
            {"budget_duration": "24h", "max_budget": 50.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 5.0, "reset_at": None},
        ],
    )

    seen_keys = []
    spend_by_window = [1.0, 10.0]  # under daily, over monthly

    async def fake_get_spend(counter_key, fallback_spend):
        seen_keys.append(counter_key)
        return spend_by_window[len(seen_keys) - 1]

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _team_multi_budget_check(team_object=team)

    err = exc_info.value
    assert err.status_code == 429
    assert "30d" in str(err)
    assert seen_keys[1] == "spend:team:team-xyz:window:30d"


@pytest.mark.asyncio
async def test_team_spend_equal_to_budget_raises():
    """spend == max_budget must raise: the comparison is >=, not >."""
    team = _make_team(
        budget_limits=[
            {"budget_duration": "24h", "max_budget": 5.0, "reset_at": None},
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=5.0,
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await _team_multi_budget_check(team_object=team)
