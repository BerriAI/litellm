"""
Unit tests for multi-budget-window enforcement on API keys.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import _team_multi_budget_check, _virtual_key_multi_budget_check


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

    async def fake_get_spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch("litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend):
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

    async def fake_get_spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch("litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend):
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


def _flush_spend_counter(monkeypatch) -> None:
    """Simulate the 2-pod issue #26672 state: Redis answers (so it is
    reachable) but the window counter key is gone, and no DB is available to
    reseed it, so the read falls all the way through to the caller's fallback."""
    flushed_cache = MagicMock()
    flushed_cache.redis_cache = MagicMock()
    flushed_cache.redis_cache.async_get_cache = AsyncMock(return_value=None)
    flushed_cache.in_memory_cache = MagicMock()
    flushed_cache.in_memory_cache.get_cache = MagicMock(return_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", flushed_cache)
    monkeypatch.setattr(ps, "prisma_client", None)


@pytest.mark.asyncio
async def test_flushed_window_counter_with_over_budget_key_spend_still_rejects(monkeypatch):
    """Issue #26672: a Redis flush losing the per-window counter must not read
    as a fresh empty window. When neither the counter nor the per-window DB
    total is readable, the key's cumulative spend (an upper bound of any
    window) must still drive the rejection."""
    _flush_spend_counter(monkeypatch)

    token = _make_valid_token(
        spend=99.0,
        budget_limits=[{"budget_duration": "1d", "max_budget": 30.0, "reset_at": None}],
    )

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _virtual_key_multi_budget_check(valid_token=token)

    assert exc_info.value.status_code == 429
    assert "1d" in str(exc_info.value)


@pytest.mark.asyncio
async def test_flushed_window_counter_with_under_budget_key_spend_admits(monkeypatch):
    """The cumulative-spend fallback must not over-block: a key whose all-time
    spend is below the window limit still admits when the counter is lost."""
    _flush_spend_counter(monkeypatch)

    token = _make_valid_token(
        spend=5.0,
        budget_limits=[{"budget_duration": "1d", "max_budget": 30.0, "reset_at": None}],
    )

    # Should not raise
    await _virtual_key_multi_budget_check(valid_token=token)


@pytest.mark.asyncio
async def test_flushed_window_counter_with_over_budget_team_spend_still_rejects(monkeypatch):
    """Same leak as the key path, for a team window: a lost counter must not
    bypass the team's per-window budget."""
    _flush_spend_counter(monkeypatch)

    team = LiteLLM_TeamTable(
        team_id="team-1",
        spend=99.0,
        budget_limits=[{"budget_duration": "1d", "max_budget": 30.0, "reset_at": None}],
    )

    with pytest.raises(litellm.BudgetExceededError):
        await _team_multi_budget_check(team_object=team)


@pytest.mark.asyncio
async def test_flushed_window_counter_flag_off_restores_cumulative_fallback_disabled(monkeypatch):
    """general_settings.cumulative_window_fallback=false restores the pre-PR
    degraded-path behavior: with neither counter nor DB readable the window
    reads 0 instead of the entity's cumulative spend."""
    _flush_spend_counter(monkeypatch)
    monkeypatch.setattr(ps, "general_settings", {"cumulative_window_fallback": False})

    token = _make_valid_token(
        spend=99.0,
        budget_limits=[{"budget_duration": "1d", "max_budget": 30.0, "reset_at": None}],
    )

    # Should not raise: the fallback is disabled, so the window reads 0.
    await _virtual_key_multi_budget_check(valid_token=token)


@pytest.mark.asyncio
async def test_flushed_window_counter_flag_default_on_keeps_the_rejection(monkeypatch):
    """Default (flag unset) keeps the fix: cumulative spend still drives the
    rejection when the counter and the DB total are both unreadable."""
    _flush_spend_counter(monkeypatch)
    monkeypatch.setattr(ps, "general_settings", {})

    token = _make_valid_token(
        spend=99.0,
        budget_limits=[{"budget_duration": "1d", "max_budget": 30.0, "reset_at": None}],
    )

    with pytest.raises(litellm.BudgetExceededError):
        await _virtual_key_multi_budget_check(valid_token=token)
