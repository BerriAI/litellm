"""
Unit tests for multi-budget-window enforcement on API keys.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _user_multi_budget_check,
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
    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check; no injection seam
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

    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check
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

    async def fake_get_spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check
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

    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check; no injection seam
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ):
        # Should not raise TypeError / KeyError — model_dump() coerces the object
        await _virtual_key_multi_budget_check(valid_token=token)


def _make_user_token(**kwargs) -> UserAPIKeyAuth:
    defaults = dict(
        user_id="user-1",
        spend=0.0,
        user_budget_limits=None,
    )
    defaults.update(kwargs)
    return UserAPIKeyAuth(**defaults)


@pytest.mark.asyncio
async def test_user_with_no_windows_passes():
    assert await _user_multi_budget_check(valid_token=_make_user_token(), team_object=None, general_settings={}) is None


@pytest.mark.asyncio
async def test_user_under_all_windows_passes():
    token = _make_user_token(
        user_budget_limits=[
            {"budget_duration": "24h", "max_budget": 10.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 100.0, "reset_at": None},
        ]
    )
    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check; no injection seam
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=1.0,
    ) as spend_mock:
        await _user_multi_budget_check(valid_token=token, team_object=None, general_settings={})

    counter_keys = [call.kwargs["counter_key"] for call in spend_mock.await_args_list]
    assert counter_keys == [
        "spend:user:user-1:window:24h",
        "spend:user:user-1:window:30d",
    ]
    assert all(call.kwargs["window_entity_type"] == "User" for call in spend_mock.await_args_list)


@pytest.mark.asyncio
async def test_user_windows_share_one_redis_batch_read(monkeypatch):
    import litellm.proxy.proxy_server as ps
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.spend_tracking.spend_counter_batch import spend_counter_batch_scope

    token = _make_user_token(
        user_budget_limits=[
            {"budget_duration": "24h", "max_budget": 10.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 100.0, "reset_at": None},
        ]
    )
    redis = MagicMock()
    redis.async_batch_get_cache = AsyncMock(
        return_value={"spend:user:user-1:window:24h": 1.0, "spend:user:user-1:window:30d": 2.0}
    )
    redis.async_get_cache = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", DualCache(redis_cache=redis))
    monkeypatch.setattr(ps, "prisma_client", None)

    with spend_counter_batch_scope(redis):
        await _user_multi_budget_check(valid_token=token, team_object=None, general_settings={})

    assert redis.async_batch_get_cache.await_count == 1
    assert sorted(redis.async_batch_get_cache.await_args.kwargs["key_list"]) == [
        "spend:user:user-1:window:24h",
        "spend:user:user-1:window:30d",
    ]
    assert redis.async_get_cache.await_count == 0


@pytest.mark.asyncio
async def test_user_over_any_window_raises():
    token = _make_user_token(
        user_budget_limits=[
            {"budget_duration": "24h", "max_budget": 50.0, "reset_at": None},
            {"budget_duration": "30d", "max_budget": 5.0, "reset_at": None},
        ]
    )

    spend_by_window = [1.0, 10.0]
    call_count = 0

    async def fake_get_spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        nonlocal call_count
        val = spend_by_window[call_count]
        call_count += 1
        return val

    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(valid_token=token, team_object=None, general_settings={})

    err = exc_info.value
    assert err.status_code == 429
    assert "30d" in str(err)
    assert "User=user-1" in str(err)


@pytest.mark.asyncio
async def test_jwt_built_token_carries_user_budget_limits_and_is_blocked():
    """JWT auth has no key; user windows must still be enforced through the
    UserAPIKeyAuth.user_budget_limits field populated from the user row."""
    token = _make_user_token(
        api_key=None,
        user_budget_limits=[
            {"budget_duration": "1d", "max_budget": 2.0, "reset_at": None},
        ],
    )
    assert token.user_budget_limits[0].max_budget == 2.0

    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check; no injection seam
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=5.0,
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _user_multi_budget_check(valid_token=token, team_object=None, general_settings={})

    assert "user-1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_user_windows_skipped_for_team_key_unless_flag_set():
    """Matches _user_max_budget_check: keys owned by a team don't inherit the
    user's windows unless apply_user_budget_to_team_keys is enabled."""
    from litellm.proxy._types import LiteLLM_TeamTable

    token = _make_user_token(user_budget_limits=[{"budget_duration": "1d", "max_budget": 2.0, "reset_at": None}])
    team = LiteLLM_TeamTable(team_id="team-1")

    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check; no injection seam
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=100.0,
    ) as spend_mock:
        await _user_multi_budget_check(valid_token=token, team_object=team, general_settings={})
    spend_mock.assert_not_awaited()

    with patch(  # test-quality-ok: get_current_spend is a lazy module import inside the check; no injection seam
        "litellm.proxy.proxy_server.get_current_spend",
        new_callable=AsyncMock,
        return_value=100.0,
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await _user_multi_budget_check(
                valid_token=token,
                team_object=team,
                general_settings={"apply_user_budget_to_team_keys": True},
            )
