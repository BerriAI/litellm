"""
Unit tests for multi-budget-window enforcement on API keys.
"""

import re
from collections.abc import Mapping
from unittest.mock import AsyncMock, patch
from typing import Final

import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import _virtual_key_multi_budget_check
from litellm.proxy.auth import auth_checks
from litellm.proxy import proxy_server
from litellm.models.user import LiteLLM_UserTable
from litellm.redis.stores.spend_counters import user_counter, user_window_counter
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spend, blocked_by",
    [
        ({user_window_counter("u", "1h"): 4.0, user_window_counter("u", "30d"): 99.0}, None),
        (
            {user_window_counter("u", "1h"): 5.0, user_window_counter("u", "30d"): 1.0},
            "over 1h budget. Spend=$5.0000, Limit=$5.00",
        ),
        (
            {user_window_counter("u", "1h"): 1.0, user_window_counter("u", "30d"): 100.0},
            "over 30d budget. Spend=$100.0000, Limit=$100.00",
        ),
        ({user_counter("u:window:1h"): 50.0, user_counter("u:window:30d"): 500.0}, None),
    ],
    ids=["both_open", "hour_full", "month_full", "lookalike_user_spend"],
)
async def test_user_is_blocked_by_either_window(
    monkeypatch: pytest.MonkeyPatch, spend: Mapping[str, float], blocked_by: str | None
) -> None:
    counters: Final = FakeSpendCounters(redis=dict(spend))
    monkeypatch.setattr(proxy_server, "spend_counter_store", counters.store())
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    user: Final = LiteLLM_UserTable(
        user_id="u",
        budget_limits=[
            {"budget_duration": "1h", "max_budget": 5.0, "reset_at": "2026-08-01T01:00:00Z"},
            {"budget_duration": "30d", "max_budget": 100.0, "reset_at": "2026-08-31T00:00:00Z"},
        ],
    )
    if blocked_by is not None:
        with pytest.raises(litellm.BudgetExceededError, match=re.escape(blocked_by)):
            await auth_checks._user_multi_budget_check(user)
        return
    await auth_checks._user_multi_budget_check(user)
    assert {key: counters.redis[key] for key in spend} == spend


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

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _virtual_key_multi_budget_check(valid_token=token)

    err = exc_info.value
    assert err.status_code == 422
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

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", side_effect=fake_get_spend
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _virtual_key_multi_budget_check(valid_token=token)

    err = exc_info.value
    assert err.status_code == 422
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
