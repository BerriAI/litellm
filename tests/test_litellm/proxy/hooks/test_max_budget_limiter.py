"""
Unit tests for the personal-budget pre-call hook.

The reservation path (added in PR #26845) atomically pre-fills the same
`spend:user:{user_id}` counter this hook reads, admitting at a strict-`<`
boundary. Re-checking with `>=` after reservation would reject requests the
reservation already admitted when the reservation fills the counter to
exactly `max_budget` (e.g. requests with no `max_tokens` cap fall back to
reserving the smallest remaining headroom).

These tests pin the skip-when-reserved behavior and guard against drift.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter


def _make_user_api_key_auth(
    user_id: str = "user-1",
    user_max_budget: float = 10.0,
    user_spend: float = 0.0,
    team_id=None,
    budget_reservation=None,
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id=user_id,
        user_max_budget=user_max_budget,
        user_spend=user_spend,
        team_id=team_id,
        budget_reservation=budget_reservation,
    )


@pytest.mark.asyncio
async def test_under_budget_passes():
    handler = _PROXY_MaxBudgetLimiter()
    user_api_key_dict = _make_user_api_key_auth(user_max_budget=10.0)

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=3.0),
    ):
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={},
            call_type="completion",
        )

    assert result is None


@pytest.mark.asyncio
async def test_over_budget_rejects_without_reservation():
    handler = _PROXY_MaxBudgetLimiter()
    user_api_key_dict = _make_user_api_key_auth(user_max_budget=10.0)

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=10.0),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=DualCache(),
                data={},
                call_type="completion",
            )

    assert exc_info.value.status_code == 429
    assert "Max budget limit reached." in exc_info.value.detail


@pytest.mark.asyncio
async def test_skips_when_user_counter_is_reserved():
    """
    Reservation atomically pre-fills `spend:user:{user_id}` and admits the
    request. The legacy `>=` check must not double-enforce on the same
    counter — that's what produced the boundary regression where a fresh
    user with no `max_tokens` cap got 429'd on their first request.
    """
    handler = _PROXY_MaxBudgetLimiter()
    user_api_key_dict = _make_user_api_key_auth(
        user_id="user-1",
        user_max_budget=10.0,
        budget_reservation={
            "reserved_cost": 10.0,
            "entries": [
                {
                    "counter_key": "spend:user:user-1",
                    "entity_type": "User",
                    "entity_id": "user-1",
                    "reserved_cost": 10.0,
                    "applied_adjustment": 0.0,
                }
            ],
            "finalized": False,
        },
    )

    # `get_current_spend` would return 10.0 here (counter pre-filled by the
    # reservation). The hook must skip without reading it.
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=10.0),
    ) as mock_get_spend:
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={},
            call_type="completion",
        )

    assert result is None
    mock_get_spend.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_skip_when_reservation_covers_a_different_counter():
    """
    A reservation that only covers e.g. `spend:team:{team_id}` (not the user
    counter) must not exempt the user-budget check.
    """
    handler = _PROXY_MaxBudgetLimiter()
    user_api_key_dict = _make_user_api_key_auth(
        user_id="user-1",
        user_max_budget=10.0,
        budget_reservation={
            "reserved_cost": 5.0,
            "entries": [
                {
                    "counter_key": "spend:team:team-x",
                    "entity_type": "Team",
                    "entity_id": "team-x",
                    "reserved_cost": 5.0,
                    "applied_adjustment": 0.0,
                }
            ],
            "finalized": False,
        },
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=10.0),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=DualCache(),
                data={},
                call_type="completion",
            )

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_team_keys_skip_personal_budget():
    handler = _PROXY_MaxBudgetLimiter()
    user_api_key_dict = _make_user_api_key_auth(
        user_max_budget=10.0,
        team_id="team-1",
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=999.0),
    ) as mock_get_spend:
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={},
            call_type="completion",
        )

    assert result is None
    mock_get_spend.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_max_budget_passes():
    handler = _PROXY_MaxBudgetLimiter()
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test",
        user_id="user-1",
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=999.0),
    ) as mock_get_spend:
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={},
            call_type="completion",
        )

    assert result is None
    mock_get_spend.assert_not_awaited()
