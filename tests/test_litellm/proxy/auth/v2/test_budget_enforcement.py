"""Verify the budget hook enforces limits on an enrichment-shaped identity.

auth_v2's enrichment populates user_max_budget/user_spend on master/JWT/OAuth
identities (which have no key row). This drives the real max_budget_limiter hook
with such an identity, mocking only the spend counter, to confirm the
enrichment -> hook chain produces a 429. Real-counter accuracy is the one piece
that still needs a live proxy.
"""

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter


@pytest.fixture
def spend(monkeypatch):
    """Control the reported current spend; disable the reservation short-circuit."""

    def _set(value):
        async def fake_get_current_spend(counter_key, fallback_spend):
            return value

        monkeypatch.setattr(
            "litellm.proxy.proxy_server.get_current_spend", fake_get_current_spend
        )
        monkeypatch.setattr(
            "litellm.proxy.spend_tracking.budget_reservation.get_reserved_counter_keys",
            lambda reservation: set(),
        )

    return _set


def _jwt_identity(**overrides):
    # Shape an enrichment-populated identity for a teamless JWT/OAuth user: no key,
    # user-level budget filled from the user row.
    base = dict(user_id="u_jwt", team_id=None, user_max_budget=10.0, user_spend=0.0)
    base.update(overrides)
    return UserAPIKeyAuth(**base)


@pytest.mark.asyncio
async def test_enriched_user_over_budget_is_blocked(spend):
    spend(15.0)
    hook = _PROXY_MaxBudgetLimiter()
    with pytest.raises(HTTPException) as exc:
        await hook.async_pre_call_hook(_jwt_identity(), DualCache(), {}, "completion")
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_enriched_user_under_budget_is_allowed(spend):
    spend(3.0)
    hook = _PROXY_MaxBudgetLimiter()
    result = await hook.async_pre_call_hook(
        _jwt_identity(), DualCache(), {}, "completion"
    )
    assert result is None  # not blocked


@pytest.mark.asyncio
async def test_no_user_budget_means_no_personal_enforcement(spend):
    spend(999.0)
    hook = _PROXY_MaxBudgetLimiter()
    result = await hook.async_pre_call_hook(
        _jwt_identity(user_max_budget=None), DualCache(), {}, "completion"
    )
    assert result is None


@pytest.mark.asyncio
async def test_team_member_personal_budget_is_skipped(spend):
    # Documents the boundary: the personal-budget hook exempts team requests, so a
    # JWT user *with* a team is governed by the team budget (item 4), not this hook.
    spend(15.0)
    hook = _PROXY_MaxBudgetLimiter()
    result = await hook.async_pre_call_hook(
        _jwt_identity(team_id="team:eng"), DualCache(), {}, "completion"
    )
    assert result is None
