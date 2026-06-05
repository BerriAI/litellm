"""Verify auth_v2 enforces team budgets by reusing v1's exact budget functions.

enforce_hierarchy_budgets calls the same _team_max_budget_check v1 uses in
common_checks (single authority). Drives it with a team over/under budget,
mocking only get_team_object and the spend counter; the global cap is inert here
(litellm.max_budget defaults to 0). Real-counter accuracy is the live piece.
"""

from types import SimpleNamespace

import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.v2.stages.budgets import enforce_hierarchy_budgets


class _Logging:
    async def budget_alerts(self, **kwargs):
        return None


def _ctx():
    return SimpleNamespace(
        prisma_client=None,
        user_api_key_cache=None,
        proxy_logging_obj=_Logging(),
        parent_otel_span=None,
    )


@pytest.fixture
def team(monkeypatch):
    def _set(max_budget, current_spend):
        team_obj = SimpleNamespace(
            team_id="t1", max_budget=max_budget, spend=0.0, organization_id=None
        )

        async def fake_get_team_object(**kwargs):
            return team_obj

        async def fake_get_current_spend(counter_key, fallback_spend):
            return current_spend

        monkeypatch.setattr(
            "litellm.proxy.auth.auth_checks.get_team_object", fake_get_team_object
        )
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.get_current_spend", fake_get_current_spend
        )

    return _set


@pytest.mark.asyncio
async def test_team_over_budget_is_blocked(team):
    team(max_budget=10.0, current_spend=15.0)
    with pytest.raises(litellm.BudgetExceededError):
        await enforce_hierarchy_budgets(
            UserAPIKeyAuth(team_id="t1"), "/chat/completions", _ctx()
        )


@pytest.mark.asyncio
async def test_team_under_budget_is_allowed(team):
    team(max_budget=10.0, current_spend=5.0)
    await enforce_hierarchy_budgets(
        UserAPIKeyAuth(team_id="t1"), "/chat/completions", _ctx()
    )


@pytest.mark.asyncio
async def test_no_team_means_no_hierarchy_cap(team):
    # A keyless/teamless identity has no team to load; over-spend is irrelevant.
    team(max_budget=10.0, current_spend=999.0)
    await enforce_hierarchy_budgets(
        UserAPIKeyAuth(team_id=None), "/chat/completions", _ctx()
    )
