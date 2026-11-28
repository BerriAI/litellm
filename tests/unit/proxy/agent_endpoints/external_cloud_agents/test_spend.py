from datetime import timedelta
from decimal import Decimal
from typing import Final

import pytest
from fastapi import BackgroundTasks

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.external_cloud_agents.endpoints import get_spend
from litellm.proxy.agent_endpoints.external_cloud_agents.models import MatchedSpend, SpendTotals
from litellm.proxy.agent_endpoints.external_cloud_agents.spend import report
from tests.unit.proxy.agent_endpoints.external_cloud_agents.conftest import Harness, NOW


def member(vendor_id: str = "member", **overrides: object) -> MatchedSpend:
    return MatchedSpend.model_validate(
        {
            "vendor_user_id": vendor_id,
            "email": "member@example.test",
            "grok_cost_usd": "2.30",
            "tokens": 35,
            "requests": 3,
            "litellm_user_id": "user",
            "litellm_gateway_spend_usd": "4.20",
            "team_ids": ("team-a", "team-b"),
            "team_names": ("Team A", "Team B"),
            **overrides,
        }
    )


def test_rollup_counts_gateway_once_for_duplicate_email_matches_and_supports_multiple_teams() -> None:
    result: Final = report("source", 30, NOW, (member(), member("second-vendor-id")))
    assert result.totals == SpendTotals(
        grok_cost_usd=Decimal("4.60"), litellm_gateway_spend_usd=Decimal("4.20"), tokens=70, requests=6
    )
    assert len(result.teams) == 1
    assert result.teams[0].team_name == "Multiple teams"
    assert tuple(team.team_id for team in result.teams[0].teams) == ("team-a", "team-b")
    assert result.teams[0].grok_cost_usd == result.totals.grok_cost_usd
    assert result.teams[0].litellm_gateway_spend_usd == result.totals.litellm_gateway_spend_usd
    assert sum(row.litellm_gateway_spend_usd for row in result.members) == result.totals.litellm_gateway_spend_usd
    assert sum(team.tokens for team in result.teams) == result.totals.tokens
    assert sum(team.requests for team in result.teams) == result.totals.requests


def test_team_attribution_preserves_exact_cost_and_usage_without_splitting() -> None:
    result: Final = report(
        "source",
        30,
        NOW,
        (member(grok_cost_usd="0.000003", litellm_gateway_spend_usd="0.000005", tokens=3, requests=1),),
    )
    assert tuple(team.grok_cost_usd for team in result.teams) == (Decimal("0.000003"),)
    assert tuple(team.litellm_gateway_spend_usd for team in result.teams) == (Decimal("0.000005"),)
    assert tuple(team.tokens for team in result.teams) == (3,)
    assert tuple(team.requests for team in result.teams) == (1,)
    assert sum(team.grok_cost_usd for team in result.teams) == result.totals.grok_cost_usd
    assert sum(team.litellm_gateway_spend_usd for team in result.teams) == result.totals.litellm_gateway_spend_usd


def test_single_team_member_is_attributed_wholly_to_that_team() -> None:
    result: Final = report("source", 30, NOW, (member(team_ids=("team-a",), team_names=("Team A",)),))
    assert len(result.teams) == 1
    assert result.teams[0].team_id == "team-a"
    assert result.teams[0].grok_cost_usd == result.totals.grok_cost_usd
    assert result.teams[0].litellm_gateway_spend_usd == result.totals.litellm_gateway_spend_usd


def test_unmatched_and_teamless_members_are_unassigned() -> None:
    unknown: Final = member("unknown", litellm_user_id=None, team_ids=(), team_names=(), litellm_gateway_spend_usd="0")
    teamless: Final = member("teamless", team_ids=(), team_names=())
    result: Final = report("source", 7, NOW, (unknown, teamless))
    assert len(result.teams) == 1
    assert result.teams[0].team_id is None
    assert result.teams[0].team_name == "Unassigned"
    assert result.teams[0].grok_cost_usd == result.totals.grok_cost_usd
    assert result.teams[0].litellm_gateway_spend_usd == Decimal("4.20")


def test_empty_spend_reports_zero_totals() -> None:
    result: Final = report("source", 30, NOW, ())
    assert result.members == ()
    assert result.teams == ()
    assert result.totals == SpendTotals()
    assert (result.end_day - result.start_day).days == 29


@pytest.mark.asyncio
@pytest.mark.parametrize("age,expected_jobs", [(None, 1), (timedelta(hours=2), 1), (timedelta(minutes=30), 0)])
async def test_lazy_sync_is_scheduled_without_waiting_for_vendor(
    harness: Harness, age: timedelta | None, expected_jobs: int
) -> None:
    assert harness.db.source is not None
    harness.db.source = harness.db.source.model_copy(
        update={
            "admin_api_key": "encrypted-key",
            "last_spend_sync_at": NOW - age if age else None,
        }
    )
    harness.db.rows = (member(),)
    tasks: Final = BackgroundTasks()
    response: Final = await get_spend(
        db=harness.db,
        auth=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        client=harness.http,
        background_tasks=tasks,
        now=NOW,
        source_id=harness.source_id,
        days=30,
    )
    assert response.totals.grok_cost_usd == Decimal("2.30")
    assert len(tasks.tasks) == expected_jobs
    assert harness.db.writes == ()
    if expected_jobs:
        await tasks()


@pytest.mark.parametrize("days", [0, 31])
def test_spend_window_rejects_out_of_range_days(harness: Harness, days: int) -> None:
    response: Final = harness.client.get(f"/v1/external_cloud_agents/spend?source_id={harness.source_id}&days={days}")
    assert response.status_code == 422
