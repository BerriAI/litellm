from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.proxy._types import TeamMemberBudgetSource
from litellm.proxy.list_api.list_framework import QueryPlan, SortKey
from litellm.proxy.management.teams.repository import TeamMemberRow
from litellm.proxy.management.teams.service import (
    TEAM_MEMBERS_LIST_SPEC,
    TeamMembersPage,
    get_team_members_list,
    member_budget_source,
)

RESET_AT: Final = datetime(2026, 11, 1, tzinfo=timezone.utc)
ROW: Final = TeamMemberRow(
    user_id="u-1",
    user_email="u1@example.com",
    user_alias="Una",
    role="admin",
    spend=1.5,
    total_spend=7.25,
    budget_id="own",
    team_default_budget_id="default",
    max_budget_in_team=20.0,
    budget_duration="30d",
    budget_reset_at=RESET_AT,
    tpm_limit=1000,
    rpm_limit=10,
    allowed_models=("gpt-a", "gpt-b"),
)
FIRST_PAGE: Final = QueryPlan(where=(), order=(SortKey(field="position", descending=False),), skip=0, take=50)


@dataclass(frozen=True, slots=True)
class RostersByTeam:
    rosters: Mapping[str, tuple[Mapping[str, object], ...]]

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        rows: Final = self.rosters.get(str(args[0]), ())
        return ({"count": len(rows)},) if "COUNT(*)" in query else rows


async def test_get_team_members_list_returns_only_the_given_teams_members_with_their_count() -> None:
    db: Final = RostersByTeam({"team-1": (ROW.model_dump(),)})

    page: Final = await get_team_members_list("team-1", FIRST_PAGE, db)

    assert page.total_count == 1
    assert [(member.user_id, member.budget_source) for member in page.members] == [("u-1", "custom")]
    assert await get_team_members_list("team-2", FIRST_PAGE, db) == TeamMembersPage(members=(), total_count=0)


@pytest.mark.parametrize(
    ("budget_id", "team_default_budget_id", "expected"),
    [
        pytest.param("own", "default", "custom", id="own-row-beside-a-default"),
        pytest.param("own", None, "custom", id="own-row-without-a-default"),
        pytest.param("default", "default", "team_default", id="linked-to-the-default-row"),
        pytest.param(None, "default", "team_default", id="no-row-follows-the-default"),
        pytest.param(None, None, "none", id="no-row-no-default"),
    ],
)
def test_member_budget_source(
    budget_id: str | None, team_default_budget_id: str | None, expected: TeamMemberBudgetSource
) -> None:
    assert member_budget_source(budget_id, team_default_budget_id) == expected


def test_list_item_carries_the_member_row_and_derives_its_budget_source() -> None:
    item: Final = TEAM_MEMBERS_LIST_SPEC.serialize(ROW)

    assert item.model_dump() == {
        "user_id": "u-1",
        "user_email": "u1@example.com",
        "user_alias": "Una",
        "role": "admin",
        "spend": 1.5,
        "total_spend": 7.25,
        "budget_id": "own",
        "budget_source": "custom",
        "max_budget_in_team": 20.0,
        "budget_duration": "30d",
        "budget_reset_at": RESET_AT,
        "tpm_limit": 1000,
        "rpm_limit": 10,
        "allowed_models": ("gpt-a", "gpt-b"),
    }
