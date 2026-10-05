from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

from litellm.proxy._types import LiteLLM_TeamTable, TeamMemberBudgetSource, UserAPIKeyAuth
from litellm.proxy.list_api.list_framework import FilterSpec, ListSpec, Scope, ScopeAll
from litellm.proxy.management.teams.access import TeamAccess
from litellm.proxy.management.teams.repository import TeamMemberRow
from litellm.proxy.management.teams.schemas import TeamMemberListItem


class Teams(Protocol):
    async def find_by_id(self, team_id: str) -> LiteLLM_TeamTable | None: ...


@dataclass(frozen=True, slots=True)
class TeamNotFound:
    team_id: str


@dataclass(frozen=True, slots=True)
class RosterHidden:
    team_id: str


@dataclass(frozen=True, slots=True)
class ReadableRoster:
    team_id: str


RosterLookup: TypeAlias = TeamNotFound | RosterHidden | ReadableRoster


async def find_readable_roster(
    team_id: str, caller: UserAPIKeyAuth, teams: Teams, team_access: TeamAccess
) -> RosterLookup:
    team: Final = await teams.find_by_id(team_id)
    if team is None:
        return TeamNotFound(team_id=team_id)
    if not await team_access.reads_roster(caller, team):
        return RosterHidden(team_id=team_id)
    return ReadableRoster(team_id=team_id)


def member_budget_source(budget_id: str | None, team_default_budget_id: str | None) -> TeamMemberBudgetSource:
    if budget_id is not None and budget_id != team_default_budget_id:
        return "custom"
    return "team_default" if team_default_budget_id is not None else "none"


def _team_member_item(row: TeamMemberRow) -> TeamMemberListItem:
    return TeamMemberListItem(
        user_id=row.user_id,
        user_email=row.user_email,
        user_alias=row.user_alias,
        role=row.role,
        spend=row.spend,
        total_spend=row.total_spend,
        budget_id=row.budget_id,
        budget_source=member_budget_source(row.budget_id, row.team_default_budget_id),
        max_budget_in_team=row.max_budget_in_team,
        budget_duration=row.budget_duration,
        budget_reset_at=row.budget_reset_at,
        tpm_limit=row.tpm_limit,
        rpm_limit=row.rpm_limit,
        allowed_models=row.allowed_models,
    )


def _roster_reader_scope(_caller: UserAPIKeyAuth) -> Scope:
    """`find_readable_roster` has already admitted the caller to this one team's roster."""
    return ScopeAll()


TEAM_MEMBERS_LIST_SPEC: Final[ListSpec[TeamMemberRow, TeamMemberListItem]] = ListSpec(
    resource="team members",
    sortable=frozenset(
        ("user_alias", "user_email", "user_id", "role", "spend", "total_spend", "max_budget_in_team", "budget_reset_at")
    ),
    searchable=frozenset(("user_id", "user_email")),
    filters=MappingProxyType({"role": FilterSpec(type=str, ops=frozenset(("eq", "in")))}),
    default_sort=(),
    default_page_size=50,
    max_page_size=100,
    scope=_roster_reader_scope,
    serialize=_team_member_item,
    tiebreaker="position",
)
