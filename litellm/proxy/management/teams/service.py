from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.proxy._types import TeamMemberBudgetSource, UserAPIKeyAuth
from litellm.proxy.list_api.list_framework import Compare, ListQuery, ListSpec, QueryPlan, Scope, ScopeAll, Within
from litellm.proxy.management.teams.repository import RawQuery, TeamMemberRow, TeamMemberRows
from litellm.proxy.management.teams.schemas import TeamMemberResponse, TeamMembersQueryParams


def member_budget_source(budget_id: str | None, team_default_budget_id: str | None) -> TeamMemberBudgetSource:
    if budget_id is not None and budget_id != team_default_budget_id:
        return "custom"
    return "team_default" if team_default_budget_id is not None else "none"


def _team_member_item(row: TeamMemberRow) -> TeamMemberResponse:
    return TeamMemberResponse(
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
    """`get_readable_team` has already admitted the caller to this one team's roster."""
    return ScopeAll()


TEAM_MEMBERS_LIST_SPEC: Final[ListSpec[TeamMemberRow, TeamMemberResponse]] = ListSpec(
    resource="team members",
    sortable=frozenset(
        ("user_alias", "user_email", "user_id", "role", "spend", "total_spend", "max_budget_in_team", "budget_reset_at")
    ),
    searchable=frozenset(("user_id", "user_email")),
    filters=MappingProxyType({}),
    default_sort=(),
    default_page_size=50,
    max_page_size=100,
    scope=_roster_reader_scope,
    serialize=_team_member_item,
    tiebreaker="position",
)


def team_members_list_query(query: TeamMembersQueryParams) -> ListQuery:
    return ListQuery(
        page=query.page,
        page_size=query.page_size,
        sort=query.sort,
        search=query.q,
        filters=(
            *(() if query.role is None else (Compare(field="role", op="eq", value=query.role),)),
            *(
                ()
                if query.role_in is None
                else (Within(field="role", values=tuple(role.strip() for role in query.role_in.split(","))),)
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class TeamMembersPage:
    members: tuple[TeamMemberResponse, ...]
    total_count: int


async def get_team_members_list(team_id: str, plan: QueryPlan, roster_db: RawQuery) -> TeamMembersPage:
    roster: Final = TeamMemberRows(db=roster_db, team_id=team_id)
    total_count: Final = await roster.count(plan.where)
    rows: Final = await roster.find_many(plan)
    return TeamMembersPage(members=tuple(_team_member_item(row) for row in rows), total_count=total_count)
