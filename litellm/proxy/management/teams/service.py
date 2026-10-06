from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.proxy._types import TeamMemberBudgetSource, UserAPIKeyAuth
from litellm.proxy.list_api.list_framework import FilterSpec, ListSpec, Scope, ScopeAll, handle_list
from litellm.proxy.management.teams.repository import RawQuery, TeamMemberRow, TeamMemberRows
from litellm.proxy.management.teams.schemas import TeamMemberListItem

if TYPE_CHECKING:
    from fastapi import Request

    from litellm.types.proxy.management_endpoints.management_v1 import ListResponse


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
    """`get_readable_team` has already admitted the caller to this one team's roster."""
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


async def get_team_members_list(
    team_id: str, request: Request, caller: UserAPIKeyAuth, roster_db: RawQuery
) -> ListResponse[TeamMemberListItem]:
    return await handle_list(
        spec=TEAM_MEMBERS_LIST_SPEC,
        executor=TeamMemberRows(db=roster_db, team_id=team_id),
        request=request,
        caller=caller,
    )
