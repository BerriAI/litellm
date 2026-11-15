"""`GET /management/v1/teams/{team_id}/members`, `POST .../members/bulk_delete` and `.../members/bulk_update`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.models.budget import LiteLLM_BudgetTable, LiteLLM_BudgetTableFull
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.proxy._types import CommonProxyErrors, TeamMemberBudgetSource, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem, reject_unknown_query_params
from litellm.proxy.list_api.list_framework import (
    FilterSpec,
    ListSpec,
    Predicate,
    QueryPlan,
    ScopeAll,
    SortKey,
    handle_list,
    order_by_sql,
    where_sql,
)
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_helpers.bulk_team_member_budgets import bulk_update_team_member_budgets
from litellm.proxy.management_helpers.bulk_user_deletion import (
    bulk_remove_team_members,
    forbidden,
    team_not_found,
)
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  # legacy decorator is untyped
)
from litellm.proxy.utils import PrismaClient, query_raw_rows
from litellm.repositories.team_repository import TeamRepository
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.management_endpoints.management_v1 import ListResponse, ProblemDetail
from litellm.types.proxy.management_endpoints.team_endpoints import (
    BulkTeamMemberBudgetUpdateRequest,
    BulkTeamMemberBudgetUpdateResponse,
    BulkTeamMemberDeleteRequest,
    BulkTeamMemberDeleteResponse,
)

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


class _RosterEntry(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str | None = None
    user_email: str | None = None
    user_alias: str | None = None
    role: str


class TeamMemberListItem(_RosterEntry):
    model_config = ConfigDict(frozen=True)

    team_id: str
    spend: float | None = 0.0
    total_spend: float | None = 0.0
    budget_id: str | None = None
    litellm_budget_table: LiteLLM_BudgetTableFull | LiteLLM_BudgetTable | None = None
    budget_source: TeamMemberBudgetSource


class _RowCount(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    count: int


_ROSTER_ENTRIES: Final = TypeAdapter(tuple[_RosterEntry, ...])
_ROW_COUNTS: Final = TypeAdapter(tuple[_RowCount, ...])

_TEAM_ROSTER: Final = """(
    SELECT roster.position AS "position",
        roster.member->>'user_id' AS "user_id",
        COALESCE(NULLIF(roster.member->>'user_email', ''), u.user_email) AS "user_email",
        u.user_alias AS "user_alias",
        roster.member->>'role' AS "role",
        COALESCE(tm.spend, 0) AS "spend",
        GREATEST(COALESCE(tm.total_spend, 0), COALESCE(tm.spend, 0)) AS "total_spend",
        COALESCE(member_budget.max_budget, default_budget.max_budget) AS "max_budget",
        member_budget.budget_reset_at AS "budget_reset_at"
    FROM "LiteLLM_TeamTable" t
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN jsonb_typeof(t.members_with_roles) = 'array' THEN t.members_with_roles ELSE '[]'::jsonb END
    ) WITH ORDINALITY AS roster(member, position)
    LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = roster.member->>'user_id'
    LEFT JOIN "LiteLLM_TeamMembership" tm ON tm.team_id = t.team_id AND tm.user_id = roster.member->>'user_id'
    LEFT JOIN "LiteLLM_BudgetTable" member_budget ON member_budget.budget_id = tm.budget_id
    LEFT JOIN "LiteLLM_BudgetTable" default_budget ON default_budget.budget_id = t.metadata->>'team_member_budget_id'
    WHERE t.team_id = $1
) AS members"""


@dataclass(frozen=True, slots=True)
class PrismaTeamMemberListExecutor:
    prisma_client: PrismaClient
    team_id: str
    active_default_budget_id: str | None

    async def count(self, where: tuple[Predicate, ...]) -> int:
        clauses, params = where_sql(where, first_index=2)
        sql: Final = f"SELECT COUNT(*) AS count FROM {_TEAM_ROSTER}" + (f" WHERE {clauses}" if clauses else "")
        counted: Final = _ROW_COUNTS.validate_python(
            await query_raw_rows(self.prisma_client.db, sql, self.team_id, *params)
        )
        return counted[0].count if counted else 0

    async def find_many(self, plan: QueryPlan) -> Sequence[TeamMemberListItem]:
        from litellm.proxy.management_endpoints.team_endpoints import get_all_team_memberships

        clauses, params = where_sql(plan.where, first_index=2)
        sql: Final = (
            f'SELECT "user_id", "user_email", "user_alias", "role" FROM {_TEAM_ROSTER}'
            + (f" WHERE {clauses}" if clauses else "")
            + f" ORDER BY {order_by_sql(plan.order)}"
            + f" LIMIT ${len(params) + 2} OFFSET ${len(params) + 3}"
        )
        page: Final = _ROSTER_ENTRIES.validate_python(
            await query_raw_rows(self.prisma_client.db, sql, self.team_id, *params, plan.take, plan.skip)
        )
        user_ids: Final = tuple(entry.user_id for entry in page if entry.user_id is not None)
        memberships: Final = (
            await get_all_team_memberships(self.prisma_client, [self.team_id], user_ids=user_ids) if user_ids else ()
        )
        membership_by_user: Final = MappingProxyType({membership.user_id: membership for membership in memberships})
        return tuple(self._item(entry, membership_by_user.get(entry.user_id or "")) for entry in page)

    def _item(self, entry: _RosterEntry, membership: LiteLLM_TeamMembership | None) -> TeamMemberListItem:
        from litellm.proxy.management_endpoints.team_endpoints import (
            member_budget_source,
        )

        membership_fields: Final[Mapping[str, object]] = (
            membership.model_dump() if membership is not None else {"team_id": self.team_id}
        )
        return TeamMemberListItem.model_validate(
            {
                **membership_fields,
                **entry.model_dump(),
                "budget_source": member_budget_source(
                    membership.budget_id if membership is not None else None, self.active_default_budget_id
                ),
            }
        )


def _serialize(row: TeamMemberListItem) -> TeamMemberListItem:
    return row


def _scope(_caller: UserAPIKeyAuth) -> ScopeAll:
    """Access is decided per team by `validate_membership` before the list runs."""
    return ScopeAll()


TEAM_MEMBERS_LIST_SPEC: Final[ListSpec[TeamMemberListItem, TeamMemberListItem]] = ListSpec(
    resource="team members",
    sortable=frozenset(
        (
            "position",
            "user_id",
            "user_email",
            "user_alias",
            "role",
            "spend",
            "total_spend",
            "max_budget",
            "budget_reset_at",
        )
    ),
    searchable=frozenset(("user_id", "user_email", "user_alias")),
    filters=MappingProxyType(
        {
            "role": FilterSpec(type=str, ops=frozenset(("eq", "in"))),
            "user_id": FilterSpec(type=str, ops=frozenset(("in",))),
        }
    ),
    default_sort=(SortKey(field="position", descending=False),),
    default_page_size=50,
    max_page_size=100,
    scope=_scope,
    serialize=_serialize,
    tiebreaker="position",
)


@router.get(
    "/teams/{team_id}/members",
    tags=["team management"],
    dependencies=(Depends(user_api_key_auth),),
    response_model=ListResponse[TeamMemberListItem],
)
async def list_team_members(
    team_id: str,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ListResponse[TeamMemberListItem]:
    """
    One team's members, paged, sortable and filterable. Readable by whoever can read the team
    through `/team/info`: proxy admins and admin viewers, a key of the team, its members, and
    admins of its organization. An unknown team is a 404.

    Each row is a roster entry (`user_id`, `user_email`, `user_alias`, `role`) with the fields
    `/team/info` reports for that member in `team_memberships` (`spend`, `total_spend`,
    `budget_id`, `litellm_budget_table`, `budget_source`). `/team/info` returns the whole roster
    on every call; this route reads one page of it.

    `sort` takes a comma-separated list of `user_id`, `user_email`, `user_alias`, `role`, `spend`,
    `total_spend`, `max_budget` (the member's own limit, else the team default), `budget_reset_at`
    or `position` (roster order, the default and the tiebreaker), each optionally prefixed with `-`
    for descending. `q` is a case-insensitive substring match on `user_id`, `user_email` and
    `user_alias`. `page_size` defaults to 50 and is capped at 100. Filters are
    `filter[role][eq|in]` and `filter[user_id][in]`.

    Example curl:
    ```
    curl --location --globoff 'http://0.0.0.0:4000/management/v1/teams/team-1/members?q=jane&sort=-spend&filter[role][eq]=user' \
        --header "Authorization: Bearer $LITELLM_MASTER_KEY"
    ```
    """
    try:
        from litellm.proxy.management_endpoints.team_endpoints import (
            existing_team_default_budget_id,
            validate_membership,
        )
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}database-not-connected",
                    title="Database not connected",
                    status=503,
                    detail=CommonProxyErrors.db_not_connected_error.value,
                )
            )

        team: Final = await TeamRepository(prisma_client).find_by_id(team_id)
        if team is None:
            raise team_not_found(team_id)
        try:
            await validate_membership(user_api_key_dict=user_api_key_dict, team_table=team)
        except HTTPException as denied:
            raise forbidden(f"Not authorized to read the members of team={team_id}.") from denied

        return await handle_list(
            spec=TEAM_MEMBERS_LIST_SPEC,
            executor=PrismaTeamMemberListExecutor(
                prisma_client=prisma_client,
                team_id=team_id,
                active_default_budget_id=await existing_team_default_budget_id(team, prisma_client),
            ),
            request=request,
            caller=user_api_key_dict,
        )

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.teams.list_team_members(): Exception occured - %s", e
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to list team members.",
            )
        )


@router.post(
    "/teams/{team_id}/members/bulk_delete",
    tags=["team management"],
    dependencies=(Depends(user_api_key_auth), Depends(reject_unknown_query_params)),
    response_model=BulkTeamMemberDeleteResponse,
)
@management_endpoint_wrapper
async def bulk_delete_team_members_action(
    team_id: str,
    data: BulkTeamMemberDeleteRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> BulkTeamMemberDeleteResponse:
    """
    Remove up to 500 members from one team in one call. Same authorization as
    `/team/member_delete`: proxy admins, the team's admins, and admins of the team's
    organization. Each member is named by exactly one of `user_id` or `user_email`;
    unknown body fields are a 422 and an unknown team is a 404.

    `data` holds one result per requested member, in request order. A row is
    `success: false` with an `error` when it names nobody on the team or repeats an
    earlier row. The roster is rewritten once, under the team's advisory lock, so a
    concurrent member_add is never overwritten from a stale read.

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/management/v1/teams/team-1/members/bulk_delete' \
        --header "Authorization: Bearer $LITELLM_MASTER_KEY" \
        --header 'Content-Type: application/json' \
        --data '{"members": [{"user_id": "user-1"}, {"user_email": "user-2@example.com"}]}'
    ```
    """
    try:
        from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

        if prisma_client is None:
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}database-not-connected",
                    title="Database not connected",
                    status=503,
                    detail=CommonProxyErrors.db_not_connected_error.value,
                )
            )

        results: Final = await bulk_remove_team_members(
            team_id=team_id,
            data=data,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        return BulkTeamMemberDeleteResponse(data=results)

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.teams.bulk_delete_team_members_action(): "
            "Exception occured - %s",
            e,
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to remove team members.",
            )
        )


@router.post(
    "/teams/{team_id}/members/bulk_update",
    tags=["team management"],
    dependencies=(Depends(user_api_key_auth), Depends(reject_unknown_query_params)),
    response_model=BulkTeamMemberBudgetUpdateResponse,
)
@management_endpoint_wrapper
async def bulk_update_team_member_budgets_action(
    team_id: str,
    data: BulkTeamMemberBudgetUpdateRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    litellm_changed_by: Annotated[
        str | None,
        Header(
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ] = None,
) -> BulkTeamMemberBudgetUpdateResponse:
    """
    Set per-member limits for up to 500 members of one team in one call. Same
    authorization and member addressing as `/team/member_update`: proxy admins, the team's
    admins, and admins of the team's organization, with each member named by exactly one of
    `user_id` or `user_email`. Unknown body fields are a 422 and an unknown team is a 404.

    Each row is a merge patch of that member's limits: a field left out is untouched, a
    field sent as null is cleared, and clearing the last limit drops the member back to the
    team default. A budget row shared by several memberships, the team default included, is
    copied for the member being patched rather than written in place, so one member's new
    cap never lands on anybody else.

    `data` holds one result per requested member, in request order, carrying the limits in
    force after the write. A row is `success: false` with an `error` when it names nobody on
    the team or repeats an earlier row. Roles are not part of this route; `/team/member_update`
    still owns them.

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/management/v1/teams/team-1/members/bulk_update' \
        --header "Authorization: Bearer $LITELLM_MASTER_KEY" \
        --header 'Content-Type: application/json' \
        --data '{"members": [{"user_id": "user-1", "max_budget_in_team": 10}, {"user_email": "user-2@example.com", "max_budget_in_team": 10, "budget_duration": "30d"}]}'
    ```
    """
    try:
        from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client, user_api_key_cache

        if prisma_client is None:
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}database-not-connected",
                    title="Database not connected",
                    status=503,
                    detail=CommonProxyErrors.db_not_connected_error.value,
                )
            )

        results: Final = await bulk_update_team_member_budgets(
            team_id=team_id,
            data=data,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
            litellm_changed_by=litellm_changed_by,
        )
        return BulkTeamMemberBudgetUpdateResponse(data=results)

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.teams.bulk_update_team_member_budgets_action(): "
            "Exception occured - %s",
            e,
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to update team member budgets.",
            )
        )
