from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final, Protocol

from fastapi import Depends, Query, Request

from litellm.proxy._types import CommonProxyErrors, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.list_api.list_framework import QueryPlan, plan_list_query, reject_repeated_params
from litellm.proxy.management.teams.authz import TeamAccess
from litellm.proxy.management.teams.exceptions import members_not_readable, team_not_found
from litellm.proxy.management.teams.schemas import TeamMembersQueryParams
from litellm.proxy.management.teams.service import TEAM_MEMBERS_LIST_SPEC, team_members_list_query
from litellm.proxy.management.users.service import PrismaOrgRoles
from litellm.repositories.team_repository import TeamRepository
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

if TYPE_CHECKING:
    from litellm.proxy.management.teams.repository import RawQuery
    from litellm.proxy.utils import PrismaClient


class Teams(Protocol):
    async def find_by_id(self, team_id: str) -> LiteLLM_TeamTable | None: ...


def get_team_access() -> TeamAccess:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    return TeamAccess(org_roles=PrismaOrgRoles(prisma_client, user_api_key_cache, proxy_logging_obj))


def _connected_prisma_client() -> PrismaClient:
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
    return prisma_client


def get_teams() -> TeamRepository:
    return TeamRepository(_connected_prisma_client())


def get_roster_db() -> RawQuery:
    return _connected_prisma_client().db


async def get_readable_team(
    team_id: str,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    teams: Annotated[Teams, Depends(get_teams)],
    team_access: Annotated[TeamAccess, Depends(get_team_access)],
) -> LiteLLM_TeamTable:
    team: Final = await teams.find_by_id(team_id)
    if team is None:
        raise team_not_found(team_id)
    if not await team_access.reads_roster(caller, team):
        raise members_not_readable(team_id)
    return team


def get_team_members_plan(
    query: Annotated[TeamMembersQueryParams, Query()],
    request: Request,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> QueryPlan:
    plan: Final = plan_list_query(TEAM_MEMBERS_LIST_SPEC, team_members_list_query(query), caller)
    if isinstance(plan, ProblemDetail):
        raise ManagementProblem(plan)
    reject_repeated_params(request)
    return plan
