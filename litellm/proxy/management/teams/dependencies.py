from __future__ import annotations

from typing import TYPE_CHECKING

from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.management.teams.access import TeamAccess
from litellm.proxy.management.users.service import PrismaOrgRoles
from litellm.repositories.team_repository import TeamRepository
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

if TYPE_CHECKING:
    from litellm.proxy.management.teams.repository import RawQuery
    from litellm.proxy.utils import PrismaClient


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
