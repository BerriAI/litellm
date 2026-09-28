from __future__ import annotations

from litellm.proxy.management.teams.access import TeamAccess
from litellm.proxy.management.users.service import PrismaOrgRoles


def get_team_access() -> TeamAccess:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    return TeamAccess(org_roles=PrismaOrgRoles(prisma_client, user_api_key_cache, proxy_logging_obj))
