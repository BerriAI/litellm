"""Which admin roles a caller holds on a team: every management route asks ``roles_on`` and reads the set."""

from __future__ import annotations

from typing import Final, NoReturn, Protocol

from fastapi import HTTPException, status

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.roles import Role

ONLY_TEAM_ADMIN: Final = frozenset({Role.TEAM_ADMIN})


class OrgRoles(Protocol):
    async def is_org_admin(self, user_id: str, organization_id: str) -> bool: ...


async def roles_on(team: LiteLLM_TeamTable, caller: UserAPIKeyAuth, orgs: OrgRoles) -> frozenset[Role]:
    """The union of admin roles ``caller`` holds on ``team``; an empty set means no access.

    A proxy admin already holds every permission, so they are answered without the roster or org lookups.
    """
    if caller.user_role == LitellmUserRoles.PROXY_ADMIN:
        return frozenset({Role.PLATFORM_ADMIN})
    held: Final = (
        (Role.TEAM_ADMIN, is_team_admin(caller, team)),
        (Role.ORG_ADMIN, await _is_org_admin(team, caller, orgs)),
    )
    return frozenset(role for role, holds in held if holds)


async def _is_org_admin(team: LiteLLM_TeamTable, caller: UserAPIKeyAuth, orgs: OrgRoles) -> bool:
    if not caller.user_id or not team.organization_id:
        return False
    return await orgs.is_org_admin(caller.user_id, team.organization_id)


def is_team_admin(user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable) -> bool:
    return any(
        member.user_id is not None and member.user_id == user_api_key_dict.user_id and member.role == "admin"
        for member in team_obj.members_with_roles
    )


def team_access_denied() -> NoReturn:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this team")
