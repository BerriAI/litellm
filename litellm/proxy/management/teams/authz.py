"""Who may act on a team: every management route asks ``TeamAccess.allows`` with the roles it accepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, NoReturn, Protocol, TypeAlias

from fastapi import HTTPException, status

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth

TeamRole: TypeAlias = Literal["proxy_admin", "org_admin", "team_admin"]
TEAM_ADMIN_ONLY: Final[frozenset[TeamRole]] = frozenset({"proxy_admin", "team_admin"})
TEAM_OR_ORG_ADMIN: Final[frozenset[TeamRole]] = frozenset({"proxy_admin", "team_admin", "org_admin"})


class OrgRoles(Protocol):
    async def is_org_admin(self, user_id: str, organization_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class TeamAccess:
    org_roles: OrgRoles

    async def allows(self, caller: UserAPIKeyAuth, team: LiteLLM_TeamTable, allow: frozenset[TeamRole]) -> bool:
        """Team admin is checked before org admin, so only callers off the roster pay for the org lookup."""
        if "proxy_admin" in allow and caller.user_role == LitellmUserRoles.PROXY_ADMIN:
            return True
        if "team_admin" in allow and is_team_admin(caller, team):
            return True
        return "org_admin" in allow and await self._is_org_admin(caller, team)

    async def strongest_role(self, caller: UserAPIKeyAuth, team: LiteLLM_TeamTable) -> TeamRole | None:
        """Org admin outranks team admin so a caller holding both keeps unrestricted edits."""
        if caller.user_role == LitellmUserRoles.PROXY_ADMIN:
            return "proxy_admin"
        if await self._is_org_admin(caller, team):
            return "org_admin"
        return "team_admin" if is_team_admin(caller, team) else None

    async def _is_org_admin(self, caller: UserAPIKeyAuth, team: LiteLLM_TeamTable) -> bool:
        if not caller.user_id or not team.organization_id:
            return False
        return await self.org_roles.is_org_admin(caller.user_id, team.organization_id)


def is_team_member(user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable) -> bool:
    return user_api_key_dict.user_id is not None and any(
        member.user_id == user_api_key_dict.user_id for member in team_obj.members_with_roles
    )


def is_team_admin(user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable) -> bool:
    return any(
        member.user_id is not None and member.user_id == user_api_key_dict.user_id and member.role == "admin"
        for member in team_obj.members_with_roles
    )


def team_access_denied() -> NoReturn:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this team")
