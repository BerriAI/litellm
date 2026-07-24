from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    PLATFORM_ADMIN = "platform_admin"
    PLATFORM_VIEWER = "platform_viewer"
    ORG_ADMIN = "org_admin"
    ORG_VIEWER = "org_viewer"
    TEAM_ADMIN = "team_admin"
    TEAM_MEMBER = "team_member"


class TeamRole(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"


_ROLE_MAP: dict[str, Role] = {
    "proxy_admin": Role.PLATFORM_ADMIN,
    "proxy_admin_viewer": Role.PLATFORM_VIEWER,
    "org_admin": Role.ORG_ADMIN,
}


def map_role(value: str | None) -> Role | None:
    """Map a LiteLLM ``user_role`` string to a platform Role."""
    if value is None:
        return None
    return _ROLE_MAP.get(value)


def team_role(role: str | None) -> TeamRole:
    return TeamRole.ADMIN if role == "admin" else TeamRole.MEMBER
