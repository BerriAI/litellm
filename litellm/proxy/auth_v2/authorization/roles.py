from __future__ import annotations

from enum import Enum
from typing import Any, List


class Role(str, Enum):
    PLATFORM_ADMIN = "platform_admin"
    PLATFORM_VIEWER = "platform_viewer"
    ORG_ADMIN = "org_admin"
    ORG_VIEWER = "org_viewer"
    TEAM_ADMIN = "team_admin"
    TEAM_MEMBER = "team_member"


_PLATFORM_ROLE_VALUES = {Role.PLATFORM_ADMIN.value, Role.PLATFORM_VIEWER.value}


def filter_claim_roles(
    roles: Any, allowed_roles: List[str], allow_platform_roles: bool
) -> List[str]:
    if not isinstance(roles, list):
        return []
    allowed = set(allowed_roles)
    filtered = [role for role in roles if role in allowed]
    if not allow_platform_roles:
        filtered = [role for role in filtered if role not in _PLATFORM_ROLE_VALUES]
    return filtered
