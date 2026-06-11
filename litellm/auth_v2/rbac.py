from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Tuple

from fastapi.security import SecurityScopes

if TYPE_CHECKING:
    from .models import Principal


class Role(str, Enum):
    PLATFORM_ADMIN = "platform_admin"
    PLATFORM_VIEWER = "platform_viewer"
    ORG_ADMIN = "org_admin"
    ORG_VIEWER = "org_viewer"
    TEAM_ADMIN = "team_admin"
    TEAM_MEMBER = "team_member"


def has_required_scopes(
    security_scopes: SecurityScopes, principal: "Principal"
) -> bool:
    return set(security_scopes.scopes).issubset(set(principal.scopes))


def has_any_role(principal: "Principal", allowed: Tuple[Role, ...]) -> bool:
    return any(role in allowed for role in principal.roles)
