"""
Types for the management endpoints

Might include fastapi/proxy requirements.txt related imports
"""

from collections.abc import Sequence
from typing import Any, Final, cast

from fastapi_sso.sso.base import OpenID

from litellm.proxy._types import LitellmUserRoles

ROLE_PRIVILEGE_ORDER: Final = (
    LitellmUserRoles.PROXY_ADMIN,
    LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    LitellmUserRoles.INTERNAL_USER,
    LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
)


def highest_privilege_role(roles: Sequence[LitellmUserRoles]) -> LitellmUserRoles | None:
    """
    Pick the most privileged role a user was granted.

    SSO providers emit multi-valued role/group claims in an arbitrary order, so resolving by
    privilege keeps a user's role stable across logins. Roles outside ROLE_PRIVILEGE_ORDER
    (org_admin, team, customer) are not comparable, so the first of those is kept.
    """
    granted: Final = frozenset(roles)
    ranked: Final = next((role for role in ROLE_PRIVILEGE_ORDER if role in granted), None)
    if ranked is not None:
        return ranked
    return roles[0] if roles else None


def _lookup_role(role_str: object) -> LitellmUserRoles | None:
    if not isinstance(role_str, str):
        return None
    result: Final = LitellmUserRoles._value2member_map_.get(role_str.lower())
    return cast(LitellmUserRoles | None, result)


def is_valid_litellm_user_role(role_str: str) -> bool:
    """
    Check if a string is a valid LitellmUserRoles enum value (case-insensitive).

    Args:
        role_str: String to validate (e.g., "proxy_admin", "PROXY_ADMIN", "internal_user")

    Returns:
        True if the string matches a valid LitellmUserRoles value, False otherwise
    """
    try:
        # Use _value2member_map_ for O(1) lookup, case-insensitive
        return role_str.lower() in LitellmUserRoles._value2member_map_
    except Exception:
        return False


def get_litellm_user_role(role_str) -> LitellmUserRoles | None:
    """
    Convert a string (or list of strings) to a LitellmUserRoles enum if valid (case-insensitive).

    Handles list inputs since some SSO providers (e.g., Keycloak) return roles
    as arrays like ["proxy_admin"] instead of plain strings. A user granted several
    roles gets the most privileged one, not whichever the provider happened to list first.

    Args:
        role_str: String or list to convert (e.g., "proxy_admin", ["proxy_admin"])

    Returns:
        LitellmUserRoles enum if valid, None otherwise
    """
    try:
        if isinstance(role_str, list):
            entries: Final[Sequence[object]] = role_str
            return highest_privilege_role(
                tuple(role for role in (_lookup_role(entry) for entry in entries) if role is not None)
            )
        return _lookup_role(role_str)
    except Exception:
        return None


class CustomOpenID(OpenID):
    team_ids: list[str]
    user_role: LitellmUserRoles | None = None
    extra_fields: dict[str, Any] | None = None
