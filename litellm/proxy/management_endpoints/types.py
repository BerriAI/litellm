"""
Types for the management endpoints

Might include fastapi/proxy requirements.txt related imports
"""

from collections.abc import Iterable, Sequence
from typing import Any, Final, cast

from fastapi_sso.sso.base import OpenID

from litellm.proxy._types import LitellmUserRoles

# Ordered highest to lowest privilege
LITELLM_USER_ROLE_HIERARCHY: Final = (
    LitellmUserRoles.PROXY_ADMIN,
    LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    LitellmUserRoles.INTERNAL_USER,
    LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
)


def highest_privilege_role(roles: Iterable[LitellmUserRoles]) -> LitellmUserRoles | None:
    """
    Pick the highest privilege role out of the roles an IdP asserted for one user.

    IdPs do not guarantee ordering within a multi-valued role claim, so a user holding
    several roles resolves to the most privileged one rather than whichever came first.
    Roles the hierarchy does not rank (org_admin, team, customer) resolve by name to stay
    deterministic.

    Args:
        roles: The roles resolved from the claim

    Returns:
        The highest privilege role, or None if `roles` is empty
    """
    resolved: Final = frozenset(roles)
    if not resolved:
        return None

    ranked: Final = next((role for role in LITELLM_USER_ROLE_HIERARCHY if role in resolved), None)
    return ranked if ranked is not None else min(resolved, key=lambda role: role.value)


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


def get_litellm_user_role(role_str: object) -> LitellmUserRoles | None:
    """
    Convert a string (or list of strings) to a LitellmUserRoles enum if valid (case-insensitive).

    Handles list inputs since some SSO providers (e.g., Keycloak) return roles
    as arrays like ["proxy_admin"] instead of plain strings. A claim carrying several
    roles resolves to the highest privilege one, so a user does not lose access just
    because the IdP listed a weaker role first.

    Args:
        role_str: String or list to convert (e.g., "proxy_admin", ["proxy_admin"])

    Returns:
        LitellmUserRoles enum if valid, None otherwise
    """
    if isinstance(role_str, (list, tuple)):
        entries: Final = cast(Sequence[object], role_str)
        return highest_privilege_role(
            role for role in (get_litellm_user_role(entry) for entry in entries) if role is not None
        )
    if not isinstance(role_str, str):
        return None
    # Use _value2member_map_ for O(1) lookup, case-insensitive
    result: Final = LitellmUserRoles._value2member_map_.get(role_str.lower())
    return cast(LitellmUserRoles | None, result)


class CustomOpenID(OpenID):
    team_ids: list[str]
    user_role: LitellmUserRoles | None = None
    extra_fields: dict[str, Any] | None = None
