"""
Types for the management endpoints

Might include fastapi/proxy requirements.txt related imports
"""

from typing import Any, Dict, List, Optional, cast

from fastapi_sso.sso.base import OpenID

from litellm.proxy._types import LitellmUserRoles


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


def get_litellm_user_role(role_str) -> Optional[LitellmUserRoles]:
    """
    Convert a string (or list of strings) to a LitellmUserRoles enum if valid (case-insensitive).

    Handles list inputs since some SSO providers (e.g., Keycloak) return roles
    as arrays like ["proxy_admin"] instead of plain strings.

    Args:
        role_str: String or list to convert (e.g., "proxy_admin", ["proxy_admin"])

    Returns:
        LitellmUserRoles enum if valid, None otherwise
    """
    try:
        if isinstance(role_str, list):
            if len(role_str) == 0:
                return None
            role_str = role_str[0]
        # Use _value2member_map_ for O(1) lookup, case-insensitive
        result = LitellmUserRoles._value2member_map_.get(role_str.lower())
        return cast(Optional[LitellmUserRoles], result)
    except Exception:
        return None


class CustomOpenID(OpenID):
    team_ids: List[str]
    user_role: Optional[LitellmUserRoles] = None
    extra_fields: Optional[Dict[str, Any]] = None
