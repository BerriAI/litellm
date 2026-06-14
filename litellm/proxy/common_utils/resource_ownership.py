from typing import List, Optional

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def is_proxy_admin(user_api_key_dict: Optional[UserAPIKeyAuth]) -> bool:
    if user_api_key_dict is None:
        return False

    return (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )


def get_resource_owner_scopes(
    user_api_key_dict: Optional[UserAPIKeyAuth],
) -> List[str]:
    """
    Return ownership scopes that may access a user-created proxy resource.

    Raw user_id is included for rows created before scope prefixes existed.
    Prefixes avoid collisions when falling back to team/org/key ownership
    for keys that do not have a user_id.

    Identity-less callers (no user_id, team_id, org_id, api_key, or token)
    return ``[]`` — they share no scope with any other caller, so access
    checks against an existing owner always fail and creates that depend
    on a primary scope must reject up front. Returning a shared sentinel
    here would let any two identity-less callers see each other's data.
    """
    if user_api_key_dict is None:
        return []

    scopes: List[str] = []

    def _add(scope: Optional[str]) -> None:
        if scope and scope not in scopes:
            scopes.append(scope)

    if user_api_key_dict.user_id:
        _add(user_api_key_dict.user_id)
        _add(f"user:{user_api_key_dict.user_id}")
    if user_api_key_dict.team_id:
        _add(f"team:{user_api_key_dict.team_id}")
    if user_api_key_dict.org_id:
        _add(f"org:{user_api_key_dict.org_id}")
    if user_api_key_dict.api_key:
        _add(f"key:{user_api_key_dict.api_key}")
    if user_api_key_dict.token:
        _add(f"key:{user_api_key_dict.token}")

    return scopes


def get_primary_resource_owner_scope(
    user_api_key_dict: Optional[UserAPIKeyAuth],
) -> Optional[str]:
    """Return the canonical owner scope to stamp on newly-created rows.

    ``None`` for identity-less callers — callers that depend on a primary
    scope to record ownership must surface that as a hard error rather
    than fall back to a shared sentinel (which would collapse every
    identity-less caller into the same logical owner).
    """
    if user_api_key_dict is None:
        return None

    if user_api_key_dict.user_id:
        return user_api_key_dict.user_id
    if user_api_key_dict.team_id:
        return f"team:{user_api_key_dict.team_id}"
    if user_api_key_dict.org_id:
        return f"org:{user_api_key_dict.org_id}"
    if user_api_key_dict.api_key:
        return f"key:{user_api_key_dict.api_key}"
    if user_api_key_dict.token:
        return f"key:{user_api_key_dict.token}"
    return None


def user_can_access_resource_owner(
    owner: Optional[str],
    user_api_key_dict: Optional[UserAPIKeyAuth],
) -> bool:
    if user_api_key_dict is None:
        return True
    if is_proxy_admin(user_api_key_dict):
        return True
    if owner is None:
        return False
    return owner in get_resource_owner_scopes(user_api_key_dict)
