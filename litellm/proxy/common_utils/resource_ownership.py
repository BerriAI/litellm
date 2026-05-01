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
    Prefixes avoid collisions when falling back to team/org/key ownership for
    keys that do not have a user_id.
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

    return scopes


def get_primary_resource_owner_scope(
    user_api_key_dict: Optional[UserAPIKeyAuth],
) -> Optional[str]:
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
