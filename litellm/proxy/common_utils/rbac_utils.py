"""
RBAC utility helpers for feature-level access control.

These helpers are used by agent and vector store endpoints to enforce
proxy-admin-configurable toggles that restrict access for internal users.
"""

from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


async def check_feature_access_for_user(
    user_api_key_dict: UserAPIKeyAuth,
    feature_name: str,
) -> None:
    """
    Raise HTTP 403 if the user's role is blocked from accessing the given feature
    by the UI settings stored in general_settings.

    Args:
        user_api_key_dict: The authenticated user.
        feature_name: Either "agents" or "vector_stores".
    """
    # Proxy admins (and view-only admins) are never blocked.
    if user_api_key_dict.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        LitellmUserRoles.PROXY_ADMIN.value,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    ):
        return

    from litellm.proxy.proxy_server import general_settings, prisma_client, user_api_key_cache

    disable_flag = f"disable_{feature_name}_for_internal_users"
    allow_team_admins_flag = f"allow_{feature_name}_for_team_admins"

    if not general_settings.get(disable_flag, False):
        # Feature is not disabled — allow all authenticated users.
        return

    # Feature is disabled.  Check if team/org admins are exempted.
    if general_settings.get(allow_team_admins_flag, False):
        from litellm.proxy.management_endpoints.common_utils import _user_has_admin_privileges

        is_admin = await _user_has_admin_privileges(
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )
        if is_admin:
            return

    raise HTTPException(
        status_code=403,
        detail={
            "error": f"Access to {feature_name} is disabled for your role. Contact your proxy admin."
        },
    )
