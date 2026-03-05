"""
RBAC utility helpers for feature-level access control.

These helpers are used by agent and vector store endpoints to enforce
proxy-admin-configurable toggles that restrict access for internal users.
"""

from typing import TYPE_CHECKING

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth

if TYPE_CHECKING:
    pass


def _is_user_team_admin_for_any_team(
    user_api_key_dict: UserAPIKeyAuth,
    teams: list,
) -> bool:
    """
    Return True if the user is an admin member in at least one of the given teams.

    Args:
        user_api_key_dict: The authenticated user.
        teams: List of Prisma team records (from litellm_teamtable.find_many).
    """
    for team in teams:
        team_obj = LiteLLM_TeamTable(**team.model_dump())
        for member in team_obj.members_with_roles:
            if (
                member.user_id is not None
                and member.user_id == user_api_key_dict.user_id
                and member.role == "admin"
            ):
                return True
    return False


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

    from litellm.proxy.proxy_server import general_settings

    disable_flag = f"disable_{feature_name}_for_internal_users"
    allow_team_admins_flag = f"allow_{feature_name}_for_team_admins"

    if not general_settings.get(disable_flag, False):
        # Feature is not disabled — allow all authenticated users.
        return

    # Feature is disabled.  Check if team admins are exempted.
    if general_settings.get(allow_team_admins_flag, False):
        is_team_admin = await _check_if_team_admin(user_api_key_dict)
        if is_team_admin:
            return

    raise HTTPException(
        status_code=403,
        detail={
            "error": f"Access to {feature_name} is disabled for your role. Contact your proxy admin."
        },
    )


async def _check_if_team_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """
    Return True if the user is a team admin in any team.
    Mirrors the logic in management_endpoints/common_utils._user_has_admin_privileges
    but scoped to team-admin check only.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None or user_api_key_dict.user_id is None:
        return False

    from litellm.caching import DualCache
    from litellm.proxy.auth.auth_checks import get_user_object

    try:
        user_obj = await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            user_id_upsert=False,
            proxy_logging_obj=None,
        )

        if user_obj is None:
            return False

        if user_obj.teams is None or len(user_obj.teams) == 0:
            return False

        teams = await prisma_client.db.litellm_teamtable.find_many(
            where={"team_id": {"in": user_obj.teams}}
        )

        return _is_user_team_admin_for_any_team(user_api_key_dict, teams)

    except Exception as e:
        verbose_proxy_logger.debug(
            f"rbac_utils: error checking team admin status for user "
            f"{user_api_key_dict.user_id}: {e}"
        )
        return False
