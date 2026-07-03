"""
Enterprise internal user management endpoints
"""


from fastapi import APIRouter, Depends, HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.internal_user_endpoints import user_api_key_auth

router = APIRouter()


@router.get(
    "/user/available_users",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def available_enterprise_users(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    For keys with `max_users` set, return the list of users that are allowed to use the key.
    """
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.proxy_server import (
        premium_user_data,
        prisma_client,
    )
    from litellm.repositories.team_repository import TeamRepository
    from litellm.repositories.user_repository import UserRepository

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    user_count = await UserRepository(prisma_client).count_billable_users()
    team_count = await TeamRepository(prisma_client).count()

    if (
        not premium_user_data
        or premium_user_data is not None
        and "max_users" not in premium_user_data
    ):
        max_users = None
    else:
        max_users = premium_user_data.get("max_users")

    if premium_user_data and "max_teams" in premium_user_data:
        max_teams = premium_user_data.get("max_teams")
    else:
        max_teams = None

    return {
        "total_users": max_users,
        "total_teams": max_teams,
        "total_users_used": user_count,
        "total_teams_used": team_count,
        "total_teams_remaining": (max_teams - team_count if max_teams else None),
        "total_users_remaining": (max_users - user_count if max_users else None),
    }
