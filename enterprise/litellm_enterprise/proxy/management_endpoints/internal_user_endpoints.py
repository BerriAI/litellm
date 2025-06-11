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
        premium_user,
        premium_user_data,
        prisma_client,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if premium_user is None:
        raise HTTPException(
            status_code=500, detail={"error": CommonProxyErrors.not_premium_user.value}
        )

    # Count number of rows in LiteLLM_UserTable
    user_count = await prisma_client.db.litellm_usertable.count()

    return {
        "total_users": premium_user_data.get("max_users")
        if premium_user_data
        else None,
        "total_users_used": user_count,
        "total_users_remaining": premium_user_data.get("max_users", 0) - user_count
        if premium_user_data
        else None,
    }
