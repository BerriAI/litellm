from datetime import timedelta
from typing import Final

from fastapi import HTTPException

import litellm
from litellm.proxy._types import CommonProxyErrors, InvitationNew, UserAPIKeyAuth
from litellm.repositories.table_repositories import InvitationLinkRepository
from litellm.repositories.user_repository import UserRepository

_FOREIGN_KEY_FAILURE: Final = "Foreign key constraint failed on the field"
_INVITER_FOREIGN_KEYS: Final = ("LiteLLM_InvitationLink_created_by_fkey", "LiteLLM_InvitationLink_updated_by_fkey")


async def create_invitation_for_user(
    data: InvitationNew,
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Create an invitation for the user to onboard to LiteLLM Admin UI.
    """
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    inviter: Final = user_api_key_dict.user_id or litellm_proxy_admin_name
    current_time: Final = litellm.utils.get_utc_datetime()
    expires_at: Final = current_time + timedelta(days=7)

    try:
        if inviter == litellm_proxy_admin_name:
            await UserRepository(prisma_client).upsert_user({"user_id": inviter})
        response: Final = await InvitationLinkRepository(prisma_client).table.create(
            data={
                "user_id": data.user_id,
                "created_at": current_time,
                "expires_at": expires_at,
                "created_by": inviter,
                "updated_at": current_time,
                "updated_by": inviter,
            }
        )
        return response
    except Exception as e:
        message: Final = str(e)
        if _FOREIGN_KEY_FAILURE in message and any(key in message for key in _INVITER_FOREIGN_KEYS):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Inviting user id={inviter} does not exist in 'LiteLLM_UserTable'. "
                    "Fix this by creating user via `/user/new`."
                },
            )
        if _FOREIGN_KEY_FAILURE in message:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "User id does not exist in 'LiteLLM_UserTable'. Fix this by creating user via `/user/new`."
                },
            )
        raise HTTPException(status_code=500, detail={"error": message})
