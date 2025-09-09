from datetime import timedelta

from fastapi import HTTPException

import litellm
from litellm.proxy._types import CommonProxyErrors, InvitationNew, UserAPIKeyAuth


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

    current_time = litellm.utils.get_utc_datetime()
    expires_at = current_time + timedelta(days=7)

    try:
        response = await prisma_client.db.litellm_invitationlink.create(
            data={
                "user_id": data.user_id,
                "created_at": current_time,
                "expires_at": expires_at,
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_at": current_time,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }  # type: ignore
        )
        return response
    except Exception as e:
        if "Foreign key constraint failed on the field" in str(e):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "User id does not exist in 'LiteLLM_UserTable'. Fix this by creating user via `/user/new`."
                },
            )
        raise HTTPException(status_code=500, detail={"error": str(e)})