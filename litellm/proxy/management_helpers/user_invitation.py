from datetime import timedelta
from typing import Final

from fastapi import HTTPException

import litellm
from litellm.proxy._types import CommonProxyErrors, InvitationNew, UserAPIKeyAuth
from litellm.repositories.table_repositories import InvitationLinkRepository


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

    creator_id = user_api_key_dict.user_id or litellm_proxy_admin_name

    # If the invitation is being created under the default admin user (e.g. master key auth
    # without prior Admin UI login), ensure the admin user row exists to prevent FK violation.
    if creator_id == litellm_proxy_admin_name:
        try:
            from litellm.repositories.user_repository import UserRepository

            await UserRepository(prisma_client).table.upsert(
                where={"user_id": litellm_proxy_admin_name},
                data={
                    "create": {
                        "user_id": litellm_proxy_admin_name,
                        "user_role": "proxy_admin",
                    },
                    "update": {},
                },
            )
        except Exception:  # noqa: S110, BLE001  # Best-effort upsert of default admin user
            pass

    current_time: Final = litellm.utils.get_utc_datetime()
    expires_at: Final = current_time + timedelta(days=7)

    try:
        response: Final = await InvitationLinkRepository(prisma_client).table.create(
            data={
                "user_id": data.user_id,
                "created_at": current_time,
                "expires_at": expires_at,
                "created_by": creator_id,
                "updated_at": current_time,
                "updated_by": creator_id,
            }
        )
        return response
    except Exception as e:  # noqa: BLE001  # Catch database exceptions to format error response
        err_str = str(e)
        if "Foreign key constraint failed on the field" in err_str:
            if "created_by" in err_str or "CreatedBy" in err_str:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"Creator user '{creator_id}' does not exist in 'LiteLLM_UserTable'."},
                )
            elif "updated_by" in err_str or "UpdatedBy" in err_str:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"Updater user '{creator_id}' does not exist in 'LiteLLM_UserTable'."},
                )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "User id does not exist in 'LiteLLM_UserTable'. Fix this by creating user via `/user/new`."
                },
            )
        raise HTTPException(status_code=500, detail={"error": err_str})
