from __future__ import annotations

from datetime import timedelta

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
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

    current_time = litellm.utils.get_utc_datetime()
    expires_at = current_time + timedelta(days=7)

    try:
        response = await InvitationLinkRepository(prisma_client).table.create(
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


def construct_invitation_link(invitation_id: str, base_url: str) -> str:
    """
    e.g. http://localhost:4000/ui/onboarding?invitation_id=7a096b3a-37c6-440f-9dd1-ba22e8043f6b
    """
    return f"{base_url.rstrip('/')}/ui/onboarding?invitation_id={invitation_id}"


async def get_user_invitation_link(user_id: str | None, base_url: str) -> str:
    """
    Return the onboarding link for `user_id`, reusing the user's most recent invitation
    or creating one if none exists.

    Falls back to `base_url` when the link cannot be built.
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_id is None or prisma_client is None:
        return base_url

    try:
        existing_invitations = await InvitationLinkRepository(prisma_client).table.find_many(
            where={"user_id": user_id},
            order={"created_at": "desc"},
        )
        invitation = (
            existing_invitations[0]
            if existing_invitations
            else await create_invitation_for_user(
                data=InvitationNew(user_id=user_id),
                user_api_key_dict=UserAPIKeyAuth(user_id=user_id),
            )
        )
    except Exception as e:
        verbose_proxy_logger.error("Unable to get/create invitation for user_id %s - %s", user_id, str(e))
        return base_url

    if invitation is None:
        return base_url

    return construct_invitation_link(invitation_id=invitation.id, base_url=base_url)
