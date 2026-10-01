import os

from fastapi import Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    if api_key == os.environ["LITELLM_MASTER_KEY"]:
        return UserAPIKeyAuth(api_key=api_key, user_role=LitellmUserRoles.PROXY_ADMIN)
    return UserAPIKeyAuth(
        api_key=api_key,
        user_role=LitellmUserRoles.INTERNAL_USER,
        end_user_id=request.headers.get("x-litellm-customer-id"),
    )
