from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    try:
        return UserAPIKeyAuth(
            api_key="best-api-key-ever",
            user_id="best-user-id-ever",
            team_id="best-team-id-ever",
        )
    except:
        raise Exception
