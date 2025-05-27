"""
Example custom auth function.

This will allow all keys starting with "my-custom-key" to pass through.
"""
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    try:
        if api_key.startswith("my-custom-key"):
            return UserAPIKeyAuth(api_key=api_key)
        else:
            raise Exception("Invalid API key")
    except Exception:
        raise Exception("Invalid API key")
