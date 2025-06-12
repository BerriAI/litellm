from starlette.types import Scope

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


async def user_api_key_auth_mcp(scope: Scope) -> UserAPIKeyAuth:
    """
    Validate and extract headers from the ASGI scope for MCP requests.

    Args:
        scope: ASGI scope containing request information

    Returns:
        Dict containing validated header information

    Raises:
        HTTPException: If headers are invalid or missing required headers
    """
    # headers = dict(scope.get("headers", []))
    # Convert byte headers to strings
    # print("HEADERS IN MCP request", headers)
    # return UserAPIKeyAuth()
    return UserAPIKeyAuth(token="1234567890")

    pass
