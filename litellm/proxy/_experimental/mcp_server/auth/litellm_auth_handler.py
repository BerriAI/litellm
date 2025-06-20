from typing import Optional

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

from litellm.proxy._types import UserAPIKeyAuth


class LiteLLMAuthenticatedUser(AuthenticatedUser):
    """
    Wrapper class to make UserAPIKeyAuth compatible with MCP's AuthenticatedUser
    """

    def __init__(self, user_api_key_auth: UserAPIKeyAuth, mcp_auth_header: Optional[str] = None):
        self.user_api_key_auth = user_api_key_auth
        self.mcp_auth_header = mcp_auth_header
