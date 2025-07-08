from typing import Optional, List

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

from litellm.proxy._types import UserAPIKeyAuth


class MCPAuthenticatedUser(AuthenticatedUser):
    """
    Wrapper class to make LiteLLM's authentication and configuration compatible with MCP's AuthenticatedUser.
    
    This class handles:
    1. User API key authentication information
    2. MCP authentication header
    3. MCP server configuration
    """

    def __init__(self, user_api_key_auth: UserAPIKeyAuth, mcp_auth_header: Optional[str] = None, mcp_servers: Optional[List[str]] = None):
        self.user_api_key_auth = user_api_key_auth
        self.mcp_auth_header = mcp_auth_header
        self.mcp_servers = mcp_servers
