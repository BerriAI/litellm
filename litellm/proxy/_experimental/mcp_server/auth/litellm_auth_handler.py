from typing import Dict, List, Optional

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

from litellm.proxy._types import UserAPIKeyAuth


class MCPAuthenticatedUser(AuthenticatedUser):
    """
    Wrapper class to make LiteLLM's authentication and configuration compatible with MCP's AuthenticatedUser.

    This class handles:
    1. User API key authentication information
    2. MCP authentication header (deprecated)
    3. MCP server configuration (can include access groups)
    4. Server-specific authentication headers
    5. OAuth2 headers
    6. Raw headers - allows forwarding specific headers to the MCP server, specified by the admin.
    """

    def __init__(
        self,
        user_api_key_auth: UserAPIKeyAuth,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        mcp_protocol_version: Optional[str] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ):
        self.user_api_key_auth = user_api_key_auth
        self.mcp_auth_header = mcp_auth_header
        self.mcp_servers = mcp_servers
        self.mcp_server_auth_headers = mcp_server_auth_headers or {}
        self.mcp_protocol_version = mcp_protocol_version
        self.oauth2_headers = oauth2_headers
        self.raw_headers = raw_headers
