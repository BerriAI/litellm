"""Exceptions raised by the LiteLLM MCP proxy."""

from typing import Optional


class MCPUpstreamAuthError(Exception):
    """Raised when an upstream MCP server returns an authentication failure
    (typically HTTP 401) and the gateway should surface it transparently to
    the client instead of swallowing it.

    Only relevant for pass-through MCP servers (see
    ``MCPServer.is_oauth_passthrough``). The gateway converts this exception
    into an HTTP 401 response on single-server routes, preserving any
    ``WWW-Authenticate`` challenge emitted by the upstream so standards-
    compliant MCP clients can trigger the upstream OAuth flow.
    """

    def __init__(
        self,
        status_code: int,
        www_authenticate: Optional[str],
        server_name: str,
    ) -> None:
        self.status_code = status_code
        self.www_authenticate = www_authenticate
        self.server_name = server_name
        super().__init__(f"Upstream MCP server {server_name!r} returned {status_code}")
