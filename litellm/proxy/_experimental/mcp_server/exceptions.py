"""Exceptions raised by the LiteLLM MCP proxy."""

from typing import Optional

from fastapi import HTTPException


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

    def to_http_exception(self) -> HTTPException:
        """Convert this upstream-auth error into an ``HTTPException`` that
        preserves the upstream status code and any ``WWW-Authenticate``
        challenge, so standards-compliant MCP clients can trigger the
        upstream OAuth flow.

        When the upstream 401 omits ``WWW-Authenticate`` (non-compliant per
        RFC 7235 §3.1) we fabricate a ``Bearer resource_metadata=`` challenge
        that points at the gateway's standard-pattern well-known endpoint for
        this server, so MCP clients can still initiate RFC 9728 discovery
        against the upstream IdP via the gateway's proxied metadata.
        """
        challenge: Optional[str] = self.www_authenticate
        if challenge is None and self.status_code == 401:
            challenge = (
                "Bearer resource_metadata="
                f'"/.well-known/oauth-protected-resource/mcp/{self.server_name}"'
            )
        return HTTPException(
            status_code=self.status_code,
            detail="Unauthorized",
            headers={"www-authenticate": challenge} if challenge else None,
        )
