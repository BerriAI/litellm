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

    def to_http_exception(
        self,
        base_url: Optional[str] = None,
        request_path: Optional[str] = None,
    ) -> HTTPException:
        """Convert this upstream-auth error into an ``HTTPException`` that
        preserves the upstream status code and any ``WWW-Authenticate``
        challenge, so standards-compliant MCP clients can trigger the
        upstream OAuth flow.

        When the upstream 401 omits ``WWW-Authenticate`` (non-compliant per
        RFC 7235 §3.1) we fabricate a ``Bearer resource_metadata=`` challenge
        that points at the gateway's well-known endpoint for this server, so
        MCP clients can still initiate RFC 9728 discovery against the upstream
        IdP via the gateway's proxied metadata. Callers must pass ``base_url``
        (the gateway origin, no trailing slash) so the fabricated URI is
        absolute as RFC 9728 §3.2 requires; if ``base_url`` is missing we
        skip fabrication entirely rather than emit a relative URI that strict
        clients reject in the Bearer challenge.

        When ``request_path`` is supplied and matches the legacy
        ``/{server_name}/mcp`` MCP transport route, the fabricated URI uses
        the matching legacy well-known form
        ``/.well-known/oauth-protected-resource/{server_name}/mcp``. Otherwise
        we default to the standard form
        ``/.well-known/oauth-protected-resource/mcp/{server_name}``. This
        keeps the ``resource_metadata`` URI aligned with the resource pattern
        the client originally targeted, matching the path-aware behaviour of
        ``_get_passthrough_resource_metadata_url`` in ``server.py``.
        """
        challenge: Optional[str] = self.www_authenticate
        if challenge is None and self.status_code == 401 and base_url:
            prefix = base_url.rstrip("/")
            if request_path and request_path.startswith(f"/{self.server_name}/mcp"):
                resource_metadata_url = (
                    f"{prefix}/.well-known/oauth-protected-resource/"
                    f"{self.server_name}/mcp"
                )
            else:
                resource_metadata_url = (
                    f"{prefix}/.well-known/oauth-protected-resource/"
                    f"mcp/{self.server_name}"
                )
            challenge = f'Bearer resource_metadata="{resource_metadata_url}"'
        detail = "Forbidden" if self.status_code == 403 else "Unauthorized"
        return HTTPException(
            status_code=self.status_code,
            detail=detail,
            headers={"www-authenticate": challenge} if challenge else None,
        )
