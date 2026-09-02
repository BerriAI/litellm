"""Exceptions raised by the LiteLLM MCP proxy."""

from typing import Final

from fastapi import HTTPException


class MCPUpstreamAuthError(Exception):
    """Raised when an upstream MCP server returns an authentication failure
    (typically HTTP 401) and the gateway should surface it transparently to
    the client instead of swallowing it.

    Relevant for MCP servers that delegate OAuth to the upstream server,
    including pass-through servers and OAuth2 servers with
    ``delegate_auth_to_upstream`` enabled. The gateway converts this exception
    into an HTTP 401 response on single-server routes, preserving any
    ``WWW-Authenticate`` challenge emitted by the upstream so standards-
    compliant MCP clients can trigger the upstream OAuth flow.
    """

    def __init__(
        self,
        status_code: int,
        www_authenticate: str | None,
        server_name: str,
    ) -> None:
        self.status_code = status_code
        self.www_authenticate = www_authenticate
        self.server_name = server_name
        super().__init__(f"Upstream MCP server {server_name!r} returned {status_code}")

    def to_http_exception(
        self,
        base_url: str | None = None,
        request_path: str | None = None,
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
        ``get_passthrough_resource_metadata_url`` in ``oauth_utils.py``.
        """
        challenge: str | None = self.www_authenticate
        if challenge is None and self.status_code == 401 and base_url:
            prefix: Final = base_url.rstrip("/")
            if request_path and request_path.startswith(f"/{self.server_name}/mcp"):
                resource_metadata_url = f"{prefix}/.well-known/oauth-protected-resource/{self.server_name}/mcp"
            else:
                resource_metadata_url = f"{prefix}/.well-known/oauth-protected-resource/mcp/{self.server_name}"
            challenge = f'Bearer resource_metadata="{resource_metadata_url}"'
        detail: Final = "Forbidden" if self.status_code == 403 else "Unauthorized"
        return HTTPException(
            status_code=self.status_code,
            detail=detail,
            headers={"www-authenticate": challenge} if challenge else None,
        )


class MCPOpenApiUpstreamError(Exception):
    """An OpenAPI-backed MCP tool's upstream answered with a non-2xx that is not a 401.

    Carries the status only. The upstream's response body is deliberately dropped rather than served
    as tool content: it crosses a trust boundary and may hold prose, urls, or an error document that
    reads as data, which is how these failures came to be reported as successful tool output. This
    matches ``outcome_wire_value``'s contract for listing faults, category and status and nothing
    else. A 401 is raised as ``MCPUpstreamAuthError`` instead, so the caller learns to
    re-authenticate; every other status stays here, mirroring the regular MCP path where a 403
    deliberately does not produce a challenge.
    """

    def __init__(self, status_code: int, server_name: str) -> None:
        self.status_code = status_code
        self.server_name = server_name
        super().__init__(f"upstream returned HTTP {status_code}")


class MCPToolResultError(Exception):
    """An MCP tool call completed with ``isError=True`` in its result.

    Never raised on the wire path: streamable HTTP MCP correctly returns tool
    failures as HTTP 200 with ``result.isError: true`` per the MCP spec. This
    exception only drives the standard failure logging (``status="failure"``
    payload, OTel ERROR span) for such results.

    Lives here rather than ``utils.py`` deliberately: tests reload ``utils``
    to re-read its env-derived constants, and a reload would fork this class
    into two identities, breaking ``isinstance`` checks against instances
    created before the reload.
    """


class MCPServerListError(Exception):
    """Carrier for a classified per-server listing fault (``faults.list_outcomes.ServerListFault``).

    Raised where a server fetch used to silently return an empty tool list, so each boundary can
    apply its own policy: the aggregate listing absorbs it into that server's outcome, while
    single-server routes relay a truthful HTTP status instead of empty-success. The fault value is
    typed as ``object`` here only to avoid a circular import with the faults package; construction
    sites always pass a ``ServerListFault``.
    """

    def __init__(self, fault: object, server_name: str) -> None:
        self.fault = fault
        self.server_name = server_name
        super().__init__(f"Listing tools from MCP server {server_name!r} failed")
