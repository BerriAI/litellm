"""
MCP OAuth2 Debug Headers
========================

Client-side debugging for MCP authentication flows.

When a client sends the ``x-litellm-mcp-debug: true`` header, LiteLLM
returns masked diagnostic headers in the response so operators can
troubleshoot OAuth2 issues without SSH access to the gateway.

Response headers returned (all values are masked for safety):

    x-mcp-debug-inbound-auth
        Which inbound auth headers were present and how they were classified.
        Example: ``x-litellm-api-key=Bearer sk-12****1234``

    x-mcp-debug-oauth2-token
        The OAuth2 token extracted from the Authorization header (masked).
        Shows ``(none)`` if absent, or flags ``SAME_AS_LITELLM_KEY`` when
        the LiteLLM API key is accidentally leaking to the MCP server.

    x-mcp-debug-auth-resolution
        Which auth priority was used for the outbound MCP call:
        ``per-request-header``, ``m2m-client-credentials``, ``static-token``,
        ``oauth2-passthrough``, or ``no-auth``.

    x-mcp-debug-outbound-url
        The upstream MCP server URL that will receive the request.

    x-mcp-debug-server-auth-type
        The ``auth_type`` configured on the MCP server (e.g. ``oauth2``,
        ``bearer_token``, ``none``).

Debugging Guide
---------------

**Common issue: LiteLLM API key leaking to the MCP server**

Symptom: ``x-mcp-debug-oauth2-token`` shows ``SAME_AS_LITELLM_KEY``.

This means the ``Authorization`` header carries the LiteLLM API key and
it's being forwarded to the upstream MCP server instead of an OAuth2 token.

Fix: Move the LiteLLM key to ``x-litellm-api-key`` so the ``Authorization``
header is free for OAuth2 discovery::

    # WRONG — blocks OAuth2 discovery
    claude mcp add --transport http my_server http://proxy/mcp/server \\
        --header "Authorization: Bearer sk-..."

    # CORRECT — LiteLLM key in dedicated header, Authorization free for OAuth2
    claude mcp add --transport http my_server http://proxy/mcp/server \\
        --header "x-litellm-api-key: Bearer sk-..." \\
        --header "x-litellm-mcp-debug: true"

**Common issue: No OAuth2 token present**

Symptom: ``x-mcp-debug-oauth2-token`` shows ``(none)`` and
``x-mcp-debug-auth-resolution`` shows ``no-auth``.

This means the client didn't go through the OAuth2 flow. Check that:
1. The ``Authorization`` header is NOT set as a static header in the client config.
2. The ``.well-known/oauth-protected-resource`` endpoint returns valid metadata.
3. The MCP server in LiteLLM config has ``auth_type: oauth2``.

**Common issue: M2M token used instead of user token**

Symptom: ``x-mcp-debug-auth-resolution`` shows ``m2m-client-credentials``.

This means the server has ``client_id``/``client_secret``/``token_url``
configured and LiteLLM is fetching a machine-to-machine token instead of
using the per-user OAuth2 token. If you want per-user tokens, remove the
client credentials from the server config.

Usage from Claude Code::

    claude mcp add --transport http my_server http://proxy/mcp/server \\
        --header "x-litellm-api-key: Bearer sk-..." \\
        --header "x-litellm-mcp-debug: true"

Usage with curl::

    curl -H "x-litellm-mcp-debug: true" \\
         -H "x-litellm-api-key: Bearer sk-..." \\
         http://localhost:4000/mcp/atlassian_mcp
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from starlette.types import Message, Send

from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

# Header the client sends to opt into debug mode
MCP_DEBUG_REQUEST_HEADER = "x-litellm-mcp-debug"

# Prefix for all debug response headers
_RESPONSE_HEADER_PREFIX = "x-mcp-debug"


class MCPDebug:
    """
    Static helper class for MCP OAuth2 debug headers.

    Provides opt-in client-side diagnostics by injecting masked
    authentication info into HTTP response headers.
    """

    # Masker: show first 6 and last 4 chars so you can distinguish token types
    # e.g. "Bearer****ef01" vs "sk-123****cdef"
    _masker = SensitiveDataMasker(
        sensitive_patterns={
            "authorization",
            "token",
            "key",
            "secret",
            "auth",
            "bearer",
        },
        visible_prefix=6,
        visible_suffix=4,
    )

    @staticmethod
    def _mask(value: Optional[str]) -> str:
        """Mask a single value for safe display in headers."""
        if not value:
            return "(none)"
        return MCPDebug._masker._mask_value(value)

    @staticmethod
    def is_debug_enabled(headers: Dict[str, str]) -> bool:
        """
        Check if the client opted into MCP debug mode.

        Looks for ``x-litellm-mcp-debug: true`` (case-insensitive) in the
        request headers.
        """
        for key, val in headers.items():
            if key.lower() == MCP_DEBUG_REQUEST_HEADER:
                return val.strip().lower() in ("true", "1", "yes")
        return False

    @staticmethod
    def resolve_auth_resolution(
        server: "MCPServer",
        mcp_auth_header: Optional[str],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        oauth2_headers: Optional[Dict[str, str]],
    ) -> str:
        """
        Determine which auth priority will be used for the outbound MCP call.

        Returns one of: ``per-request-header``, ``m2m-client-credentials``,
        ``static-token``, ``oauth2-passthrough``, or ``no-auth``.
        """
        from litellm.types.mcp import MCPAuth

        has_server_specific = bool(
            mcp_server_auth_headers
            and (
                mcp_server_auth_headers.get(server.alias or "")
                or mcp_server_auth_headers.get(server.server_name or "")
            )
        )
        if has_server_specific or mcp_auth_header:
            return "per-request-header"
        if server.has_client_credentials:
            return "m2m-client-credentials"
        if server.authentication_token:
            return "static-token"
        if oauth2_headers and server.auth_type == MCPAuth.oauth2:
            return "oauth2-passthrough"
        return "no-auth"

    @staticmethod
    def build_debug_headers(
        *,
        inbound_headers: Dict[str, str],
        oauth2_headers: Optional[Dict[str, str]],
        litellm_api_key: Optional[str],
        auth_resolution: str,
        server_url: Optional[str],
        server_auth_type: Optional[str],
    ) -> Dict[str, str]:
        """
        Build masked debug response headers.

        Parameters
        ----------
        inbound_headers : dict
            Raw headers received from the MCP client.
        oauth2_headers : dict or None
            Extracted OAuth2 headers (``{"Authorization": "Bearer ..."}``).
        litellm_api_key : str or None
            The LiteLLM API key extracted from ``x-litellm-api-key`` or
            ``Authorization`` header.
        auth_resolution : str
            Which auth priority was selected for the outbound call.
        server_url : str or None
            Upstream MCP server URL.
        server_auth_type : str or None
            The ``auth_type`` configured on the server (e.g. ``oauth2``).

        Returns
        -------
        dict
            Headers to include in the response (all values masked).
        """
        debug: Dict[str, str] = {}

        # --- Inbound auth summary ---
        inbound_parts = []
        for hdr_name in ("x-litellm-api-key", "authorization", "x-mcp-auth"):
            for k, v in inbound_headers.items():
                if k.lower() == hdr_name:
                    inbound_parts.append(f"{hdr_name}={MCPDebug._mask(v)}")
                    break
        debug[f"{_RESPONSE_HEADER_PREFIX}-inbound-auth"] = (
            "; ".join(inbound_parts) if inbound_parts else "(none)"
        )

        # --- OAuth2 token ---
        oauth2_token = (oauth2_headers or {}).get("Authorization")
        if oauth2_token and litellm_api_key:
            oauth2_raw = oauth2_token.removeprefix("Bearer ").strip()
            litellm_raw = litellm_api_key.removeprefix("Bearer ").strip()
            if oauth2_raw == litellm_raw:
                debug[f"{_RESPONSE_HEADER_PREFIX}-oauth2-token"] = (
                    f"{MCPDebug._mask(oauth2_token)} "
                    f"(SAME_AS_LITELLM_KEY - likely misconfigured)"
                )
            else:
                debug[f"{_RESPONSE_HEADER_PREFIX}-oauth2-token"] = MCPDebug._mask(
                    oauth2_token
                )
        else:
            debug[f"{_RESPONSE_HEADER_PREFIX}-oauth2-token"] = MCPDebug._mask(
                oauth2_token
            )

        # --- Auth resolution ---
        debug[f"{_RESPONSE_HEADER_PREFIX}-auth-resolution"] = auth_resolution

        # --- Server info ---
        debug[f"{_RESPONSE_HEADER_PREFIX}-outbound-url"] = server_url or "(unknown)"
        debug[f"{_RESPONSE_HEADER_PREFIX}-server-auth-type"] = (
            server_auth_type or "(none)"
        )

        return debug

    @staticmethod
    def wrap_send_with_debug_headers(
        send: Send, debug_headers: Dict[str, str]
    ) -> Send:
        """
        Return a new ASGI ``send`` callable that injects *debug_headers*
        into the ``http.response.start`` message.
        """

        async def _send_with_debug(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                for k, v in debug_headers.items():
                    headers.append((k.encode(), v.encode()))
                message = {**message, "headers": headers}
            await send(message)

        return _send_with_debug

    @staticmethod
    def maybe_build_debug_headers(
        *,
        raw_headers: Optional[Dict[str, str]],
        scope: Dict,
        mcp_servers: Optional[List[str]],
        mcp_auth_header: Optional[str],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        oauth2_headers: Optional[Dict[str, str]],
        client_ip: Optional[str],
    ) -> Dict[str, str]:
        """
        Build debug headers if debug mode is enabled, otherwise return empty dict.

        This is the single entry point called from the MCP request handler.
        """
        if not raw_headers or not MCPDebug.is_debug_enabled(raw_headers):
            return {}

        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server_url: Optional[str] = None
        server_auth_type: Optional[str] = None
        auth_resolution = "no-auth"

        for server_name in mcp_servers or []:
            server = global_mcp_server_manager.get_mcp_server_by_name(
                server_name, client_ip=client_ip
            )
            if server:
                server_url = server.url
                server_auth_type = server.auth_type
                auth_resolution = MCPDebug.resolve_auth_resolution(
                    server, mcp_auth_header, mcp_server_auth_headers, oauth2_headers
                )
                break

        scope_headers = MCPRequestHandler._safe_get_headers_from_scope(scope)
        litellm_key = MCPRequestHandler.get_litellm_api_key_from_headers(
            scope_headers
        )

        return MCPDebug.build_debug_headers(
            inbound_headers=raw_headers,
            oauth2_headers=oauth2_headers,
            litellm_api_key=litellm_key,
            auth_resolution=auth_resolution,
            server_url=server_url,
            server_auth_type=server_auth_type,
        )
