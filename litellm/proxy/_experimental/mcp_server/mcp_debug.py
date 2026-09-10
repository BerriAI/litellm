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
        ``oauth2-passthrough``, ``stored-user-token``, ``token-exchange``,
        ``id-jag``, ``aws-sigv4``, ``extra-headers``, or ``no-auth``.
        ``unresolved`` means no outcome was available before the first response
        frame; ``multiple`` means several servers resolved credentials;
        ``not-applicable`` covers stdio; ``resolution-failed`` is a resolver error.

    x-mcp-debug-auth-resolutions
        For multiple servers, a JSON map of server IDs to resolution labels.
        At most 32 entries are included; x-mcp-debug-auth-resolutions-truncated
        is true when additional servers were omitted. No credentials are included.

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

``no-auth`` means the resolved upstream client carries no authentication.
An absent inbound OAuth2 token does not imply the user skipped OAuth: the gateway
can retrieve a stored per-user token, reported as ``stored-user-token``.
``unresolved`` is used when a stream starts before credential resolution, or a
request (such as initialization or a cached tool listing) resolves no credential.
Debug reporting does not fetch credentials or delay a streaming frame to resolve them.
``extra-headers`` identifies supplied headers that won over the resolver or were
the only headers supplied; their values are never inspected to guess a scheme.
``per-request-header`` denotes a legacy credential override, including a BYOK
credential supplied by the gateway; it does not imply a caller-supplied token.

**Common issue: M2M token used instead of user token**

Symptom: ``x-mcp-debug-auth-resolution`` shows ``m2m-client-credentials``.

This means the server has ``client_id``/``client_secret``/``token_url``
configured and LiteLLM is fetching a machine-to-machine token instead of
using the per-user OAuth2 token. For gateway-stored per-user tokens,
configure ``oauth2_flow: authorization_code``.

Usage from Claude Code::

    claude mcp add --transport http my_server http://proxy/mcp/server \\
        --header "x-litellm-api-key: Bearer sk-..." \\
        --header "x-litellm-mcp-debug: true"

Usage with curl::

    curl -H "x-litellm-mcp-debug: true" \\
         -H "x-litellm-api-key: Bearer sk-..." \\
         http://localhost:4000/mcp/atlassian_mcp
"""

import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from starlette.requests import HTTPConnection
from starlette.types import Message, Send

from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import AuthResolution

# Header the client sends to opt into debug mode
MCP_DEBUG_REQUEST_HEADER: Final = "x-litellm-mcp-debug"

# Prefix for all debug response headers
_RESPONSE_HEADER_PREFIX: Final = "x-mcp-debug"


MCP_AUTH_DIAGNOSTICS_SCOPE_KEY: Final = "litellm.mcp.auth_diagnostics"


def record_auth_resolution(server_id: str, source: AuthResolution) -> None:
    from mcp.server.lowlevel.server import request_ctx

    context: Final[object] = request_ctx.get(None)
    request: Final[object] = getattr(context, "request", None)
    if isinstance(request, HTTPConnection):
        diagnostics: Final[object] = request.scope.get(MCP_AUTH_DIAGNOSTICS_SCOPE_KEY)
        if isinstance(diagnostics, MCPAuthDiagnostics):
            diagnostics.record(server_id, source)


class MCPAuthDiagnostics:
    def __init__(self) -> None:
        self._outcomes: tuple[tuple[str, AuthResolution], ...] = ()

    def record(self, server_id: str, resolution: AuthResolution) -> None:
        self._outcomes = tuple(item for item in self._outcomes if item[0] != server_id) + ((server_id, resolution),)

    def resolution(self) -> str:
        match self._outcomes:
            case ():
                return AuthResolution.unresolved.value
            case ((_, source),):
                return source.value
            case _:
                return AuthResolution.multiple.value

    def headers(self) -> Mapping[str, str]:
        if len(self._outcomes) <= 1:
            return MappingProxyType({"x-mcp-debug-auth-resolution": self.resolution()})
        return MappingProxyType(
            {
                "x-mcp-debug-auth-resolution": AuthResolution.multiple.value,
                "x-mcp-debug-auth-resolutions": json.dumps(
                    {
                        server_id: source.value for server_id, source in self._outcomes[:32]
                    },  # mutable-ok: JSON encoder requires a concrete dict
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                **(
                    MappingProxyType({"x-mcp-debug-auth-resolutions-truncated": "true"})
                    if len(self._outcomes) > 32
                    else MappingProxyType({})
                ),
            }
        )


class _DiagnosticSend:
    def __init__(self, send: Send, headers: Mapping[str, str], resolution: Callable[[], Mapping[str, str]]) -> None:
        self._send = send
        self._headers = headers
        self._resolution = resolution
        self._start: Message | None = None

    async def __call__(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self._start = message
            return
        if self._start is not None:
            start: Final = self._start
            self._start = None
            headers: Final = MappingProxyType({**self._headers, **self._resolution()})
            await self._send(
                {  # mutable-ok: ASGI send consumes a mutable message mapping
                    **start,
                    "headers": tuple(start.get("headers", ()))
                    + tuple((key.encode(), value.encode()) for key, value in headers.items()),
                }
            )
        await self._send(message)


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
    def _mask(value: str | None) -> str:
        """Mask a single value for safe display in headers."""
        if not value:
            return "(none)"
        return MCPDebug._masker._mask_value(value)

    @staticmethod
    def is_debug_enabled(headers: dict[str, str]) -> bool:
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
    def build_debug_headers(
        *,
        inbound_headers: dict[str, str],
        oauth2_headers: dict[str, str] | None,
        litellm_api_key: str | None,
        auth_resolution: str,
        server_url: str | None,
        server_auth_type: str | None,
    ) -> dict[str, str]:
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
        debug: Final[dict[str, str]] = {}

        # --- Inbound auth summary ---
        inbound_parts: Final = []
        for hdr_name in ("x-litellm-api-key", "authorization", "x-mcp-auth"):
            for k, v in inbound_headers.items():
                if k.lower() == hdr_name:
                    inbound_parts.append(f"{hdr_name}={MCPDebug._mask(v)}")
                    break
        debug[f"{_RESPONSE_HEADER_PREFIX}-inbound-auth"] = "; ".join(inbound_parts) if inbound_parts else "(none)"

        # --- OAuth2 token ---
        oauth2_token: Final = (oauth2_headers or {}).get("Authorization")
        if oauth2_token and litellm_api_key:
            oauth2_raw: Final = oauth2_token.removeprefix("Bearer ").strip()
            litellm_raw: Final = litellm_api_key.removeprefix("Bearer ").strip()
            if oauth2_raw == litellm_raw:
                debug[f"{_RESPONSE_HEADER_PREFIX}-oauth2-token"] = (
                    f"{MCPDebug._mask(oauth2_token)} (SAME_AS_LITELLM_KEY - likely misconfigured)"
                )
            else:
                debug[f"{_RESPONSE_HEADER_PREFIX}-oauth2-token"] = MCPDebug._mask(oauth2_token)
        else:
            debug[f"{_RESPONSE_HEADER_PREFIX}-oauth2-token"] = MCPDebug._mask(oauth2_token)

        # --- Auth resolution ---
        debug[f"{_RESPONSE_HEADER_PREFIX}-auth-resolution"] = auth_resolution

        # --- Server info ---
        debug[f"{_RESPONSE_HEADER_PREFIX}-outbound-url"] = server_url or "(unknown)"
        debug[f"{_RESPONSE_HEADER_PREFIX}-server-auth-type"] = server_auth_type or "(none)"

        return debug

    @staticmethod
    def wrap_send_with_debug_headers(
        send: Send,
        debug_headers: Mapping[str, str],
        resolution: Callable[[], Mapping[str, str]] | None = None,
        *,
        request_method: str | None = None,
    ) -> Send:
        """
        Return a new ASGI ``send`` callable that injects *debug_headers*
        into the ``http.response.start`` message.
        """

        if resolution is not None and request_method == "POST":
            return _DiagnosticSend(send, debug_headers, resolution)

        async def _send_with_debug(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: Final = list(message.get("headers", []))
                for k, v in debug_headers.items():
                    headers.append((k.encode(), v.encode()))
                message = {**message, "headers": headers}
            await send(message)

        return _send_with_debug

    @staticmethod
    def maybe_build_debug_headers(
        *,
        raw_headers: dict[str, str] | None,
        scope: dict,
        mcp_servers: list[str] | None,
        oauth2_headers: dict[str, str] | None,
        client_ip: str | None,
    ) -> dict[str, str]:
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

        server_url: str | None = None
        server_auth_type: str | None = None
        auth_resolution: Final = AuthResolution.unresolved.value

        for server_name in mcp_servers or []:
            server = global_mcp_server_manager.get_mcp_server_by_name(server_name, client_ip=client_ip)
            if server:
                server_url = server.url
                server_auth_type = server.auth_type
                break

        scope_headers: Final = MCPRequestHandler._safe_get_headers_from_scope(scope)
        litellm_key: Final = MCPRequestHandler.get_litellm_api_key_from_headers(scope_headers)

        return MCPDebug.build_debug_headers(
            inbound_headers=raw_headers,
            oauth2_headers=oauth2_headers,
            litellm_api_key=litellm_key,
            auth_resolution=auth_resolution,
            server_url=server_url,
            server_auth_type=server_auth_type,
        )
