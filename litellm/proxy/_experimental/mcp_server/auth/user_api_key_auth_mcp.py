import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, cast

from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import Scope
from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.oauth_utils import (
    get_request_base_url,
    well_known_root_suffix,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
    BridgeEnvelopeAdmitted,
    BridgeEnvelopeInvalid,
    NotBridgeEnvelope,
    envelope_keys_from_master_key,
    is_bridge_envelope_shaped,
    resolve_bridge_envelope,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
    EnvelopeIdentity,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    is_session_bearer_shaped,
)
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_TeamTable,
    ProxyException,
    SpecialHeaders,
    SpecialMCPServerName,
    SpecialMCPServerNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.auth.user_api_key_auth import (
    _run_centralized_common_checks,
    user_api_key_auth,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.common_utils.user_api_key_cache import get_management_object_ttl
from litellm.repositories.table_repositories import (
    AgentsRepository,
    MCPServerRepository,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _parse_mcp_server_names_from_path(path: str, mcp_servers_header: Optional[List[str]] = None) -> Optional[List[str]]:
    """Resolve the single MCP server name a cold-start passthrough bypass may
    target. Delegates parsing to
    :meth:`MCPRequestHandler._extract_target_server_names_from_path` so the
    names used here always match the names downstream routing uses; returns
    ``None`` whenever the bypass must not activate (aggregate ``/mcp``,
    multi-server CSV paths, or any other unrecognized path).

    Also fails closed when the ``x-mcp-servers`` header introduces any server
    not present in the path-derived target set. Downstream routing for
    ``/mcp/...`` paths overrides the header with path-derived names, but a
    header/path mismatch here is a sign of a confused or hostile caller —
    refuse the cold-start bypass rather than admit anonymously based on the
    path while the header advertises a stricter, non-passthrough target."""
    servers = MCPRequestHandler._extract_target_server_names_from_path(path)
    if len(servers) != 1:
        verbose_logger.debug(
            "MCP cold-start: path %r resolved to %r; passthrough 401 bypass "
            "requires exactly one target and will not activate",
            path,
            servers,
        )
        return None
    if mcp_servers_header is not None and (set(mcp_servers_header) - set(servers)):
        verbose_logger.debug(
            "MCP cold-start: x-mcp-servers header %r introduces target(s) not "
            "in path-derived set %r; passthrough 401 bypass will not activate",
            mcp_servers_header,
            servers,
        )
        return None
    return servers


def _is_mcp_passthrough_cold_start(mcp_servers: Optional[List[str]], client_ip: Optional[str]) -> bool:
    """True only when EVERY targeted server is a pass-through server with no
    auth headers — the cold-start OAuth discovery case per RFC 9728 / MCP
    Authorization spec. Lets the route handler's 401 emitter produce the
    spec-compliant WWW-Authenticate challenge instead of surfacing a generic
    admission error.

    Uses "all" semantics (mirrors
    :meth:`MCPRequestHandler._target_servers_delegate_auth_to_upstream`): one
    non-passthrough target in a co-targeted set must not flip the bypass open
    for the others. Fails closed when any target cannot be resolved."""
    if not mcp_servers:
        return False
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    for name in mcp_servers:
        server = global_mcp_server_manager.get_mcp_server_by_name(name, client_ip=client_ip)
        if server is None or not getattr(server, "is_oauth_passthrough", False):
            return False
    return True


def _is_litellm_auth_admission_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPException):
        return exc.status_code == 401
    if isinstance(exc, ProxyException):
        try:
            return int(exc.code) == 401
        except (TypeError, ValueError):
            return False
    return False


def _has_client_supplied_mcp_auth(
    mcp_auth_header: Optional[str],
    mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
) -> bool:
    return bool(mcp_auth_header) or bool(mcp_server_auth_headers)


def _is_mcp_admitted_user_subject(user_api_key_auth: UserAPIKeyAuth | None) -> bool:
    """True when this auth is a keyless subject admitted by the gateway session / bridge user
    path, as opposed to a JWT or other keyless auth that merely lacks a ``team_id``.

    Reads the server-only ``UserAPIKeyAuth.mcp_admitted_user_subject`` field, set exclusively by
    ``_reload_admitted_user`` at admission. It is deliberately NOT a ``metadata`` key: virtual-key
    metadata is caller-controlled at key creation, so a metadata marker could be forged on a
    personal key to gain the team-inherited grant union or to dodge the caller-Authorization
    egress scrub. This field cannot be set from caller input."""
    return user_api_key_auth is not None and user_api_key_auth.mcp_admitted_user_subject is True


def _is_aggregate_mcp_scope(route: str, mcp_servers: list[str] | None) -> bool:
    """True when a request targets the aggregate ``/mcp`` endpoint rather than any named
    server. Named targets arrive either through ``x-mcp-servers`` (``mcp_servers``) or a
    path segment (``/mcp/{server}`` / ``/{server}/mcp``); the aggregate scope has neither.
    The gateway-DCR session arm and challenge fire only here, so a per-server flow is never
    affected."""
    if mcp_servers:
        return False
    return len(MCPRequestHandler._extract_target_server_names_from_path(route)) == 0


def _is_aggregate_gateway_dcr_challenge_scope(
    route: str,
    mcp_servers: list[str] | None,
    mcp_auth_header: str | None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None,
    exc: Exception,
) -> bool:
    """True when an unauthenticated request to the aggregate ``/mcp`` endpoint
    should receive the RFC 9728 401 challenge that advertises the gateway as
    the authorization server.

    Fires only for a genuine 401 on the aggregate scope: any named target
    (path or ``x-mcp-servers``) belongs to the per-server challenge paths, and
    client-supplied MCP auth headers mean the caller is not a cold-start DCR
    client. Fails closed to the original admission error otherwise."""
    if not _is_litellm_auth_admission_error(exc):
        return False
    if _has_client_supplied_mcp_auth(mcp_auth_header, mcp_server_auth_headers):
        return False
    return _is_aggregate_mcp_scope(route, mcp_servers)


def _aggregate_gateway_dcr_challenge(request: Request, invalid_token: bool) -> HTTPException:
    """The RFC 9728 challenge for the aggregate endpoint: points the client at
    the gateway's own protected-resource metadata so a DCR client discovers
    the gateway as its authorization server and starts the sign-in flow.

    ``invalid_token`` adds the RFC 6750 error code for a request that DID
    present a bearer that failed admission (expired or revoked), telling
    spec-compliant clients to re-authorize rather than retry; a request with
    no credentials at all gets the bare challenge per RFC 6750 section 3.1."""
    error_attr = 'error="invalid_token", ' if invalid_token else ""
    resource_metadata_url = (
        f"{get_request_base_url(request)}/.well-known/oauth-protected-resource{well_known_root_suffix()}/mcp"
    )
    return HTTPException(
        status_code=401,
        detail={
            "error": "authentication_required",
            "message": "Authenticate with the gateway to use the MCP endpoint.",
        },
        headers={"WWW-Authenticate": f'Bearer {error_attr}resource_metadata="{resource_metadata_url}"'},
    )


def _admission_failure_fallback(
    request: Request,
    request_route: str,
    mcp_servers: list[str] | None,
    mcp_auth_header: str | None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None,
    exc: Exception,
    bearer_presented: bool,
) -> UserAPIKeyAuth:
    """Map a failed LiteLLM admission to its anonymous fallback or challenge.

    Two fallbacks exist, both gated on a genuine 401 with no client-supplied
    MCP auth headers. The pass-through cold start (RFC 9728 / MCP
    Authorization spec discovery return) admits anonymously so the route's
    401 emitter can produce the per-server challenge. The aggregate
    gateway-DCR scope converts the failure into the gateway's own
    resource_metadata challenge, with the RFC 6750 ``invalid_token`` error
    code when the caller DID present a bearer (an expired gateway session
    must re-authorize, not retry a dead token). Anything else re-raises the
    original admission error unchanged."""
    mcp_servers_from_path = _parse_mcp_server_names_from_path(request_route, mcp_servers)
    if (
        mcp_servers_from_path is not None
        and not _has_client_supplied_mcp_auth(mcp_auth_header, mcp_server_auth_headers)
        and _is_litellm_auth_admission_error(exc)
        and _is_mcp_passthrough_cold_start(
            mcp_servers_from_path,
            client_ip=IPAddressUtils.get_mcp_client_ip(request),
        )
    ):
        verbose_logger.debug("MCP pass-through cold start: deferring admission to route 401 emitter")
        return UserAPIKeyAuth()
    if _is_aggregate_gateway_dcr_challenge_scope(
        route=request_route,
        mcp_servers=mcp_servers,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        exc=exc,
    ):
        raise _aggregate_gateway_dcr_challenge(request, invalid_token=bearer_presented) from exc
    raise exc


class MCPRequestHandler:
    """
    Class to handle MCP request processing, including:
    1. Authentication via LiteLLM API keys
    2. MCP server configuration and routing
    3. Header extraction and validation

    Utilizes the main `user_api_key_auth` function to validate authentication
    """

    LITELLM_API_KEY_HEADER_NAME_PRIMARY = SpecialHeaders.custom_litellm_api_key.value
    LITELLM_API_KEY_HEADER_NAME_SECONDARY = SpecialHeaders.openai_authorization.value

    # This is the header to use if you want LiteLLM to use this header for authenticating to the MCP server
    LITELLM_MCP_AUTH_HEADER_NAME = SpecialHeaders.mcp_auth.value

    LITELLM_MCP_SERVERS_HEADER_NAME = SpecialHeaders.mcp_servers.value

    LITELLM_MCP_ACCESS_GROUPS_HEADER_NAME = SpecialHeaders.mcp_access_groups.value

    @staticmethod
    async def process_mcp_request(
        scope: Scope,
    ) -> Tuple[
        UserAPIKeyAuth,
        Optional[str],
        Optional[List[str]],
        Optional[Dict[str, Dict[str, str]]],
        Optional[Dict[str, str]],
        Optional[Dict[str, str]],
    ]:
        """
        Process and validate MCP request headers from the ASGI scope.
        This includes:
        1. Extracting and validating authentication headers
        2. Processing MCP server configuration
        3. Handling MCP-specific headers
        4. Handling oauth2 headers
        5. Raw headers - allows forwarding specific headers to the MCP server, specified by the admin.

        Args:
            scope: ASGI scope containing request information

        Returns:
            UserAPIKeyAuth containing validated authentication information
            mcp_auth_header: Optional[str] MCP auth header to be passed to the MCP server (deprecated)
            mcp_servers: Optional[List[str]] List of MCP servers and access groups to use
            mcp_server_auth_headers: Optional[Dict[str, str]] Server-specific auth headers in format {server_alias: auth_value}
            oauth2_headers: Optional[Dict[str, str]] OAuth2 headers
            raw_headers: Optional[Dict[str, str]] Raw headers to be forwarded to the MCP server
        Raises:
            HTTPException: If headers are invalid or missing required headers
        """
        headers = MCPRequestHandler._safe_get_headers_from_scope(scope)

        # Check if there is an explicit LiteLLM API key (primary header)
        has_explicit_litellm_key = headers.get(MCPRequestHandler.LITELLM_API_KEY_HEADER_NAME_PRIMARY) is not None

        litellm_api_key = MCPRequestHandler.get_litellm_api_key_from_headers(headers) or ""

        # Get the old mcp_auth_header for backward compatibility
        mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(headers)

        # Get the new server-specific auth headers
        mcp_server_auth_headers = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)

        # Get the oauth2 headers
        oauth2_headers = MCPRequestHandler._get_oauth2_headers_from_headers(headers)

        # Parse MCP servers from header
        mcp_servers_header = headers.get(MCPRequestHandler.LITELLM_MCP_SERVERS_HEADER_NAME)
        verbose_logger.debug(f"Raw MCP servers header: {mcp_servers_header}")
        mcp_servers = None
        if mcp_servers_header is not None:
            try:
                mcp_servers = [s.strip() for s in mcp_servers_header.split(",") if s.strip()]
                verbose_logger.debug(f"Parsed MCP servers: {mcp_servers}")
            except Exception as e:
                verbose_logger.debug(f"Error parsing mcp_servers header: {e}")
                mcp_servers = None
            if mcp_servers_header == "" or (mcp_servers is not None and len(mcp_servers) == 0):
                mcp_servers = []
        # Create a proper Request object with mock body method to avoid ASGI receive channel issues
        request = Request(scope=scope)

        async def mock_body():
            return b"{}"

        request.body = mock_body  # type: ignore
        # Inline import — auth_utils participates in a proxy import cycle.
        from litellm.proxy.auth.auth_utils import (  # noqa: PLC0415
            get_request_route,
        )

        request_route = get_request_route(request)
        # Only OAuth metadata routes registered under /.well-known/ are public.
        if request_route.startswith("/.well-known/"):
            validated_user_api_key_auth = UserAPIKeyAuth()
        elif has_explicit_litellm_key:
            # An explicit x-litellm-api-key is always a LiteLLM credential, even
            # for a delegated server, so validate it: identity / spend / rate
            # limits resolve and any stored upstream token can be forwarded.
            validated_user_api_key_auth = await user_api_key_auth(api_key=litellm_api_key, request=request)
        elif MCPRequestHandler._target_servers_delegate_auth_to_upstream(
            path=request_route,
            mcp_servers=mcp_servers,
            client_ip=IPAddressUtils.get_mcp_client_ip(request),
        ):
            # Operator opted this oauth2 server into upstream-delegated auth: the
            # client authenticates directly with the upstream MCP server, so any
            # Authorization bearer is an upstream token, never a LiteLLM key. Skip
            # LiteLLM validation entirely — covering both the no-credential
            # discovery request and the authenticated call carrying the upstream
            # bearer — so a tool call that succeeds never carries a phantom 401
            # auth span; the bearer is forwarded upstream unchanged. Gated by
            # _target_servers_delegate_auth_to_upstream, which returns True only
            # when EVERY target is auth_type=oauth2 with delegate_auth_to_upstream
            # set; fails closed otherwise.
            validated_user_api_key_auth = UserAPIKeyAuth()
        elif MCPRequestHandler._target_servers_are_true_passthrough(
            path=request_route,
            mcp_servers=mcp_servers,
            client_ip=IPAddressUtils.get_mcp_client_ip(request),
        ):
            validated_user_api_key_auth = UserAPIKeyAuth()
        elif (
            (
                bridge_delegate_target := MCPRequestHandler._single_dcr_bridge_delegate_target(
                    path=request_route,
                    mcp_servers=mcp_servers,
                    client_ip=IPAddressUtils.get_mcp_client_ip(request),
                )
            )
            is not None
            and oauth2_headers
            and is_bridge_envelope_shaped(oauth2_headers["Authorization"])
        ):
            # A single DCR-bridge oauth_delegate target carrying an envelope-shaped
            # Authorization: open the envelope, admit under its recovered identity, and
            # inject the inner upstream token for egress. A non-envelope bearer on the same
            # server is NOT admitted here — it falls through to the oauth2 arm, which 401s.
            validated_user_api_key_auth, mcp_server_auth_headers = await MCPRequestHandler._admit_dcr_bridge_delegate(
                server=bridge_delegate_target,
                authorization_value=oauth2_headers["Authorization"],
                mcp_server_auth_headers=mcp_server_auth_headers,
                request=request,
                route=request_route,
            )
        elif (
            _is_aggregate_mcp_scope(request_route, mcp_servers)
            and oauth2_headers
            and is_session_bearer_shaped(oauth2_headers["Authorization"])
        ):
            # A gateway DCR session bearer at the aggregate /mcp scope: open the
            # identity-only session token and admit under the live litellm user it
            # references. A session-shaped bearer that does not open fails closed with
            # the aggregate invalid_token challenge; a non-session bearer never reaches
            # here (is_session_bearer_shaped is false) and falls through to the oauth2 arm.
            validated_user_api_key_auth = await MCPRequestHandler._admit_gateway_session(
                authorization_value=oauth2_headers["Authorization"],
                request=request,
                route=request_route,
            )
        elif oauth2_headers:
            # Authorization on a non-delegated server: the bearer must be a real
            # LiteLLM credential, so a failed validation is a genuine 401/403 and
            # propagates unless a fallback in _admission_failure_fallback applies.
            try:
                validated_user_api_key_auth = await user_api_key_auth(api_key=litellm_api_key, request=request)
            except (HTTPException, ProxyException) as e:
                validated_user_api_key_auth = _admission_failure_fallback(
                    request=request,
                    request_route=request_route,
                    mcp_servers=mcp_servers,
                    mcp_auth_header=mcp_auth_header,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    exc=e,
                    bearer_presented=True,
                )
        else:
            try:
                validated_user_api_key_auth = await user_api_key_auth(api_key=litellm_api_key, request=request)
            except (HTTPException, ProxyException) as exc:
                validated_user_api_key_auth = _admission_failure_fallback(
                    request=request,
                    request_route=request_route,
                    mcp_servers=mcp_servers,
                    mcp_auth_header=mcp_auth_header,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    exc=exc,
                    bearer_presented=False,
                )

        # Leak-defense (single chokepoint): a gateway admission credential — the session bearer or the
        # bridge envelope — is NEVER a valid upstream MCP token. Scrub it from EVERY egress header context
        # (top-level Authorization, the deprecated `x-mcp-auth`, and per-server `x-mcp-{alias}-authorization`)
        # so no client-forwarded, OBO-subject, or passthrough path can send it upstream, where a hostile
        # server could capture and replay it against the aggregate endpoint as this user. Anchored to the
        # credential SHAPE, so a legitimate upstream/passthrough token (never session- or envelope-shaped)
        # is forwarded unchanged; per-server vaulted credentials (resolved at egress) are unaffected.
        raw_headers = dict(headers)
        (
            oauth2_headers,
            raw_headers,
            mcp_auth_header,
            mcp_server_auth_headers,
        ) = MCPRequestHandler._scrub_gateway_admission_credentials(
            admitted=_is_mcp_admitted_user_subject(validated_user_api_key_auth),
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
        )

        return (
            validated_user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
        )

    @staticmethod
    def _is_gateway_admission_credential(value: str | None) -> bool:
        """True when a header value is a gateway admission credential — a session bearer (``llm_session_`` /
        ``llm_srefresh_``) or a bridge envelope. Such a value proves who signed in to the GATEWAY; it is
        never a valid credential for an UPSTREAM MCP server, so it must never be forwarded, where a hostile
        upstream could capture and replay it against the aggregate ``/mcp`` endpoint as this user."""
        return value is not None and (is_session_bearer_shaped(value) or is_bridge_envelope_shaped(value))

    @staticmethod
    def _scrub_gateway_admission_credentials(
        admitted: bool,
        oauth2_headers: dict[str, str] | None,
        raw_headers: dict[str, str],
        mcp_auth_header: str | None,
        mcp_server_auth_headers: dict[str, dict[str, str]] | None,
    ) -> tuple[dict[str, str] | None, dict[str, str], str | None, dict[str, dict[str, str]] | None]:
        """Remove any gateway admission credential from EVERY egress header context, keyed on the credential
        SHAPE: the top-level ``Authorization`` (``oauth2_headers`` + ``raw_headers``), the deprecated
        ``x-mcp-auth`` (``mcp_auth_header``), and per-server ``x-mcp-{alias}-authorization``
        (``mcp_server_auth_headers``). A legitimate upstream/passthrough token is never session- or
        envelope-shaped, so it is forwarded unchanged; the per-server token the bridge arm injects is the
        real upstream credential (also not gateway-shaped), so it survives. An admitted subject's top-level
        Authorization IS the admission bearer, so it is dropped unconditionally as defense-in-depth even
        though it is already gateway-shaped."""
        cred = MCPRequestHandler._is_gateway_admission_credential

        # 1. Top-level Authorization → oauth2_headers.
        authz = oauth2_headers.get("Authorization") if oauth2_headers else None
        if admitted or cred(authz):
            oauth2_headers = None

        # 2. raw_headers: drop the admitted subject's Authorization, and ANY header whose value is a
        #    gateway credential (covers x-mcp-auth and x-mcp-{alias}-authorization in their raw form).
        raw_headers = {
            k: v for k, v in raw_headers.items() if not ((admitted and k.lower() == "authorization") or cred(v))
        }

        # 3. Deprecated x-mcp-auth value.
        if cred(mcp_auth_header):
            mcp_auth_header = None

        # 4. Per-server x-mcp-{alias}-authorization values (drop the value, then any now-empty server dict).
        if mcp_server_auth_headers:
            stripped = {
                alias: {h: val for h, val in hdrs.items() if not cred(val)}
                for alias, hdrs in mcp_server_auth_headers.items()
            }
            mcp_server_auth_headers = {alias: hdrs for alias, hdrs in stripped.items() if hdrs}

        return oauth2_headers, raw_headers, mcp_auth_header, mcp_server_auth_headers

    @staticmethod
    def _extract_target_server_names_from_path(path: str) -> List[str]:
        """
        Extract the target MCP server name(s) from the standard MCP transport
        URL patterns: ``/mcp/{server_name_or_csv}[/...]`` and
        ``/{server_name}/mcp[/...]``. Returns ``[]`` for any other path so
        callers fail closed when the target cannot be resolved.

        Mirrors the regex-based parser in ``server.py::_get_mcp_servers_in_path``
        so the names used for auth gating match the names used for downstream
        filtering. Without this alignment, an attacker could craft
        ``/mcp/<delegated_server>/<garbage>`` so that auth treats the request
        as targeting the delegate server (bypassing LiteLLM auth) while
        downstream filtering sees a different (non-existent) target and falls
        back to the caller's full allowed-server set.

        REST/admin endpoints, OAuth2 server endpoints
        (``/{server_name}/authorize``, ``/token`` etc.), and ``.well-known``
        discovery routes intentionally fall through — those flows do not need
        OAuth2 token passthrough. Clients aggregating multiple servers should
        use ``x-mcp-servers`` on a path that does not encode a target.
        """
        # ``/{server_name}/mcp[/...]`` form — single server. The literal
        # ``mcp`` must be the second segment (not the first, which would be
        # the ``/mcp/...`` form handled below). This branch must stay in sync
        # with ``server.py::_get_mcp_servers_in_path``, which also accepts the
        # un-rewritten form (some entry points may skip the
        # ``dynamic_mcp_route`` rewrite).
        segments = [s for s in path.split("/") if s]
        if len(segments) >= 2 and segments[1] == "mcp" and segments[0] != "mcp":
            return [segments[0]]

        # ``/mcp/...`` form — server name(s) may contain a slash (e.g.
        # ``custom_solutions/user_123``) and may be a comma-separated list.
        # Use the same parsing logic as ``_get_mcp_servers_in_path`` so the
        # parsed names match downstream routing.
        mcp_path_match = re.match(r"^/mcp/([^?#]+)(?:\?.*)?(?:#.*)?$", path)
        if not mcp_path_match:
            return []
        servers_and_path = mcp_path_match.group(1)
        if not servers_and_path:
            return []

        if "," in servers_and_path:
            # Comma-separated servers, possibly followed by a trailing path.
            path_match = re.search(r"/([^/,]+(?:/[^/,]+)*)$", servers_and_path)
            if path_match:
                servers_part = servers_and_path[: -(len(path_match.group(1)) + 1)]
            else:
                servers_part = servers_and_path
            return [s.strip() for s in servers_part.split(",") if s.strip()]

        # Single-server case — server name may contain at most one slash.
        single_server_match = re.match(r"^([^/]+(?:/[^/]+)?)(?:/.*)?$", servers_and_path)
        if single_server_match:
            return [single_server_match.group(1)]
        return [servers_and_path]

    @staticmethod
    def _target_servers_delegate_auth_to_upstream(
        path: str, mcp_servers: Optional[List[str]], client_ip: Optional[str]
    ) -> bool:
        """
        True only when EVERY MCP server the request targets is configured for
        ``auth_type == oauth2`` AND has ``delegate_auth_to_upstream=True``.
        Fails closed when any target does not opt in or cannot be resolved.

        Used by :meth:`process_mcp_request` to skip LiteLLM API-key/SSO auth
        entirely (PKCE passthrough) so the client authenticates directly with
        the upstream MCP server. Mixed-target requests (e.g. one delegated +
        one non-delegated server) fall back to normal LiteLLM auth.
        """
        # Inline imports avoid a circular dependency: mcp_server_manager imports
        # from this module.
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth

        # Must mirror the downstream header-vs-path override
        # (``extract_mcp_auth_context``) or an attacker could set
        # ``x-mcp-servers`` to a delegate-enabled server while the URL path
        # targets a non-delegate server, skipping LiteLLM auth for it.
        target_names = MCPRequestHandler._resolve_target_server_names(path=path, mcp_servers_header=mcp_servers)
        if not target_names:
            return False

        for name in target_names:
            server = global_mcp_server_manager.get_mcp_server_by_name(name, client_ip=client_ip)
            if server is None or server.auth_type != MCPAuth.oauth2:
                return False
            # `is True` is intentional: opt-in must be an explicit boolean
            # True. A MagicMock attribute (in tests) or any other truthy
            # non-bool must not silently enable the bypass.
            if getattr(server, "delegate_auth_to_upstream", False) is not True:
                return False
            # Never delegate for M2M (client_credentials) servers: LiteLLM
            # fetches the upstream token automatically using stored credentials,
            # so allowing anonymous bypass would let any external caller invoke
            # tools authenticated as LiteLLM's service account.
            #
            # Resolve the flow rather than reading has_client_credentials directly:
            # this is a security gate, and a legacy row whose oauth2_flow was never
            # stamped still carries the M2M credential shape (client_id/secret +
            # token_url, no authorization_url). Treating an unstamped-but-M2M-shaped
            # row as non-M2M here would reopen the anonymous bypass the explicit
            # column no longer closes on its own. Shares the one resolution helper
            # with the egress backstop and the anonymous-delegate allowlist; all fail
            # closed on the ambiguous shape and are removed together once no null rows
            # remain. A pure-PKCE delegate server (no stored credentials) resolves to a
            # non-M2M flow and keeps its bypass.
            if MCPServerManager.effective_oauth2_flow(server) == "client_credentials":
                return False
        return True

    @staticmethod
    def _target_servers_are_true_passthrough(
        path: str, mcp_servers: Optional[list[str]], client_ip: Optional[str]
    ) -> bool:
        """
        True only when EVERY MCP server the request targets is ``auth_type == true_passthrough``.
        Fails closed when any target does not opt in or cannot be resolved.

        Used by :meth:`process_mcp_request` to skip LiteLLM admission auth entirely: the gateway is a
        transparent proxy and the caller's ``Authorization`` is an upstream token, never a LiteLLM key.
        Mirrors :meth:`_target_servers_delegate_auth_to_upstream`; a mixed-target request keeps normal auth.
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth

        target_names = MCPRequestHandler._resolve_target_server_names(path=path, mcp_servers_header=mcp_servers)
        if not target_names:
            return False

        for name in target_names:
            server = global_mcp_server_manager.get_mcp_server_by_name(name, client_ip=client_ip)
            if server is None or server.auth_type != MCPAuth.true_passthrough:
                return False
        return True

    @staticmethod
    def _single_dcr_bridge_delegate_target(
        path: str, mcp_servers: Optional[List[str]], client_ip: Optional[str]
    ) -> Optional[MCPServer]:
        """The one DCR-bridge ``oauth_delegate`` server this request targets, or ``None``.

        Returns the server only when EXACTLY ONE target resolves and it is both
        ``is_oauth_delegate`` and ``is_dcr_bridge``. Fails closed (``None``) on a
        multi-target request, an unresolved target, or a non-matching server, so the
        envelope admission arm never fires for an aggregate scope or a server that did not
        opt into the bridge. Mirrors :meth:`_target_servers_are_true_passthrough`.
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        target_names = MCPRequestHandler._resolve_target_server_names(path=path, mcp_servers_header=mcp_servers)
        if len(target_names) != 1:
            return None
        server = global_mcp_server_manager.get_mcp_server_by_name(target_names[0], client_ip=client_ip)
        if server is None or not server.is_oauth_delegate or not server.is_dcr_bridge:
            return None
        # Egress resolves the injected per-server token only by alias / server_name; a server with
        # neither cannot receive the forwarded token, so fail closed rather than admit-and-drop.
        if not (server.server_name or server.alias):
            return None
        return server

    @staticmethod
    async def _admit_dcr_bridge_delegate(
        server: MCPServer,
        authorization_value: str,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        request: Request,
        route: str,
    ) -> Tuple[UserAPIKeyAuth, Optional[Dict[str, Dict[str, str]]]]:
        """Open the bridge envelope and admit the caller under the live key it references.

        The envelope's signature proves the user authenticated when it was minted, but
        authorization is resolved fresh here rather than trusted from the envelope: the
        sealed ``key_hash`` reloads the current ``UserAPIKeyAuth`` record, and the admitted
        identity then runs through the standard pipeline's centralized policy gate, so the
        key's present restrictions and revocation state gate the request instead of a
        snapshot frozen at mint time. The inner upstream token is injected under the
        server's per-server auth-header key so egress forwards it via the
        ``PassthroughConfig`` override; the envelope ``Authorization`` the leak-defense
        strips never reaches the upstream. A new headers dict is returned rather than
        mutating the input. Fails closed with a 401 on an invalid or expired envelope, or
        when the referenced key is missing, blocked, or expired, its owner is
        SCIM-deactivated, or the centralized policy gate rejects it (blocked team or
        project, org or budget limits).

        The sealed token is keyed alias-first, matching the order egress resolves
        (``lookup_mcp_server_auth_in_headers`` tries ``alias`` before ``server_name``). Keying
        under ``server_name`` would leave a caller-supplied ``x-mcp-{alias}-authorization`` at the
        higher-priority alias slot, pairing the admitted identity with an attacker's upstream
        credential; the alias-keyed injection overwrites any such caller value.
        """
        from litellm.proxy.proxy_server import master_key

        if not master_key:
            raise HTTPException(status_code=500, detail="Server misconfigured: master_key is not set")

        await MCPRequestHandler._run_pre_db_read_auth_checks(request=request, route=route)

        keys = envelope_keys_from_master_key(master_key)
        result = resolve_bridge_envelope(authorization_value, keys, datetime.now(timezone.utc), server.server_id)
        match result:
            case BridgeEnvelopeAdmitted():
                header_key = server.alias or server.server_name
                if header_key is None:
                    raise HTTPException(status_code=500, detail="Server misconfigured: MCP server has no routable name")
                admitted = await MCPRequestHandler._reload_admitted_principal(result.identity)
                await MCPRequestHandler._enforce_admitted_live_policy(admitted=admitted, request=request, route=route)
                injected = {header_key: {"Authorization": result.upstream_authorization.get_secret_value()}}
                new_headers = {**(mcp_server_auth_headers or {}), **injected}
                return admitted, new_headers
            case BridgeEnvelopeInvalid() | NotBridgeEnvelope():
                raise HTTPException(status_code=401, detail="Invalid or expired credential")
            case _:
                assert_never(result)

    @staticmethod
    async def _admit_gateway_session(
        authorization_value: str,
        request: Request,
        route: str,
    ) -> UserAPIKeyAuth:
        """Open a gateway DCR session bearer and admit the live litellm user it references.

        The custody sibling of :meth:`_admit_dcr_bridge_delegate`: the session token seals
        no upstream credential (those are vaulted per user and resolved at egress), so this
        admits identity only and injects no per-server header. The token's signature proves
        the user signed in when it was minted, but authorization is resolved fresh here, the
        sealed ``user_id`` reloads the current user record through the SAME
        :meth:`_reload_admitted_user` the bridge user-subject path uses, and the admitted
        identity runs through the centralized policy gate, so the user's present team, org,
        budget, and SCIM state gate the request rather than a snapshot frozen at mint time.

        Fails closed with the aggregate ``invalid_token`` challenge on an expired, tampered,
        or foreign token, on a refresh token presented at the tool edge, and when the
        referenced user is missing, deactivated, or rejected by the policy gate. The
        pre-DB gates (size, IP, route allowlist) run first, mirroring the bridge arm and the
        standard pipeline, so a caller blocked by IP or route is turned away before any
        crypto or DB read."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
            NotSessionBearer,
            SessionBearerAdmitted,
            SessionBearerInvalid,
            resolve_session_bearer,
            session_keys_from_master_key,
        )
        from litellm.proxy.proxy_server import master_key

        if not master_key:
            raise HTTPException(status_code=500, detail="Server misconfigured: master_key is not set")

        await MCPRequestHandler._run_pre_db_read_auth_checks(request=request, route=route)

        keys = session_keys_from_master_key(master_key)
        result = resolve_session_bearer(authorization_value, keys, datetime.now(timezone.utc))
        match result:
            case SessionBearerAdmitted():
                try:
                    admitted = await MCPRequestHandler._reload_admitted_user(result.principal.user_id)
                    await MCPRequestHandler._enforce_admitted_live_policy(
                        admitted=admitted, request=request, route=route
                    )
                except HTTPException as exc:
                    # A cryptographically valid bearer whose referenced user is now missing or
                    # SCIM-deactivated is an invalid_token at the aggregate scope: relay the RFC 9728
                    # challenge so the DCR client re-authorizes, matching the SessionBearerInvalid
                    # arm, instead of a bare 401 with no WWW-Authenticate. A 503 (DB outage) is a
                    # transient availability failure, not an auth failure, so it passes through.
                    if exc.status_code == 401:
                        raise _aggregate_gateway_dcr_challenge(request, invalid_token=True) from exc
                    raise
                return admitted
            case SessionBearerInvalid():
                raise _aggregate_gateway_dcr_challenge(request, invalid_token=True)
            case NotSessionBearer():
                # Unreachable: the arm is entered only for an is_session_bearer_shaped
                # value. Kept for match exhaustiveness and fails closed regardless.
                raise _aggregate_gateway_dcr_challenge(request, invalid_token=True)
            case _:
                assert_never(result)

    @staticmethod
    async def _run_pre_db_read_auth_checks(request: Request, route: str) -> None:
        """Run the proxy-wide gates ``user_api_key_auth`` applies before any key lookup: the
        request-size and body-safety limits, the IP allowlist, and the ``general_settings``
        route allowlist. The envelope arm bypasses ``user_api_key_auth`` (it opens the envelope
        and reloads the identity itself), so without this a caller blocked by IP or hitting a
        proxy route the allowlist forbids would be admitted through an envelope where the same
        principal presented on the normal MCP admission path would be rejected. Runs before the
        envelope crypto so a disallowed caller is turned away before any work, mirroring the
        standard pipeline's pre-DB ordering. Violations raise the gate's own status (an IP or
        route block is a 403, an oversized body its own limit error)."""
        from litellm.proxy.auth.auth_utils import pre_db_read_auth_checks

        await pre_db_read_auth_checks(
            request=request,
            request_data=await _read_request_body(request=request),
            route=route,
        )

    @staticmethod
    async def _reload_admitted_principal(identity: EnvelopeIdentity) -> UserAPIKeyAuth:
        """Reload the live litellm record the envelope's subject references.

        Dispatches on the sealed subject type: a ``key_hash`` reloads the virtual key that
        minted the envelope (the scripted two-header client that presents a litellm key at the
        token endpoint), a ``user_id`` reloads the user that authenticated interactively (the
        DCR client, whose SSO login at the bridged authorize yields a user, not a key). Both
        return a ``UserAPIKeyAuth`` the caller runs through the centralized policy gate, so
        team/project/org/budget/SCIM enforcement is identical to the principal presenting
        itself directly."""
        match identity.subject_type:
            case "key_hash":
                return await MCPRequestHandler._reload_admitted_key(identity.subject)
            case "user_id":
                return await MCPRequestHandler._reload_admitted_user(identity.subject)
            case _:
                assert_never(identity.subject_type)

    @staticmethod
    async def _reload_admitted_user(user_id: str) -> UserAPIKeyAuth:
        """Reload the live user an interactively-minted envelope references and admit them as
        themselves.

        The DCR client authenticates via SSO at the bridged authorize, which yields a user
        subject rather than a virtual key, so the envelope admits under the user's own
        identity: the reloaded ``user_id``, the user's own MCP object permission, and the user's
        ``org_id`` ride on the returned ``UserAPIKeyAuth``, and the SAME ``get_allowed_mcp_servers``
        the key path uses then computes which servers the user may reach, so the user's litellm MCP
        grants and access groups gate the request exactly as a key's do. Because the returned auth is
        stamped ``mcp_admitted_user_subject`` (below), ``get_allowed_mcp_servers`` unions the servers the
        user reaches through ANY of their teams on top of these direct grants — a ``UserAPIKeyAuth``
        pins one ``team_id`` but a user belongs to many, so the team fan-out happens off the marker, not
        the single ``team_id``. Each source is bounded by ITS OWN org: the user's direct grants by the
        bound ``org_id`` (their primary org), and each team's grant by that team's owning org inside
        ``_allowed_mcp_servers_for_single_team`` — so a user who spans organizations does not leak one
        org's servers past another org's ceiling. The caller's centralized policy gate enforces the
        user's live budget and org state, and a SCIM-deactivated owner fails closed.

        Error handling mirrors the key path's retryable-503 contract, but ``get_user_object`` defeats a
        type-based check: where ``get_key_object`` raises a typed ``ProxyException`` for a missing key
        and lets a DB outage propagate raw, ``get_user_object`` catches every DB failure and re-raises a
        bare ``ValueError``, so a missing user and a real outage look identical and the original error
        survives only as ``__context__``. ``_raise_503_if_db_unavailable`` therefore walks the cause
        chain: a transient DB outage still surfaces as a retryable 503, while a missing user, or any
        other non-outage resolution failure, fails closed as a 401 rather than an opaque 500. The
        object-permission load shares this one boundary, so an outage there is classified the same
        way (``get_object_permission`` itself swallows a failed load to ``None``, matching how
        ``get_key_object`` best-effort-loads a key's object permission)."""
        from litellm.proxy.auth.auth_checks import get_object_permission, get_user_object
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Server misconfigured: no database connection")
        try:
            user_object = await get_user_object(
                user_id=user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_id_upsert=False,
            )
            # Resolve the user's own MCP object permission (get_user_object does not load it) so the shared
            # get_allowed_mcp_servers can grant the user their litellm-granted servers. Reuses the same
            # get_object_permission resolver the key and team paths use; no permission logic is duplicated.
            object_permission = user_object.object_permission if user_object is not None else None
            if user_object is not None and object_permission is None and user_object.object_permission_id:
                object_permission = await get_object_permission(
                    object_permission_id=user_object.object_permission_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                )
        except (ProxyException, HTTPException):
            raise HTTPException(status_code=401, detail="Invalid or expired credential") from None
        except Exception as e:  # noqa: BLE001  # a DB outage anywhere in the resolution is a retryable 503, not an opaque 500; anything else fails closed as 401
            MCPRequestHandler._raise_503_if_db_unavailable(e)
            raise HTTPException(status_code=401, detail="Invalid or expired credential") from None
        if user_object is None:
            raise HTTPException(status_code=401, detail="Invalid or expired credential")
        if isinstance(user_object.metadata, dict) and user_object.metadata.get("scim_active") is False:
            raise HTTPException(status_code=401, detail="Invalid or expired credential")
        admitted = UserAPIKeyAuth(
            user_id=user_object.user_id,
            user_role=user_object.user_role,
            org_id=user_object.organization_id,
            object_permission=object_permission,
            object_permission_id=user_object.object_permission_id,
            # Copy the live user's rate limits, exactly as the standard user-subject auth path does
            # (user_api_key_auth.py). The parallel limiter reads these off the auth object rather than
            # re-fetching, and treats None as sys.maxsize (unlimited), so a keyless admitted user with
            # them unset would invoke tools past their configured user RPM/TPM.
            #
            # Rate-limit model for the keyless admitted subject: bounded by their USER rpm/tpm
            # (copied here; those descriptors key off user_id, which is set) AND by the per-server
            # mcp_rpm_limit of EVERY team it reaches servers through, stamped below. Per-KEY MCP
            # limits genuinely do not apply, because there is no key.
            user_tpm_limit=user_object.tpm_limit,
            user_rpm_limit=user_object.rpm_limit,
        )
        # Set the server-only admission marker AFTER construction: the before-validator strips it
        # from any validated input, so a post-construction assignment is the only way to set it, and
        # caller-supplied data (key metadata, JWT claims) can never forge it.
        admitted.mcp_admitted_user_subject = True
        # Carry each granting team's per-server MCP rpm limit. A key is pinned to one team so the
        # limiter reads team_metadata directly; this subject reaches servers through several teams
        # under its own identity, so without this the team ceiling silently does not apply to it and
        # a cross-team user outruns every team's mcp_rpm_limit. Resolved from the same roster-checked
        # sources the grant union uses, so a team can only throttle what it actually granted.
        admitted.mcp_source_team_rpm_limits = await MCPRequestHandler._admitted_subject_team_rpm_limits(admitted)
        return admitted

    @staticmethod
    async def _admitted_subject_team_rpm_limits(auth: UserAPIKeyAuth) -> dict[str, dict[str, int]] | None:
        """``team_id -> mcp_rpm_limit`` for every team this subject reaches servers through, with each
        team's map filtered to the servers THAT team's grant actually reaches.

        A limit rides the same scope as the access it bounds: a team's throttle exists to cap usage of
        the access the team granted, so a roster team whose grant does not reach a server (not granted,
        blocked, org-forbidden, opted out) must not be charged when the user reaches that server
        through a DIFFERENT team — otherwise this user's calls drain a bucket shared by that team's own
        keys for access the team never provided. The grant scope comes from the SAME
        ``get_allowed_mcp_servers(source)`` call authorization uses, so the throttle scope cannot
        diverge from the access scope. Limit maps are keyed by server name/alias (the limiter matches
        on the called server's name) while grants are ids, so each key is resolved through
        ``expand_permission_list`` — the one existing name->id owner — before the membership check.

        Returns None when no team contributes an applicable limit, so the limiter adds no descriptors
        rather than empty ones. A lookup failure narrows to None rather than raising: rate limiting
        must not be able to deny a request that authorization already allowed, and the user's own
        rpm/tpm still bounds them."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        try:
            limits: dict[str, dict[str, int]] = {}
            source_grants = await MCPRequestHandler.admitted_source_grants(auth)
            for source, granted_ids in source_grants:
                if not source.team_id:
                    continue
                team_obj = await MCPRequestHandler._roster_team_object(source.team_id, auth)
                team_limit = (team_obj.metadata or {}).get("mcp_rpm_limit") if team_obj is not None else None
                if not isinstance(team_limit, dict) or not team_limit:
                    continue
                applicable: dict[str, int] = {}
                for server_name, rpm in team_limit.items():
                    for server_id in global_mcp_server_manager.expand_permission_list([server_name]):
                        if server_id not in granted_ids:
                            continue
                        # Charge ONLY the source the call is attributed to — the same single source
                        # billing picks, from the same owner. Adding a descriptor for every granting
                        # team let one cross-team user drain several teams' SHARED buckets at once,
                        # blocking their other members for access those teams did not provide on
                        # this call; and when the user's OWN grant reaches the server, no team
                        # provided it, so no team bucket is charged at all.
                        attributed = await MCPRequestHandler.attributing_source_for_server(
                            auth, server_id, source_grants=source_grants
                        )
                        if attributed is not None and attributed.team_id == source.team_id:
                            applicable[server_name] = rpm
                            break
                if applicable:
                    limits[source.team_id] = applicable
            return limits or None
        except Exception as e:  # noqa: BLE001  # throttling metadata must never fail an allowed request
            verbose_logger.warning(f"Failed to resolve per-team MCP rpm limits for admitted subject: {str(e)}")
            return None

    @staticmethod
    async def _reload_admitted_key(key_hash: str) -> UserAPIKeyAuth:
        """Reload the live key record an admitted envelope references and re-check live policy.

        Resolving the current ``UserAPIKeyAuth`` (cache first, then DB) is what stops the
        envelope from carrying frozen authority: the key's present team/org/object-permission
        restrictions ride on the returned object, and a key that has since been deleted,
        blocked, or expired fails closed with a 401 here rather than being admitted as an
        unrestricted identity. ``get_key_object`` raises for a hash with no key row; a
        blocked or expired row is rejected explicitly because ``get_key_object`` resolves a
        row without applying those checks (the main ``user_api_key_auth`` pipeline enforces
        them downstream, which this admission path bypasses). The owner's SCIM state is the
        other builder-inline check mirrored here, so IdP offboarding revokes every envelope
        minted under the user's keys rather than leaving them live until expiry. Team,
        project, org, and budget state are NOT re-checked here; the caller runs the admitted
        identity through ``_enforce_admitted_live_policy`` for those.
        """
        from litellm.proxy.auth.auth_checks import get_key_object
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Server misconfigured: no database connection")
        try:
            key_object = await get_key_object(
                hashed_token=key_hash,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
            )
        except (ProxyException, HTTPException):
            raise HTTPException(status_code=401, detail="Invalid or expired credential") from None
        except Exception as e:  # noqa: BLE001  # a DB outage during reload is a retryable 503, not an opaque 500
            MCPRequestHandler._raise_503_if_db_unavailable(e)
            raise
        if not MCPRequestHandler._admitted_key_is_active(key_object):
            raise HTTPException(status_code=401, detail="Invalid or expired credential")
        await MCPRequestHandler._reject_if_admitted_owner_scim_deactivated(key_object)
        return key_object

    @staticmethod
    def _raise_503_if_db_unavailable(e: Exception) -> None:
        """Raise a retryable 503 when ``e`` means the auth database is unreachable, else return so the
        caller applies its own fail-closed mapping. A DB outage must not masquerade as an auth failure
        (401) or surface as an opaque 500; the caller retries. Mirrors ``UserAPIKeyAuthExceptionHandler``,
        which renders a service-unavailable database error as 503 on the standard pipeline.

        Classifies across the ``__cause__``/``__context__`` chain, not just ``e`` itself: ``get_user_object``
        re-raises every DB failure as a bare ``ValueError``, so a type-based check on the top exception
        would miss a real outage wrapped inside it."""
        from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler

        if PrismaDBExceptionHandler.is_database_service_unavailable_error_in_chain(e):
            raise HTTPException(
                status_code=503,
                detail="Service Unavailable, the authentication database is temporarily unreachable. Please retry shortly.",
            ) from None

    @staticmethod
    async def _reject_if_admitted_owner_scim_deactivated(key_object: UserAPIKeyAuth) -> None:
        """Fail closed with a 401 when the key's owning user was deactivated via SCIM.

        The standard pipeline enforces this inline in ``_user_api_key_auth_builder`` rather
        than in ``common_checks``, so the centralized policy gate does not cover it; without
        this mirror, IdP offboarding would leave the user's already-minted envelopes live
        until expiry. A failed user lookup skips the gate (fail-open), matching the builder:
        this is the one deliberately fail-open check in an otherwise fail-closed arm, so a
        transient DB outage during this lookup admits the request rather than rejecting it,
        keeping parity with how the standard pipeline treats the same lookup failure."""
        if key_object.user_id is None:
            return
        from litellm.proxy.auth.auth_checks import get_user_object
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        try:
            user_object = await get_user_object(
                user_id=key_object.user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_id_upsert=False,
            )
        except Exception as e:  # noqa: BLE001  # mirror the builder's fail-open user lookup; DB errors are of any type
            verbose_logger.debug(f"bridge admission: user lookup failed, skipping SCIM gate: {e}")
            user_object = None
        if user_object is None or not isinstance(user_object.metadata, dict):
            return
        if user_object.metadata.get("scim_active") is False:
            raise HTTPException(status_code=401, detail="Invalid or expired credential")

    @staticmethod
    async def _enforce_admitted_live_policy(admitted: UserAPIKeyAuth, request: Request, route: str) -> None:
        """Run the standard pipeline's authorization checks over the admitted identity.

        Mirrors the ``user_api_key_auth`` wrapper between the builder and its return: clear the
        request-scoped ``budget_reservation`` on the reloaded identity, run the route gate
        (``RouteChecks.should_call_route``) to enforce the identity's ``allowed_routes`` and any
        disabled/admin-only route, then run ``_run_centralized_common_checks`` (the same gate every
        builder path funnels through) for team-block, project-block, org, and budget. The route gate
        closes a bypass: a key barred from MCP routes could otherwise mint an envelope at the token
        endpoint (not itself an MCP route) and replay it against MCP, because the centralized checks
        treat MCP as an inference route and never re-check ``allowed_routes``.

        Failures surface with the status the standard pipeline would give them, mirroring
        ``UserAPIKeyAuthExceptionHandler``: a disallowed route is the route gate's own 403, an
        over-budget identity is a 429, a sub-check that raised its own ``HTTPException``/
        ``ProxyException`` keeps that status, a transient database outage is a retryable 503, and
        only a genuinely unresolvable failure (a blocked team/project raises a bare ``Exception``,
        same as the standard pipeline's fallback) becomes the fail-closed 401. Collapsing every
        failure to 401 was misleading: it told an over-budget but validly-authenticated caller their
        credential was invalid, which on a DCR client reads as broken auth and can trigger a
        pointless re-authorize loop that cannot fix a budget problem, and it masked a DB outage as an
        auth error."""
        from litellm.proxy.auth.route_checks import RouteChecks

        admitted.budget_reservation = None
        try:
            RouteChecks.should_call_route(route=route, valid_token=admitted, request=request)
            await _run_centralized_common_checks(
                user_api_key_auth_obj=admitted,
                request=request,
                request_data=await _read_request_body(request=request),
                route=route,
            )
        except (HTTPException, ProxyException):
            raise
        except litellm.BudgetExceededError as e:
            raise HTTPException(status_code=getattr(e, "status_code", 429), detail=str(e)) from None
        except Exception as e:  # noqa: BLE001  # untyped gate failure: retryable 503 for a DB outage, else fail closed 401
            MCPRequestHandler._raise_503_if_db_unavailable(e)
            raise HTTPException(status_code=401, detail="Invalid or expired credential") from None

    @staticmethod
    def _admitted_key_is_active(key_object: UserAPIKeyAuth) -> bool:
        """False when the referenced key is blocked or past its expiry, so a revoked key
        cannot be admitted through its still-unexpired envelope. Mirrors the active-key gate
        the bridge token endpoint applies at mint time."""
        if key_object.blocked is True:
            return False
        expires = key_object.expires
        if expires is None:
            return True
        expiry = expires if isinstance(expires, datetime) else datetime.fromisoformat(expires)
        if expiry.tzinfo is None or expiry.tzinfo.utcoffset(expiry) is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry >= datetime.now(timezone.utc)

    @staticmethod
    def _resolve_target_server_names(path: str, mcp_servers_header: Optional[List[str]]) -> List[str]:
        """
        Resolve the target MCP server names exactly as downstream routing
        does (``server.py::extract_mcp_auth_context``).

        For ``/mcp/...`` paths, downstream routing **overrides** any
        ``x-mcp-servers`` header value with the path-derived names. Mirror
        that here so an attacker cannot use a permissive header value to
        flip an auth gate while the path targets a stricter server
        (header/path TOCTOU). For non-``/mcp/...`` paths (where the path
        does not encode targets), fall back to the header.
        """
        path_targets = MCPRequestHandler._extract_target_server_names_from_path(path)
        if path_targets:
            return path_targets
        # Path did not resolve to /mcp/... targets — trust the header
        # (including an explicitly empty list, which means "no targets").
        return mcp_servers_header if mcp_servers_header is not None else []

    @staticmethod
    def _get_mcp_auth_header_from_headers(headers: Headers) -> Optional[str]:
        """
        Get the header passed to LiteLLM to pass to downstream MCP servers

        By default litellm will check for the header `x-mcp-auth` by setting one of the following:
            1. `LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME` as an environment variable
            2. `mcp_client_side_auth_header_name` in the general settings on the config.yaml file

        Support this auth: https://docs.litellm.ai/docs/mcp#using-your-mcp-with-client-side-credentials

        If you want to use a different header name, you can set the `LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME` in the secret manager or `mcp_client_side_auth_header_name` in the general settings.

        DEPRECATED: This method is deprecated in favor of server-specific auth headers using the format x-mcp-{{server_alias}}-{{header_name}} instead.
        """
        mcp_client_side_auth_header_name: str = MCPRequestHandler._get_mcp_client_side_auth_header_name()
        auth_header = headers.get(mcp_client_side_auth_header_name)
        if auth_header:
            verbose_logger.warning(
                f"The '{mcp_client_side_auth_header_name}' header is deprecated. "
                f"Please use server-specific auth headers in the format 'x-mcp-{{server_alias}}-{{header_name}}' instead."
            )
        return auth_header

    @staticmethod
    def _get_mcp_server_auth_headers_from_headers(
        headers: Headers,
    ) -> Dict[str, Dict[str, str]]:
        """
        Parse server-specific MCP auth headers from the request headers.

        Looks for headers in the format: x-mcp-{server_alias}-{header_name}
        Examples:
        - x-mcp-github-authorization: Bearer token123
        - x-mcp-zapier-x-api-key: api_key_456
        - x-mcp-deepwiki-authorization: Basic base64_encoded_creds

        Returns:
            Dict[str, Dict[str, str]]: Mapping of server alias to header dict
        """
        server_auth_headers: Dict[str, Dict[str, str]] = {}
        prefix = "x-mcp-"

        for header_name, header_value in headers.items():
            if header_name.lower().startswith(prefix):
                # Skip the access groups header as it's not a server auth header
                if (
                    header_name.lower() == MCPRequestHandler.LITELLM_MCP_ACCESS_GROUPS_HEADER_NAME.lower()
                    or header_name.lower() == MCPRequestHandler.LITELLM_MCP_SERVERS_HEADER_NAME.lower()
                ):
                    continue

                # Extract server_alias and header_name from x-mcp-{server_alias}-{header_name}
                remaining = header_name[len(prefix) :].lower()
                if "-" in remaining:
                    # Split on the first dash to separate server_alias from header_name
                    parts = remaining.split("-", 1)
                    if len(parts) == 2:
                        server_alias, auth_header_name = parts

                        # Convert common header names to proper case
                        if auth_header_name == "authorization":
                            auth_header_name = "Authorization"

                        # Initialize server dict if not exists
                        if server_alias not in server_auth_headers:
                            server_auth_headers[server_alias] = {}

                        server_auth_headers[server_alias][auth_header_name] = header_value
                        verbose_logger.debug(
                            f"Found server auth header: {server_alias} -> {auth_header_name}: {header_value[:10]}..."
                        )

        return server_auth_headers

    @staticmethod
    def _get_oauth2_headers_from_headers(headers: Headers) -> Dict[str, str]:
        """
        Get the oauth2 headers from the request headers.
        """
        oauth2_headers = {}
        for header_name, header_value in headers.items():
            if header_name.lower().startswith("authorization"):
                oauth2_headers["Authorization"] = header_value
        return oauth2_headers

    @staticmethod
    def _get_mcp_client_side_auth_header_name() -> str:
        """
        Get the header name used to pass the MCP auth header to the MCP server

        By default litellm will check for the header `x-mcp-auth` by setting one of the following:
            1. `LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME` as an environment variable
            2. `mcp_client_side_auth_header_name` in the general settings on the config.yaml file
        """
        from litellm.proxy.proxy_server import general_settings
        from litellm.secret_managers.main import get_secret_str

        MCP_CLIENT_SIDE_AUTH_HEADER_NAME: str = MCPRequestHandler.LITELLM_MCP_AUTH_HEADER_NAME
        if get_secret_str("LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME") is not None:
            MCP_CLIENT_SIDE_AUTH_HEADER_NAME = (
                get_secret_str("LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME") or MCP_CLIENT_SIDE_AUTH_HEADER_NAME
            )
        elif general_settings.get("mcp_client_side_auth_header_name") is not None:
            MCP_CLIENT_SIDE_AUTH_HEADER_NAME = (
                general_settings.get("mcp_client_side_auth_header_name") or MCP_CLIENT_SIDE_AUTH_HEADER_NAME
            )
        return MCP_CLIENT_SIDE_AUTH_HEADER_NAME

    @staticmethod
    def get_litellm_api_key_from_headers(headers: Headers) -> Optional[str]:
        """
        Get the Litellm API key from the headers using case-insensitive lookup

        1. Check if `x-litellm-api-key` is in the headers
        2. If not, check if `Authorization` is in the headers

        Args:
            headers: Starlette Headers object that handles case insensitivity
        """
        # Headers object handles case insensitivity automatically
        api_key = headers.get(MCPRequestHandler.LITELLM_API_KEY_HEADER_NAME_PRIMARY)
        if api_key:
            return api_key

        auth_header = headers.get(MCPRequestHandler.LITELLM_API_KEY_HEADER_NAME_SECONDARY)
        if auth_header:
            return auth_header

        return None

    @staticmethod
    def _safe_get_headers_from_scope(scope: Scope) -> Headers:
        """
        Safely extract headers from ASGI scope using Starlette's Headers class
        which handles case insensitivity and proper header parsing.

        ASGI headers are in format: List[List[bytes, bytes]]
        We need to convert them to the format Headers expects.

        Collapsing the ASGI list into a dict keeps the last value for a duplicated
        header name, so a request carrying more than one ``Authorization`` is
        rejected first: for the client-forwarded token modes the gateway relays the
        caller's ``Authorization`` upstream, so a duplicate would make which token is
        forwarded ambiguous (and diverge from what admission inspected). Multiple
        ``Authorization`` headers is malformed for bearer auth anyway (RFC 9110: not
        a comma-combinable field), so fail closed with a 400.
        """
        raw_headers = scope.get("headers", [])
        MCPRequestHandler._reject_duplicate_authorization(raw_headers)
        try:
            # ASGI headers are list of [name: bytes, value: bytes] pairs
            # Convert bytes to strings and create dict for Headers constructor
            headers_dict = {name.decode("latin-1"): value.decode("latin-1") for name, value in raw_headers}
            return Headers(headers_dict)
        except (UnicodeDecodeError, AttributeError, TypeError) as e:
            verbose_logger.exception(f"Error getting headers from scope: {e}")
            # Return empty Headers object with empty dict
            return Headers({})

    @staticmethod
    def _reject_duplicate_authorization(raw_headers: object) -> None:
        """Raise 400 when the raw ASGI headers carry more than one ``Authorization`` header."""
        if not isinstance(raw_headers, (list, tuple)):
            return
        count = 0
        for entry in raw_headers:
            if not isinstance(entry, (list, tuple)) or len(entry) < 1:
                continue
            name = entry[0]
            if isinstance(name, (bytes, bytearray)) and bytes(name).lower() == b"authorization":
                count += 1
            elif isinstance(name, str) and name.lower() == "authorization":
                count += 1
        if count > 1:
            raise HTTPException(
                status_code=400,
                detail="Multiple Authorization headers are not allowed",
            )

    @staticmethod
    async def get_allowed_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        *,
        keyless_source: bool = False,
    ) -> List[str]:
        """
        Get list of allowed MCP servers for the given user/key based on permissions.

        Permission hierarchy (all rules are intersections):
        1. Get allowed servers from key permissions
        2. Get allowed servers from team permissions (key inherits from team, or
           intersection; or inherits nothing when require_key_mcp_access_defined
           is enabled, making the team a ceiling rather than a default)
        3. Get allowed servers from end_user permissions (intersected if set)
        4. Get allowed servers from agent permissions (intersected if set)
        5. Get allowed servers from org permissions — org acts as a ceiling: if the org
           has an explicit MCP server list, the combined key/team/end_user/agent result is
           capped to that list.  If the org has no list, no extra restriction is applied.

        Returns:
            List[str]: List of allowed MCP servers by server id
        """
        from litellm.proxy.proxy_server import general_settings

        try:
            # A keyless admitted subject is resolved entirely per source, BEFORE any single-source
            # rule runs here. Ordering matters: the no_mcp_servers opt-out below reads the caller's
            # own object_permission, so leaving it above this branch let a user's own opt-out zero
            # their TEAMS' grants too — the sources are independent, and an opt-out on one of them
            # must silence only that one (it is applied per source, inside the recursive call).
            if _is_mcp_admitted_user_subject(user_api_key_auth) and user_api_key_auth is not None:
                return await MCPRequestHandler._resolve_admitted_subject_servers(user_api_key_auth)

            # Get allowed servers from key and team
            allowed_mcp_servers_for_key = await MCPRequestHandler._get_allowed_mcp_servers_for_key(user_api_key_auth)

            # The key explicitly opted out of every MCP server. This overrides
            # team inheritance and additive grants (mirrors no-default-models).
            if SpecialMCPServerNames.no_mcp_servers.value in allowed_mcp_servers_for_key:
                return []

            allowed_mcp_servers_for_team = await MCPRequestHandler._get_allowed_mcp_servers_for_team(user_api_key_auth)

            key_access_group_grants = await MCPRequestHandler._get_key_access_group_mcp_server_extras(user_api_key_auth)

            #########################################################
            # Calculate key/team allowed servers using inheritance and intersection logic
            #########################################################
            key_set = set(allowed_mcp_servers_for_key)
            team_set = set(allowed_mcp_servers_for_team)
            grants_set = set(key_access_group_grants)

            has_lower_level_mcp_restrictions = bool(key_set or team_set or grants_set)

            # 1. Key/team ceiling. An empty set means "this level does not restrict".
            if not team_set:
                base = key_set  # no team restriction
            elif not key_set:
                # A key that grants no MCP servers of its own inherits the
                # team's by default. With require_key_mcp_access_defined the
                # team is a ceiling rather than a default, so the key must
                # grant servers explicitly (or via an access group) to reach
                # any — it inherits none. That ceiling is for VIRTUAL KEYS that
                # can declare their own access; a keyless gateway/bridge-admitted
                # user has no key to declare access on — team membership IS their
                # only access path — so the flag must not zero their team grants.
                # A keyless admitted subject returned above and never reaches this virtual-key ceiling,
                # so require_key_mcp_access_defined can only ever zero a real key's inherited team grants.
                # ``keyless_source`` marks one grant source of an admitted subject, which has no key
                # to declare access on, so the flag must not zero its team grants.
                require_key_access = (
                    general_settings.get("require_key_mcp_access_defined", False) and not keyless_source
                )
                base = team_set if not require_key_access else set()
            else:
                base = key_set & team_set  # both restrict → intersect

            # 2. Add the key's access-group grants on top. These are additive:
            # attaching a group to the key grants its servers regardless of the
            # team ceiling.
            allowed_mcp_servers: List[str] = list(base | grants_set)

            #########################################################
            # Check end_user permissions if end_user_id is set
            #########################################################
            if user_api_key_auth and user_api_key_auth.end_user_id:
                allowed_mcp_servers_for_end_user = await MCPRequestHandler._get_allowed_mcp_servers_for_end_user(
                    user_api_key_auth
                )

                # If end_user has explicit MCP server permissions, apply intersection
                if len(allowed_mcp_servers_for_end_user) > 0:
                    has_lower_level_mcp_restrictions = True
                    verbose_logger.debug(
                        f"End user {user_api_key_auth.end_user_id} has explicit MCP permissions: {allowed_mcp_servers_for_end_user}"
                    )

                    # Always apply intersection: key/team AND end_user
                    # This ensures end_user can only access servers that both they AND their key/team are authorized for
                    filtered_servers = []
                    for _mcp_server in allowed_mcp_servers:
                        if _mcp_server in allowed_mcp_servers_for_end_user:
                            filtered_servers.append(_mcp_server)
                    allowed_mcp_servers = filtered_servers
                    verbose_logger.debug(
                        f"Applied end_user intersection filter. Final allowed servers: {allowed_mcp_servers}"
                    )
                # If flag is enabled but end_user has no permissions, block all access
                elif general_settings.get("require_end_user_mcp_access_defined", False):
                    verbose_logger.debug(
                        f"require_end_user_mcp_access_defined=True and end_user {user_api_key_auth.end_user_id} has no MCP permissions - blocking MCP access"
                    )
                    return []

            #########################################################
            # Check agent permissions if agent_id is set on the key
            #########################################################
            if user_api_key_auth and user_api_key_auth.agent_id:
                allowed_mcp_servers_for_agent = await MCPRequestHandler._get_allowed_mcp_servers_for_agent(
                    user_api_key_auth
                )
                if len(allowed_mcp_servers_for_agent) > 0:
                    has_lower_level_mcp_restrictions = True
                    # Intersect: agent can only use servers allowed by BOTH key/team AND agent config
                    allowed_mcp_servers = [s for s in allowed_mcp_servers if s in allowed_mcp_servers_for_agent]
                    verbose_logger.debug(
                        f"Applied agent intersection filter. Final allowed servers: {allowed_mcp_servers}"
                    )

            #########################################################
            # Apply org-level ceiling if org_id is set
            #########################################################
            allowed_mcp_servers = await MCPRequestHandler._apply_primary_org_ceiling(
                allowed_mcp_servers,
                user_api_key_auth,
                has_lower_level_mcp_restrictions,
                keyless_source=keyless_source,
            )

            return list(set(allowed_mcp_servers))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers: {str(e)}")
            return []

    @staticmethod
    async def _apply_primary_org_ceiling(
        allowed_mcp_servers: list[str],
        user_api_key_auth: UserAPIKeyAuth | None,
        has_lower_level_mcp_restrictions: bool,
        keyless_source: bool = False,
    ) -> list[str]:
        """Cap the resolved server list by this caller's org ceiling. If the org names an explicit MCP
        list, lower-level restrictions are intersected with it, else the org list becomes the ceiling.
        No org, or an empty org list, leaves the result unchanged.

        ``keyless_source`` marks one grant source of a keyless admitted subject and governs BOTH
        org divergences, because they are the same fact about that caller shape.

        First, what an UNRESOLVABLE ceiling means. A virtual key keeps the
        long-standing fail-open behavior (a DB blip must not lock working keys out mid-incident). A
        keyless admitted subject fails CLOSED, because its only org bound is this ceiling: silently
        dropping it on a transient fault would widen a cross-org user to servers their team's org
        forbids, which is a privilege escalation rather than an availability blip.

        Second, whether the org list may SUBSTITUTE for absent lower-level grants. For a key it may
        (that is the key ceiling model). For a source it may only ever intersect, because the
        admitted model is a union of grants and a ceiling that grants is not a ceiling."""
        if not (user_api_key_auth and user_api_key_auth.org_id):
            return allowed_mcp_servers
        allowed_mcp_servers_for_org = await MCPRequestHandler._get_allowed_mcp_servers_for_org(user_api_key_auth)
        if allowed_mcp_servers_for_org is None:
            verbose_logger.warning(
                f"MCP org ceiling unresolved for org_id={user_api_key_auth.org_id!r}; "
                f"{'denying (keyless admitted subject)' if keyless_source else 'leaving uncapped (key auth)'}"
            )
            return [] if keyless_source else allowed_mcp_servers
        if len(allowed_mcp_servers_for_org) == 0:
            return allowed_mcp_servers
        if has_lower_level_mcp_restrictions or keyless_source:
            # Lower-level restrictions exist, so org can only cap them.
            #
            # A keyless admitted source ALWAYS takes this arm: its model is a union of GRANTS, so an
            # org list may only narrow what a source already grants, never become one. Letting it
            # substitute would hand every admitted user with an org_id that org's whole server list
            # without any direct or team grant — a ceiling silently acting as a grant.
            capped = [s for s in allowed_mcp_servers if s in allowed_mcp_servers_for_org]
        else:
            # No lower-level restrictions → org list becomes the ceiling.
            capped = allowed_mcp_servers_for_org
        verbose_logger.debug(f"Applied org ceiling filter. Final allowed servers: {capped}")
        return capped

    @staticmethod
    def _scoped_source_auth(
        auth: UserAPIKeyAuth,
        *,
        team_id: str | None,
        org_id: str | None,
        carry_user_grants: bool,
    ) -> UserAPIKeyAuth:
        """A plain, UNMARKED auth describing ONE grant source of an admitted subject.

        Only the fields the resolver actually consults are carried. Everything else is left at its
        default on purpose: ``api_key``/``token`` stay unset (this is not a key), budget, spend and
        rate-limit fields stay unset because the admitted subject's own user-level limits are what
        the request is metered against and cloning them per source would show the limiter N copies of
        the same descriptor, and ``user_role`` stays unset because an admin role would grant every
        server if this auth ever reached the server-manager wrapper. The admission marker cannot be
        set through the constructor at all (a before-validator pops it), so each source is resolved
        as an ordinary caller and cannot re-enter the admitted path.
        """
        scoped = UserAPIKeyAuth(
            user_id=auth.user_id,
            team_id=team_id,
            org_id=org_id,
            parent_otel_span=auth.parent_otel_span,
        )
        if carry_user_grants:
            # The user's OWN grants. A team source deliberately carries none of these: the resolver
            # loads that team's object_permission and access groups from team_id itself, and mixing
            # the user's in would widen the team source with grants the team never made.
            scoped.object_permission = auth.object_permission
            scoped.object_permission_id = auth.object_permission_id
            scoped.access_group_ids = auth.access_group_ids
        return scoped

    @staticmethod
    async def _admitted_subject_sources(auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
        """The independent sources a keyless admitted subject reaches MCP servers through: their own
        direct grants, plus every team they are a live roster member of.

        Each team source carries that TEAM's org as its ``org_id``, which is what makes the canonical
        resolver apply the team's OWN owning-org ceiling to it — a cross-org user's teams are each
        bounded by their own org rather than by the caller's home org. A team with no organization
        falls back to the user's org so it is bounded rather than unbounded.

        Roster membership is checked HERE because it is a property of the source list, not of any one
        resolution: a key is structurally pinned to a team it belongs to, while a user's cached
        ``teams`` array can name a team whose ``members_with_roles`` no longer contains them (SCIM
        group sync, or cache lag after a team_member_delete), and JWT auth can rewrite that array
        outright. The roster is the source of truth for revocation.
        """
        from litellm.proxy.proxy_server import prisma_client

        sources = [
            MCPRequestHandler._scoped_source_auth(auth, team_id=None, org_id=auth.org_id, carry_user_grants=True)
        ]
        if not auth.user_id or prisma_client is None:
            return sources
        for team_id in await MCPRequestHandler._resolve_user_team_ids(auth.user_id, auth):
            team_obj = await MCPRequestHandler._roster_team_object(team_id, auth)
            if team_obj is None:
                continue
            sources.append(
                MCPRequestHandler._scoped_source_auth(
                    auth,
                    team_id=team_id,
                    org_id=team_obj.organization_id or auth.org_id,
                    carry_user_grants=False,
                )
            )
        return sources

    @staticmethod
    async def _roster_team_object(team_id: str, auth: UserAPIKeyAuth) -> LiteLLM_TeamTable | None:
        """The team row for ``team_id``, but ONLY when ``auth``'s user is a live roster member of it.

        The single owner of "is this team really one of this subject's sources", so the grant union
        and the per-team rate limits cannot disagree about which teams count. A team lingering in the
        user's cached ``teams`` array whose ``members_with_roles`` no longer lists them returns None
        here, which is what revokes both its grants and its throttle in one place."""
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None or not auth.user_id:
            return None
        try:
            team_obj: LiteLLM_TeamTable | None = await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:  # noqa: BLE001  # per-source isolation: one team's blip must not deny the others
            # The unit of fault isolation is the SOURCE: a team that cannot be resolved contributes
            # nothing this request (fail closed for that team alone — access only ever narrows),
            # while the user's own grants and every other resolvable team stand. Raising here
            # instead would collapse the whole union to deny-all because one team's row was
            # momentarily unreadable, on the servers, tools and throttle axes alike.
            verbose_logger.warning(f"MCP admitted-subject source team {team_id!r} unresolvable, skipping: {str(e)}")
            return None
        if team_obj is None:
            return None
        member_user_ids = {getattr(m, "user_id", None) for m in (team_obj.members_with_roles or [])} - {None}
        if auth.user_id not in member_user_ids:
            return None
        # A team over its own max budget — or owned by an org over ITS budget — is not a live
        # grantor, exactly as it is not for a virtual key pinned to it (common_checks rejects that
        # key outright). Enforced with the SAME owners the key path uses (_team_max_budget_check /
        # _organization_max_budget_check, cross-pod Redis-first spend), targeted at the TEAM's org
        # via the scoped source view, so a cross-org team is judged by its own org's budget. This is
        # budget ENFORCEMENT of an already-exceeded state; ATTRIBUTION of new spend stays with the
        # user (documented deferral) — the two are different questions. Sitting here, no consumer of
        # the source list (servers, tools, throttle stamping) can ever see an over-budget team.
        from litellm.exceptions import BudgetExceededError
        from litellm.proxy.auth.auth_checks import (
            _organization_max_budget_check,
            _team_max_budget_check,
        )

        source_view = MCPRequestHandler._scoped_source_auth(
            auth, team_id=team_id, org_id=team_obj.organization_id or auth.org_id, carry_user_grants=False
        )
        try:
            await _team_max_budget_check(
                team_object=team_obj, valid_token=source_view, proxy_logging_obj=proxy_logging_obj
            )
            await _organization_max_budget_check(
                valid_token=source_view,
                team_object=team_obj,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
        except BudgetExceededError as e:
            verbose_logger.info(f"MCP admitted-subject source team {team_id!r} over budget, not a grantor: {str(e)}")
            return None
        except Exception as e:  # noqa: BLE001  # per-source isolation: a budget-check fault narrows, never raises
            verbose_logger.warning(f"MCP budget check failed for source team {team_id!r}, skipping source: {str(e)}")
            return None
        return team_obj

    @staticmethod
    async def admitted_source_grants(auth: UserAPIKeyAuth) -> list[tuple[UserAPIKeyAuth, set[str]]]:
        """``(source, the servers that source grants)`` for every source of an admitted subject.

        THE owner of "which source reaches which server". The reachable union, the per-team throttle
        scope, the tool union and billing attribution are all just different reads of this one
        answer — computing it separately per consumer is how they drift (a throttle map scoped by
        roster instead of by grant charged unrelated teams' buckets)."""
        return [
            (source, set(await MCPRequestHandler.get_allowed_mcp_servers(source, keyless_source=True)))
            for source in await MCPRequestHandler._admitted_subject_sources(auth)
        ]

    @staticmethod
    async def _resolve_admitted_subject_servers(auth: UserAPIKeyAuth) -> list[str]:
        """Union of what each of the admitted subject's sources reaches, each answered by the
        canonical resolver so no rule is reimplemented for this caller shape."""
        reachable: set[str] = set()
        for _source, granted in await MCPRequestHandler.admitted_source_grants(auth):
            reachable.update(granted)
        return list(reachable)

    @staticmethod
    async def billing_auth_for_tool_call(auth: UserAPIKeyAuth, tool_name: str) -> UserAPIKeyAuth:
        """The auth object a tool call's SPEND should be recorded against.

        Returns ``auth`` unchanged for every caller that is not a keyless admitted subject, so key
        and JWT billing is byte-identical. For an admitted subject whose call is reached through a
        team's grant, returns a copy carrying that team's ``team_id`` and its owning ``org_id`` so
        the team's budget accumulates and the correct organization is charged.

        Inert rather than wrong when the target server cannot be resolved from the tool name (a
        display-name override, or a REST caller passing server_id with an unprefixed name): billing
        then falls back to today's user-level attribution instead of guessing a team. Resolution
        reuses the manager's own tool-name lookup rather than re-deriving prefix rules that live
        there."""
        if not _is_mcp_admitted_user_subject(auth):
            return auth
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            server = global_mcp_server_manager._get_mcp_server_from_tool_name(tool_name)
            if server is None:
                return auth
            source = await MCPRequestHandler.attributing_source_for_server(auth, server.server_id)
            if source is None or not source.team_id:
                return auth
            billed = auth.model_copy()
            billed.team_id = source.team_id
            billed.org_id = source.org_id
            return billed
        except Exception as e:  # noqa: BLE001  # attribution must never fail an authorized call
            verbose_logger.warning(f"MCP billing attribution failed for {tool_name!r}, billing the user: {str(e)}")
            return auth

    @staticmethod
    async def attributing_source_for_server(
        auth: UserAPIKeyAuth,
        server_id: str,
        source_grants: list[tuple[UserAPIKeyAuth, set[str]]] | None = None,
    ) -> UserAPIKeyAuth | None:
        """The source a billable call to ``server_id`` is attributed to, or None to bill the caller
        as themselves (their own grant reaches it, or nothing does).

        A keyless admitted subject carries no ``team_id``, so downstream spend skipped team updates
        entirely and charged the user's PRIMARY org — a team-derived call neither accumulated its
        team's budget (so that budget could never begin to block) nor charged the org that owns the
        granting team. Attribution restores both.

        The rule: a user's OWN grant is not "through a team", so it bills the user. Otherwise the
        call is billed to a granting team — deterministically the lowest ``team_id`` when several
        grant the same server, so the choice is stable, reproducible and auditable rather than
        dependent on dict ordering. Reads the one grant owner, so the team that gets billed is
        always a team that actually granted the server."""
        source_grants = source_grants or await MCPRequestHandler.admitted_source_grants(auth)
        granting = [(source, granted) for source, granted in source_grants if server_id in granted]
        if not granting:
            return None
        for source, _granted in granting:
            if source.team_id is None:
                return None  # the user's own grant reaches it: their spend, their org
        return min((source for source, _ in granting), key=lambda s: s.team_id or "")

    @staticmethod
    async def _resolve_admitted_subject_tools(server_id: str, auth: UserAPIKeyAuth) -> list[str] | None:
        """Effective tool allowlist on ``server_id`` for an admitted subject, as the union over the
        sources that actually grant that server.

        A source that does not grant the server contributes nothing, so its tool rules cannot leak
        onto a server reached through a different source. A source that grants the server with no
        tool restriction means the user can use every tool on it, so allow-all wins the union. When
        no source grants the server the result is ``[]`` — deny all, fail closed."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        # An OPEN channel (operator-opened allow_all_keys, the user's own BYOM submission) makes the
        # server REACHABLE through the user themselves — no grant source names it, so without this
        # the union below would return [] and leave it listable but uninvokable. Reachability is ALL
        # it confers: it is not a waiver of the ceilings that bound the server. The user's own
        # mcp_tool_permissions and their org's tool ceiling still bind, which is what a virtual key
        # on the same allow_all server gets (its key_tools and _apply_agent_and_org_tool_ceilings
        # both run). Returning None here instead skipped both and let a session holder invoke tools
        # their own or their org's policy excludes.
        reachable_via_open_channel = server_id in await global_mcp_server_manager.operator_open_server_ids(auth)

        allowed: set[str] = set()
        for source, granted in await MCPRequestHandler.admitted_source_grants(auth):
            # The open channel is evaluated against the user's OWN source (team_id is None), so that
            # source's restrictions apply to it; a team's rules never ride an open-channel server.
            if server_id not in granted and not (reachable_via_open_channel and source.team_id is None):
                continue
            tools = await MCPRequestHandler.get_allowed_tools_for_server(server_id, source, keyless_source=True)
            if tools is None:
                return None
            allowed.update(tools)
        return sorted(allowed)

    @staticmethod
    def _get_key_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ):
        """
        Get key object_permission - already loaded by get_key_object() in main auth flow.

        Note: object_permission is automatically populated when the key is fetched via
        get_key_object() in litellm/proxy/auth/auth_checks.py
        """
        if not user_api_key_auth:
            return None

        return user_api_key_auth.object_permission

    @staticmethod
    async def _get_team_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ):
        """
        Get team object_permission - automatically loaded by get_team_object() in main auth flow.

        Note: object_permission is automatically populated when the team is fetched via
        get_team_object() in litellm/proxy/auth/auth_checks.py
        """
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        verbose_logger.debug(
            f"MCP team permission lookup: team_id={user_api_key_auth.team_id if user_api_key_auth else None}"
        )
        if not user_api_key_auth or not user_api_key_auth.team_id or not prisma_client:
            return None

        if user_api_key_auth.team_id == UI_TEAM_ID:
            return None

        # Get the team object (which has object_permission already loaded)
        team_obj: Optional[LiteLLM_TeamTable] = await get_team_object(
            team_id=user_api_key_auth.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_auth.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )

        if not team_obj:
            return None

        return team_obj.object_permission

    @staticmethod
    async def get_allowed_tools_for_server(
        server_id: str,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        *,
        keyless_source: bool = False,
    ) -> Optional[List[str]]:
        """
        Get list of allowed tool names for a specific server based on key/team permissions.
        Follows same inheritance logic as get_allowed_mcp_servers.

        Args:
            server_id: Server ID to check permissions for
            user_api_key_auth: User auth

        Returns:
            List[str] if restrictions exist, None if no restrictions (allow all)
        """
        if not user_api_key_auth:
            return None

        try:
            # FIRST statement, mirroring get_allowed_mcp_servers: a keyless admitted subject is
            # resolved per grant source and shares NOTHING with the single-credential prelude below.
            # Ordering is the invariant, not a nicety — when this branch sat after the prelude, a
            # fault in a lookup the subject never uses (its own mcp_toolsets, its team_obj_perm) hit
            # the fail-closed handler and denied tools its teams did grant. Nothing that resolves a
            # single credential's scope may run before this line.
            if _is_mcp_admitted_user_subject(user_api_key_auth):
                return await MCPRequestHandler._resolve_admitted_subject_tools(server_id, user_api_key_auth)

            # Get key and team object permissions (already loaded in main auth flow)
            key_obj_perm = MCPRequestHandler._get_key_object_permission(user_api_key_auth)
            team_obj_perm = await MCPRequestHandler._get_team_object_permission(user_api_key_auth)

            # Extract tool permissions for this server. Dict keys may be
            # server_ids OR names/aliases; normalize to server_id-keyed form
            # before lookup so a name-based key does not silently drop its
            # tool restrictions when server_id is the resolved uuid.
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            key_direct_tools = (
                global_mcp_server_manager.expand_tool_permissions(key_obj_perm.mcp_tool_permissions).get(server_id)
                if key_obj_perm
                else None
            )

            # Tools granted through the key's toolsets restrict this server exactly
            # as direct tool permissions do; union with any direct grants so the
            # tool-level check sees the key's full effective tool scope
            key_toolset_ids = (key_obj_perm.mcp_toolsets or []) if key_obj_perm else []
            key_toolset_tools = (
                (await global_mcp_server_manager.resolve_toolset_tool_permissions(toolset_ids=key_toolset_ids)).get(
                    server_id
                )
                if key_toolset_ids
                else None
            )

            key_tools = (
                list(set(key_direct_tools or []) | set(key_toolset_tools or []))
                if key_direct_tools is not None or key_toolset_tools is not None
                else None
            )
            team_tools = (
                global_mcp_server_manager.expand_tool_permissions(team_obj_perm.mcp_tool_permissions).get(server_id)
                if team_obj_perm
                else None
            )

            # A keyless gateway/bridge-admitted user has no single team_id, so team_obj_perm above is
            # None and the single-team lookup yields allow-all — silently dropping every team's
            # per-server tool exclusions. Resolve it as the union over the sources that grant the
            # Apply same inheritance logic as get_allowed_mcp_servers
            if team_tools:
                if key_tools:
                    # Both have restrictions → intersection
                    allowed_tools = list(set(team_tools) & set(key_tools))
                else:
                    # Only team has restrictions → inherit from team
                    allowed_tools = team_tools
            else:
                # No team restrictions → use key restrictions
                allowed_tools = cast(List[str], key_tools)

            return await MCPRequestHandler._apply_agent_and_org_tool_ceilings(
                allowed_tools, server_id, user_api_key_auth, keyless_source=keyless_source
            )

        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed tools for server: {str(e)}")
            # Fail CLOSED for a keyless admitted subject: ANY error resolving the tool allowlist
            # (multi-team fan-out, org/agent lookups) must deny the server's tools ([]) for this
            # request rather than collapse to allow-all (None), mirroring the fail-closed server
            # path. Key/JWT auth keeps its prior allow-all-on-error behavior.
            #
            # keyless_source matters as much as the marker: each source of an admitted subject is
            # resolved through an UNMARKED auth, so without it a fault under a source returned None,
            # and None wins the union as allow-all — dropping every team and org tool ceiling on a
            # blip. The marker alone only covers a fault raised before the fan-out.
            return [] if (keyless_source or _is_mcp_admitted_user_subject(user_api_key_auth)) else None

    @staticmethod
    async def _apply_agent_and_org_tool_ceilings(
        allowed_tools: list[str] | None,
        server_id: str,
        user_api_key_auth: UserAPIKeyAuth,
        keyless_source: bool = False,
    ) -> list[str] | None:
        """Narrow a key/team tool allowlist by the agent's tool permissions and the caller's org tool
        ceiling. Each level only ever intersects, and None at a level means "no restriction from this
        level".

        An UNRESOLVABLE org ceiling (``_get_org_object_permission`` raises: the org names a permission
        that cannot be loaded) is decided here, per caller shape, mirroring the servers axis: a
        virtual key keeps its long-standing fail-open — the org step is skipped and the key/team/agent
        restrictions already computed STAND (letting the raise escape would collapse them to
        allow-all, which is fail-open WIDER than before the fault). A keyless admitted source
        re-raises, and the outer handler denies tools for that one source while the subject's other
        sources stand — its only org bound is this ceiling, so skipping it would widen access."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        if user_api_key_auth.agent_id:
            # Pre-fetch agent object_permission once to avoid a duplicate DB query.
            agent_obj_perm = await MCPRequestHandler._get_agent_object_permission(user_api_key_auth)
            agent_tools = await MCPRequestHandler._get_agent_tool_permissions_for_server(
                server_id=server_id,
                user_api_key_auth=user_api_key_auth,
                agent_object_permission=agent_obj_perm,
            )
            if agent_tools is not None:
                allowed_tools = (
                    list(set(allowed_tools) & set(agent_tools)) if allowed_tools is not None else agent_tools
                )

        if user_api_key_auth.org_id:
            # _get_org_object_permission uses user_api_key_cache, so this is not a fresh DB round-trip
            # when get_allowed_mcp_servers was already called.
            try:
                org_obj_perm = await MCPRequestHandler._get_org_object_permission(user_api_key_auth)
            except Exception as e:  # noqa: BLE001  # unresolvable org ceiling, decided per caller shape
                if keyless_source:
                    raise
                verbose_logger.warning(
                    f"MCP org tool ceiling unresolvable for org_id={user_api_key_auth.org_id!r}; "
                    f"skipping org intersect, key/team/agent restrictions stand: {str(e)}"
                )
                return allowed_tools
            org_tools = (
                global_mcp_server_manager.expand_tool_permissions(org_obj_perm.mcp_tool_permissions).get(server_id)
                if org_obj_perm and org_obj_perm.mcp_tool_permissions
                else None
            )
            if org_tools is not None:
                allowed_tools = (
                    list(set(allowed_tools) & set(org_tools)) if allowed_tools is not None else list(org_tools)
                )

        return allowed_tools

    @staticmethod
    async def is_tool_allowed_for_server(
        tool_name: str,
        server_id: str,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> bool:
        """
        Check if a specific tool is allowed for a server based on key/team permissions.

        Args:
            tool_name: Name of the tool to check
            server_id: Server ID
            user_api_key_auth: User auth

        Returns:
            True if allowed, False if blocked
        """
        allowed_tools = await MCPRequestHandler.get_allowed_tools_for_server(
            server_id=server_id,
            user_api_key_auth=user_api_key_auth,
        )

        # None means no restrictions (allow all)
        if allowed_tools is None:
            return True

        # Empty list means no tools allowed
        if not allowed_tools:
            return False

        # Check if tool is in allowed list
        return tool_name in allowed_tools

    @staticmethod
    def is_tool_allowed(
        allowed_mcp_servers: List[str],
        server_name: str,
    ) -> bool:
        """
        Check if the tool is allowed for the given user/key based on permissions
        """
        if len(allowed_mcp_servers) == 0:
            return False
        elif server_name in allowed_mcp_servers:
            return True
        return False

    @staticmethod
    async def _get_key_access_group_mcp_server_extras(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Resolve the key's unified `access_group_ids` (LiteLLM_AccessGroupTable) to
        MCP server IDs as additive grants: a group attached to the key extends the
        key's allowed servers on top of the key/team ceiling rather than being
        capped by the team. Attaching the group to the key is itself the grant —
        no `assigned_key_ids` / `assigned_team_ids` re-check. Tag-style
        `mcp_access_groups` (per-server tags) live in the key's object_permission
        scope, not here.
        """
        if user_api_key_auth is None:
            return []
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )
            from litellm.proxy.auth.auth_checks import (
                _get_mcp_server_ids_from_access_groups,
            )
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            raw_server_ids = await _get_mcp_server_ids_from_access_groups(
                access_group_ids=user_api_key_auth.access_group_ids or [],
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            if not raw_server_ids:
                return []
            # Permission entries may be server_ids OR names/aliases — expand to ids.
            return global_mcp_server_manager.expand_permission_list(raw_server_ids)
        except Exception as e:
            verbose_logger.warning(f"Failed to get key access group MCP server grants: {str(e)}")
            return []

    @staticmethod
    async def _get_allowed_mcp_servers_for_key(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get the key's own MCP ceiling from its object_permission
        (mcp_servers, tag-style mcp_access_groups, mcp_tool_permissions).

        Unified key.access_group_ids are NOT resolved here — they are additive
        grants handled by _get_key_access_group_mcp_server_extras and unioned on
        top of the key/team ceiling, so they must not enter this scope (which is
        intersected against the team).
        """
        if user_api_key_auth is None:
            return []
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )
            from litellm.proxy.auth.auth_checks import (
                get_object_permission,
            )
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            # Get key object permission (already loaded in main auth flow, or fetch from DB)
            key_object_permission = MCPRequestHandler._get_key_object_permission(user_api_key_auth)
            if key_object_permission is None and user_api_key_auth.object_permission_id and prisma_client is not None:
                key_object_permission = await get_object_permission(
                    object_permission_id=user_api_key_auth.object_permission_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=user_api_key_auth.parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
            if key_object_permission is None:
                return []

            # Sentinel opt-out: surface it unexpanded so the caller can short-circuit
            # to zero servers instead of inheriting the team.
            if SpecialMCPServerNames.no_mcp_servers.value in (key_object_permission.mcp_servers or []):
                return [SpecialMCPServerNames.no_mcp_servers.value]

            # Permission entries may be server_ids OR names/aliases — expand to ids.
            direct_mcp_servers = global_mcp_server_manager.expand_permission_list(
                key_object_permission.mcp_servers or []
            )

            # Get MCP servers from access groups
            access_group_servers = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                key_object_permission.mcp_access_groups or []
            )

            # servers referenced in tool permissions should also be accessible
            tool_perm_servers = list(
                global_mcp_server_manager.expand_tool_permissions(key_object_permission.mcp_tool_permissions).keys()
            )

            # servers referenced by the key's toolset grants are part of the key's
            # scope on every path (list, call, REST), subject to the same team/org
            # ceilings as any other key-level grant
            toolset_ids = key_object_permission.mcp_toolsets or []
            toolset_servers = (
                list((await global_mcp_server_manager.resolve_toolset_tool_permissions(toolset_ids=toolset_ids)).keys())
                if toolset_ids
                else []
            )

            # Combine all lists
            all_servers = direct_mcp_servers + access_group_servers + tool_perm_servers + toolset_servers
            return list(set(all_servers))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers for key: {str(e)}")
            return []

    @staticmethod
    async def _get_allowed_mcp_servers_for_team(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> list[str]:
        """Get allowed MCP servers a caller inherits from the team it is pinned to.

        Exactly one team, or none. A subject that reaches servers through SEVERAL teams does not
        fan out here: it is resolved one source per team in ``_resolve_admitted_subject_servers``,
        and each of those sources pins a single ``team_id`` before reaching this point. Keeping the
        fan-out here as well would be a second multi-team path to drift from that one.
        """
        team_ids = await MCPRequestHandler._team_ids_for_mcp_grant(user_api_key_auth)
        if not team_ids:
            return []
        return await MCPRequestHandler._allowed_mcp_servers_for_single_team(team_ids[0], user_api_key_auth)

    @staticmethod
    async def _team_ids_for_mcp_grant(user_api_key_auth: UserAPIKeyAuth | None) -> list[str]:
        """The team ids whose MCP grants a caller inherits.

        A caller with an explicit ``team_id`` uses that single team; every other caller inherits no
        team grants. That covers key auth and JWT auth (a keyless ``user_id`` auth with no team_id,
        which must NOT silently gain the union across every team the user belongs to), and it covers
        each single-source auth an admitted subject fans out into — those pin a team_id, so they land
        on the first branch. The admitted subject itself never reaches here: it resolves per source
        in ``_resolve_admitted_subject_servers`` before this point. The ``UI_TEAM_ID`` sentinel
        resolves to no teams exactly as before."""
        if user_api_key_auth is None or not user_api_key_auth.team_id:
            return []
        return [] if user_api_key_auth.team_id == UI_TEAM_ID else [user_api_key_auth.team_id]

    @staticmethod
    async def _resolve_user_team_ids(user_id: str, user_api_key_auth: UserAPIKeyAuth) -> list[str]:
        """The distinct team ids a user belongs to, from the live user record. Returns [] on
        no DB, a missing user, or any resolution failure so a lookup blip narrows access
        rather than raising; the caller's direct grants still apply."""
        from litellm.proxy.auth.auth_checks import get_user_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            return []
        try:
            user_object = await get_user_object(
                user_id=user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_id_upsert=False,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:  # noqa: BLE001  # a team-resolution blip narrows access, never raises
            verbose_logger.warning(f"Failed to resolve user teams for MCP grant: {str(e)}")
            return []
        if user_object is None or not user_object.teams:
            return []
        return list(dict.fromkeys(t for t in user_object.teams if t and t != UI_TEAM_ID))

    @staticmethod
    async def _team_granted_servers(team_obj: LiteLLM_TeamTable, team_access_group_servers: list[str]) -> set[str]:
        """The raw MCP-server set a team grants (before any org ceiling): its object_permission (direct
        ``mcp_servers``, the ``all_proxy_servers`` sentinel → the full registry, legacy access groups,
        tool-perm-referenced servers) unioned with its unified ``access_group_ids`` servers."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        object_permissions = team_obj.object_permission
        if object_permissions is None:
            return set(team_access_group_servers)
        if SpecialMCPServerName.all_proxy_servers.value in (object_permissions.mcp_servers or []):
            return set(global_mcp_server_manager.get_registry().keys())
        legacy_access_group_servers = await MCPRequestHandler._get_mcp_servers_from_access_groups(
            object_permissions.mcp_access_groups or []
        )
        return (
            set(global_mcp_server_manager.expand_permission_list(object_permissions.mcp_servers or []))
            | set(legacy_access_group_servers)
            | set(global_mcp_server_manager.expand_tool_permissions(object_permissions.mcp_tool_permissions).keys())
            | set(team_access_group_servers)
        )

    @staticmethod
    async def _allowed_mcp_servers_for_single_team(
        team_id: str,
        user_api_key_auth: UserAPIKeyAuth | None,
    ) -> list[str]:
        """Allowed MCP servers granted by ONE team (its raw grant, then capped by the team's own org
        for a keyless admitted subject).

        Unions two sources:
        - Legacy team.object_permission (mcp_servers, mcp_access_groups,
          mcp_tool_permissions).
        - Unified team.access_group_ids → access_group.access_mcp_server_ids.
          Mirrors the model-side pattern in can_team_access_model — the group
          is already attached to the team, so the team relationship is itself
          the gate (no assigned_team_ids check needed here).
        """
        try:
            from litellm.proxy.auth.auth_checks import (
                _get_mcp_server_ids_from_access_groups,
                get_team_object,
            )
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            if not team_id or team_id == UI_TEAM_ID or prisma_client is None:
                return []

            parent_otel_span = user_api_key_auth.parent_otel_span if user_api_key_auth is not None else None
            team_obj: LiteLLM_TeamTable | None = await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_obj is None:
                return []
            if team_obj.blocked:
                # A blocked team grants nothing. The central policy gate enforces this for a key
                # pinned to a single team_id, but a keyless admitted identity (no team_id) unions
                # across all of its teams and would otherwise inherit a blocked team's MCP grants.
                return []
            team_access_group_servers = await _get_mcp_server_ids_from_access_groups(
                access_group_ids=team_obj.access_group_ids or [],
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

            servers = await MCPRequestHandler._team_granted_servers(team_obj, team_access_group_servers)
            return list(servers)
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers for team: {str(e)}")
            return []

    @staticmethod
    async def _get_org_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ):
        """
        Get org object_permission via the established ``get_org_object`` /
        ``get_object_permission`` helpers so MCP requests share the same
        ``user_api_key_cache`` entries as the rest of the proxy.
        """
        from litellm.proxy.auth.auth_checks import (
            OrganizationNotFoundError,
            get_object_permission,
            get_org_object,
        )
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if not user_api_key_auth or not user_api_key_auth.org_id:
            return None

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return None

        # An ABSENT org is a determinate fact, not a failure: a team's organization_id can point at a
        # row that was deleted or has not synced yet, and get_org_object raises for that. It places no
        # ceiling, exactly as a key with a dangling org_id is not locked out.
        try:
            org_obj = await get_org_object(
                org_id=user_api_key_auth.org_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except OrganizationNotFoundError as e:
            # CONFIRMED absent (deleted org, not-yet-synced organization_id): a determinate fact, so
            # it places no ceiling. Every OTHER exception is an operational failure and propagates —
            # caught upstream as an unresolvable ceiling, which denies for a keyless source and stays
            # fail-open for a key. Catching bare Exception here treated a DB outage as "no org", which
            # silently dropped a real org's ceiling for exactly as long as the outage lasted.
            verbose_logger.debug(f"MCP org ceiling: org {user_api_key_auth.org_id!r} does not exist: {e}")
            return None

        if org_obj is None or not org_obj.object_permission_id:
            return None

        # From here the org NAMES a permission. Failing to read it is INDETERMINATE, so it must not
        # collapse into the same None that means "no ceiling" -- that is what would silently drop a
        # real ceiling on a transient fault. Raise and let each caller pick fail-open or fail-closed.
        object_permission = await get_object_permission(
            object_permission_id=org_obj.object_permission_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_auth.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
        if object_permission is None:
            raise ValueError(
                f"org {user_api_key_auth.org_id!r} names object_permission_id "
                f"{org_obj.object_permission_id!r} which could not be loaded"
            )
        return object_permission

    @staticmethod
    async def _get_allowed_mcp_servers_for_org(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get allowed MCP servers for an organization.

        Returns the MCP servers from the org's object_permission.
        An empty result means the org places no restriction (allow-all from this level).
        """
        try:
            object_permissions = await MCPRequestHandler._get_org_object_permission(user_api_key_auth)

            if object_permissions is None:
                return []

            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            # Expand names/aliases to canonical server IDs (consistent with key/team/end-user path)
            direct_mcp_servers = global_mcp_server_manager.expand_permission_list(object_permissions.mcp_servers or [])

            access_group_servers = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                object_permissions.mcp_access_groups or []
            )

            tool_perm_servers = list(
                global_mcp_server_manager.expand_tool_permissions(object_permissions.mcp_tool_permissions).keys()
            )

            all_servers = direct_mcp_servers + access_group_servers + tool_perm_servers
            return list(set(all_servers))
        except Exception as e:
            # None = the org ceiling could NOT be resolved, which is not the same fact as [] = the
            # org places no restriction. Collapsing the two is what let a transient DB fault silently
            # remove an org's ceiling; the caller picks fail-open or fail-closed from this signal.
            verbose_logger.warning(f"Failed to get allowed MCP servers for org: {str(e)}")
            return None

    @staticmethod
    async def _get_allowed_mcp_servers_for_end_user(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get allowed MCP servers for an end user.

        Returns the MCP servers from the end_user's object_permission.
        """
        from litellm.proxy.auth.auth_checks import get_end_user_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if not user_api_key_auth or not user_api_key_auth.end_user_id:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            # Use optimized get_end_user_object function with caching
            end_user_obj = await get_end_user_object(
                end_user_id=user_api_key_auth.end_user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
                route="/mcp",
            )

            if end_user_obj is None or end_user_obj.object_permission is None:
                return []

            # Permission entries may be server_ids OR names/aliases — expand to ids.
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            direct_mcp_servers = global_mcp_server_manager.expand_permission_list(
                end_user_obj.object_permission.mcp_servers or []
            )

            # Get MCP servers from access groups
            access_group_servers = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                end_user_obj.object_permission.mcp_access_groups or []
            )

            # servers referenced in tool permissions should also be accessible
            tool_perm_servers = list(
                global_mcp_server_manager.expand_tool_permissions(
                    end_user_obj.object_permission.mcp_tool_permissions
                ).keys()
            )

            # Combine all lists
            all_servers = direct_mcp_servers + access_group_servers + tool_perm_servers
            return list(set(all_servers))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers for end_user: {str(e)}")
            return []

    # Sentinel stored in cache when an agent has no object_permission, so we
    # don't re-query the DB on every MCP request for that agent.
    _AGENT_NO_PERMISSION_SENTINEL = "__agent_no_mcp_permission__"

    @staticmethod
    async def _get_agent_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ):
        """
        Get agent object_permission via the established ``get_object_permission``
        helper. Caches the ``agent_id -> object_permission_id`` mapping so we
        avoid re-reading the agent row on every request, and reuses the shared
        ``object_permission_id`` cache populated by the org / team / key paths.
        """
        from litellm.proxy.auth.auth_checks import get_object_permission
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if not user_api_key_auth or not user_api_key_auth.agent_id:
            return None

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return None

        agent_id = user_api_key_auth.agent_id
        cache_key = f"agent_object_permission_id:{agent_id}"

        try:
            object_permission_id: Optional[str] = await user_api_key_cache.async_get_cache(key=cache_key)

            if object_permission_id == MCPRequestHandler._AGENT_NO_PERMISSION_SENTINEL:
                return None

            if object_permission_id is None:
                agent_row = await AgentsRepository(prisma_client).table.find_unique(
                    where={"agent_id": agent_id},
                )
                object_permission_id = (
                    getattr(agent_row, "object_permission_id", None) if agent_row is not None else None
                )
                await user_api_key_cache.async_set_cache(
                    key=cache_key,
                    value=object_permission_id or MCPRequestHandler._AGENT_NO_PERMISSION_SENTINEL,
                    ttl=get_management_object_ttl(user_api_key_cache),
                )
                if not object_permission_id:
                    return None

            return await get_object_permission(
                object_permission_id=object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_logger.warning(f"Failed to get agent object permission: {str(e)}")
            return None

    @staticmethod
    async def _get_allowed_mcp_servers_for_agent(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        agent_object_permission=None,
    ) -> List[str]:
        """
        Get allowed MCP servers for an agent (from the agent's object_permission).

        Returns the MCP servers from the agent's object_permission.
        If agent has no object_permission, returns [] (no extra restriction).

        Args:
            user_api_key_auth: User auth with agent_id
            agent_object_permission: Pre-fetched object_permission to avoid duplicate DB query.
                If None, will be fetched from DB.
        """
        if not user_api_key_auth or not user_api_key_auth.agent_id:
            return []

        try:
            obj_perm = agent_object_permission
            if obj_perm is None:
                obj_perm = await MCPRequestHandler._get_agent_object_permission(user_api_key_auth)
            if obj_perm is None:
                return []

            direct_mcp_servers = getattr(obj_perm, "mcp_servers", None) or []
            if isinstance(direct_mcp_servers, str):
                direct_mcp_servers = []
            mcp_access_groups = getattr(obj_perm, "mcp_access_groups", None) or []
            if isinstance(mcp_access_groups, str):
                mcp_access_groups = []

            # Permission entries may be server_ids OR names/aliases — expand to ids.
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            expanded_direct_servers = global_mcp_server_manager.expand_permission_list(list(direct_mcp_servers))

            access_group_servers = await MCPRequestHandler._get_mcp_servers_from_access_groups(mcp_access_groups)
            all_servers = expanded_direct_servers + access_group_servers
            return list(set(all_servers))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers for agent: {str(e)}")
            return []

    @staticmethod
    async def _get_agent_tool_permissions_for_server(
        server_id: str,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        agent_object_permission=None,
    ) -> Optional[List[str]]:
        """
        Get allowed tool names for a server from the agent's object_permission.
        Returns None if agent has no tool restrictions for this server.

        Args:
            server_id: Server ID to check permissions for
            user_api_key_auth: User auth with agent_id
            agent_object_permission: Pre-fetched object_permission to avoid duplicate DB query.
                If None, will be fetched from DB.
        """
        if not user_api_key_auth or not user_api_key_auth.agent_id:
            return None

        try:
            obj_perm = agent_object_permission
            if obj_perm is None:
                obj_perm = await MCPRequestHandler._get_agent_object_permission(user_api_key_auth)
            if obj_perm is None:
                return None

            mcp_tool_permissions = getattr(obj_perm, "mcp_tool_permissions", None)
            if not mcp_tool_permissions or not isinstance(mcp_tool_permissions, dict):
                return None
            # Dict keys may be server_ids OR names/aliases; normalize before lookup.
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            tools = global_mcp_server_manager.expand_tool_permissions(mcp_tool_permissions).get(server_id)
            return list(tools) if tools else None
        except Exception as e:
            verbose_logger.warning(f"Failed to get agent tool permissions for server: {str(e)}")
            return None

    @staticmethod
    def _get_config_server_ids_for_access_groups(config_mcp_servers, access_groups: List[str]) -> Set[str]:
        """
        Helper to get server_ids from config-loaded servers that match any of the given access groups.
        """
        server_ids: Set[str] = set()
        for server_id, server in config_mcp_servers.items():
            if server.access_groups:
                if any(group in server.access_groups for group in access_groups):
                    server_ids.add(server_id)
        return server_ids

    @staticmethod
    async def _get_db_server_ids_for_access_groups(prisma_client, access_groups: List[str]) -> Set[str]:
        """
        Helper to get server_ids from DB servers that match any of the given access groups.
        """
        server_ids: Set[str] = set()
        if access_groups and prisma_client is not None:
            try:
                mcp_servers = await MCPServerRepository(prisma_client).table.find_many(
                    where={"mcp_access_groups": {"hasSome": access_groups}}
                )
                for server in mcp_servers:
                    server_ids.add(server.server_id)
            except Exception as e:
                verbose_logger.debug(f"Error getting MCP servers from access groups: {e}")
        return server_ids

    @staticmethod
    async def _get_mcp_servers_from_access_groups(
        access_groups: List[str],
    ) -> List[str]:
        """
        Resolve MCP access groups to server IDs by querying BOTH the MCP server table (DB) AND config-loaded servers
        """
        from litellm.proxy.proxy_server import prisma_client

        try:
            # Import here to avoid circular import
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            # Use the new helper for config-loaded servers
            server_ids = MCPRequestHandler._get_config_server_ids_for_access_groups(
                global_mcp_server_manager.config_mcp_servers, access_groups
            )

            # Use the new helper for DB servers
            db_server_ids = await MCPRequestHandler._get_db_server_ids_for_access_groups(prisma_client, access_groups)
            server_ids.update(db_server_ids)

            return list(server_ids)
        except Exception as e:
            verbose_logger.warning(f"Failed to get MCP servers from access groups: {str(e)}")
            return []

    @staticmethod
    async def get_mcp_access_groups(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get list of MCP access groups for the given user/key based on permissions
        """
        access_groups: List[str] = []
        access_groups_for_key = await MCPRequestHandler._get_mcp_access_groups_for_key(user_api_key_auth)
        access_groups_for_team = await MCPRequestHandler._get_mcp_access_groups_for_team(user_api_key_auth)

        #########################################################
        # If team has access groups, then key must have a subset of the team's access groups
        #########################################################
        if len(access_groups_for_team) > 0:
            for access_group in access_groups_for_key:
                if access_group in access_groups_for_team:
                    access_groups.append(access_group)
        else:
            access_groups = access_groups_for_key

        return list(set(access_groups))

    @staticmethod
    async def _get_mcp_access_groups_for_key(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        from litellm.proxy.auth.auth_checks import get_object_permission
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.object_permission_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            key_object_permission = await get_object_permission(
                object_permission_id=user_api_key_auth.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if key_object_permission is None:
                return []

            return key_object_permission.mcp_access_groups or []
        except Exception as e:
            verbose_logger.warning(f"Failed to get MCP access groups for key: {str(e)}")
            return []

    @staticmethod
    async def _get_mcp_access_groups_for_team(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get MCP access groups for the team
        """
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.team_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        if user_api_key_auth.team_id == UI_TEAM_ID:
            return []

        try:
            team_obj: Optional[LiteLLM_TeamTable] = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_obj is None:
                verbose_logger.debug("team_obj is None")
                return []

            object_permissions = team_obj.object_permission
            if object_permissions is None:
                return []

            return object_permissions.mcp_access_groups or []
        except Exception as e:
            verbose_logger.warning(f"Failed to get MCP access groups for team: {str(e)}")
            return []

    @staticmethod
    def get_mcp_access_groups_from_headers(headers: Headers) -> Optional[List[str]]:
        """
        Extract and parse the x-mcp-access-groups header as a list of strings.
        """
        mcp_access_groups_header = headers.get(MCPRequestHandler.LITELLM_MCP_ACCESS_GROUPS_HEADER_NAME)
        if mcp_access_groups_header is not None:
            try:
                return [s.strip() for s in mcp_access_groups_header.split(",") if s.strip()]
            except Exception:
                return None
        return None

    @staticmethod
    def get_mcp_access_groups_from_scope(scope: Scope) -> Optional[List[str]]:
        """
        Extract and parse the x-mcp-access-groups header from an ASGI scope.
        """
        headers = MCPRequestHandler._safe_get_headers_from_scope(scope)
        return MCPRequestHandler.get_mcp_access_groups_from_headers(headers)
