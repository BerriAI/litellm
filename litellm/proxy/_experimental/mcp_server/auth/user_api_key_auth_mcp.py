import re
from typing import Dict, List, Optional, Set, Tuple, cast

from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import Scope

from litellm._logging import verbose_logger
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
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import get_management_object_ttl
from litellm.repositories.table_repositories import (
    AgentsRepository,
    MCPServerRepository,
)


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
        elif oauth2_headers:
            # Authorization on a non-delegated server: the bearer must be a real
            # LiteLLM credential, so a failed validation is a genuine 401/403 and
            # propagates. The sole anonymous fallback is the auth_type=none
            # pass-through cold-start (RFC 9728 discovery return), gated on a 401
            # so a recognized-but-forbidden key still fails closed.
            client_ip = IPAddressUtils.get_mcp_client_ip(request)
            try:
                validated_user_api_key_auth = await user_api_key_auth(api_key=litellm_api_key, request=request)
            except (HTTPException, ProxyException) as e:
                # ProxyException.code is normalized to str (possibly "None"), so
                # compare both int and str forms rather than coercing.
                status = e.status_code if isinstance(e, HTTPException) else e.code
                is_unauthenticated = status in (401, "401")
                mcp_servers_from_path = _parse_mcp_server_names_from_path(request_route, mcp_servers)
                if (
                    is_unauthenticated
                    and mcp_servers_from_path is not None
                    and not _has_client_supplied_mcp_auth(
                        mcp_auth_header,
                        mcp_server_auth_headers,
                    )
                    and _is_mcp_passthrough_cold_start(mcp_servers_from_path, client_ip=client_ip)
                ):
                    verbose_logger.debug(
                        "MCP pass-through return: forwarding Authorization as upstream OAuth token for delegated auth"
                    )
                    validated_user_api_key_auth = UserAPIKeyAuth()
                else:
                    raise
        else:
            try:
                validated_user_api_key_auth = await user_api_key_auth(api_key=litellm_api_key, request=request)
            except (HTTPException, ProxyException) as exc:
                # Cold-start MCP OAuth discovery: RFC 9728 / MCP Authorization spec
                # require unauthenticated requests to protected resources to receive
                # 401 + WWW-Authenticate. Defer to _raise_preemptive_401_for_unauthenticated_servers
                # for pass-through servers instead of surfacing a generic admission error.
                mcp_servers_from_path = _parse_mcp_server_names_from_path(request_route, mcp_servers)
                client_ip = IPAddressUtils.get_mcp_client_ip(request)
                if (
                    mcp_servers_from_path is not None
                    and not _has_client_supplied_mcp_auth(
                        mcp_auth_header,
                        mcp_server_auth_headers,
                    )
                    and _is_litellm_auth_admission_error(exc)
                    and _is_mcp_passthrough_cold_start(mcp_servers_from_path, client_ip=client_ip)
                ):
                    verbose_logger.debug("MCP pass-through cold start: deferring admission to route 401 emitter")
                    validated_user_api_key_auth = UserAPIKeyAuth()
                else:
                    raise

        return (
            validated_user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            dict(headers),
        )

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
        """
        try:
            # ASGI headers are list of [name: bytes, value: bytes] pairs
            raw_headers = scope.get("headers", [])
            # Convert bytes to strings and create dict for Headers constructor
            headers_dict = {name.decode("latin-1"): value.decode("latin-1") for name, value in raw_headers}
            return Headers(headers_dict)
        except (UnicodeDecodeError, AttributeError, TypeError) as e:
            verbose_logger.exception(f"Error getting headers from scope: {e}")
            # Return empty Headers object with empty dict
            return Headers({})

    @staticmethod
    async def get_allowed_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
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
                # any — it inherits none.
                base = set() if general_settings.get("require_key_mcp_access_defined", False) else team_set
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
            if user_api_key_auth and user_api_key_auth.org_id:
                allowed_mcp_servers_for_org = await MCPRequestHandler._get_allowed_mcp_servers_for_org(
                    user_api_key_auth
                )
                if len(allowed_mcp_servers_for_org) > 0:
                    if has_lower_level_mcp_restrictions:
                        # Lower-level restrictions exist, so org can only cap them.
                        allowed_mcp_servers = [s for s in allowed_mcp_servers if s in allowed_mcp_servers_for_org]
                    else:
                        # No lower-level restrictions → org list becomes the ceiling
                        allowed_mcp_servers = allowed_mcp_servers_for_org
                    verbose_logger.debug(f"Applied org ceiling filter. Final allowed servers: {allowed_mcp_servers}")

            return list(set(allowed_mcp_servers))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers: {str(e)}")
            return []

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

            key_tools = (
                global_mcp_server_manager.expand_tool_permissions(key_obj_perm.mcp_tool_permissions).get(server_id)
                if key_obj_perm
                else None
            )
            team_tools = (
                global_mcp_server_manager.expand_tool_permissions(team_obj_perm.mcp_tool_permissions).get(server_id)
                if team_obj_perm
                else None
            )

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

            # Intersect with agent's tool permissions if agent_id is set
            if user_api_key_auth.agent_id:
                # Pre-fetch agent object_permission once to avoid duplicate DB query
                agent_obj_perm = await MCPRequestHandler._get_agent_object_permission(user_api_key_auth)
                agent_tools = await MCPRequestHandler._get_agent_tool_permissions_for_server(
                    server_id=server_id,
                    user_api_key_auth=user_api_key_auth,
                    agent_object_permission=agent_obj_perm,
                )
                if agent_tools is not None:
                    if allowed_tools is not None:
                        allowed_tools = list(set(allowed_tools) & set(agent_tools))
                    else:
                        allowed_tools = agent_tools

            # Apply org-level tool ceiling if org_id is set
            if user_api_key_auth.org_id:
                # _get_org_object_permission uses user_api_key_cache, so this is not a
                # fresh DB round-trip when get_allowed_mcp_servers was already called.
                org_obj_perm = await MCPRequestHandler._get_org_object_permission(user_api_key_auth)
                org_tools = (
                    global_mcp_server_manager.expand_tool_permissions(org_obj_perm.mcp_tool_permissions).get(server_id)
                    if org_obj_perm and org_obj_perm.mcp_tool_permissions
                    else None
                )
                if org_tools is not None:
                    if allowed_tools is not None:
                        allowed_tools = list(set(allowed_tools) & set(org_tools))
                    else:
                        allowed_tools = list(org_tools)

            return allowed_tools

        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed tools for server: {str(e)}")
            return None

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

            # Combine all lists
            all_servers = direct_mcp_servers + access_group_servers + tool_perm_servers
            return list(set(all_servers))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers for key: {str(e)}")
            return []

    @staticmethod
    async def _get_allowed_mcp_servers_for_team(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get allowed MCP servers for a team.

        Unions two sources:
        - Legacy team.object_permission (mcp_servers, mcp_access_groups,
          mcp_tool_permissions).
        - Unified team.access_group_ids → access_group.access_mcp_server_ids.
          Mirrors the model-side pattern in can_team_access_model — the group
          is already attached to the team, so the team relationship is itself
          the gate (no assigned_team_ids check needed here).
        """
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )
            from litellm.proxy.auth.auth_checks import (
                _get_mcp_server_ids_from_access_groups,
                get_team_object,
            )
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            if user_api_key_auth is None or not user_api_key_auth.team_id or prisma_client is None:
                return []

            if user_api_key_auth.team_id == UI_TEAM_ID:
                return []

            team_obj: Optional[LiteLLM_TeamTable] = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_obj is None:
                return []

            team_access_group_servers = await _get_mcp_server_ids_from_access_groups(
                access_group_ids=team_obj.access_group_ids or [],
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

            object_permissions = team_obj.object_permission
            if object_permissions is None:
                return list(set(team_access_group_servers))

            if SpecialMCPServerName.all_proxy_servers.value in (object_permissions.mcp_servers or []):
                return list(global_mcp_server_manager.get_registry().keys())

            direct_mcp_servers = global_mcp_server_manager.expand_permission_list(object_permissions.mcp_servers or [])

            legacy_access_group_servers = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                object_permissions.mcp_access_groups or []
            )

            tool_perm_servers = list(
                global_mcp_server_manager.expand_tool_permissions(object_permissions.mcp_tool_permissions).keys()
            )

            all_servers = (
                direct_mcp_servers + legacy_access_group_servers + tool_perm_servers + team_access_group_servers
            )
            return list(set(all_servers))
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
        from litellm.proxy.auth.auth_checks import get_object_permission, get_org_object
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

        try:
            org_obj = await get_org_object(
                org_id=user_api_key_auth.org_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )

            if org_obj is None or not org_obj.object_permission_id:
                return None

            return await get_object_permission(
                object_permission_id=org_obj.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_logger.warning(f"Failed to get org object permission: {str(e)}")
            return None

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
            verbose_logger.warning(f"Failed to get allowed MCP servers for org: {str(e)}")
            return []

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
