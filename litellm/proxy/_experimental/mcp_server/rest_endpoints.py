import asyncio
import importlib
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.exceptions import (
    MCPServerListError,
    MCPUpstreamAuthError,
)
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
    classify_list_exception,
    list_fault_http_status,
)
from litellm.proxy._experimental.mcp_server.ui_session_utils import (
    build_effective_auth_contexts,
)
from litellm.proxy._experimental.mcp_server.utils import (
    MCPMissingUserEnvVarsError,
    get_server_prefix,
    merge_mcp_headers,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers
from litellm.types.mcp import MCPAuth
from litellm.types.utils import CallTypes

MCP_AVAILABLE: bool = True
try:
    importlib.import_module("mcp")
except ImportError as e:
    verbose_logger.debug(f"MCP module not found: {e}")
    MCP_AVAILABLE = False


router = APIRouter(
    prefix="/mcp-rest",
    tags=["mcp"],
)


def _connection_error_message(exc: BaseException) -> str:
    if isinstance(exc, httpx.LocalProtocolError):
        return (
            "Failed to connect to MCP server: a request header is malformed. "
            "Check static headers for leading/trailing spaces or illegal characters."
        )
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return (
            "Failed to connect to MCP server: the server is unreachable. Check the URL and that the server is running."
        )
    if isinstance(exc, httpx.TimeoutException):
        return "Failed to connect to MCP server: the connection timed out."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"Failed to connect to MCP server: it returned HTTP {exc.response.status_code}."
    return "Failed to connect to MCP server. Check proxy logs for details."


if MCP_AVAILABLE:
    from mcp.types import Tool as MCPTool

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        _UPSTREAM_OAUTH_DISCOVERY_AUTH_TYPES,
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        get_request_base_url,
    )
    from litellm.proxy._experimental.mcp_server.server import (
        ListMCPToolsRestAPIResponseObject,
        MCPInfo,
        MCPServer,
        _apply_toolset_scope,
        _fire_mcp_tool_call_logging,
        _tool_name_matches,
        execute_mcp_tool,
        filter_tools_by_allowed_tools,
    )

    ########################################################
    ############ MCP Server REST API Routes #################
    async def _safe_fire_mcp_tool_call_logging(
        logging_obj: Optional[Any],
        result: Any,
        start_time: datetime,
        end_time: datetime,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        request_data: Optional[Mapping[str, object]] = None,
    ) -> None:
        if logging_obj is None:
            return
        logging_results = await asyncio.gather(
            _fire_mcp_tool_call_logging(
                logging_obj,
                result,
                start_time,
                end_time,
                user_api_key_auth=user_api_key_auth,
                request_data=request_data,
            ),
            return_exceptions=True,
        )
        logging_error = logging_results[0]
        if isinstance(logging_error, asyncio.CancelledError):
            raise logging_error
        if isinstance(logging_error, BaseException):
            verbose_logger.warning("MCP tool call logging failed (continuing): %s", logging_error)

    def _relay_upstream_auth_http_exception(e: MCPUpstreamAuthError, request: Request) -> HTTPException:
        """Convert a client-forwarded pass-through upstream 401 into an HTTPException that preserves the
        upstream WWW-Authenticate, so a standards-compliant MCP client can run the upstream OAuth flow
        instead of the generic 500 the endpoint catch-all would return."""
        return e.to_http_exception(
            base_url=get_request_base_url(request),
            request_path=request.scope.get("_original_path") or request.url.path,
        )

    async def _handle_virtual_mcp_tool(
        request: Request,
        data: Dict[str, Any],
        tool_name: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Any:
        """Handle the virtual ``mcp_tool_search`` / ``mcp_tool_call`` REST tools (gated on
        ``mcp_tool_search_enabled``). Kept out of ``call_tool_rest_api`` so that endpoint stays a single
        dispatch. An upstream 401 raised by the virtual ``mcp_tool_call`` propagates unhandled to the
        caller's ``except MCPUpstreamAuthError`` relay, the same as the direct call path."""
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._experimental.mcp_server.tool_search import (
            MCP_TOOL_SEARCH_TOOL_NAME,
            coerce_top_k,
            handle_mcp_tool_call,
            handle_mcp_tool_search,
        )
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
        from litellm.proxy.proxy_server import general_settings, proxy_config, proxy_logging_obj

        if not getattr(getattr(user_api_key_dict, "object_permission", None), "mcp_tool_search_enabled", False):
            raise HTTPException(
                status_code=403,
                detail={"error": "forbidden", "message": f"{tool_name} requires mcp_tool_search_enabled on the key"},
            )
        tool_arguments = data.get("arguments") or {}
        rest_client_ip = IPAddressUtils.get_mcp_client_ip(request)
        (
            virtual_mcp_auth_header,
            virtual_mcp_server_auth_headers,
            virtual_raw_headers,
        ) = _extract_mcp_headers_from_request(request, MCPRequestHandler)
        virtual_oauth2_headers = MCPRequestHandler._get_oauth2_headers_from_headers(request.headers)
        if tool_name == MCP_TOOL_SEARCH_TOOL_NAME:
            return await handle_mcp_tool_search(
                query=tool_arguments.get("query", ""),
                top_k=coerce_top_k(tool_arguments.get("top_k", 5)),
                user_api_key_dict=user_api_key_dict,
                client_ip=rest_client_ip,
                mcp_auth_header=virtual_mcp_auth_header,
                mcp_server_auth_headers=virtual_mcp_server_auth_headers,
                oauth2_headers=virtual_oauth2_headers,
                raw_headers=virtual_raw_headers,
            )
        # MCP_TOOL_CALL_TOOL_NAME: run the same pre-call pipeline as the normal path so the tool
        # execution is spend-logged and guardrail-checked.
        (_, virtual_logging_obj) = await ProxyBaseLLMRequestProcessing(data=data).common_processing_pre_call_logic(
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
            route_type=CallTypes.call_mcp_tool.value,
            proxy_logging_obj=proxy_logging_obj,
            general_settings=general_settings,
        )
        _tool_start_time = datetime.now()
        result = await handle_mcp_tool_call(
            tool_name=tool_arguments.get("tool_name", ""),
            arguments=tool_arguments.get("arguments") or {},
            user_api_key_dict=user_api_key_dict,
            client_ip=rest_client_ip,
            mcp_auth_header=virtual_mcp_auth_header,
            mcp_server_auth_headers=virtual_mcp_server_auth_headers,
            oauth2_headers=virtual_oauth2_headers,
            raw_headers=virtual_raw_headers,
            litellm_logging_obj=virtual_logging_obj,
        )
        await _safe_fire_mcp_tool_call_logging(
            virtual_logging_obj,
            result,
            _tool_start_time,
            datetime.now(),
            user_api_key_auth=user_api_key_dict,
            request_data=data,
        )
        return result

    def _get_server_auth_header(
        server,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        mcp_auth_header: Optional[str],
    ) -> Optional[Union[Dict[str, str], str]]:
        """Helper function to get server-specific auth header with case-insensitive matching."""
        from litellm.proxy._experimental.mcp_server.utils import (
            lookup_mcp_server_auth_in_headers,
        )

        if mcp_server_auth_headers:
            server_auth = lookup_mcp_server_auth_in_headers(
                mcp_server_auth_headers,
                alias=getattr(server, "alias", None),
                server_name=getattr(server, "server_name", None),
            )
            if server_auth is not None:
                return server_auth
        return mcp_auth_header

    def _get_oauth2_server_ids(allowed_server_ids: List[str]) -> Set[str]:
        """Return the subset of *allowed_server_ids* whose servers use OAuth2 auth.

        Used as a cheap pre-flight check to skip bulk credential fetching when no
        OAuth2 servers are involved in the current request.
        """
        return {
            sid
            for sid in allowed_server_ids
            if getattr(global_mcp_server_manager.get_mcp_server_by_id(sid), "auth_type", None) == MCPAuth.oauth2
        }

    async def _get_user_oauth_extra_headers(
        server,
        user_api_key_dict: UserAPIKeyAuth,
        prefetched_creds: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, str]]:
        """
        For OAuth2 servers, look up the user's stored access token and return it
        as extra_headers {"Authorization": "Bearer <token>"} so that it reaches
        the MCP server the same way the admin "Add MCP / Authorize and Fetch" flow does.
        Returns None for non-OAuth2 servers or when no credential is stored.

        Args:
            prefetched_creds: Optional dict keyed by server_id with credential payloads.
                              When provided, avoids a per-server DB round-trip.
        """
        if getattr(server, "auth_type", None) != MCPAuth.oauth2:
            return None
        user_id = getattr(user_api_key_dict, "user_id", None)
        server_id = getattr(server, "server_id", None)
        if not user_id or not server_id:
            return None
        try:
            from litellm.proxy._experimental.mcp_server.db import (
                get_user_oauth_credential,
                resolve_valid_user_oauth_token,
            )

            prisma_client = None
            if prefetched_creds is not None:
                cred = prefetched_creds.get(server_id)
            else:
                from litellm.proxy.utils import get_prisma_client_or_throw

                prisma_client = get_prisma_client_or_throw(
                    "Database not connected. Connect a database to use OAuth2 MCP tools."
                )
                cred = await get_user_oauth_credential(prisma_client, user_id, server_id)
            cred = await resolve_valid_user_oauth_token(
                user_id=user_id,
                server=server,
                cred=cred,
                prisma_client=prisma_client,
            )
            if cred and cred.get("access_token"):
                return {"Authorization": f"Bearer {cred['access_token']}"}
        except Exception as e:
            verbose_logger.warning(
                f"_get_user_oauth_extra_headers: failed to retrieve credential for "
                f"user={user_id} server={server_id}: {e}"
            )
        return None

    async def _prefetch_user_oauth_creds(
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch all OAuth2 credentials for the user in a single DB query.

        Returns a dict keyed by server_id. Used to avoid N+1 DB queries when
        iterating over multiple OAuth2 MCP servers.
        """
        user_id = getattr(user_api_key_dict, "user_id", None)
        if not user_id:
            return {}
        try:
            from litellm.proxy._experimental.mcp_server.db import (
                list_user_oauth_credentials,
            )
            from litellm.proxy.utils import get_prisma_client_or_throw

            prisma_client = get_prisma_client_or_throw(
                "Database not connected. Connect a database to use OAuth2 MCP tools."
            )
            creds = await list_user_oauth_credentials(prisma_client, user_id)
            return {c["server_id"]: c for c in creds if "server_id" in c}
        except Exception as e:
            verbose_logger.warning(f"_prefetch_user_oauth_creds: failed to prefetch for user={user_id}: {e}")
            return {}

    async def _get_bulk_user_oauth_headers(
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Dict[str, Dict[str, str]]:
        """
        Fetch ALL OAuth2 credentials for the current user in a single DB query and
        return a mapping of server_id → {"Authorization": "Bearer <token>"}.

        This is the batch alternative to calling _get_user_oauth_extra_headers
        per-server inside a loop (N+1 DB queries).
        """
        user_id = getattr(user_api_key_dict, "user_id", None)
        if not user_id:
            return {}
        try:
            from litellm.proxy._experimental.mcp_server.db import (
                list_user_oauth_credentials,
            )
            from litellm.proxy.utils import get_prisma_client_or_throw

            prisma_client = get_prisma_client_or_throw(
                "Database not connected. Connect a database to use OAuth2 MCP tools."
            )
            creds = await list_user_oauth_credentials(prisma_client, user_id)
            return {
                c["server_id"]: {"Authorization": f"Bearer {c['access_token']}"}
                for c in creds
                if c.get("access_token") and c.get("server_id")
            }
        except Exception:
            verbose_logger.debug("Failed to bulk-fetch OAuth credentials", exc_info=True)
            return {}

    def _create_tool_response_objects(tools, server: MCPServer):
        """Helper function to create tool response objects.

        Enriches the server's ``mcp_info`` with ``server_id`` and ``alias`` so
        REST clients can map the internal ``server_name`` to the user-facing
        alias without needing access to the ``mcp_routes``-gated server listing.
        """
        enriched_mcp_info: MCPInfo = {
            **(server.mcp_info or {}),
            "server_id": server.server_id,
            "alias": server.alias,
        }
        return [
            ListMCPToolsRestAPIResponseObject(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.inputSchema,
                mcp_info=enriched_mcp_info,
            )
            for tool in tools
        ]

    def _extract_mcp_headers_from_request(
        request: Request,
        mcp_request_handler_cls,
    ) -> tuple:
        """
        Extract MCP auth headers from HTTP request.

        Returns:
            Tuple of (mcp_auth_header, mcp_server_auth_headers, raw_headers)
        """
        headers = request.headers
        raw_headers = dict(headers)
        mcp_auth_header = mcp_request_handler_cls._get_mcp_auth_header_from_headers(headers)
        mcp_server_auth_headers = mcp_request_handler_cls._get_mcp_server_auth_headers_from_headers(headers)
        return mcp_auth_header, mcp_server_auth_headers, raw_headers

    def _resolve_mcp_server_id_for_rest(
        server_id: str,
        allowed_server_ids: Union[Set[str], List[str]],
        client_ip: Optional[str] = None,
    ) -> str:
        """
        Map REST ``server_id`` (UUID, server_name, or alias) to canonical server_id.

        tools/list already did this; tools/call must match so clients can pass
        server names like ``order_status_mcp`` instead of only UUIDs.
        """
        allowed = set(allowed_server_ids)
        if server_id in allowed:
            return server_id
        by_name = global_mcp_server_manager.get_mcp_server_by_name(server_id, client_ip=client_ip)
        if by_name is not None and by_name.server_id in allowed:
            return by_name.server_id
        return server_id

    async def _resolve_allowed_mcp_servers_with_ip_filter(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth,
        server_id: str,
    ) -> Tuple[List[MCPServer], str]:
        """
        Resolve allowed MCP servers for a tool call with IP filtering.

        Args:
            request: The HTTP request object
            user_api_key_dict: The user's API key auth object
            server_id: The server ID to validate access for

        Returns:
            Tuple of (allowed MCPServer objects, canonical server_id)

        Raises:
            HTTPException: If the server_id is not allowed or not found
        """
        # Get all auth contexts
        auth_contexts = await build_effective_auth_contexts(user_api_key_dict)

        # Collect allowed server IDs from all contexts, then apply IP filtering
        _rest_client_ip = IPAddressUtils.get_mcp_client_ip(request)
        allowed_server_ids_set = set()
        for auth_context in auth_contexts:
            servers = await global_mcp_server_manager.get_allowed_mcp_servers(
                user_api_key_auth=auth_context,
            )
            allowed_server_ids_set.update(servers)

        allowed_server_ids_set = set(
            global_mcp_server_manager.filter_server_ids_by_ip(list(allowed_server_ids_set), _rest_client_ip)
        )

        canonical_server_id = _resolve_mcp_server_id_for_rest(server_id, allowed_server_ids_set, _rest_client_ip)

        if canonical_server_id not in allowed_server_ids_set:
            _server = global_mcp_server_manager.get_mcp_server_by_id(
                server_id
            ) or global_mcp_server_manager.get_mcp_server_by_name(server_id)
            if (
                _server is not None
                and _rest_client_ip is not None
                and not global_mcp_server_manager._is_server_accessible_from_ip(_server, _rest_client_ip)
            ):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "ip_filtering",
                        "message": (
                            f"MCP server '{server_id}' is not accessible from your IP address "
                            f"({_rest_client_ip}). This server is restricted to internal "
                            "networks only. To make it externally accessible, set "
                            "'available_on_public_internet: true' in the server configuration."
                        ),
                    },
                )
            if _server is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "server_not_found",
                        "message": f"MCP server '{server_id}' was not found",
                    },
                )
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "access_denied",
                    "message": f"The key is not allowed to access server {server_id}",
                },
            )

        # Build allowed_mcp_servers list (only include allowed servers)
        allowed_mcp_servers: List[MCPServer] = []
        for allowed_server_id in allowed_server_ids_set:
            server = global_mcp_server_manager.get_mcp_server_by_id(allowed_server_id)
            if server is not None:
                allowed_mcp_servers.append(server)

        from litellm.proxy._experimental.mcp_server.mcp_trust_scoring import (
            apply_trust_filter_to_allowed_mcp_servers,
            assert_requested_server_passes_trust_filter,
        )

        allowed_mcp_servers = list(
            await apply_trust_filter_to_allowed_mcp_servers(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
                allowed_mcp_servers
            )
        )
        assert_requested_server_passes_trust_filter(
            filtered_servers=allowed_mcp_servers,
            server_id=canonical_server_id,
        )

        return allowed_mcp_servers, canonical_server_id

    async def _get_tools_for_single_server(
        server,
        server_auth_header,
        raw_headers: Optional[Dict[str, str]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        apply_tool_filters: bool = True,
    ):
        """Helper function to get tools for a single server.

        When ``apply_tool_filters`` is False the raw server catalog is returned
        without the allowed_tools/disallowed_tools gate or the per-key tool
        permissions. This is the admin-only configuration view; every runtime
        path keeps the default True so callable tools stay filtered.
        """
        tools = await global_mcp_server_manager._get_tools_from_server(
            server=server,
            mcp_auth_header=server_auth_header,
            extra_headers=extra_headers,
            add_prefix=False,
            raw_headers=raw_headers,
            user_api_key_auth=user_api_key_auth,
        )

        if not apply_tool_filters:
            return _create_tool_response_objects(tools, server)

        # Always apply allowed_tools/disallowed_tools so the blacklist is
        # enforced even when no allowlist is set (matches the SSE/HTTP path).
        tools = filter_tools_by_allowed_tools(tools, server)

        # Filter by the key's effective tool permissions through the same
        # primitive the MCP protocol path uses (direct grants, toolset grants,
        # and team/agent/org ceilings), so REST listing cannot drift from it
        if user_api_key_auth:
            from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
                MCPRequestHandler,
            )

            allowed_tools_for_server = await MCPRequestHandler.get_allowed_tools_for_server(
                server_id=server.server_id,
                user_api_key_auth=user_api_key_auth,
            )
            if allowed_tools_for_server is not None:
                tools = [tool for tool in tools if _tool_name_matches(tool.name, allowed_tools_for_server)]

        return _create_tool_response_objects(tools, server)

    async def _resolve_allowed_mcp_servers_for_tool_call(
        user_api_key_dict: UserAPIKeyAuth,
        server_id: str,
    ) -> List[MCPServer]:
        """Resolve allowed MCP servers for the given user and validate server_id access."""
        auth_contexts = await build_effective_auth_contexts(user_api_key_dict)
        allowed_server_ids_set = set()
        for auth_context in auth_contexts:
            servers = await global_mcp_server_manager.get_allowed_mcp_servers(user_api_key_auth=auth_context)
            allowed_server_ids_set.update(servers)
        if server_id not in allowed_server_ids_set:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "access_denied",
                    "message": f"The key is not allowed to access server {server_id}",
                },
            )
        allowed_mcp_servers: List[MCPServer] = []
        for allowed_server_id in allowed_server_ids_set:
            server = global_mcp_server_manager.get_mcp_server_by_id(allowed_server_id)
            if server is not None:
                allowed_mcp_servers.append(server)

        from litellm.proxy._experimental.mcp_server.mcp_trust_scoring import (
            apply_trust_filter_to_allowed_mcp_servers,
            assert_requested_server_passes_trust_filter,
        )

        filtered_servers = list(
            await apply_trust_filter_to_allowed_mcp_servers(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
                allowed_mcp_servers
            )
        )
        assert_requested_server_passes_trust_filter(
            filtered_servers=filtered_servers,
            server_id=server_id,
        )

        return filtered_servers

    async def _list_tools_for_single_server(
        server_id: str,
        allowed_server_ids: List[str],
        rest_client_ip: Optional[str],
        mcp_server_auth_headers: dict,
        mcp_auth_header: Optional[str],
        raw_headers_from_request: dict,
        user_api_key_dict: UserAPIKeyAuth,
        apply_tool_filters: bool = True,
    ) -> dict:
        """Handle tool listing for a single server_id request."""
        # Resolve a server name to its UUID if needed
        _name_resolved = None
        if server_id not in allowed_server_ids:
            _name_resolved = global_mcp_server_manager.get_mcp_server_by_name(server_id)
            if _name_resolved is not None and _name_resolved.server_id in set(allowed_server_ids):
                server_id = _name_resolved.server_id

        if server_id not in allowed_server_ids:
            _server = global_mcp_server_manager.get_mcp_server_by_id(server_id) or _name_resolved
            if (
                _server is not None
                and rest_client_ip is not None
                and not global_mcp_server_manager._is_server_accessible_from_ip(_server, rest_client_ip)
            ):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "ip_filtering",
                        "message": (
                            f"MCP server '{server_id}' is not accessible from your IP address "
                            f"({rest_client_ip}). This server is restricted to internal "
                            "networks only. To make it externally accessible, set "
                            "'available_on_public_internet: true' in the server configuration."
                        ),
                    },
                )
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "access_denied",
                    "message": f"The key is not allowed to access server {server_id}",
                },
            )
        server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
        if server is None:
            return {
                "tools": [],
                "error": "server_not_found",
                "message": f"Server with id {server_id} not found",
            }

        server_auth_header = _get_server_auth_header(server, mcp_server_auth_headers, mcp_auth_header)
        user_oauth_extra_headers = await _get_user_oauth_extra_headers(server, user_api_key_dict)

        try:
            list_tools_result = await _get_tools_for_single_server(
                server,
                server_auth_header,
                raw_headers_from_request,
                user_api_key_dict,
                extra_headers=user_oauth_extra_headers,
                apply_tool_filters=apply_tool_filters,
            )
        except MCPUpstreamAuthError:
            # Surface the upstream 401/403 to the caller so it can emit the
            # matching status code and WWW-Authenticate challenge; that is what
            # lets standards-compliant MCP clients run the upstream OAuth flow.
            raise
        except MCPServerListError as e:
            fault = classify_list_exception(e)
            verbose_logger.info(f"Listing tools from {server.name} failed with a {fault.tag} fault")
            raise HTTPException(
                status_code=list_fault_http_status(fault),
                detail={
                    "error": fault.tag,
                    "message": f"Failed to list tools from server {get_server_prefix(server)}",
                },
            ) from e
        except Exception as e:
            verbose_logger.exception(f"Error getting tools from {server.name}: {e}")
            return {
                "tools": [],
                "error": "server_error",
                "message": f"Failed to get tools from server {server.name}: {str(e)}",
            }
        return {
            "tools": list_tools_result,
            "error": None,
            "message": "Successfully retrieved tools",
        }

    def _as_query_str(value: Any) -> Optional[str]:
        """Coerce an Optional[str] Query param to str|None, dropping unresolved FastAPI defaults."""
        return value if isinstance(value, str) else None

    async def _resolve_toolset_scope(
        toolset_name: Optional[str],
        user_api_key_dict: UserAPIKeyAuth,
    ) -> UserAPIKeyAuth:
        """Resolve ``toolset_name`` to its scoped ``UserAPIKeyAuth``, or return unchanged."""
        if not toolset_name:
            return user_api_key_dict

        from litellm.proxy.utils import get_prisma_client_or_throw

        prisma_client = get_prisma_client_or_throw("Database not available. Connect a database to your proxy")
        toolset = await global_mcp_server_manager.get_toolset_by_name_cached(prisma_client, toolset_name)
        if toolset is None:
            raise HTTPException(
                status_code=404,
                detail=f"Toolset '{toolset_name}' not found",
            )
        return await _apply_toolset_scope(user_api_key_dict, toolset.toolset_id)

    @router.get("/tools/list", dependencies=[Depends(user_api_key_auth)])
    async def list_tool_rest_api(
        request: Request,
        server_id: Optional[str] = Query(None, description="The server id to list tools for"),
        mcp_server_name: Optional[str] = Query(
            None, description="Filter tools to a single MCP server by name or alias"
        ),
        toolset_name: Optional[str] = Query(None, description="Filter tools to a single toolset by name"),
        include_disabled_tools: bool = Query(
            False,
            description=(
                "Admin only. Return the full server tool catalog without the "
                "allowed_tools filter or per-key tool permissions, so the MCP "
                "settings UI can configure the allowlist. Ignored for non-admins."
            ),
        ),
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> dict:
        """
        List all available tools with information about the server they belong to.

        Example response:
        {
            "tools": [
                {
                    "name": "create_zap",
                    "description": "Create a new zap",
                    "inputSchema": "tool_input_schema",
                    "mcp_info": {
                        "server_name": "zapier",
                        "logo_url": "https://www.zapier.com/logo.png",
                        "server_id": "a1b2c3d4-...",
                        "alias": "zapier_prod",
                    }
                }
            ],
            "error": null,
            "message": "Successfully retrieved tools"
        }
        """
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        try:
            mcp_server_name = _as_query_str(mcp_server_name)
            toolset_name = _as_query_str(toolset_name)

            # The full catalog (allowlist filter skipped) is admin-only so the
            # REST endpoint can't be used to enumerate deliberately-disabled tools.
            apply_tool_filters = not (
                include_disabled_tools and user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
            )

            user_api_key_dict = await _resolve_toolset_scope(toolset_name, user_api_key_dict)

            if server_id is None:
                server_id = mcp_server_name

            if (
                apply_tool_filters
                and server_id is None
                and toolset_name is None
                and getattr(
                    getattr(user_api_key_dict, "object_permission", None),
                    "mcp_tool_search_enabled",
                    False,
                )
            ):
                from litellm.proxy._experimental.mcp_server.tool_search import (
                    get_virtual_tool_definitions,
                )

                return {
                    "tools": get_virtual_tool_definitions(),
                    "error": None,
                    "message": "Successfully retrieved tools",
                }

            # Extract auth headers from request
            headers = request.headers
            raw_headers_from_request = dict(headers)
            mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(headers)
            mcp_server_auth_headers = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)

            auth_contexts = await build_effective_auth_contexts(user_api_key_dict)

            _rest_client_ip = IPAddressUtils.get_mcp_client_ip(request)

            allowed_server_ids_set = set()
            for auth_context in auth_contexts:
                servers = await global_mcp_server_manager.get_allowed_mcp_servers(
                    user_api_key_auth=auth_context,
                )
                allowed_server_ids_set.update(servers)

            (
                allowed_server_ids,
                _ip_blocked_count,
            ) = global_mcp_server_manager.filter_server_ids_by_ip_with_info(
                list(allowed_server_ids_set), _rest_client_ip
            )

            from litellm.proxy._experimental.mcp_server.mcp_trust_scoring import (
                apply_trust_filter_to_allowed_mcp_servers,
                get_mcp_trust_scoring_client,
            )

            trust_client = get_mcp_trust_scoring_client()
            if trust_client is not None and trust_client.enabled:
                allowed_mcp_server_objects = [
                    server
                    for allowed_server_id in allowed_server_ids
                    if (
                        server := global_mcp_server_manager.get_mcp_server_by_id(
                            allowed_server_id
                        )
                    )
                    is not None
                ]
                trusted_servers = await apply_trust_filter_to_allowed_mcp_servers(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
                    allowed_mcp_server_objects
                )
                allowed_server_ids = [server.server_id for server in trusted_servers]

            list_tools_result = []
            error_message = None

            # If server_id is specified, only query that specific server
            if server_id:
                return await _list_tools_for_single_server(
                    server_id=server_id,
                    allowed_server_ids=allowed_server_ids,
                    rest_client_ip=_rest_client_ip,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    mcp_auth_header=mcp_auth_header,
                    raw_headers_from_request=raw_headers_from_request,
                    user_api_key_dict=user_api_key_dict,
                    apply_tool_filters=apply_tool_filters,
                )
            else:
                if not allowed_server_ids:
                    if _ip_blocked_count > 0:
                        raise HTTPException(
                            status_code=403,
                            detail={
                                "error": "ip_filtering",
                                "message": (
                                    f"No MCP tools are available for your IP address ({_rest_client_ip}). "
                                    f"{_ip_blocked_count} server(s) are restricted to internal networks only. "
                                    "To make servers externally accessible, set "
                                    "'available_on_public_internet: true' in the server configuration."
                                ),
                            },
                        )
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "access_denied",
                            "message": "The key is not allowed to access any MCP servers.",
                        },
                    )

                # Pre-fetch OAuth credentials only when at least one allowed server uses OAuth2,
                # to avoid an unnecessary DB round-trip on requests with no OAuth2 MCP servers.
                prefetched_oauth_creds = (
                    await _prefetch_user_oauth_creds(user_api_key_dict)
                    if _get_oauth2_server_ids(allowed_server_ids)
                    else {}
                )

                # Query all servers the user has access to
                errors = []
                for allowed_server_id in allowed_server_ids:
                    server = global_mcp_server_manager.get_mcp_server_by_id(allowed_server_id)
                    if server is None:
                        continue

                    server_auth_header = _get_server_auth_header(server, mcp_server_auth_headers, mcp_auth_header)
                    user_oauth_extra_headers = await _get_user_oauth_extra_headers(
                        server,
                        user_api_key_dict,
                        prefetched_creds=prefetched_oauth_creds,
                    )

                    try:
                        tools_result = await _get_tools_for_single_server(
                            server,
                            server_auth_header,
                            raw_headers_from_request,
                            user_api_key_dict,
                            extra_headers=user_oauth_extra_headers,
                            apply_tool_filters=apply_tool_filters,
                        )
                        list_tools_result.extend(tools_result)
                    except Exception as e:
                        verbose_logger.exception(f"Error getting tools from {server.name}: {e}")
                        errors.append(
                            f"{get_server_prefix(server)}: {classify_list_exception(e).tag}"
                            if isinstance(e, (MCPServerListError, MCPUpstreamAuthError))
                            else f"{get_server_prefix(server)}: {str(e)}"
                        )
                        continue

                if errors and not list_tools_result:
                    error_message = "Failed to get tools from servers: " + "; ".join(errors)

            return {
                "tools": list_tools_result,
                "error": "partial_failure" if error_message else None,
                "message": (error_message if error_message else "Successfully retrieved tools"),
            }

        except MCPUpstreamAuthError as e:
            # Surface upstream pass-through 401/403 challenges to the client so
            # standards-compliant MCP clients can run the upstream OAuth flow.
            raise e.to_http_exception(
                base_url=get_request_base_url(request),
                request_path=request.scope.get("_original_path") or request.url.path,
            )
        except HTTPException as http_exc:
            if http_exc.status_code == status.HTTP_404_NOT_FOUND or server_id:
                # Single-server requests relay the truthful status (a 502/504 upstream fault must
                # not masquerade as a 200 empty-success body); only the multi-server aggregate
                # keeps the legacy error-dict response shape below.
                raise
            # Internal access/IP 403s keep the legacy error-dict response shape
            # so the existing contract stays intact.
            verbose_logger.exception("HTTPException in list_tool_rest_api: %s", str(http_exc))
            return {
                "tools": [],
                "error": "unexpected_error",
                "message": (f"An unexpected error occurred: {http_exc.detail}"),
            }
        except Exception as e:
            verbose_logger.exception("Unexpected error in list_tool_rest_api: %s", str(e))
            return {
                "tools": [],
                "error": "unexpected_error",
                "message": f"An unexpected error occurred: {str(e)}",
            }

    @router.post("/tools/call", dependencies=[Depends(user_api_key_auth)])
    async def call_tool_rest_api(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        REST API to call a specific MCP tool with the provided arguments
        """
        from fastapi import HTTPException

        from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )
        from litellm.proxy.proxy_server import (
            general_settings,
            proxy_config,
            proxy_logging_obj,
        )

        try:
            data = await request.json()

            tool_name = data.get("name")
            tool_arguments = data.get("arguments") or {}

            from litellm.proxy._experimental.mcp_server.tool_search import (
                MCP_TOOL_CALL_TOOL_NAME,
                MCP_TOOL_SEARCH_TOOL_NAME,
            )

            if tool_name in (MCP_TOOL_SEARCH_TOOL_NAME, MCP_TOOL_CALL_TOOL_NAME):
                return await _handle_virtual_mcp_tool(request, data, tool_name, user_api_key_dict)

            # Validate required parameters early
            server_id = data.get("server_id")
            if not server_id:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_parameter",
                        "message": "server_id is required in request body",
                    },
                )

            if not tool_name:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_parameter",
                        "message": "name is required in request body",
                    },
                )

            proxy_base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
            (
                data,
                logging_obj,
            ) = await proxy_base_llm_response_processor.common_processing_pre_call_logic(
                request=request,
                user_api_key_dict=user_api_key_dict,
                proxy_config=proxy_config,
                route_type=CallTypes.call_mcp_tool.value,
                proxy_logging_obj=proxy_logging_obj,
                general_settings=general_settings,
            )

            # Extract MCP auth headers from request and add to data dict
            (
                mcp_auth_header,
                mcp_server_auth_headers,
                raw_headers_from_request,
            ) = _extract_mcp_headers_from_request(request, MCPRequestHandler)
            if mcp_auth_header:
                data["mcp_auth_header"] = mcp_auth_header
            if mcp_server_auth_headers:
                data["mcp_server_auth_headers"] = mcp_server_auth_headers
            data["raw_headers"] = raw_headers_from_request

            # Extract user_api_key_auth from metadata and add to top level
            # call_mcp_tool expects user_api_key_auth as a top-level parameter
            if "metadata" in data and "user_api_key_auth" in data["metadata"]:
                data["user_api_key_auth"] = data["metadata"]["user_api_key_auth"]

            # Resolve allowed MCP servers with IP filtering
            (
                allowed_mcp_servers,
                canonical_server_id,
            ) = await _resolve_allowed_mcp_servers_with_ip_filter(request, user_api_key_dict, server_id)

            # Look up per-user OAuth headers for this server (mirrors list_tool_rest_api).
            user_oauth_extra_headers: Optional[Dict[str, str]] = None
            target_server = next(
                (s for s in allowed_mcp_servers if s.server_id == canonical_server_id),
                None,
            )
            if target_server is not None:
                user_oauth_extra_headers = await _get_user_oauth_extra_headers(target_server, user_api_key_dict)

            # Call execute_mcp_tool directly (permission checks already done)
            _tool_start_time = datetime.now()
            result = await execute_mcp_tool(
                name=tool_name,
                arguments=tool_arguments,
                allowed_mcp_servers=allowed_mcp_servers,
                start_time=_tool_start_time,
                user_api_key_auth=data.get("user_api_key_auth"),
                mcp_auth_header=data.get("mcp_auth_header"),
                mcp_server_auth_headers=data.get("mcp_server_auth_headers"),
                oauth2_headers=user_oauth_extra_headers or data.get("oauth2_headers"),
                raw_headers=data.get("raw_headers"),
                litellm_logging_obj=data.get("litellm_logging_obj"),
                requested_server_id=canonical_server_id,
            )
            await _safe_fire_mcp_tool_call_logging(
                logging_obj,
                result,
                _tool_start_time,
                datetime.now(),
                user_api_key_auth=user_api_key_dict,
                request_data=data,
            )
            return result
        except MCPMissingUserEnvVarsError as e:
            verbose_logger.info(
                "MCP tool call missing per-user env vars: server_id=%s missing=%s",
                e.server_id,
                e.missing,
            )
            raise HTTPException(
                status_code=412,
                detail={
                    "error": "missing_user_env_vars",
                    "message": str(e),
                    "server_id": e.server_id,
                    "server_name": e.server_name,
                    "missing": e.missing,
                    "setup_url": e.setup_url,
                },
            )
        except BlockedPiiEntityError as e:
            verbose_logger.error(f"BlockedPiiEntityError in MCP tool call: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "blocked_pii_entity",
                    "message": str(e),
                    "entity_type": getattr(e, "entity_type", None),
                    "guardrail_name": getattr(e, "guardrail_name", None),
                },
            )
        except GuardrailRaisedException as e:
            verbose_logger.error(f"GuardrailRaisedException in MCP tool call: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "guardrail_violation",
                    "message": str(e),
                    "guardrail_name": getattr(e, "guardrail_name", None),
                },
            )
        except MCPUpstreamAuthError as e:
            # A client-forwarded pass-through upstream 401 from either the direct or the virtual call
            # branch. Relay it as a 401 + WWW-Authenticate so the MCP client can re-run upstream OAuth,
            # and log at info: an expected caller-must-reauth signal, not an operator-actionable error.
            verbose_logger.info(f"MCP tool call relaying upstream HTTP {e.status_code}")
            raise _relay_upstream_auth_http_exception(e, request)
        except HTTPException as e:
            # Locally generated denials (tool/server permission, IP filtering, BYOK) stay at error level
            # so restriction probing keeps full monitoring visibility; the relayed upstream 401 above is
            # the only status demoted to info.
            verbose_logger.error(f"HTTPException in MCP tool call: {str(e)}")
            raise e
        except Exception as e:
            verbose_logger.exception(f"Unexpected error in MCP tool call: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_server_error",
                    "message": f"An unexpected error occurred: {str(e)}",
                },
            )

    ########################################################
    # MCP Connection testing routes
    # /health -> Test if we can connect to the MCP server
    # /health/tools/list -> List tools from MCP server
    # For these routes users will dynamically pass the MCP connection params, they don't need to be on the MCP registry
    ########################################################
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        NewMCPServerRequest,
        _inherit_credentials_from_existing_server,
    )

    def _extract_credentials(
        request: NewMCPServerRequest,
    ) -> tuple:
        """
        Extract OAuth credentials from the nested ``request.credentials`` dict.

        Returns:
            (client_id, client_secret, scopes) — any value may be ``None``.
        """
        creds = request.credentials if isinstance(request.credentials, dict) else {}
        client_id: Optional[str] = creds.get("client_id")
        client_secret: Optional[str] = creds.get("client_secret")
        scopes_raw = creds.get("scopes")
        scopes: Optional[List[str]] = scopes_raw if isinstance(scopes_raw, list) else None
        return client_id, client_secret, scopes

    async def _execute_with_mcp_client(
        request: NewMCPServerRequest,
        operation: Callable[..., Awaitable[Any]],
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Create a temporary MCP client from *request*, run *operation*, and return the result.

        For M2M OAuth servers (those with ``client_id``, ``client_secret``, and
        ``token_url``), the incoming ``oauth2_headers`` are dropped so that
        ``resolve_mcp_auth`` can auto-fetch a token via ``client_credentials``.

        Args:
            request: MCP server configuration submitted by the UI.
            operation: Async callable that receives the created client and returns a result dict.
            mcp_auth_header: Pre-resolved credential header (API-key / bearer token).
            oauth2_headers: Headers extracted from the incoming request (may contain the
                litellm API key — must NOT be forwarded for M2M servers).
            raw_headers: Raw request headers forwarded for stdio env construction.

        Returns:
            The dict returned by *operation*, or an error dict on failure.
        """
        try:
            client_id, client_secret, scopes = _extract_credentials(request)

            _oauth2_flow: Optional[Literal["client_credentials", "authorization_code"]] = request.oauth2_flow or (
                "client_credentials" if client_id and client_secret and request.token_url else None
            )
            # client_credentials requires token_url to fetch a token; without it the
            # incoming auth header would be dropped with nothing to replace it.
            if _oauth2_flow == "client_credentials" and not request.token_url:
                _oauth2_flow = None

            server_model = MCPServer(
                server_id=request.server_id or "",
                name=request.alias or request.server_name or "",
                url=request.url,
                transport=request.transport,
                auth_type=request.auth_type,
                mcp_info=request.mcp_info,
                command=request.command,
                args=request.args,
                env=request.env,
                static_headers=request.static_headers,
                client_id=client_id,
                client_secret=client_secret,
                issuer=request.issuer,
                token_url=request.token_url,
                scopes=scopes,
                authorization_url=request.authorization_url,
                registration_url=request.registration_url,
                oauth2_flow=_oauth2_flow,
                instructions=request.instructions,
            )

            stdio_env = global_mcp_server_manager._build_stdio_env(server_model, raw_headers)

            # For M2M OAuth servers, drop the incoming Authorization header so that
            # resolve_mcp_auth can auto-fetch a token via client_credentials.
            effective_oauth2_headers = None if server_model.has_client_credentials else oauth2_headers

            # Interactive authorization_code tools preview: the operator holds a just-authorized
            # token but it is not persisted yet. Resolve it through the v2 resolver via a one-shot
            # presented store - the same path runtime uses for the stored token - rather than the
            # caller-override path _create_mcp_client refuses for authorization_code. The bare token
            # becomes the upstream credential, so it is not also forwarded as a caller header. Gated
            # to the v2-mapped oauth2 case (to_server_spec non-None); M2M (client_credentials),
            # delegate/passthrough, and token-exchange are unaffected.
            from litellm.proxy._experimental.mcp_server.outbound_credentials import (  # noqa: PLC0415
                UpstreamCredentialProvider,
            )
            from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415
                to_server_spec,
            )
            from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (  # noqa: PLC0415
                OAuthToken,
            )
            from litellm.proxy._experimental.mcp_server.outbound_credentials.presented_token_store import (  # noqa: PLC0415
                PresentedOAuthTokenStore,
            )

            forwarded_authorization = (
                effective_oauth2_headers.get("Authorization") if effective_oauth2_headers else None
            )
            preview_cred_provider = (
                UpstreamCredentialProvider(
                    oauth_token_store=PresentedOAuthTokenStore(
                        OAuthToken(
                            access_token=forwarded_authorization[7:]
                            if forwarded_authorization[:7].lower() == "bearer "
                            else forwarded_authorization
                        )
                    )
                )
                if (
                    server_model.auth_type == MCPAuth.oauth2
                    and forwarded_authorization is not None
                    and to_server_spec(server_model) is not None
                )
                else None
            )

            merged_headers = merge_mcp_headers(
                extra_headers=(None if preview_cred_provider else effective_oauth2_headers),
                static_headers=request.static_headers,
            )

            client = await global_mcp_server_manager._create_mcp_client(
                server=server_model,
                mcp_auth_header=mcp_auth_header,
                extra_headers=merged_headers,
                stdio_env=stdio_env,
                cred_provider=preview_cred_provider,
            )

            return await operation(client)

        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except BaseException as e:
            verbose_logger.error("Error in MCP operation: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": True,
                "message": _connection_error_message(e),
            }

    async def _preview_openapi_tools(spec_path: str) -> dict:
        """Generate tool previews from an OpenAPI spec without creating a server."""
        from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
            _OPENAPI_TOOL_NAME_MAX_LEN,
            build_input_schema,
            load_openapi_spec_async,
            resolve_operation_params,
            sanitize_openapi_tool_name,
        )

        try:
            spec = await load_openapi_spec_async(spec_path)
            paths = spec.get("paths", {})
            components = spec.get("components", {})
            tools: List[dict] = []
            used_names: set = set()
            for path, path_item in paths.items():
                for method in ("get", "post", "put", "delete", "patch"):
                    operation = path_item.get(method)
                    if operation is None:
                        continue

                    resolved_op = resolve_operation_params(operation, path_item, components)

                    raw_op_id = operation.get("operationId", f"{method}_{path}")
                    # Match what register_tools_from_openapi does so the preview
                    # the user sees in the dashboard equals the names that get
                    # registered (and shipped to LLM providers, which enforce
                    # ^[a-zA-Z0-9_-]+$). See sanitize_openapi_tool_name docstring.
                    op_id = sanitize_openapi_tool_name(raw_op_id)

                    unique = op_id
                    n = 1
                    while unique in used_names:
                        n += 1
                        suffix = f"_{n}"
                        unique = op_id[: _OPENAPI_TOOL_NAME_MAX_LEN - len(suffix)] + suffix
                    op_id = unique
                    used_names.add(op_id)
                    summary = operation.get("summary", "")
                    description = operation.get("description", summary)
                    input_schema = build_input_schema(resolved_op)
                    tools.append(
                        {
                            "name": op_id,
                            "description": description or summary or f"{method.upper()} {path}",
                            "inputSchema": input_schema,
                        }
                    )
            return {
                "tools": tools,
                "error": None,
                "message": f"Found {len(tools)} tools from OpenAPI spec",
            }
        except Exception as e:
            verbose_logger.error("Error previewing OpenAPI tools: %s", e, exc_info=True)
            return {
                "tools": [],
                "error": True,
                "message": f"Failed to load OpenAPI spec: {e}",
            }

    @router.post("/test/connection", dependencies=[Depends(user_api_key_auth)])
    async def test_connection(
        request: Request,
        new_mcp_server_request: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Test if we can connect to the provided MCP server before adding it
        """
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "User does not have permission to test MCP server connections. Only PROXY_ADMIN users can perform this action."
                },
            )

        async def _test_connection_operation(client):
            async def _noop(session):
                return "ok"

            await client.run_with_session(_noop)
            return {"status": "ok"}

        return await _execute_with_mcp_client(
            new_mcp_server_request,
            _test_connection_operation,
            raw_headers=_safe_get_request_headers(request),
        )

    @router.post("/test/tools/list", dependencies=[Depends(user_api_key_auth)])
    async def test_tools_list(
        request: Request,
        new_mcp_server_request: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Preview tools available from MCP server before adding it
        """
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "User does not have permission to test MCP server tools. Only PROXY_ADMIN users can perform this action."
                },
            )

        new_mcp_server_request = _inherit_credentials_from_existing_server(new_mcp_server_request)

        # For OpenAPI spec servers, generate tools from the spec directly
        if new_mcp_server_request.spec_path:
            return await _preview_openapi_tools(new_mcp_server_request.spec_path)

        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        headers = request.headers

        mcp_auth_header: Optional[str] = None
        if new_mcp_server_request.auth_type in {
            MCPAuth.api_key,
            MCPAuth.bearer_token,
            MCPAuth.basic,
            MCPAuth.authorization,
        }:
            credentials = getattr(new_mcp_server_request, "credentials", None)
            if isinstance(credentials, dict):
                mcp_auth_header = credentials.get("auth_value")

        # Authorization doubles as the admission fallback (LITELLM_API_KEY_HEADER_NAME_SECONDARY):
        # when the primary x-litellm-api-key header is absent, the Authorization value is the
        # caller's LiteLLM key, not an upstream token, and must never be forwarded upstream.
        oauth2_headers: Optional[Dict[str, str]] = None
        if new_mcp_server_request.auth_type in _UPSTREAM_OAUTH_DISCOVERY_AUTH_TYPES and headers.get(
            MCPRequestHandler.LITELLM_API_KEY_HEADER_NAME_PRIMARY
        ):
            oauth2_headers = MCPRequestHandler._get_oauth2_headers_from_headers(headers)

        async def _list_tools_operation(client):
            async def _list_tools_session_operation(session):
                return await session.list_tools()

            list_tools_response = await client.run_with_session(_list_tools_session_operation)
            list_tools_result: List[MCPTool] = list_tools_response.tools
            model_dumped_tools: List[dict] = [tool.model_dump() for tool in list_tools_result]
            return {
                "tools": model_dumped_tools,
                "error": None,
                "message": "Successfully retrieved tools",
            }

        return await _execute_with_mcp_client(
            new_mcp_server_request,
            _list_tools_operation,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=_safe_get_request_headers(request),
        )
