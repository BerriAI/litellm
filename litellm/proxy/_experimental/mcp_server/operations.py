"""Shared MCP operation policy and dispatch."""

import asyncio
import traceback
import types
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final, NoReturn, TypeAlias, overload

from fastapi import HTTPException
from mcp import ReadResourceResult, Resource
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    GetPromptRequest,
    GetPromptRequestParams,
    GetPromptResult,
    InputRequiredResult,
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListResourceTemplatesRequest,
    ListResourceTemplatesResult,
    ListToolsRequest,
    ListToolsResult,
    PaginatedRequestParams,
    Prompt,
    ReadResourceRequest,
    ReadResourceRequestParams,
    ResourceTemplate,
    TextContent,
)
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl, ConfigDict, Field, TypeAdapter
from typing_extensions import ReadOnly, TypedDict, assert_never

from litellm._logging import verbose_logger
from litellm.constants import (
    MAXIMUM_TRACEBACK_LINES_TO_LOG,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.byok_credential_cache import (
    byok_credential_cache,
    byok_credential_cache_key,
    cache_byok_credential,
    get_cached_byok_credential,
)
from litellm.proxy._experimental.mcp_server.contracts import (
    AuthorizedToolCall,
    OperationContext,
    ProgressCallback,
)
from litellm.proxy._experimental.mcp_server.db import OAuthCredentialPayload
from litellm.proxy._experimental.mcp_server.exceptions import (
    MCPToolResultError,
    MCPUpstreamAuthError,
)
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
    SERVER_OUTCOMES_META_KEY,
    AggregateToolListing,
    ServerListOk,
    ServerOutcome,
    classify_list_exception,
    outcome_wire_value,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _caller_authorization_fans_out,
    _client_forwarded_authorization_headers,
    _resolve_openapi_tool_auth,
    _should_strip_caller_authorization,
    global_mcp_server_manager,
)
from litellm.proxy._experimental.mcp_server.oauth_utils import (
    _redact_mcp_resource_url,
    get_byok_www_authenticate,
)
from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
    _request_auth_header,
    _request_extra_headers,
    _request_resolved_auth_headers,
)
from litellm.proxy._experimental.mcp_server.result_conversion import (
    WireCompat,
    complete_call_tool_result,
    handler_outcome,
    to_call_tool_result,
)
from litellm.proxy._experimental.mcp_server.tool_registry import (
    global_mcp_tool_registry,
)
from litellm.proxy._experimental.mcp_server.utils import (
    MCP_TOOL_PREFIX_SEPARATOR,
    MCPMissingUserEnvVarsError,
    add_server_prefix_to_name,
    build_synthetic_mcp_request,
    extract_mcp_tool_result_error_message,
    get_server_prefix,
    is_tool_name_prefixed,
    iter_known_server_prefixes,
    logging_safe_mcp_headers,
    match_known_tool_name,
    normalize_server_name,
    split_server_prefix_from_name,
    strip_known_server_prefix,
)
from litellm.proxy._types import (
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    publish_auth_cache_invalidation,
)
from litellm.proxy.litellm_pre_call_utils import (
    LiteLLMProxyRequestSetup,
    get_chain_id_from_headers,
)
from litellm.types.mcp import (
    DEFAULT_CREDENTIAL_HEADER,
    MCPAuth,
    without_header,
)
from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPServer
from litellm.types.utils import CallTypes, StandardLoggingMCPToolCall
from litellm.utils import Rules, client, function_setup

__all__ = (
    "_MCP_CREDENTIAL_REQUEST_FIELDS",
    "ListMCPToolsRestAPIResponseObject",
    "MCPInfo",
    "MCPServer",
    "_McpDeniedDetail",
    "_aggregate_server_key",
    "_build_virtual_call_logging_obj",
    "_check_byok_credential",
    "_client_has_passthrough_authorization",
    "_client_has_per_server_auth_header",
    "_dispatch_virtual_mcp_tool",
    "_fire_mcp_tool_call_logging",
    "_get_allowed_mcp_servers",
    "_get_allowed_mcp_servers_from_mcp_server_names",
    "_get_byok_credential",
    "_get_prompts_from_mcp_servers",
    "_get_resource_templates_from_mcp_servers",
    "_get_resources_from_mcp_servers",
    "_get_standard_logging_mcp_tool_call",
    "_get_tools_from_mcp_servers",
    "_get_user_oauth_extra_headers_from_db",
    "_handle_local_mcp_tool",
    "_handle_managed_mcp_tool",
    "_http_detail_message",
    "_invalidate_byok_cred_cache",
    "_list_mcp_prompts",
    "_list_mcp_resource_templates",
    "_list_mcp_resources",
    "_list_mcp_tools",
    "_list_tools_before_first_call",
    "_mcp_session_id_from_headers",
    "_merge_gateway_initialize_instructions",
    "_prefetch_oauth_creds_for_user",
    "_prepare_mcp_server_headers",
    "_raise_if_initialize_grants_no_mcp_servers",
    "_resolve_display_name_to_original",
    "_run_post_mcp_call_guardrails",
    "_server_answers_to",
    "_tool_name_matches",
    "apply_tool_overrides",
    "call_mcp_tool",
    "execute_mcp_tool",
    "filter_tools_by_allowed_tools",
    "filter_tools_by_key_team_permissions",
    "fire_mcp_tool_call_failure_logging",
    "mcp_get_prompt",
    "mcp_read_resource",
    "raise_denied_scoped_mcp_access",
)


async def _invalidate_byok_cred_cache(user_id: str, server_id: str) -> None:
    """Drop a stored-or-deleted BYOK credential from this worker's cache and from every peer worker's."""
    cache_key: Final = byok_credential_cache_key(user_id, server_id)
    byok_credential_cache.delete_cache(cache_key)
    await publish_auth_cache_invalidation(cache_key=cache_key)


def _mcp_session_id_from_headers(
    raw_headers: dict[str, str] | None,
) -> str | None:
    """The ``mcp-session-id`` of a stateful MCP session, read case-insensitively
    from the request headers. ``None`` for stateless calls (no such header)."""
    if not raw_headers:
        return None
    for key, value in raw_headers.items():
        if isinstance(key, str) and key.lower() == "mcp-session-id":
            return value or None
    return None


class ListMCPToolsRestAPIResponseObject(MCPTool):
    """
    Object returned by the /tools/list REST API route.
    """

    mcp_info: MCPInfo | None = Field(default=None, alias="mcp_info")
    model_config = ConfigDict(arbitrary_types_allowed=True)


async def _build_virtual_call_logging_obj(
    name: str,
    arguments: dict[str, object],
    user_api_key_auth: UserAPIKeyAuth,
    raw_headers: Mapping[str, str] | None = None,
    client_ip: str | None = None,
) -> LiteLLMLoggingObj | None:
    """Run the pre-call pipeline (guardrails + logging setup) for a virtual
    mcp_tool_call so the SSE path spend-logs like the REST path."""
    from litellm.proxy.common_request_processing import (
        ProxyBaseLLMRequestProcessing,
    )
    from litellm.proxy.proxy_server import (
        general_settings,
        proxy_config,
        proxy_logging_obj,
    )

    request: Final = build_synthetic_mcp_request(
        path="/mcp/tools/call",
        raw_headers=raw_headers,
        client_ip=client_ip,
    )
    _, virtual_logging_obj = await ProxyBaseLLMRequestProcessing(
        data={"name": name, "arguments": arguments}
    ).common_processing_pre_call_logic(
        request=request,
        user_api_key_dict=user_api_key_auth,
        proxy_config=proxy_config,
        route_type=CallTypes.call_mcp_tool.value,
        proxy_logging_obj=proxy_logging_obj,
        general_settings=general_settings,
    )
    return virtual_logging_obj


async def _dispatch_virtual_mcp_tool(
    name: str,
    arguments: dict[str, object] | None,
    user_api_key_auth: UserAPIKeyAuth | None,
    client_ip: str | None,
    mcp_servers: list[str] | None = None,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    mcp_proxy_mode: bool = False,
) -> CallToolResult | None:
    """Handle the mcp_tool_search / mcp_tool_call virtual tools.

    Returns a CallToolResult when ``name`` is a virtual tool, else ``None`` so
    the caller falls through to normal tool routing.
    """
    from litellm.llms.litellm_proxy.skills.skill_search import DEFAULT_SKILL_SEARCH_TOP_K
    from litellm.proxy._experimental.mcp_server.tool_search import (
        AGENT_SEARCH_TOOL_NAME,
        DEFAULT_AGENT_SEARCH_TOP_K,
        MCP_PROXY_CALL_TOOL_NAME,
        MCP_PROXY_TOOL_NAMES,
        MCP_TOOL_SEARCH_TOOL_NAME,
        SKILL_SEARCH_TOOL_NAME,
        VIRTUAL_TOOL_NAMES,
        coerce_top_k,
        handle_agent_search,
        handle_mcp_proxy_tool,
        handle_mcp_tool_call,
        handle_mcp_tool_search,
        handle_skill_search,
    )

    if mcp_proxy_mode and name not in MCP_PROXY_TOOL_NAMES:
        return CallToolResult(
            content=[  # mutable-ok: MCP result content
                TextContent(type="text", text=f"Tool {name} is unavailable on /mcp/proxy")
            ],
            is_error=True,
        )

    if mcp_proxy_mode and name in MCP_PROXY_TOOL_NAMES:
        assert user_api_key_auth is not None
        proxy_call_start: Final = datetime.now()  # noqa: DTZ005  # logging pipeline uses naive datetimes
        proxy_logging_obj: Final = (
            await _build_virtual_call_logging_obj(
                name=name,
                arguments=arguments or {},  # mutable-ok: logging pipeline payload
                user_api_key_auth=user_api_key_auth,
                raw_headers=raw_headers,
                client_ip=client_ip,
            )
            if name == MCP_PROXY_CALL_TOOL_NAME
            else None
        )
        try:
            proxy_result: Final = await handle_mcp_proxy_tool(
                name=name,
                arguments=arguments or {},  # mutable-ok: proxy handler payload
                user_api_key_dict=user_api_key_auth,
                client_ip=client_ip,
                mcp_servers=mcp_servers,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                litellm_logging_obj=proxy_logging_obj,
            )
        except Exception as exc:
            if proxy_logging_obj is not None:
                from litellm.proxy.proxy_server import proxy_logging_obj as request_logging_obj

                failure_end: Final = datetime.now()  # noqa: DTZ005  # matches the logging pipeline start time
                failure_traceback: Final = traceback.format_exc(limit=MAXIMUM_TRACEBACK_LINES_TO_LOG)
                try:
                    proxy_logging_obj.failure_handler(exc, failure_traceback, proxy_call_start, failure_end)
                    await proxy_logging_obj.async_failure_handler(exc, failure_traceback, proxy_call_start, failure_end)
                    if not isinstance(exc, MCPUpstreamAuthError):
                        await request_logging_obj.post_call_failure_hook(
                            request_data={  # mutable-ok: failure hook mutates its request payload
                                "name": name,
                                "arguments": arguments,
                                "litellm_logging_obj": proxy_logging_obj,
                            },
                            original_exception=exc,
                            user_api_key_dict=user_api_key_auth,
                            route="/mcp/call_tool",
                            traceback_str=failure_traceback,
                        )
                except Exception:  # noqa: BLE001  # a failing failure hook must not mask the tool call's own error
                    verbose_logger.exception("Error logging failed MCP proxy tool call")
            raise
        if proxy_logging_obj is not None:
            return await _fire_mcp_tool_call_logging(
                logging_obj=proxy_logging_obj,
                result=proxy_result,
                start_time=proxy_call_start,
                end_time=datetime.now(),  # noqa: DTZ005  # matches the logging pipeline start time
                user_api_key_auth=user_api_key_auth,
                request_data=types.MappingProxyType({"name": name, "arguments": arguments}),
            )
        return proxy_result

    if name not in VIRTUAL_TOOL_NAMES:
        return None

    if not getattr(
        getattr(user_api_key_auth, "object_permission", None),
        "mcp_tool_search_enabled",
        False,
    ):
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Tool {name} requires mcp_tool_search_enabled on the key",
                )
            ],
            is_error=True,
        )

    args: Final = arguments or {}
    if name == MCP_TOOL_SEARCH_TOOL_NAME:
        return await handle_mcp_tool_search(
            query=TypeAdapter(str).validate_python(args.get("query", "")),
            top_k=coerce_top_k(args.get("top_k", 5)),
            user_api_key_dict=user_api_key_auth,
            client_ip=client_ip,
            mcp_servers=mcp_servers,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
        )

    assert user_api_key_auth is not None  # guaranteed by the flag check above
    if name == AGENT_SEARCH_TOOL_NAME:
        return await handle_agent_search(
            query=str(args.get("query", "")),
            top_k=coerce_top_k(args.get("top_k", DEFAULT_AGENT_SEARCH_TOP_K), default=DEFAULT_AGENT_SEARCH_TOP_K),
            user_api_key_dict=user_api_key_auth,
        )
    if name == SKILL_SEARCH_TOOL_NAME:
        return await handle_skill_search(
            query=str(args.get("query", "")),
            top_k=coerce_top_k(args.get("top_k", DEFAULT_SKILL_SEARCH_TOP_K), default=DEFAULT_SKILL_SEARCH_TOP_K),
            user_api_key_dict=user_api_key_auth,
        )
    virtual_logging_obj: Final = await _build_virtual_call_logging_obj(
        name=name,
        arguments=args,
        user_api_key_auth=user_api_key_auth,
        raw_headers=raw_headers,
        client_ip=client_ip,
    )
    tool_request: Final = CallToolRequestParams.model_validate(
        types.MappingProxyType({"name": args.get("tool_name", ""), "arguments": args.get("arguments") or {}})
    )
    return await handle_mcp_tool_call(
        tool_name=tool_request.name,
        arguments=tool_request.arguments or {},
        user_api_key_dict=user_api_key_auth,
        client_ip=client_ip,
        mcp_servers=mcp_servers,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        litellm_logging_obj=virtual_logging_obj,
    )


async def _get_allowed_mcp_servers_from_mcp_server_names(
    mcp_servers: Sequence[str] | None,
    allowed_mcp_servers: list[MCPServer],
) -> list[MCPServer]:
    """
    Get the filtered MCP servers from the MCP server names.

    Fails closed when ``mcp_servers`` is explicitly provided (path- or
    header-derived) but none of the names resolve to a server alias or
    access group the caller can access. The previous behavior returned
    the full ``allowed_mcp_servers`` set, which silently widened scope
    when a client targeted ``/mcp/<unknown>/`` and made URL/header
    namespacing appear to work when it did not.
    """

    filtered_server: Final[dict[str, MCPServer]] = {}
    # Filter servers based on mcp_servers parameter if provided
    if mcp_servers is not None:
        for server_or_group in mcp_servers:
            server_name_matched = False

            for server in allowed_mcp_servers:
                if server and _server_answers_to(server, server_or_group):
                    filtered_server[server.server_id] = server
                    server_name_matched = True
                    break

            if not server_name_matched:
                try:
                    access_group_server_ids = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                        [server_or_group]
                    )
                    # Only include servers that the user has access to
                    for server_id in access_group_server_ids:
                        for server in allowed_mcp_servers:
                            if server_id == server.server_id:
                                filtered_server[server.server_id] = server
                except Exception as e:
                    verbose_logger.debug("Could not resolve '%s' as access group: %s", server_or_group, e)

    if filtered_server:
        return list(filtered_server.values())

    if mcp_servers is not None:
        # Caller asked for a specific scope but nothing resolved. Fail
        # closed so URL/header namespacing cannot silently fall back to
        # the caller's full allowed-server set.
        verbose_logger.debug(
            "MCP scope filter resolved to no servers for requested names %s; returning empty list (fail-closed).",
            mcp_servers,
        )
        return []

    return allowed_mcp_servers


def _http_detail_message(detail: object) -> str:
    return str(detail.get("error")) if isinstance(detail, dict) and detail.get("error") else str(detail)


def _server_answers_to(server: MCPServer, name: str) -> bool:
    requested: Final = name.lower()
    return any(requested == known.lower() for known in iter_known_server_prefixes(server) if known)


async def raise_denied_scoped_mcp_access(
    requested_names: Sequence[str],
    user_api_key_auth: UserAPIKeyAuth | None,
    client_ip: str | None = None,
) -> None:
    """A scoped request (``/mcp/<name>`` path or ``x-mcp-servers`` header) resolved to zero
    allowed servers, so the denial must be loud: a silent 200 with no tools reads as a healthy
    server with no tools. Unknown, unauthorized, and access-group names all share one generic
    error so scoping cannot probe which servers exist; the agent variant fires only when the
    same request resolves once the agent binding is stripped, proving the binding caused the veto."""
    agent_id: Final = user_api_key_auth.agent_id if user_api_key_auth else None
    if user_api_key_auth is not None and agent_id:
        resolved_without_agent: Final = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth.model_copy(update=types.MappingProxyType({"agent_id": None})),
            mcp_servers=requested_names,
            client_ip=client_ip,
        )

        def _resolved_to_server(name: str) -> bool:
            return any(_server_answers_to(server, name) for server in resolved_without_agent)

        vetoed_server: Final = next((name for name in requested_names if _resolved_to_server(name)), None)
        if vetoed_server is not None:
            agent_denial: Final[_McpDeniedDetail] = {
                "error": (
                    f"MCP server '{vetoed_server}' is not available to this key: the key is bound to "
                    f"agent '{agent_id}', whose MCP grants do not include this server. Add the server "
                    f"to the agent's object_permission.mcp_servers (edit the agent in the Admin UI or "
                    f"PATCH /v1/agents/{agent_id}), or use a key that is not bound to the agent."
                )
            }
            raise HTTPException(status_code=403, detail=agent_denial)
        vetoed_group: Final = next(
            (
                name
                for name in requested_names
                if not _resolved_to_server(name)
                and any(name in (server.access_groups or ()) for server in resolved_without_agent)
            ),
            None,
        )
        if vetoed_group is not None:
            group_denial: Final[_McpDeniedDetail] = {
                "error": (
                    f"MCP access group '{vetoed_group}' is not available to this key: the key is bound to "
                    f"agent '{agent_id}', whose MCP grants do not include it. Add the group to the "
                    f"agent's object_permission.mcp_access_groups (edit the agent in the Admin UI or "
                    f"PATCH /v1/agents/{agent_id}), or use a key that is not bound to the agent."
                )
            }
            raise HTTPException(status_code=403, detail=group_denial)
    generic_denial: Final[_McpDeniedDetail] = {
        "error": f"The key is not allowed to access the requested MCP servers: {', '.join(requested_names)}"
    }
    raise HTTPException(status_code=403, detail=generic_denial)


def _tool_name_matches(tool_name: str, filter_list: list[str], mcp_server: MCPServer) -> bool:
    """
    Check if a tool name matches any name in the filter list.

    Reads the same owner the server-level permission checks use, so discovery hides
    exactly what dispatch refuses. ``mcp_server`` is required: guessing the boundary
    at the first separator mismatches every tool on a server whose prefix contains
    the separator.
    """
    bare_name: Final = strip_known_server_prefix(tool_name, mcp_server)
    return match_known_tool_name(bare_name, mcp_server, filter_list) is not None


def filter_tools_by_allowed_tools(
    tools: list[MCPTool],
    mcp_server: MCPServer,
) -> list[MCPTool]:
    """
    Filter tools by allowed/disallowed tools configuration.

    If allowed_tools is set, only tools in that list are returned.
    If disallowed_tools is set, tools in that list are excluded.
    Tool names are matched with and without server prefixes for flexibility.

    Args:
        tools: List of tools to filter
        mcp_server: Server configuration with allowed_tools/disallowed_tools

    Returns:
        Filtered list of tools
    """
    from litellm.proxy._experimental.mcp_server.utils import (
        server_applies_tool_allowlist,
    )

    tools_to_return = tools

    # Filter by allowed_tools (whitelist)
    if server_applies_tool_allowlist(mcp_server):
        if not mcp_server.allowed_tools:
            return []
        tools_to_return = [
            tool for tool in tools if _tool_name_matches(tool.name, mcp_server.allowed_tools, mcp_server)
        ]

    # Filter by disallowed_tools (blacklist)
    if mcp_server.disallowed_tools:
        tools_to_return = [
            tool
            for tool in tools_to_return
            if not _tool_name_matches(tool.name, mcp_server.disallowed_tools, mcp_server)
        ]

    return tools_to_return


def apply_tool_overrides(
    tools: list[MCPTool],
    mcp_server: MCPServer,
) -> list[MCPTool]:
    """Apply admin-configured display name/description overrides to tools.

    Overrides are keyed by the unprefixed tool name, same convention as
    allowed_tools configuration.
    """
    display_name_map: Final = mcp_server.tool_name_to_display_name or {}
    description_map: Final = mcp_server.tool_name_to_description or {}
    if not display_name_map and not description_map:
        return tools

    for tool in tools:
        unprefixed = strip_known_server_prefix(tool.name, mcp_server)
        lookup_key = unprefixed or tool.name
        if lookup_key in display_name_map:
            tool.name = display_name_map[lookup_key]
        if lookup_key in description_map:
            tool.description = description_map[lookup_key]
    return tools


async def _get_allowed_mcp_servers(
    user_api_key_auth: UserAPIKeyAuth | None,
    mcp_servers: Sequence[str] | None,
    client_ip: str | None = None,
) -> list[MCPServer]:
    """Return allowed MCP servers for a request after applying filters.

    Args:
        user_api_key_auth: The authenticated user's API key info.
        mcp_servers: Optional list of server names to filter to.
        client_ip: Client IP for IP-based access control. If None, falls back to
                  auth context. Pass explicitly from request handlers for safety.
    Note: If client_ip is None and auth context is not set, IP filtering is skipped.
          This is intentional for internal callers but may indicate a bug if called
          from a request handler without proper context setup.
    """
    allowed_mcp_server_ids = await global_mcp_server_manager.get_allowed_mcp_servers(user_api_key_auth)
    (
        allowed_mcp_server_ids,
        _ip_blocked,
    ) = global_mcp_server_manager.filter_server_ids_by_ip_with_info(allowed_mcp_server_ids, client_ip)
    verbose_logger.debug(
        "MCP IP filter: client_ip=%s, allowed_server_ids=%s",
        client_ip,
        allowed_mcp_server_ids,
    )
    if _ip_blocked > 0:
        verbose_logger.debug(
            "MCP IP filtering: %d server(s) are not accessible from client IP %s "
            "because they are restricted to internal networks. "
            "No tools from those servers will be returned. "
            "To expose a server externally, set 'available_on_public_internet: true' "
            "in its configuration.",
            _ip_blocked,
            client_ip,
        )
    allowed_mcp_servers: list[MCPServer] = []
    for allowed_mcp_server_id in allowed_mcp_server_ids:
        mcp_server = global_mcp_server_manager.get_mcp_server_by_id(allowed_mcp_server_id)
        if mcp_server is not None:
            # Apply the request-time oauth2_flow backstop for legacy null rows.
            mcp_server = MCPServerManager.resolve_oauth2_flow_for_request(mcp_server)
            allowed_mcp_servers.append(mcp_server)

    if mcp_servers is not None:
        allowed_mcp_servers = await _get_allowed_mcp_servers_from_mcp_server_names(
            mcp_servers=mcp_servers,
            allowed_mcp_servers=allowed_mcp_servers,
        )

    return allowed_mcp_servers


def _client_has_per_server_auth_header(
    server: MCPServer,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None,
) -> bool:
    """True if the request carries a per-server ``x-mcp-{alias}-authorization``
    header for this server. This is the multi-server binding: it names one
    upstream, so it is unambiguously the caller's upstream token regardless of
    auth mode (never the LiteLLM admission credential).

    Resolves through the same ``lookup_mcp_server_auth_in_headers`` egress uses, so
    the connect gate and egress agree on which per-server header names match: a
    dashboard client sends ``x-mcp-{sanitize_mcp_alias_for_header(alias)}-authorization``,
    and matching only the raw alias here would 401 a token egress would forward.
    """
    if not mcp_server_auth_headers:
        return False
    from litellm.proxy._experimental.mcp_server.utils import (
        lookup_mcp_server_auth_in_headers,
    )

    server_headers: Final = lookup_mcp_server_auth_in_headers(
        mcp_server_auth_headers,
        alias=server.alias,
        server_name=server.server_name,
        access_groups=server.access_groups,
    )
    if isinstance(server_headers, str):
        return bool(server_headers.strip())
    if isinstance(server_headers, dict):
        return any(isinstance(hk, str) and hk.lower() == "authorization" for hk in server_headers)
    return False


def _client_has_passthrough_authorization(
    server: MCPServer,
    oauth2_headers: dict[str, str] | None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None,
) -> bool:
    """True if the incoming request already carries an ``Authorization``
    header the gateway will forward to this pass-through server.

    The client may supply the bearer as either the top-level
    ``Authorization`` header (surfaced via ``oauth2_headers``) or a
    per-server ``x-mcp-auth-<alias>`` style header (surfaced via
    ``mcp_server_auth_headers``). Either form skips the pre-emptive 401.
    """
    if oauth2_headers:
        for k in oauth2_headers:
            if k.lower() == "authorization":
                return True
    return _client_has_per_server_auth_header(server, mcp_server_auth_headers)


async def _get_user_oauth_extra_headers_from_db(
    server: MCPServer,
    user_api_key_auth: UserAPIKeyAuth | None,
    prefetched_creds: 'Mapping[str, "OAuthCredentialPayload"] | None' = None,
) -> dict[str, str] | None:
    """Stored OAuth2 token for (user, server) as an ``Authorization: Bearer`` header, or None.

    Thin wrapper over ``resolve_user_oauth_access_token`` (Redis cache, else DB + refresh);
    ``prefetched_creds`` skips the per-server Redis/DB lookups for the batch path.
    """
    if server.auth_type != MCPAuth.oauth2 or user_api_key_auth is None:
        return None
    from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
        resolve_user_oauth_access_token,
    )

    token: Final = await resolve_user_oauth_access_token(
        getattr(user_api_key_auth, "user_id", None), server, prefetched_creds
    )
    return {"Authorization": f"Bearer {token}"} if token else None


async def _prefetch_oauth_creds_for_user(
    user_api_key_auth: UserAPIKeyAuth | None,
) -> dict[str, "OAuthCredentialPayload"]:
    """Fetch all OAuth2 credentials for the user in one DB query.

    Returns a dict keyed by server_id to avoid N+1 queries in asyncio.gather loops.
    """
    user_id: Final[str | None] = getattr(user_api_key_auth, "user_id", None) if user_api_key_auth else None
    if not user_id:
        return {}
    try:
        from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
            list_user_oauth_credentials,
        )
        from litellm.proxy.utils import get_prisma_client_or_throw  # noqa: PLC0415

        prisma_client: Final = get_prisma_client_or_throw(
            "Database not connected. Connect a database to use OAuth2 MCP tools."
        )
        creds: Final = await list_user_oauth_credentials(prisma_client, user_id)
        return {c["server_id"]: c for c in creds if "server_id" in c}
    except Exception:
        verbose_logger.warning("_prefetch_oauth_creds_for_user: failed to prefetch OAuth credentials")
        return {}


def _prepare_mcp_server_headers(
    server: MCPServer,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None,
    mcp_auth_header: str | None,
    oauth2_headers: dict[str, str] | None,
    raw_headers: dict[str, str] | None,
    user_api_key_auth: UserAPIKeyAuth | None = None,
    scope_servers: list[MCPServer] | None = None,
) -> tuple[dict[str, str] | str | None, dict[str, str] | None]:
    """Build auth and extra headers for a server.

    ``scope_servers`` is the full server list a fan-out handler iterates. Passing it lets the
    client-forwarded token modes withhold the caller's request-wide ``Authorization`` when
    another server in the scope would also receive it (``_caller_authorization_fans_out``);
    explicitly-addressed operations leave it None. Per-server ``x-mcp-{alias}-authorization``
    headers are unaffected — they bind one token to one server and are the multi-server shape.
    """
    server_auth_header: dict[str, str] | str | None = None
    if mcp_server_auth_headers:
        from litellm.proxy._experimental.mcp_server.utils import (
            lookup_mcp_server_auth_in_headers,
        )

        server_auth_header = lookup_mcp_server_auth_in_headers(
            mcp_server_auth_headers,
            alias=server.alias,
            server_name=server.server_name,
            access_groups=server.access_groups,
        )

    extra_headers: dict[str, str] | None = None
    is_client_forwarded_mode: Final = server.is_client_forwarded_token
    # In a multi-server listing scope the request-wide Authorization can only carry one token,
    # so it is withheld from a client-forwarded server when another server in scope also consumes
    # it (RFC 9700 cross-resource replay); such scopes must bind per-server via
    # x-mcp-{alias}-authorization. The decision is computed once so BOTH the forwarding branch and
    # the extra_headers copy loop below honor it — otherwise a server that lists Authorization in
    # extra_headers would re-copy the withheld bearer from raw_headers and replay it anyway.
    withhold_forwarded_authorization: Final = is_client_forwarded_mode and _caller_authorization_fans_out(
        server, scope_servers
    )
    if server.auth_type == MCPAuth.oauth2:
        # For OAuth2 M2M servers, upstream Authorization must come from
        # client_credentials token fetch, never from caller headers.
        if server.has_client_credentials:
            extra_headers = None
        else:
            # Copy to avoid mutating the original dict (important for parallel fetching)
            extra_headers = oauth2_headers.copy() if oauth2_headers else None
            # Migrated authorization_code: the v2 resolver injects the stored per-user
            # token, so drop the caller-forwarded Authorization (apply-if-absent would
            # otherwise let it shadow the resolved token). Delegate keeps it. Centralized
            # via _should_strip_caller_authorization to match _call_regular_mcp_tool.
            if extra_headers and _should_strip_caller_authorization(
                mcp_server=server,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
            ):
                extra_headers = without_header(extra_headers, DEFAULT_CREDENTIAL_HEADER)
    elif is_client_forwarded_mode:
        if not withhold_forwarded_authorization:
            extra_headers = _client_forwarded_authorization_headers(
                mcp_server=server,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
            )

    if server.extra_headers and raw_headers:
        if extra_headers is None:
            extra_headers = {}

        normalized_raw_headers: Final = {str(k).lower(): v for k, v in raw_headers.items() if isinstance(k, str)}

        # Centralized strip decision shared with
        # ``MCPServerManager._call_regular_mcp_tool`` so the two
        # code paths cannot drift on this security-sensitive choice.
        # See ``_should_strip_caller_authorization`` for the rules.
        strip_caller_authorization: Final = _should_strip_caller_authorization(
            mcp_server=server,
            raw_headers=raw_headers,
            user_api_key_auth=user_api_key_auth,
        )

        for header in server.extra_headers:
            if not isinstance(header, str):
                continue
            if header.lower() == "authorization" and (strip_caller_authorization or withhold_forwarded_authorization):
                continue
            header_value = normalized_raw_headers.get(header.lower())
            if header_value is None:
                continue
            extra_headers[header] = header_value

    # Reset to None if no headers were actually added
    if extra_headers is not None and len(extra_headers) == 0:
        extra_headers = None

    if server_auth_header is None:
        server_auth_header = mcp_auth_header

    return server_auth_header, extra_headers


def _merge_gateway_initialize_instructions(
    allowed_mcp_servers: list[MCPServer],
) -> str | None:
    """YAML/DB override, else upstream text (prefetch on init, or list_tools / health_check / call_tool cache)."""
    if not allowed_mcp_servers:
        return None

    texts: Final[list[tuple[str, str]]] = []
    for server in allowed_mcp_servers:
        label = server.alias or server.server_name or server.name or server.server_id or "mcp"
        if server.instructions and server.instructions.strip():
            texts.append((label, server.instructions.strip()))
            continue
        if server.spec_path:
            continue
        cached = global_mcp_server_manager._upstream_initialize_instructions_by_server_id.get(server.server_id)
        if cached and cached.strip():
            texts.append((label, cached.strip()))

    if not texts:
        return None
    if len(texts) == 1:
        return texts[0][1]
    return "\n\n---\n\n".join(f"[{lbl}]\n{txt}" for lbl, txt in texts)


async def _raise_if_initialize_grants_no_mcp_servers(
    allowed: Sequence[MCPServer],
    user_api_key_auth: UserAPIKeyAuth | None,
    mcp_servers: Sequence[str] | None,
    client_ip: str | None,
) -> None:
    if allowed or user_api_key_auth is None or not user_api_key_auth.api_key:
        return
    if mcp_servers:
        await raise_denied_scoped_mcp_access(
            requested_names=mcp_servers,
            user_api_key_auth=user_api_key_auth,
            client_ip=client_ip,
        )
    no_servers_denial: Final[_McpDeniedDetail] = {
        "error": (
            "The key has no MCP servers granted, or none of its granted servers is loaded and allowed for "
            "this client IP. Grant servers or access groups to the key, its team, or its organization "
            "(object_permission.mcp_servers), check the server's allowed IPs, and reconnect."
        )
    }
    raise HTTPException(status_code=403, detail=no_servers_denial)


def _aggregate_server_key(server: MCPServer) -> str:
    """The client-visible key for a server in listing outcomes and spend metadata: the same
    display prefix (alias, or the short prefix when that mode is enabled) the caller already
    sees on the tool names. Canonical internal server names never key a caller-readable
    surface; when the display naming deliberately hides them, the outcome keys must too."""
    return get_server_prefix(server) or "unknown"


async def _get_tools_from_mcp_servers(
    user_api_key_auth: UserAPIKeyAuth | None,
    mcp_auth_header: str | None,
    mcp_servers: list[str] | None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    log_list_tools_to_spendlogs: bool = False,
    list_tools_log_source: str | None = None,
    litellm_trace_id: str | None = None,
    request_tags: list[str] | None = None,
    client_ip: str | None = None,
    mcp_proxy_mode: bool = False,
) -> AggregateToolListing:
    """
    Helper method to fetch tools from MCP servers based on server filtering criteria.

    Args:
        user_api_key_auth: User authentication info for access control
        mcp_auth_header: Optional auth header for MCP server (deprecated)
        mcp_servers: Optional list of server names/aliases to filter by
        mcp_server_auth_headers: Optional dict of server-specific auth headers
        oauth2_headers: Optional dict of oauth2 headers

    Returns:
        AggregateToolListing: Combined tools from filtered servers plus each server's
        classified listing outcome
    """

    list_tools_start_time: Final = datetime.now()
    litellm_logging_obj: LiteLLMLoggingObj | None = None
    list_tools_request_data: dict[str, object] = {}

    if log_list_tools_to_spendlogs:
        # This is intentionally minimal: only async_success_handler / post_call_failure_hook
        rules_obj: Final = Rules()
        list_tools_call_id: Final = str(uuid.uuid4())
        # Derive trace_id from raw_headers when not explicitly passed (same as A2A / MCP call_tool)
        effective_litellm_trace_id: Final = litellm_trace_id or get_chain_id_from_headers(raw_headers)
        spend_logs_metadata: Final[dict[str, object]] = {
            "mcp_operation": "list_tools",
        }
        if isinstance(list_tools_log_source, str):
            spend_logs_metadata["source"] = list_tools_log_source
        if isinstance(mcp_servers, list):
            spend_logs_metadata["requested_mcp_servers"] = mcp_servers

        list_tools_request_data = {
            "model": "MCP: list_tools",
            "call_type": CallTypes.list_mcp_tools.value,
            "litellm_call_id": list_tools_call_id,
            "litellm_trace_id": effective_litellm_trace_id,
            "metadata": {
                "spend_logs_metadata": spend_logs_metadata,
                "headers": logging_safe_mcp_headers(raw_headers),
                **({"tags": request_tags} if request_tags else {}),
            },
            # Provide a small input payload for standard logging
            "input": [
                {
                    "role": "system",
                    "content": {
                        "mcp_operation": "list_tools",
                        "requested_mcp_servers": mcp_servers,
                    },
                }
            ],
        }

        # Attach user identifiers using the standard helper
        if user_api_key_auth is not None:
            LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
                data=list_tools_request_data,
                user_api_key_dict=user_api_key_auth,
                _metadata_variable_name="metadata",
            )

            user_identifier: Final = getattr(user_api_key_auth, "end_user_id", None) or getattr(
                user_api_key_auth, "user_id", None
            )
            if user_identifier:
                list_tools_request_data["user"] = user_identifier

        try:
            litellm_logging_obj, _ = function_setup(
                original_function="list_mcp_tools",
                is_async_call=False,
                rules_obj=rules_obj,
                start_time=list_tools_start_time,
                **list_tools_request_data,
            )
            if litellm_logging_obj:
                litellm_logging_obj.call_type = CallTypes.list_mcp_tools.value
                litellm_logging_obj.model = "MCP: list_tools"
        except Exception as logging_error:
            verbose_logger.debug("Failed to initialize logging for MCP list_tools: %s", logging_error)
            litellm_logging_obj = None

    try:
        allowed_mcp_servers: Final = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
            client_ip=client_ip,
        )
        if mcp_servers and not allowed_mcp_servers:
            await raise_denied_scoped_mcp_access(
                requested_names=mcp_servers,
                user_api_key_auth=user_api_key_auth,
                client_ip=client_ip,
            )

        # Pre-fetch OAuth credentials only when at least one server uses OAuth2,
        # to avoid an unnecessary DB round-trip on requests with no OAuth2 MCP servers.
        _has_oauth2_server = any(getattr(s, "auth_type", None) == MCPAuth.oauth2 for s in allowed_mcp_servers)
        _prefetched_oauth_creds: Final = (
            await _prefetch_oauth_creds_for_user(user_api_key_auth) if _has_oauth2_server else {}
        )

        async def _fetch_and_filter_server_tools(
            server: MCPServer,
        ) -> "tuple[list[MCPTool], ServerOutcome]":
            """Fetch and filter tools from a single server, classifying any failure into that
            server's outcome so the aggregate can keep serving the healthy subset without a
            broken server masquerading as an empty one."""
            if server is None:
                return [], ServerListOk(tool_count=0)

            server_auth_header, extra_headers = _prepare_mcp_server_headers(
                server=server,
                mcp_server_auth_headers=mcp_server_auth_headers,
                mcp_auth_header=mcp_auth_header,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
                scope_servers=allowed_mcp_servers,
            )

            # Prefer server-stored per-user OAuth when configured, so a stale
            # Authorization header from the MCP client cannot override Redis/DB
            # (same issue as call_tool in mcp_server_manager: VS Code caches tokens).
            from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415
                to_server_spec,
            )

            # A server migrated to the v2 resolver gets its token from the resolver at connect
            # time; building it here would double-resolve and be shadowed by the v2 graft. The
            # preemptive 401 already challenged a missing token, so one exists for the connect.
            migrated_to_v2: Final = to_server_spec(server) is not None
            if (
                not migrated_to_v2
                and server.auth_type == MCPAuth.oauth2
                and getattr(server, "needs_user_oauth_token", False)
                and user_api_key_auth is not None
            ):
                db_headers: Final = await _get_user_oauth_extra_headers_from_db(
                    server,
                    user_api_key_auth,
                    prefetched_creds=_prefetched_oauth_creds,
                )
                if db_headers:
                    extra_headers = db_headers

            # If still no OAuth2 token, fall back to pre-fetched creds (non-stale-client path)
            elif not migrated_to_v2 and extra_headers is None and server.auth_type == MCPAuth.oauth2:
                extra_headers = await _get_user_oauth_extra_headers_from_db(
                    server,
                    user_api_key_auth,
                    prefetched_creds=_prefetched_oauth_creds,
                )

            if server.is_byok and server.auth_type != MCPAuth.oauth2 and server_auth_header is None:
                server_auth_header = await _get_byok_credential(server, user_api_key_auth)

            try:
                tools: Final = await global_mcp_server_manager._get_tools_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    extra_headers=extra_headers,
                    add_prefix=True,  # Always add server prefix
                    raw_headers=raw_headers,
                    client_ip=client_ip,
                    user_api_key_auth=user_api_key_auth,
                    oauth2_headers=oauth2_headers,
                )
                filtered_tools = filter_tools_by_allowed_tools(tools, server)

                filtered_tools = await filter_tools_by_key_team_permissions(
                    tools=filtered_tools,
                    server_id=server.server_id,
                    user_api_key_auth=user_api_key_auth,
                )

                if mcp_proxy_mode:
                    from litellm.proxy._experimental.mcp_server.tool_search import with_mcp_proxy_identity

                    filtered_tools = [  # mutable-ok: MCP tool pipeline
                        with_mcp_proxy_identity(tool, server.server_id) for tool in filtered_tools
                    ]
                else:
                    filtered_tools = apply_tool_overrides(filtered_tools, server)

                verbose_logger.debug(
                    "Successfully fetched %s tools from server %s, %s after filtering",
                    len(tools),
                    server.name,
                    len(filtered_tools),
                )
                return filtered_tools, ServerListOk(tool_count=len(filtered_tools))
            except MCPUpstreamAuthError as e:
                # Absorb so one unauthenticated server does not empty every other server's
                # tools. Surfacing the upstream 401 to the client as a re-auth challenge is
                # intentionally not done here: raising from this list handler cannot produce a
                # 401 + WWW-Authenticate (the MCP session manager serializes it as a JSON-RPC
                # error). Single-server routes surface it via the request-scope preemptive
                # check in _raise_preemptive_401_for_unauthenticated_servers instead.
                verbose_logger.debug("MCP list_tools: omitting %s; it needs upstream auth", server.name)
                return [], classify_list_exception(e)
            except Exception as e:
                verbose_logger.exception("Error getting tools from server %s: %s", server.name, e)
                return [], classify_list_exception(e)

        # Fetch tools from all servers in parallel
        tasks: Final = [_fetch_and_filter_server_tools(server) for server in allowed_mcp_servers]
        results: Final = await asyncio.gather(*tasks)

        # Flatten results into single list
        all_tools: Final[list[MCPTool]] = [tool for tools, _ in results for tool in tools]
        server_outcomes: Final[dict[str, ServerOutcome]] = {
            _aggregate_server_key(server): outcome
            for server, (_, outcome) in zip(allowed_mcp_servers, results)
            if server is not None
        }

        # If logging is enabled, enrich spend_logs_metadata with counts
        if litellm_logging_obj:
            per_server_tool_counts: Final[dict[str, int]] = {
                _aggregate_server_key(server): len(server_tools)
                for server, (server_tools, _) in zip(allowed_mcp_servers, results)
                if server is not None
            }

            metadata_dict: Final = litellm_logging_obj.model_call_details.get("metadata")
            if isinstance(metadata_dict, dict):
                spend_meta = metadata_dict.get("spend_logs_metadata")
                if not isinstance(spend_meta, dict):
                    spend_meta = {}
                    metadata_dict["spend_logs_metadata"] = spend_meta
                spend_meta["allowed_server_count"] = len(allowed_mcp_servers)
                spend_meta["tool_count_total"] = len(all_tools)
                spend_meta["per_server_tool_counts"] = per_server_tool_counts
                spend_meta["per_server_list_outcomes"] = {
                    key: outcome_wire_value(outcome) for key, outcome in server_outcomes.items()
                }

            end_time: Final = datetime.now()
            try:
                await litellm_logging_obj.async_success_handler(
                    result=[tool.model_dump(mode="json") if isinstance(tool, MCPTool) else tool for tool in all_tools],
                    start_time=list_tools_start_time,
                    end_time=end_time,
                )
            except Exception as log_exc:
                # list_tools responses must not be dropped due to non-blocking
                # observability/serialization failures.
                verbose_logger.warning(
                    "MCP list_tools success logging failed (continuing): %s",
                    log_exc,
                )

        verbose_logger.info("Successfully fetched %s tools total from all MCP servers", len(all_tools))

        return AggregateToolListing(tools=all_tools, outcomes=server_outcomes)
    except Exception as e:
        # Only fire failure hook if logging was requested for this list-tools execution
        if log_list_tools_to_spendlogs and user_api_key_auth is not None:
            try:
                from litellm.proxy.proxy_server import proxy_logging_obj

                if proxy_logging_obj:
                    traceback_str: Final = traceback.format_exc(limit=MAXIMUM_TRACEBACK_LINES_TO_LOG)
                    await proxy_logging_obj.post_call_failure_hook(
                        request_data=list_tools_request_data or {},
                        original_exception=e,
                        user_api_key_dict=user_api_key_auth,
                        route="/mcp/list_tools",
                        traceback_str=traceback_str,
                    )
            except Exception:
                verbose_logger.debug("Failed to log MCP list_tools failure via post_call_failure_hook")
        raise


async def _get_prompts_from_mcp_servers(
    user_api_key_auth: UserAPIKeyAuth | None,
    mcp_auth_header: str | None,
    mcp_servers: list[str] | None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> list[Prompt]:
    """
    Helper method to fetch prompt from MCP servers based on server filtering criteria.

    Args:
        user_api_key_auth: User authentication info for access control
        mcp_auth_header: Optional auth header for MCP server (deprecated)
        mcp_servers: Optional list of server names/aliases to filter by
        mcp_server_auth_headers: Optional dict of server-specific auth headers
        oauth2_headers: Optional dict of oauth2 headers

    Returns:
        List[Prompt]: Combined list of prompts from filtered servers
    """

    allowed_mcp_servers: Final = await _get_allowed_mcp_servers(
        user_api_key_auth=user_api_key_auth,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
    )

    # Get prompts from each allowed server
    all_prompts: Final = []
    for server in allowed_mcp_servers:
        if server is None:
            continue

        server_auth_header, extra_headers = _prepare_mcp_server_headers(
            server=server,
            mcp_server_auth_headers=mcp_server_auth_headers,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            user_api_key_auth=user_api_key_auth,
            scope_servers=allowed_mcp_servers,
        )

        try:
            prompts = await global_mcp_server_manager.get_prompts_from_server(
                server=server,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=server_auth_header,
                extra_headers=extra_headers,
                add_prefix=True,  # Always add server prefix
                raw_headers=raw_headers,
                client_ip=client_ip,
            )

            all_prompts.extend(prompts)

            verbose_logger.debug("Successfully fetched %s prompts from server %s", len(prompts), server.name)
        except Exception as e:
            verbose_logger.exception("Error getting prompts from server %s: %s", server.name, e)
            # Continue with other servers instead of failing completely

    verbose_logger.info("Successfully fetched %s prompts total from all MCP servers", len(all_prompts))

    return all_prompts


async def _get_resources_from_mcp_servers(
    user_api_key_auth: UserAPIKeyAuth | None,
    mcp_auth_header: str | None,
    mcp_servers: list[str] | None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> list[Resource]:
    """Fetch resources from allowed MCP servers."""

    allowed_mcp_servers: Final = await _get_allowed_mcp_servers(
        user_api_key_auth=user_api_key_auth,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
    )

    all_resources: Final[list[Resource]] = []
    for server in allowed_mcp_servers:
        if server is None:
            continue

        server_auth_header, extra_headers = _prepare_mcp_server_headers(
            server=server,
            mcp_server_auth_headers=mcp_server_auth_headers,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            user_api_key_auth=user_api_key_auth,
            scope_servers=allowed_mcp_servers,
        )

        try:
            resources = await global_mcp_server_manager.get_resources_from_server(
                server=server,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=server_auth_header,
                extra_headers=extra_headers,
                add_prefix=True,  # Always add server prefix
                raw_headers=raw_headers,
                client_ip=client_ip,
            )
            all_resources.extend(resources)

            verbose_logger.debug("Successfully fetched %s resources from server %s", len(resources), server.name)
        except Exception as e:
            verbose_logger.exception("Error getting resources from server %s: %s", server.name, e)

    verbose_logger.info("Successfully fetched %s resources total from all MCP servers", len(all_resources))

    return all_resources


async def _get_resource_templates_from_mcp_servers(
    user_api_key_auth: UserAPIKeyAuth | None,
    mcp_auth_header: str | None,
    mcp_servers: list[str] | None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> list[ResourceTemplate]:
    """Fetch resource templates from allowed MCP servers."""

    allowed_mcp_servers: Final = await _get_allowed_mcp_servers(
        user_api_key_auth=user_api_key_auth,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
    )

    all_resource_templates: Final[list[ResourceTemplate]] = []
    for server in allowed_mcp_servers:
        if server is None:
            continue

        server_auth_header, extra_headers = _prepare_mcp_server_headers(
            server=server,
            mcp_server_auth_headers=mcp_server_auth_headers,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            user_api_key_auth=user_api_key_auth,
            scope_servers=allowed_mcp_servers,
        )

        try:
            resource_templates = await global_mcp_server_manager.get_resource_templates_from_server(
                server=server,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=server_auth_header,
                extra_headers=extra_headers,
                add_prefix=True,  # Always add server prefix
                raw_headers=raw_headers,
                client_ip=client_ip,
            )
            all_resource_templates.extend(resource_templates)
            verbose_logger.debug(
                "Successfully fetched %s resource templates from server %s",
                len(resource_templates),
                server.name,
            )
        except Exception as e:
            verbose_logger.exception(
                "Error getting resource templates from server %s: %s",
                server.name,
                str(e),
            )

    verbose_logger.info(
        "Successfully fetched %s resource templates total from all MCP servers",
        len(all_resource_templates),
    )

    return all_resource_templates


async def filter_tools_by_key_team_permissions(
    tools: list[MCPTool],
    server_id: str,
    user_api_key_auth: UserAPIKeyAuth | None,
) -> list[MCPTool]:
    """
    Filter tools based on key/team mcp_tool_permissions.

    Note: Tool names in the DB are stored without server prefixes,
    but tool names from MCP servers are prefixed. We need to strip
    the prefix before comparing.
    """
    # Filter by key/team tool-level permissions
    allowed_tool_names: Final = await MCPRequestHandler.get_allowed_tools_for_server(
        server_id=server_id,
        user_api_key_auth=user_api_key_auth,
    )

    # Tools arrive prefixed with the server's own prefix; strip exactly that
    # prefix (resolved from the server) rather than the first separator, so a
    # prefix containing the separator still reduces to the stored bare name.
    server: Final = global_mcp_server_manager.get_mcp_server_by_id(server_id)
    return [
        t
        for t in tools
        if MCPRequestHandler.tool_is_granted(strip_known_server_prefix(t.name, server), allowed_tool_names)
    ]


async def _list_mcp_tools(
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    log_list_tools_to_spendlogs: bool = False,
    list_tools_log_source: str | None = None,
    client_ip: str | None = None,
    mcp_proxy_mode: bool = False,
) -> AggregateToolListing:
    """
    List all available MCP tools.

    Args:
        user_api_key_auth: User authentication info for access control
        mcp_auth_header: Optional auth header for MCP server (deprecated)
        mcp_servers: Optional list of server names/aliases to filter by
        mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
        client_ip: Client IP for IP-based server access control

    Returns:
        AggregateToolListing: Combined tools from all accessible servers plus each server's
        classified listing outcome
    """

    try:
        listing: Final = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            log_list_tools_to_spendlogs=log_list_tools_to_spendlogs,
            list_tools_log_source=list_tools_log_source,
            client_ip=client_ip,
            mcp_proxy_mode=mcp_proxy_mode,
        )
        verbose_logger.debug("Successfully fetched %s tools from managed MCP servers", len(listing.tools))
        return listing
    except HTTPException:
        raise
    except Exception as e:
        verbose_logger.exception("Error getting tools from managed MCP servers: %s", e)
        # Continue with an empty listing instead of failing completely
        return AggregateToolListing(tools=[], outcomes={})


async def _list_mcp_prompts(
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> list[Prompt]:
    """
    List all available MCP prompts.

    Args:
        user_api_key_auth: User authentication info for access control
        mcp_auth_header: Optional auth header for MCP server (deprecated)
        mcp_servers: Optional list of server names/aliases to filter by
        mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}

    Returns:
        List[Prompt]: Combined list of tools from all accessible servers
    """
    # Get tools from managed MCP servers with error handling
    managed_prompts = []
    try:
        managed_prompts = await _get_prompts_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )
        verbose_logger.debug("Successfully fetched %s prompts from managed MCP servers", len(managed_prompts))
    except Exception as e:
        verbose_logger.exception("Error getting tools from managed MCP servers: %s", e)
        # Continue with empty managed tools list instead of failing completely

    return managed_prompts


async def _list_mcp_resources(
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> list[Resource]:
    """List all available MCP resources."""

    managed_resources: list[Resource] = []
    try:
        managed_resources = await _get_resources_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )
        verbose_logger.debug("Successfully fetched %s resources from managed MCP servers", len(managed_resources))
    except Exception as e:
        verbose_logger.exception("Error getting resources from managed MCP servers: %s", e)

    return managed_resources


async def _list_mcp_resource_templates(
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> list[ResourceTemplate]:
    """List all available MCP resource templates."""

    managed_resource_templates: list[ResourceTemplate] = []
    try:
        managed_resource_templates = await _get_resource_templates_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )
        verbose_logger.debug(
            "Successfully fetched %s resource templates from managed MCP servers",
            len(managed_resource_templates),
        )
    except Exception as e:
        verbose_logger.exception(
            "Error getting resource templates from managed MCP servers: %s",
            str(e),
        )

    return managed_resource_templates


def _resolve_display_name_to_original(
    name: str,
    allowed_mcp_servers: list[MCPServer],
) -> str:
    """Translate a display-name override back to the original prefixed tool name.

    When a client received a customised display name from tools/list (e.g.
    "Get Pet") it will call tools/call with that same string.  We need to
    reverse-map it to the original prefixed name (e.g.
    "petstore_mcp-getPetById") before any routing or permission logic runs.
    """
    for server in allowed_mcp_servers:
        display_map = server.tool_name_to_display_name or {}
        for unprefixed_name, display_name in display_map.items():
            if display_name == name:
                return add_server_prefix_to_name(unprefixed_name, get_server_prefix(server))
    return name


async def _get_byok_credential(
    mcp_server: MCPServer,
    user_api_key_auth: UserAPIKeyAuth | None,
) -> str | None:
    """Retrieve the stored BYOK credential for a user+server pair, served from the worker cache within its TTL."""
    if not mcp_server.is_byok:
        return None
    user_id: Final = (user_api_key_auth.user_id if user_api_key_auth else None) or ""
    if not user_id:
        return None

    cached: Final = get_cached_byok_credential(user_id, mcp_server.server_id)
    if cached is not None:
        return cached.credential

    from litellm.proxy._experimental.mcp_server.db import get_user_credential
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None
    credential: Final = await get_user_credential(
        prisma_client=prisma_client,
        user_id=user_id,
        server_id=mcp_server.server_id,
    )
    cache_byok_credential(user_id, mcp_server.server_id, credential)
    return credential


async def _check_byok_credential(
    mcp_server: MCPServer,
    user_api_key_auth: UserAPIKeyAuth | None,
) -> None:
    """
    If the MCP server is BYOK-enabled, verify that the requesting user has a
    stored credential.  When no credential is found, raise an HTTP 401 with a
    WWW-Authenticate header that points the MCP client to our OAuth metadata
    endpoint so it can drive the authorization flow.
    """
    if not mcp_server.is_byok:
        return

    user_id: Final = (user_api_key_auth.user_id if user_api_key_auth else None) or ""
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "byok_auth_required",
                "server_id": mcp_server.server_id,
                "server_name": mcp_server.server_name or mcp_server.name,
                "message": "User identity is required for BYOK servers",
            },
            headers={"WWW-Authenticate": get_byok_www_authenticate()},
        )

    cached: Final = get_cached_byok_credential(user_id, mcp_server.server_id)
    if cached is not None:
        if cached.credential is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "byok_auth_required",
                    "server_id": mcp_server.server_id,
                    "server_name": mcp_server.server_name or mcp_server.name,
                    "message": (
                        "No stored credential found for this BYOK server. "
                        "Complete the OAuth authorization flow to provide your API key."
                    ),
                },
                headers={"WWW-Authenticate": get_byok_www_authenticate()},
            )
        return

    from litellm.proxy._experimental.mcp_server.db import get_user_credential
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        # Fail closed on DB unavailability: returning here previously
        # bypassed the ownership check and let any proxy-authenticated
        # caller invoke BYOK tools during outage windows.
        raise HTTPException(
            status_code=503,
            detail={
                "error": "byok_auth_unavailable",
                "server_id": mcp_server.server_id,
                "server_name": mcp_server.server_name or mcp_server.name,
                "message": "BYOK credential check requires a database connection.",
            },
        )

    credential: Final = await get_user_credential(
        prisma_client=prisma_client,
        user_id=user_id,
        server_id=mcp_server.server_id,
    )
    cache_byok_credential(user_id, mcp_server.server_id, credential)
    if credential is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "byok_auth_required",
                "server_id": mcp_server.server_id,
                "server_name": mcp_server.server_name or mcp_server.name,
                "message": (
                    "No stored credential found for this BYOK server. "
                    "Complete the OAuth authorization flow to provide your API key."
                ),
            },
            headers={"WWW-Authenticate": get_byok_www_authenticate()},
        )


def _challenge_missing_token_exchange_subject(
    server: MCPServer | None,
    requested_server: MCPServer | None,
    allowed_mcp_servers: list[MCPServer],
    user_api_key_auth: UserAPIKeyAuth | None,
    oauth2_headers: dict[str, str] | None,
    raw_headers: dict[str, str] | None,
) -> None:
    """Raise the RFC 9728 challenge when a token-exchange server is called without a subject token.

    The listing that fills a cold catalog absorbs the upstream 401 by design, so without this
    check a missing subject surfaces as an unknown-tool error instead of the challenge the
    warm path already raises. Gated to servers the key may reach so an unauthorized caller
    learns nothing about the catalog.
    """
    if server is None or server.auth_type != MCPAuth.oauth2_token_exchange:
        return
    if requested_server is not None and requested_server.server_id != server.server_id:
        return
    if all(allowed.server_id != server.server_id for allowed in allowed_mcp_servers):
        return
    if global_mcp_server_manager._extract_subject_token(oauth2_headers, raw_headers, user_api_key_auth) is not None:
        return
    from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415  # lazy: adapter pulls MCP subgraph
        raise_token_exchange_challenge,
    )
    from litellm.proxy.middleware.per_request_root_path_middleware import (  # noqa: PLC0415  # lazy: middleware imports proxy utils
        get_request_root_path,
    )

    raise_token_exchange_challenge(server, root_path=get_request_root_path())


async def _list_tools_before_first_call(
    server: MCPServer | None,
    tool_name: str,
    allowed_mcp_servers: list[MCPServer],
    user_api_key_auth: UserAPIKeyAuth | None,
    mcp_auth_header: str | None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None,
    oauth2_headers: dict[str, str] | None,
    raw_headers: dict[str, str] | None,
    client_ip: str | None = None,
) -> None:
    """List ``server`` with the caller's own credentials when it does not yet expose ``tool_name`` here.

    The startup fill skips a server whose upstream wants the caller's token, and mcp 2 no
    longer lists before an uncached tools/call, so a worker that has not served tools/list
    for this caller would otherwise answer 404 for a tool the caller can see. Gating on the
    requested tool, not on any prior listing, keeps callers with different upstream catalogs
    from masking each other.
    """
    if server is None or global_mcp_server_manager.server_exposes_tool(server, tool_name):
        return
    if all(allowed.server_id != server.server_id for allowed in allowed_mcp_servers):
        return
    try:
        await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=[server.server_id],
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )
    except Exception as e:  # noqa: BLE001  # best effort: resolution below answers as it did before
        verbose_logger.debug("MCP tools/call: listing %s before its first call failed: %s", server.name, e)


async def execute_mcp_tool(
    name: str,
    arguments: dict[str, object],
    allowed_mcp_servers: list[MCPServer],
    start_time: datetime,
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    host_progress_callback: ProgressCallback | None = None,
    guardrail_context: Mapping[str, object] | None = None,
    client_ip: str | None = None,
    wire_compat: WireCompat = WireCompat.LEGACY,
    **kwargs: object,  # kwargs-ok: preserves the existing REST and decorated logging call contract
) -> CallToolResult | InputRequiredResult:
    context: Final = prepare_context(
        user_api_key_auth=user_api_key_auth,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        client_ip=client_ip,
        wire_compat=wire_compat,
    )
    operation: Final = AuthorizedToolCall(
        name=name,
        arguments=arguments,
        allowed_mcp_servers=tuple(allowed_mcp_servers),
        start_time=start_time,
        host_progress_callback=host_progress_callback,
        guardrail_context=guardrail_context,
        logging_data=types.MappingProxyType(kwargs),
    )
    return await GatewayOperations().execute(operation, context)


async def _execute_mcp_tool(
    name: str,
    arguments: dict[str, object],
    allowed_mcp_servers: list[MCPServer],
    start_time: datetime,
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    host_progress_callback: ProgressCallback | None = None,
    guardrail_context: Mapping[str, object] | None = None,
    client_ip: str | None = None,
    wire_compat: WireCompat = WireCompat.LEGACY,
    **kwargs: Any,
) -> CallToolResult | InputRequiredResult:
    """
    Execute MCP tool.

    This function assumes permission checks have already been performed.

    Args:
        name: Tool name (may include server prefix)
        arguments: Tool arguments
        allowed_mcp_servers: Pre-validated list of servers the user can access
        start_time: Start time for logging
        user_api_key_auth: Optional user API key auth for logging
        mcp_auth_header: Optional MCP auth header
        mcp_server_auth_headers: Optional server-specific auth headers
        oauth2_headers: Optional OAuth2 headers
        raw_headers: Optional raw HTTP headers
        **kwargs: Additional arguments (e.g., litellm_logging_obj)

    Returns:
        CallToolResult: Tool execution result
    """
    # Track resolved MCP server for both permission checks and dispatch
    mcp_server: MCPServer | None = None
    requested_server_id: Final[str | None] = kwargs.get("requested_server_id")

    # If the client called with a display-name override (e.g. "Get Pet"),
    # translate it back to the original prefixed name before any routing.
    name = _resolve_display_name_to_original(name, allowed_mcp_servers)

    # Remove prefix from tool name for logging and processing
    original_tool_name, server_name = split_server_prefix_from_name(name)

    requested_server: MCPServer | None = None
    if requested_server_id:
        requested_server = next(
            (s for s in allowed_mcp_servers if s.server_id == requested_server_id),
            None,
        )

    name_is_prefixed = False
    if requested_server is not None and MCP_TOOL_PREFIX_SEPARATOR in name:
        all_registry_prefixes: Final[set[str]] = set()
        for registry_server in global_mcp_server_manager.get_registry().values():
            for known_prefix in iter_known_server_prefixes(registry_server):
                all_registry_prefixes.add(normalize_server_name(known_prefix))
        name_is_prefixed = is_tool_name_prefixed(name, known_server_prefixes=all_registry_prefixes)

    first_call_target: Final = (
        requested_server
        if requested_server is not None and not name_is_prefixed
        else global_mcp_server_manager.server_owning_tool_name_prefix(name)
    )
    first_call_tool_name: Final = (
        name
        if first_call_target is None or (requested_server is not None and not name_is_prefixed)
        else strip_known_server_prefix(name, first_call_target)
    )
    _challenge_missing_token_exchange_subject(
        server=first_call_target,
        requested_server=requested_server,
        allowed_mcp_servers=allowed_mcp_servers,
        user_api_key_auth=user_api_key_auth,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
    )
    await _list_tools_before_first_call(
        server=first_call_target,
        tool_name=first_call_tool_name,
        allowed_mcp_servers=allowed_mcp_servers,
        user_api_key_auth=user_api_key_auth,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        client_ip=client_ip,
    )

    if requested_server is not None and not name_is_prefixed:
        # REST callers may pass server_id with the upstream tool name (no
        # LiteLLM prefix). The first segment is not a registered server
        # prefix, so the whole string is the upstream tool name and may
        # legitimately contain the separator (e.g. "text-to-speech").
        # server_id is authoritative for routing and auth.
        mcp_server = requested_server
        server_name = requested_server.name
        original_tool_name = name
    else:
        # Resolve from tool name (MCP JSON-RPC or prefixed REST tool names).
        mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)
        if mcp_server is None and requested_server is not None:
            for known_prefix in iter_known_server_prefixes(requested_server):
                candidate = global_mcp_server_manager._get_mcp_server_from_tool_name(
                    add_server_prefix_to_name(name, known_prefix)
                )
                if candidate is not None:
                    mcp_server = candidate
                    break
        if mcp_server is not None:
            server_name = mcp_server.name
            original_tool_name = strip_known_server_prefix(name, mcp_server)

        if requested_server is not None:
            if mcp_server is not None and mcp_server.server_id != requested_server.server_id:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "tool_server_mismatch",
                        "message": (
                            f"Tool '{name}' belongs to MCP server "
                            f"'{mcp_server.name}' but request specified "
                            f"server_id for '{requested_server.name}'."
                        ),
                    },
                )
            if mcp_server is None:
                mcp_server = requested_server
                server_name = requested_server.name
                original_tool_name = strip_known_server_prefix(name, requested_server)

    # Only enforce server-level permissions when we can resolve a server
    if server_name:
        if not MCPRequestHandler.is_tool_allowed(
            allowed_mcp_servers=[server.name for server in allowed_mcp_servers],
            server_name=server_name,
        ):
            raise HTTPException(
                status_code=403,
                detail="User not allowed to call this tool.",
            )

    standard_logging_mcp_tool_call: Final[StandardLoggingMCPToolCall] = _get_standard_logging_mcp_tool_call(
        name=original_tool_name,  # Use original name for logging
        arguments=arguments,
        server_name=server_name,
        session_id=_mcp_session_id_from_headers(raw_headers),
    )
    litellm_logging_obj: Final[LiteLLMLoggingObj | None] = kwargs.get("litellm_logging_obj", None)
    if litellm_logging_obj:
        litellm_logging_obj.model_call_details["mcp_tool_call_metadata"] = standard_logging_mcp_tool_call
        litellm_logging_obj.model = f"MCP: {name}"
        litellm_logging_obj.model_call_details["model"] = f"MCP: {name}"
    # Resolve the MCP server early so BYOK checks and credential injection
    # apply to ALL dispatch paths (local tool registry AND managed MCP server).
    if mcp_server is None:
        mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)

    if mcp_server:
        standard_logging_mcp_tool_call["mcp_server_cost_info"] = (mcp_server.mcp_info or {}).get("mcp_server_cost_info")
        if litellm_logging_obj:
            litellm_logging_obj.model_call_details["mcp_tool_call_metadata"] = standard_logging_mcp_tool_call

        # BYOK: retrieve the stored per-user credential.  A single DB call
        # both checks existence and fetches the value, avoiding a double query.
        if mcp_server.is_byok and not mcp_auth_header:
            byok_cred: Final = await _get_byok_credential(mcp_server, user_api_key_auth)
            if byok_cred is None:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "byok_auth_required",
                        "server_id": mcp_server.server_id,
                        "server_name": mcp_server.server_name or mcp_server.name,
                        "message": (
                            "No stored credential found for this BYOK server. "
                            "Complete the OAuth authorization flow to provide your API key."
                        ),
                    },
                    headers={"WWW-Authenticate": get_byok_www_authenticate()},
                )
            mcp_auth_header = byok_cred
        elif mcp_server.is_byok:
            # External auth header supplied; still enforce user-identity check.
            await _check_byok_credential(mcp_server, user_api_key_auth)

    # Check if tool exists in local registry first (for OpenAPI-based tools)
    # These tools are registered with their prefixed names
    #########################################################
    local_tool: Final = global_mcp_tool_registry.get_tool(name)
    if local_tool:
        # OpenAPI-backed tools used to bypass `pre_call_tool_check` —
        # only the managed path ran allowed/banned-tool checks, key/team
        # tool permissions, and parameter validation. Run the same checks
        # before dispatching to the local registry. Refuse the call if
        # we cannot resolve a server: tools registered via
        # openapi_to_mcp_generator are always tied to a server, so a
        # missing mcp_server here means the tool->server mapping has
        # not finished initializing or the registry entry is orphaned.
        # Skipping the check would re-open the same authorization gap.
        if mcp_server is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"MCP server for tool '{name}' is not available; "
                    "refusing to dispatch without authorization checks. "
                    "Retry once the server is registered."
                ),
            )

        # `pre_call_tool_check` calls into `proxy_logging_obj` for the
        # pre-call guardrail hooks, so source it from the canonical
        # `proxy_server` module the same way `_handle_managed_mcp_tool`
        # does. `kwargs.get("proxy_logging_obj")` is None on the MCP
        # entry path and would crash with AttributeError after the
        # security checks pass.
        from litellm.proxy.proxy_server import proxy_logging_obj

        hook_result = await global_mcp_server_manager.pre_call_tool_check(
            name=original_tool_name,
            arguments=arguments or {},
            server_name=server_name or mcp_server.name,
            user_api_key_auth=user_api_key_auth,
            proxy_logging_obj=proxy_logging_obj,
            server=mcp_server,
            raw_headers=raw_headers,
            litellm_logging_obj=litellm_logging_obj,
            guardrail_context=guardrail_context,
        )
        # `pre_call_tool_check` may return guardrail-modified
        # arguments; honor them on the local path too.
        if isinstance(hook_result, dict) and "arguments" in hook_result:
            arguments = hook_result["arguments"]

        verbose_logger.debug("Executing local registry tool: %s", name)
        # The credential rides ContextVars because the tool function has its
        # headers baked into the closure at registration time.
        auth_header_value, openapi_forwarded_headers, upstream_credential = _resolve_openapi_tool_auth(
            mcp_server=mcp_server,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            raw_headers=raw_headers,
            user_api_key_auth=user_api_key_auth,
        )
        (
            resolved_auth_headers,
            forwarded_headers,
        ) = await global_mcp_server_manager.resolve_openapi_upstream_auth(
            mcp_server=mcp_server,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            mcp_auth_header=upstream_credential,
            user_api_key_auth=user_api_key_auth,
            forwarded_headers=openapi_forwarded_headers,
        )

        _auth_token: Final = _request_auth_header.set(auth_header_value)
        _extra_token: Final = _request_extra_headers.set(forwarded_headers)
        _resolved_token: Final = _request_resolved_auth_headers.set(resolved_auth_headers)
        try:
            response = await _handle_local_mcp_tool(name, arguments, wire_compat)
        finally:
            _request_auth_header.reset(_auth_token)
            _request_extra_headers.reset(_extra_token)
            _request_resolved_auth_headers.reset(_resolved_token)

    # Try managed MCP server tool (the name is bare; the prefix boundary was
    # already resolved above against this server's registered prefixes)
    # Primary and recommended way to use external MCP servers
    #########################################################
    elif mcp_server:
        response = await _handle_managed_mcp_tool(
            server_name=server_name,
            name=original_tool_name,
            arguments=arguments,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
            litellm_logging_obj=litellm_logging_obj,
            guardrail_context=guardrail_context,
            host_progress_callback=host_progress_callback,
            wire_compat=wire_compat,
        )

    # Fall back to local tool registry with original name (legacy support)
    #########################################################
    # Deprecated: Local MCP Server Tool
    #########################################################
    else:
        # Gate only what can actually dispatch. When the unprefixed name is
        # not in the registry either, `_handle_local_mcp_tool` below reports
        # 404 and nothing runs, so demanding a server here would turn every
        # unknown tool name into a misleading 503.
        if global_mcp_tool_registry.get_tool(original_tool_name) is not None:
            # `mcp_server` is None here because the tool name is not in the
            # tool -> server mapping, but the name still carries a prefix
            # that the server-level check above compared against the
            # caller's `allowed_mcp_servers` by exact `name`. So the named
            # server is in that list and can carry the tool-level checks,
            # even with the mapping cold. Resolve it from
            # `allowed_mcp_servers` rather than the registry: the registry
            # would happily return a server the caller holds no grant for,
            # and matching anything other than `name` would accept a server
            # the check never validated.
            prefix_server: Final = next(
                (candidate for candidate in allowed_mcp_servers if candidate.name == server_name),
                None,
            )
            if prefix_server is None:
                # A non-empty prefix that passed the server-level check
                # always matches here, so this arm only fires when the
                # prefix was empty, which is exactly the case that check
                # skips. Fail closed rather than dispatch with no server to
                # evaluate a tool ceiling against.
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"MCP server for tool '{original_tool_name}' is not available; "
                        "refusing to dispatch without authorization checks. "
                        "Retry once the server is registered."
                    ),
                )

            from litellm.proxy.proxy_server import proxy_logging_obj

            hook_result = await global_mcp_server_manager.pre_call_tool_check(
                name=original_tool_name,
                arguments=arguments,
                server_name=server_name,
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                server=prefix_server,
                raw_headers=raw_headers,
                litellm_logging_obj=litellm_logging_obj,
                guardrail_context=guardrail_context,
            )
            if "arguments" in hook_result:
                arguments = hook_result["arguments"]  # pyright: ignore[reportAny]  # hook returns untyped args

        response = await _handle_local_mcp_tool(original_tool_name, arguments, wire_compat)

    converted: Final = to_call_tool_result(response, wire_compat)
    if isinstance(converted, InputRequiredResult):
        return converted
    return await _run_post_mcp_call_guardrails(
        result=converted,
        litellm_logging_obj=litellm_logging_obj,
        user_api_key_auth=user_api_key_auth,
        request_data=kwargs,
    )


async def _run_post_mcp_call_guardrails(
    result: CallToolResult,
    litellm_logging_obj: LiteLLMLoggingObj | None,
    user_api_key_auth: UserAPIKeyAuth | None,
    request_data: Mapping[str, object],
) -> CallToolResult:
    """Run ``post_mcp_call`` guardrails over an executed tool result.

    Lives on ``execute_mcp_tool``'s return path rather than inside
    ``_fire_mcp_tool_call_logging`` so enforcement never depends on logging
    being configured, and so every dispatch route gets it: the MCP protocol
    handler, the REST endpoint, and tool search all funnel through here.
    A guardrail that rejects the result raises, matching ``pre_mcp_call``.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj

    if proxy_logging_obj is None:
        return result
    return await proxy_logging_obj.post_mcp_call_hook(
        response=result,
        request_data=(
            litellm_logging_obj.model_call_details if litellm_logging_obj is not None else dict(request_data)
        ),
        user_api_key_dict=user_api_key_auth,
    )


def suppress_completed_success_logging(logging_obj: LiteLLMLoggingObj) -> None:
    """An interim ``InputRequiredResult`` is not a completed call, so the ``@client`` wrapper
    on ``call_mcp_tool`` must not run the success handlers for it when the coroutine returns."""
    logging_obj.has_run_logging(event_type="sync_success")
    logging_obj.has_run_logging(event_type="async_success")


async def _fire_mcp_tool_call_logging(
    logging_obj: LiteLLMLoggingObj,
    result: CallToolResult,
    start_time: datetime,
    end_time: datetime,
    user_api_key_auth: UserAPIKeyAuth | None = None,
    request_data: Mapping[str, object] | None = None,
) -> CallToolResult:
    """Fire post-call logging for an executed MCP tool call, returning the result to send.

    The returned result is what the caller must forward to the client: a
    ``post_mcp_call`` guardrail may rewrite the tool output (e.g. mask
    sensitive values) or reject it, in which case its exception propagates.
    Guardrails run before the success/failure logging so the masked text, not
    the raw one, is what gets logged.

    A result with ``is_error=True`` is logged as a failure (``status="failure"``
    payload, so OTel marks the span ERROR) while the HTTP wire behavior stays
    200 + ``isError: true`` per the MCP spec. The error check runs after
    ``async_post_mcp_tool_call_hook`` because guardrails may flip the result
    to ``is_error=True`` in that hook. Raised exceptions never reach here (the
    ``@client`` wrapper and ``call_mcp_tool``'s except path log those), so
    this cannot double-log a failure.

    ``request_data`` may carry credential-bearing fields (the REST path puts
    ``raw_headers``, ``mcp_auth_header``, ``mcp_server_auth_headers``, and
    ``oauth2_headers`` at the top level of its data dict), so those are
    stripped before the dict is handed to ``post_call_failure_hook``
    callbacks.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj

    logging_obj.post_call(original_response=result)
    result = await logging_obj.async_post_mcp_tool_call_hook(
        kwargs=logging_obj.model_call_details,
        response_obj=result,
        start_time=start_time,
        end_time=end_time,
    )
    logging_obj.call_type = CallTypes.call_mcp_tool.value
    error_message: Final = extract_mcp_tool_result_error_message(result)
    if error_message is None:
        await logging_obj.async_success_handler(result=result, start_time=start_time, end_time=end_time)
        return result

    logging_obj.has_run_logging(event_type="sync_success")
    logging_obj.has_run_logging(event_type="async_success")
    tool_error: Final = MCPToolResultError(error_message)
    logging_obj.failure_handler(tool_error, "", start_time, end_time)
    await logging_obj.async_failure_handler(tool_error, "", start_time, end_time)

    if user_api_key_auth is None:
        return result

    if proxy_logging_obj:
        sanitized_request_data: Final = {
            key: value for key, value in (request_data or {}).items() if key not in _MCP_CREDENTIAL_REQUEST_FIELDS
        }
        await proxy_logging_obj.post_call_failure_hook(
            request_data=sanitized_request_data,
            original_exception=tool_error,
            user_api_key_dict=user_api_key_auth,
            route="/mcp/call_tool",
        )
    return result


async def fire_mcp_tool_call_failure_logging(
    logging_obj: LiteLLMLoggingObj | None,
    exception: Exception,
    start_time: datetime,
    user_api_key_auth: UserAPIKeyAuth | None,
    request_data: Mapping[str, object],
) -> None:
    """Failure logging shared by the ``/mcp`` path and the REST endpoint. Call from
    inside the ``except`` block so the traceback is still available.

    The failure handlers run first because ``_ProxyDBLogger.async_post_call_failure_hook``
    builds the failure spend-log row from the ``standard_logging_object`` they produce;
    both gate on ``should_run_logging``, so the ``@client`` wrapper does not log twice.
    A relayed upstream 401 (``MCPUpstreamAuthError``) is an expected caller-must-reauth
    signal and skips ``post_call_failure_hook``, which fires the ``llm_exceptions`` alert.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj

    traceback_str: Final = traceback.format_exc(limit=MAXIMUM_TRACEBACK_LINES_TO_LOG)
    if logging_obj is not None:
        end_time: Final = datetime.now()  # noqa: DTZ005  # naive to match `start_time`, which it is subtracted from
        logging_obj.failure_handler(exception, traceback_str, start_time, end_time)
        await logging_obj.async_failure_handler(exception, traceback_str, start_time, end_time)

    if isinstance(exception, MCPUpstreamAuthError) or not proxy_logging_obj or user_api_key_auth is None:
        return
    sanitized_request_data: Final = {
        key: value for key, value in request_data.items() if key not in _MCP_CREDENTIAL_REQUEST_FIELDS
    }
    await proxy_logging_obj.post_call_failure_hook(
        request_data=sanitized_request_data,
        original_exception=exception,
        user_api_key_dict=user_api_key_auth,
        route="/mcp/call_tool",
        traceback_str=traceback_str,
    )


@client
async def call_mcp_tool(
    name: str,
    arguments: dict[str, object] | None = None,
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
    wire_compat: WireCompat = WireCompat.LEGACY,
    **kwargs: Any,
) -> CallToolResult | InputRequiredResult:
    """
    Call a specific tool with the provided arguments (handles prefixed tool names).

    A modern ``InputRequiredResult`` is an interim answer, so it is returned as is and skips the
    completed-call logging below.
    """
    start_time: Final = datetime.now()
    litellm_logging_obj: Final[LiteLLMLoggingObj | None] = kwargs.get("litellm_logging_obj", None)

    try:
        if arguments is None:
            raise HTTPException(status_code=400, detail="Request arguments are required")

        ## CHECK IF USER IS ALLOWED TO CALL THIS TOOL
        allowed_mcp_server_ids: Final = await global_mcp_server_manager.get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
        )

        allowed_mcp_servers: list[MCPServer] = []
        for allowed_mcp_server_id in allowed_mcp_server_ids:
            allowed_server = global_mcp_server_manager.get_mcp_server_by_id(allowed_mcp_server_id)
            if allowed_server is not None:
                # Same request-time oauth2_flow backstop the listing path applies,
                # so a null-flow M2M-shape row is treated as M2M on tool calls too.
                allowed_server = MCPServerManager.resolve_oauth2_flow_for_request(allowed_server)
                allowed_mcp_servers.append(allowed_server)

        allowed_mcp_servers = await _get_allowed_mcp_servers_from_mcp_server_names(
            mcp_servers=mcp_servers,
            allowed_mcp_servers=allowed_mcp_servers,
        )
        if mcp_servers and not allowed_mcp_servers:
            await raise_denied_scoped_mcp_access(
                requested_names=mcp_servers,
                user_api_key_auth=user_api_key_auth,
                client_ip=client_ip,
            )
        if not allowed_mcp_servers:
            raise HTTPException(
                status_code=403,
                detail="User not allowed to call this tool.",
            )

        # Delegate to execute_mcp_tool for execution
        response = await execute_mcp_tool(
            name=name,
            arguments=arguments,
            allowed_mcp_servers=allowed_mcp_servers,
            start_time=start_time,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
            wire_compat=wire_compat,
            **kwargs,
        )
    except Exception as e:
        await fire_mcp_tool_call_failure_logging(litellm_logging_obj, e, start_time, user_api_key_auth, kwargs)
        raise

    if isinstance(response, InputRequiredResult):
        if litellm_logging_obj:
            suppress_completed_success_logging(litellm_logging_obj)
        return response
    if litellm_logging_obj:
        response = await _fire_mcp_tool_call_logging(
            logging_obj=litellm_logging_obj,
            result=response,
            start_time=start_time,
            end_time=datetime.now(),
            user_api_key_auth=user_api_key_auth,
            request_data=kwargs,
        )
    return response


async def mcp_get_prompt(
    name: str,
    arguments: dict[str, str] | None = None,
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> GetPromptResult:
    """
    Fetch a specific MCP prompt, handling both prefixed and unprefixed names.
    """
    allowed_mcp_servers: Final = await _get_allowed_mcp_servers(
        user_api_key_auth=user_api_key_auth,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
    )

    if not allowed_mcp_servers:
        raise HTTPException(
            status_code=403,
            detail="User not allowed to get this prompt.",
        )

    # Extract server name from prefixed prompt name
    original_prompt_name, server_name = split_server_prefix_from_name(name)

    server: Final = next((s for s in allowed_mcp_servers if s.name == server_name), None)
    if server is None:
        raise HTTPException(
            status_code=403,
            detail="User not allowed to get this prompt.",
        )

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=mcp_server_auth_headers,
        mcp_auth_header=mcp_auth_header,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        user_api_key_auth=user_api_key_auth,
    )

    return await global_mcp_server_manager.get_prompt_from_server(
        server=server,
        user_api_key_auth=user_api_key_auth,
        prompt_name=original_prompt_name,
        arguments=arguments,
        mcp_auth_header=server_auth_header,
        extra_headers=extra_headers,
        raw_headers=raw_headers,
        client_ip=client_ip,
    )


async def mcp_read_resource(
    url: AnyUrl,
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> ReadResourceResult:
    """Read resource contents from upstream MCP servers."""

    allowed_mcp_servers: Final = await _get_allowed_mcp_servers(
        user_api_key_auth=user_api_key_auth,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
    )

    if not allowed_mcp_servers:
        raise HTTPException(
            status_code=403,
            detail="User not allowed to read this resource.",
        )

    if len(allowed_mcp_servers) != 1:
        raise HTTPException(
            status_code=400,
            detail=("Multiple MCP servers configured; read_resource currently supports exactly one allowed server."),
        )

    server: Final = allowed_mcp_servers[0]

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=mcp_server_auth_headers,
        mcp_auth_header=mcp_auth_header,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        user_api_key_auth=user_api_key_auth,
    )

    return await global_mcp_server_manager.read_resource_from_server(
        server=server,
        user_api_key_auth=user_api_key_auth,
        url=url,
        mcp_auth_header=server_auth_header,
        extra_headers=extra_headers,
        raw_headers=raw_headers,
        client_ip=client_ip,
    )


def _get_standard_logging_mcp_tool_call(
    name: str,
    arguments: dict[str, object],
    server_name: str | None,
    session_id: str | None = None,
) -> StandardLoggingMCPToolCall:
    mcp_server: Final = global_mcp_server_manager._get_mcp_server_from_tool_name(
        add_server_prefix_to_name(name, server_name) if server_name else name
    )
    namespaced_tool_name: Final = f"{server_name}/{name}" if server_name else name
    if mcp_server:
        mcp_info: Final = mcp_server.mcp_info or {}
        return StandardLoggingMCPToolCall(
            name=name,
            arguments=arguments,
            mcp_server_name=mcp_info.get("server_name"),
            mcp_server_logo_url=mcp_info.get("logo_url"),
            namespaced_tool_name=namespaced_tool_name,
            mcp_session_id=session_id,
            mcp_auth_mode=mcp_server.auth_type,
            mcp_server_resource=_redact_mcp_resource_url(mcp_server.url),
        )
    else:
        return StandardLoggingMCPToolCall(
            name=name,
            arguments=arguments,
            namespaced_tool_name=namespaced_tool_name,
            mcp_session_id=session_id,
        )


async def _handle_managed_mcp_tool(
    server_name: str,
    name: str,
    arguments: dict[str, object],
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    litellm_logging_obj: LiteLLMLoggingObj | None = None,
    host_progress_callback: ProgressCallback | None = None,
    guardrail_context: Mapping[str, object] | None = None,
    client_ip: str | None = None,
    wire_compat: WireCompat = WireCompat.LEGACY,
) -> CallToolResult | InputRequiredResult:
    """Handle tool execution for managed server tools"""
    # Import here to avoid circular import
    from litellm.proxy.proxy_server import proxy_logging_obj

    call_tool_result: Final = await global_mcp_server_manager.call_tool(
        server_name=server_name,
        name=name,
        arguments=arguments,
        user_api_key_auth=user_api_key_auth,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        client_ip=client_ip,
        proxy_logging_obj=proxy_logging_obj,
        host_progress_callback=host_progress_callback,
        litellm_logging_obj=litellm_logging_obj,
        guardrail_context=guardrail_context,
        wire_compat=wire_compat,
    )
    verbose_logger.debug("CALL TOOL RESULT: %s", call_tool_result)
    return call_tool_result


async def _handle_local_mcp_tool(
    name: str, arguments: dict[str, object], wire_compat: WireCompat = WireCompat.LEGACY
) -> CallToolResult:
    """Execute a local-registry tool and report whether it succeeded.

    Returns the result rather than bare content because the verdict is part of it: the content
    alone cannot say whether the handler failed, so callers used to stamp is_error=False on every
    outcome and an upstream rejection was served as tool output.

    A failure is reported as ``is_error=True`` here rather than raised, because the REST surface
    turns an unrecognized exception into a 500 and an upstream 403 or 429 is not a gateway crash.
    ``MCPUpstreamAuthError`` is the exception: it propagates so the caller is told to
    re-authenticate, which both renderers already know how to say.

    Note: Local tools don't use prefixes, so we use the original name
    """
    import inspect

    tool: Final = global_mcp_tool_registry.get_tool(name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

    try:
        if inspect.iscoroutinefunction(tool.handler):
            result = await tool.handler(**arguments)
        else:
            result = tool.handler(**arguments)
    except MCPUpstreamAuthError:
        raise
    except Exception as e:
        verbose_logger.exception("Error executing local tool %s: %s", name, e)
        return CallToolResult(
            content=[TextContent(text=f"Error: {e}", type="text")],  # mutable-ok: MCP result content
            is_error=True,
        )
    return complete_call_tool_result(handler_outcome(result), wire_compat)


_MCP_CREDENTIAL_REQUEST_FIELDS: Final = frozenset(
    {
        "raw_headers",
        "mcp_auth_header",
        "mcp_server_auth_headers",
        "oauth2_headers",
        "user_api_key_auth",
    }
)


class _McpDeniedDetail(TypedDict):
    error: ReadOnly[str]


async def _execute_handle_list_tools(
    context: OperationContext, params: PaginatedRequestParams, host_progress_callback: ProgressCallback | None = None
) -> ListToolsResult:
    try:
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        ) = context.legacy_auth()
        verbose_logger.debug("MCP list_tools - User API Key Auth from context: %s", user_api_key_auth)
        verbose_logger.debug("MCP list_tools - MCP servers from context: %s", mcp_servers)
        verbose_logger.debug(
            "MCP list_tools - MCP server auth headers: %s",
            list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None,
        )
        from mcp.types import Tool

        from litellm.proxy._experimental.mcp_server.tool_search import (
            get_mcp_proxy_tool_definitions,
            get_virtual_tool_definitions,
        )

        if context.mcp_proxy_mode:
            return ListToolsResult(tools=[Tool.model_validate(d) for d in get_mcp_proxy_tool_definitions()])
        if getattr(
            getattr(user_api_key_auth, "object_permission", None),
            "mcp_tool_search_enabled",
            False,
        ):
            return ListToolsResult(tools=[Tool.model_validate(d) for d in get_virtual_tool_definitions()])

        # Get mcp_servers from context variable
        verbose_logger.debug("MCP list_tools - Calling _list_mcp_tools")
        listing: Final = await _list_mcp_tools(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            log_list_tools_to_spendlogs=True,
            list_tools_log_source="mcp_protocol",
            client_ip=_client_ip,
        )
        verbose_logger.info("MCP list_tools - Successfully returned %s tools", len(listing.tools))
        if not listing.outcomes:
            return ListToolsResult(tools=listing.tools)
        outcome_meta: Final = {
            SERVER_OUTCOMES_META_KEY: {key: outcome_wire_value(outcome) for key, outcome in listing.outcomes.items()}
        }
        return ListToolsResult.model_validate({"tools": listing.tools, "_meta": outcome_meta})
    except HTTPException as e:
        from mcp.shared.exceptions import MCPError
        from mcp.types import INVALID_REQUEST

        raise MCPError(code=INVALID_REQUEST, message=_http_detail_message(e.detail)) from e
    except Exception as e:
        verbose_logger.exception("Error in list_tools endpoint: %s", e)
        # Return empty list instead of failing completely
        # This prevents the HTTP stream from failing and allows the client to get a response
        return ListToolsResult(tools=[])  # mutable-ok: MCP result payload


async def _execute_mcp_server_tool_call(
    context: OperationContext, params: CallToolRequestParams, host_progress_callback: ProgressCallback | None = None
) -> CallToolResult | InputRequiredResult:
    from mcp.types import CallToolResult

    from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import proxy_config

    (
        user_api_key_auth,
        mcp_auth_header,
        mcp_servers,
        mcp_server_auth_headers,
        oauth2_headers,
        raw_headers,
        _client_ip,
    ) = context.legacy_auth()
    verbose_logger.debug(
        "MCP mcp_server_tool_call - user_api_key_auth=%s, user_role=%s",
        user_api_key_auth,
        getattr(user_api_key_auth, "user_role", "N/A"),
    )

    verbose_logger.debug("MCP mcp_server_tool_call - User API Key Auth from context: %s", user_api_key_auth)

    try:
        # Inside this try so virtual-tool errors convert to isError
        # CallToolResult instead of raising out of the protocol handler.
        virtual_tool_result: Final = await _dispatch_virtual_mcp_tool(
            name=params.name,
            arguments=params.arguments,
            user_api_key_auth=user_api_key_auth,
            client_ip=_client_ip,
            mcp_servers=mcp_servers,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            mcp_proxy_mode=context.mcp_proxy_mode,
        )
        if virtual_tool_result is not None:
            return virtual_tool_result

        # Create a body date for logging
        body_data: Final = {"name": params.name, "arguments": params.arguments}  # mutable-ok: logging payload
        # Set trace/session id from raw_headers so spend logs and logging_obj stay consistent (same as A2A)
        chain_id: Final = get_chain_id_from_headers(raw_headers)
        if chain_id:
            body_data["litellm_trace_id"] = chain_id
            body_data["litellm_session_id"] = chain_id

        request: Final = build_synthetic_mcp_request(
            path="/mcp/tools/call",
            raw_headers=raw_headers,
            client_ip=_client_ip,
        )
        if user_api_key_auth is not None:
            data = await add_litellm_data_to_request(
                data=body_data,
                request=request,
                # Bill a team-derived call to the team that granted it. A keyless admitted
                # subject carries no team_id, so spend skipped team updates entirely and
                # charged the user's PRIMARY org — the granting team's budget never
                # accumulated (so it could never begin to block) and, cross-org, the wrong
                # organization was charged. This is the ACCOUNTING half; the enforcement
                # half (an already-over-budget team stops granting) lives in the source gate.
                # Authorization is unaffected: it ran before this, and the union is resolved
                # from the untouched auth object passed to call_mcp_tool below.
                user_api_key_dict=await MCPRequestHandler.billing_auth_for_tool_call(
                    user_api_key_auth, tool_name=params.name
                ),
                proxy_config=proxy_config,
            )
        else:
            data = body_data

        response: Final = await call_mcp_tool(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=_client_ip,
            host_progress_callback=host_progress_callback,
            wire_compat=context.wire_compat,
            **data,  # for logging
        )
    except MCPMissingUserEnvVarsError as e:
        verbose_logger.info(
            "MCP mcp_server_tool_call missing per-user env vars: server_id=%s missing=%s",
            e.server_id,
            e.missing,
        )
        return CallToolResult(
            content=[TextContent(text=str(e), type="text")],
            is_error=True,
        )
    except BlockedPiiEntityError as e:
        verbose_logger.error("BlockedPiiEntityError in MCP tool call: %s", e)
        return CallToolResult(
            content=[
                TextContent(
                    text=f"Error: Blocked PII entity detected - {e}",
                    type="text",
                )
            ],
            is_error=True,
        )
    except GuardrailRaisedException as e:
        verbose_logger.error("GuardrailRaisedException in MCP tool call: %s", e)
        return CallToolResult(
            content=[TextContent(text=f"Error: Guardrail violation - {e}", type="text")],
            is_error=True,
        )
    except HTTPException as e:
        verbose_logger.error("HTTPException in MCP tool call: %s", e)
        return CallToolResult(
            content=[TextContent(text=f"Error: {_http_detail_message(e.detail)}", type="text")],
            is_error=True,
        )
    except MCPUpstreamAuthError as e:
        # The MCP session manager serializes handler exceptions as JSON-RPC errors, so a
        # mid-session tool call cannot emit a raw 401 + WWW-Authenticate the way the REST
        # call path and the connect-time preemptive check do. Return an explicit isError
        # naming the upstream status (at info level, not a traceback) so the client still
        # learns it must re-authenticate upstream and expected pass-through 401s don't spam.
        verbose_logger.info("Upstream auth failure calling MCP tool: HTTP %s", e.status_code)
        return CallToolResult(
            content=[
                TextContent(
                    text=f"Error: upstream authentication required (HTTP {e.status_code})",
                    type="text",
                )
            ],
            is_error=True,
        )
    except Exception as e:
        verbose_logger.exception("MCP mcp_server_tool_call - error: %s", e)
        return CallToolResult(
            content=[TextContent(text=f"Error: {e}", type="text")],
            is_error=True,
        )

    return response


async def _execute_list_prompts(
    context: OperationContext, params: PaginatedRequestParams, host_progress_callback: ProgressCallback | None = None
) -> ListPromptsResult:
    if context.mcp_proxy_mode:
        _reject_mcp_proxy_operation()
    try:
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        ) = context.legacy_auth()
        verbose_logger.debug("MCP list_prompts - User API Key Auth from context: %s", user_api_key_auth)
        verbose_logger.debug("MCP list_prompts - MCP servers from context: %s", mcp_servers)
        verbose_logger.debug(
            "MCP list_prompts - MCP server auth headers: %s",
            list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None,
        )
        # Get mcp_servers from context variable
        verbose_logger.debug("MCP list_prompts - Calling _list_prompts")
        prompts: Final = await _list_mcp_prompts(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=_client_ip,
        )
        verbose_logger.info("MCP list_prompts - Successfully returned %s prompts", len(prompts))
        return ListPromptsResult(prompts=prompts)
    except Exception as e:
        verbose_logger.exception("Error in list_prompts endpoint: %s", e)
        # Return empty list instead of failing completely
        # This prevents the HTTP stream from failing and allows the client to get a response
        return ListPromptsResult(prompts=[])  # mutable-ok: MCP result payload


async def _execute_get_prompt(
    context: OperationContext, params: GetPromptRequestParams, host_progress_callback: ProgressCallback | None = None
) -> GetPromptResult:
    if context.mcp_proxy_mode:
        _reject_mcp_proxy_operation()
    (
        user_api_key_auth,
        mcp_auth_header,
        mcp_servers,
        mcp_server_auth_headers,
        oauth2_headers,
        raw_headers,
        _client_ip,
    ) = context.legacy_auth()

    verbose_logger.debug("MCP mcp_server_tool_call - User API Key Auth from context: %s", user_api_key_auth)
    return await mcp_get_prompt(
        name=params.name,
        arguments=params.arguments,
        user_api_key_auth=user_api_key_auth,
        mcp_auth_header=mcp_auth_header,
        mcp_servers=mcp_servers,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        client_ip=_client_ip,
    )


async def _execute_list_resources(
    context: OperationContext, params: PaginatedRequestParams, host_progress_callback: ProgressCallback | None = None
) -> ListResourcesResult:
    if context.mcp_proxy_mode:
        _reject_mcp_proxy_operation()
    try:
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        ) = context.legacy_auth()
        verbose_logger.debug("MCP list_resources - User API Key Auth from context: %s", user_api_key_auth)
        verbose_logger.debug("MCP list_resources - MCP servers from context: %s", mcp_servers)
        verbose_logger.debug(
            "MCP list_resources - MCP server auth headers: %s",
            list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None,
        )

        resources: Final = await _list_mcp_resources(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=_client_ip,
        )
        verbose_logger.info("MCP list_resources - Successfully returned %s resources", len(resources))
        return ListResourcesResult(resources=resources)
    except Exception as e:
        verbose_logger.exception("Error in list_resources endpoint: %s", e)
        return ListResourcesResult(resources=[])  # mutable-ok: MCP result payload


async def _execute_list_resource_templates(
    context: OperationContext, params: PaginatedRequestParams, host_progress_callback: ProgressCallback | None = None
) -> ListResourceTemplatesResult:
    if context.mcp_proxy_mode:
        _reject_mcp_proxy_operation()
    try:
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        ) = context.legacy_auth()
        verbose_logger.debug("MCP list_resource_templates - User API Key Auth from context: %s", user_api_key_auth)
        verbose_logger.debug("MCP list_resource_templates - MCP servers from context: %s", mcp_servers)
        verbose_logger.debug(
            "MCP list_resource_templates - MCP server auth headers: %s",
            list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None,
        )

        resource_templates: Final = await _list_mcp_resource_templates(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=_client_ip,
        )
        verbose_logger.info(
            "MCP list_resource_templates - Successfully returned %s resource templates", len(resource_templates)
        )
        return ListResourceTemplatesResult(resource_templates=resource_templates)
    except Exception as e:
        verbose_logger.exception("Error in list_resource_templates endpoint: %s", e)
        return ListResourceTemplatesResult(resource_templates=[])  # mutable-ok: MCP result payload


async def _execute_read_resource(
    context: OperationContext, params: ReadResourceRequestParams, host_progress_callback: ProgressCallback | None = None
) -> ReadResourceResult:
    if context.mcp_proxy_mode:
        _reject_mcp_proxy_operation()
    (
        user_api_key_auth,
        mcp_auth_header,
        mcp_servers,
        mcp_server_auth_headers,
        oauth2_headers,
        raw_headers,
        _client_ip,
    ) = context.legacy_auth()

    read_resource_result: Final = await mcp_read_resource(
        url=params.uri,
        user_api_key_auth=user_api_key_auth,
        mcp_auth_header=mcp_auth_header,
        mcp_servers=mcp_servers,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        client_ip=_client_ip,
    )

    return read_resource_result


def _reject_mcp_proxy_operation() -> NoReturn:
    from mcp.shared.exceptions import MCPError
    from mcp.types import METHOD_NOT_FOUND

    raise MCPError(code=METHOD_NOT_FOUND, message="Operation unavailable on /mcp/proxy")


def prepare_context(
    user_api_key_auth: UserAPIKeyAuth | None = None,
    mcp_auth_header: str | None = None,
    mcp_servers: Sequence[str] | None = None,
    mcp_server_auth_headers: Mapping[str, Mapping[str, str]] | None = None,
    oauth2_headers: Mapping[str, str] | None = None,
    raw_headers: Mapping[str, str] | None = None,
    client_ip: str | None = None,
    mcp_proxy_mode: bool = False,
    wire_compat: WireCompat = WireCompat.LEGACY,
) -> OperationContext:
    return OperationContext(
        _caller=user_api_key_auth,
        mcp_auth_header=mcp_auth_header,
        mcp_servers=tuple(mcp_servers) if mcp_servers is not None else None,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        client_ip=client_ip,
        mcp_proxy_mode=mcp_proxy_mode,
        wire_compat=wire_compat,
    )


GatewayOperation: TypeAlias = (
    AuthorizedToolCall
    | ListToolsRequest
    | CallToolRequest
    | ListPromptsRequest
    | GetPromptRequest
    | ListResourcesRequest
    | ListResourceTemplatesRequest
    | ReadResourceRequest
)
GatewayResult: TypeAlias = (
    ListToolsResult
    | CallToolResult
    | InputRequiredResult
    | ListPromptsResult
    | GetPromptResult
    | ListResourcesResult
    | ListResourceTemplatesResult
    | ReadResourceResult
)


class GatewayOperations:
    def __init__(self, host_progress_callback: ProgressCallback | None = None) -> None:
        self._host_progress_callback = host_progress_callback

    @overload
    async def execute(
        self, operation: AuthorizedToolCall, context: OperationContext
    ) -> CallToolResult | InputRequiredResult: ...

    @overload
    async def execute(self, operation: ListToolsRequest, context: OperationContext) -> ListToolsResult: ...

    @overload
    async def execute(
        self, operation: CallToolRequest, context: OperationContext
    ) -> CallToolResult | InputRequiredResult: ...

    @overload
    async def execute(self, operation: ListPromptsRequest, context: OperationContext) -> ListPromptsResult: ...

    @overload
    async def execute(self, operation: GetPromptRequest, context: OperationContext) -> GetPromptResult: ...

    @overload
    async def execute(self, operation: ListResourcesRequest, context: OperationContext) -> ListResourcesResult: ...

    @overload
    async def execute(
        self, operation: ListResourceTemplatesRequest, context: OperationContext
    ) -> ListResourceTemplatesResult: ...

    @overload
    async def execute(self, operation: ReadResourceRequest, context: OperationContext) -> ReadResourceResult: ...

    async def execute(self, operation: GatewayOperation, context: OperationContext) -> GatewayResult:
        match operation:
            case AuthorizedToolCall():
                auth, token, _servers, server_headers, oauth_headers, headers, _client_ip = context.legacy_auth()
                return await _execute_mcp_tool(
                    name=operation.name,
                    arguments=dict(operation.arguments),  # mutable-ok: existing tool hooks own mutable argument data
                    allowed_mcp_servers=list(operation.allowed_mcp_servers),
                    start_time=operation.start_time,
                    user_api_key_auth=auth,
                    mcp_auth_header=token,
                    mcp_server_auth_headers=server_headers,
                    oauth2_headers=oauth_headers,
                    raw_headers=headers,
                    client_ip=_client_ip,
                    host_progress_callback=operation.host_progress_callback,
                    guardrail_context=operation.guardrail_context,
                    wire_compat=context.wire_compat,
                    **operation.logging_data,
                )
            case ListToolsRequest(params=params):
                return await _execute_handle_list_tools(
                    context, params or PaginatedRequestParams(), self._host_progress_callback
                )
            case CallToolRequest(params=params):
                return await _execute_mcp_server_tool_call(context, params, self._host_progress_callback)
            case ListPromptsRequest(params=params):
                return await _execute_list_prompts(
                    context, params or PaginatedRequestParams(), self._host_progress_callback
                )
            case GetPromptRequest(params=params):
                return await _execute_get_prompt(context, params, self._host_progress_callback)
            case ListResourcesRequest(params=params):
                return await _execute_list_resources(
                    context, params or PaginatedRequestParams(), self._host_progress_callback
                )
            case ListResourceTemplatesRequest(params=params):
                return await _execute_list_resource_templates(
                    context, params or PaginatedRequestParams(), self._host_progress_callback
                )
            case ReadResourceRequest(params=params):
                return await _execute_read_resource(context, params, self._host_progress_callback)
            case _:
                return assert_never(operation)
