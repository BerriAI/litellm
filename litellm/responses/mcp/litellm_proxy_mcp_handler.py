import traceback
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, TypeAlias, overload

from openai.types.chat import ChatCompletionToolParam
from openai.types.responses.function_tool_param import FunctionToolParam
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, Unpack
from typing_extensions import TypedDict as ExtraItemsTypedDict

from litellm._internal_context import nested_tool_call_failure
from litellm._logging import verbose_logger
from litellm.constants import MAXIMUM_TRACEBACK_LINES_TO_LOG
from litellm.litellm_core_utils.allowed_tools_tool_choice import (
    NO_GATEWAY_TOOLS,
    GatewayTool,
    GatewayTools,
    allowed_tools_tool_choice_kept_on_follow_up,
    parse_allowed_tools_tool_choice,
    tool_calls_the_choice_excludes,
)
from litellm.litellm_core_utils.core_helpers import attribute_or_none
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.mcp_server.catalog import catalog_operation, global_manager
from litellm.proxy.mcp_server.result_conversion import url_elicitation_as_error_result
from litellm.proxy.mcp_server.utils import (
    iter_known_server_prefixes,
    iter_known_tool_name_spellings,
    logging_safe_mcp_headers,
    match_known_tool_name,
    split_server_prefix_from_name,
    strip_known_server_prefix,
)
from litellm.responses.main import aresponses  # noqa: TID251  # inner call must skip the MCP gateway that invoked it
from litellm.responses.mcp.tool_precheck import (
    LITELLM_PROXY_MCP_SERVER_URL_PREFIX,
    may_name_litellm_mcp_gateway,
    names_gateway_explicitly,
    names_whole_gateway,
    proxy_path_mcp_name,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.llms.openai import ToolParam as ResponsesToolParam
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import (
    CallTypes,
    Choices,
    ModelResponse,
    StandardLoggingMCPToolCall,
)
from litellm.utils import Rules, function_setup

if TYPE_CHECKING:
    from mcp.types import CallToolResult
    from mcp.types import Tool as MCPTool

    from litellm.proxy.mcp_server.ui_session_utils import GrantedToolsetIds
    from litellm.proxy.utils import ProxyLogging
    from litellm.responses.mcp.direct_url_claims import AllowedTools, DirectTool
    from litellm.responses.mcp.mcp_streaming_iterator import MCPEnhancedStreamingIterator
    from litellm.responses.mcp.openai_mcp_items import OpenAIStreamShaper
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
else:
    MCPTool = Any

# NOTE: We intentionally keep ToolParam as a broad type here to avoid tight coupling
ToolParam: TypeAlias = Mapping[str, object]
SplitTools: TypeAlias = tuple[list[ToolParam], list[Any]]


class MCPToolResult(ExtraItemsTypedDict):
    tool_call_id: str | None
    result: str
    name: str | None
    error: ReadOnly[NotRequired[str]]


class _AresponsesOptions(ExtraItemsTypedDict, total=False, extra_items=object):
    pass


def _failed_tool_result(
    tool_call_id: str | None, message: str, name: str | None, reports_errors: bool
) -> MCPToolResult:
    return (
        MCPToolResult(tool_call_id=tool_call_id, result=message, name=name, error=message)
        if reports_errors
        else MCPToolResult(tool_call_id=tool_call_id, result=message, name=name)
    )


def _registered_mcp_servers() -> Collection[MCPServer]:
    from litellm.proxy.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    return global_mcp_server_manager.get_registry().values()


def _registry_serves(name: str, servers: Collection[MCPServer]) -> bool:
    requested: Final = name.lower()
    return any(
        requested in (known.lower() for known in (*iter_known_server_prefixes(server), server.name))
        or name in (server.access_groups or ())
        for server in servers
    )


async def read_saved_toolset(name: str) -> bool | Exception:
    try:
        from litellm.proxy.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return False
        return await global_mcp_server_manager.get_toolset_by_name_cached(prisma_client, name) is not None
    except Exception as e:
        return e


async def _toolset_exists(name: str) -> bool:
    found: Final = await read_saved_toolset(name)
    if isinstance(found, Exception):
        verbose_logger.debug("Could not resolve '%s' as toolset: %s", name, found)
        return False
    return found


@catalog_operation(global_manager)
async def _gateway_served_names(
    names: Collection[str],
    servers: Callable[[], Collection[MCPServer]] = _registered_mcp_servers,
    toolset_exists: Callable[[str], Awaitable[bool]] = _toolset_exists,
) -> frozenset[str]:
    registered: Final = tuple(servers()) if names else ()
    return frozenset([name for name in names if _registry_serves(name, registered) or await toolset_exists(name)])


async def _served_mcp_path_names(
    tools: Collection[ToolParam], served_names: Callable[[Collection[str]], Awaitable[frozenset[str]]]
) -> frozenset[str]:
    names: Final = frozenset(name for name in map(proxy_path_mcp_name, tools) if name is not None)
    return await served_names(names) if names else frozenset[str]()


_ITEM_FIELDS: Final = TypeAdapter(dict[str, object])
_CONTENT_PARTS: Final = TypeAdapter(tuple[object, ...])


def _output_item_fields(output_item: object) -> Mapping[str, object] | None:
    if isinstance(output_item, BaseModel):
        return output_item.model_dump(exclude_none=True)
    if isinstance(output_item, dict):
        return _ITEM_FIELDS.validate_python(output_item)
    return None


def _is_blank_text_part(part: object) -> bool:
    if not isinstance(part, dict):
        return part == ""
    fields: Final = _ITEM_FIELDS.validate_python(part)
    return fields.get("type") == "output_text" and not fields.get("text")


def _replayed_function_call(fields: Mapping[str, object]) -> Mapping[str, object] | None:
    call_id: Final = fields.get("call_id") or fields.get("id")
    name: Final = fields.get("name")
    if not call_id or not name:
        return None
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": fields.get("arguments")}


def _replayed_message(fields: Mapping[str, object]) -> Mapping[str, object] | None:
    content: Final = fields.get("content", ())
    parts: Final = _CONTENT_PARTS.validate_python(content) if isinstance(content, list) else (content,)
    visible: Final = tuple(part for part in parts if not _is_blank_text_part(part))
    if not visible:
        return None
    return {"type": "message", "role": "assistant", "content": list(visible)}


def _replayed_output_item(output_item: object, preserve_reasoning: bool) -> Mapping[str, object] | None:
    fields: Final = _output_item_fields(output_item)
    if fields is None:
        return None
    match fields.get("type"):
        case "function_call":
            return _replayed_function_call(fields)
        case "reasoning":
            return fields if preserve_reasoning else None
        case "message":
            return _replayed_message(fields)
        case _:
            return None


def _history_items(original_input: str | ResponseInputParam | None) -> tuple[object, ...]:
    if not original_input:
        return ()
    if isinstance(original_input, str):
        return ({"type": "message", "role": "user", "content": original_input},)
    if isinstance(original_input, list):
        return tuple(original_input)
    return (original_input,)


def _served_tool(tool_name: str, server_name: str) -> tuple[MCPServer | None, str]:
    from litellm.proxy.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy.mcp_server.server import (
        _resolve_display_name_to_original,
    )

    mcp_server: Final = global_mcp_server_manager.get_mcp_server_by_name(
        server_name
    ) or global_mcp_server_manager.get_mcp_server_from_tool_name(tool_name)
    resolved_tool_name: Final = _resolve_display_name_to_original(tool_name, [mcp_server]) if mcp_server else tool_name
    return mcp_server, strip_known_server_prefix(resolved_tool_name, mcp_server)


def admitted_by_server_allowed_tools(
    mcp_tools: list[MCPTool], tool_server_map: dict[str, str], allowed_tools: Mapping[str, "AllowedTools"]
) -> tuple[list[MCPTool], dict[str, str]]:
    if not allowed_tools:
        return mcp_tools, tool_server_map
    admitted: Final = frozenset(
        tool.name for tool in mcp_tools if _server_admits(tool, tool_server_map.get(tool.name, ""), allowed_tools)
    )
    return (
        [tool for tool in mcp_tools if tool.name in admitted],
        {name: server_name for name, server_name in tool_server_map.items() if name in admitted},
    )


def _server_admits(tool: MCPTool, server_name: str, allowed_tools: Mapping[str, "AllowedTools"]) -> bool:
    server, tool_name = _served_tool(tool.name, server_name)
    allowed: Final = None if server is None else allowed_tools.get(server.server_id)
    return allowed is None or allowed.admits(tool.model_copy(update={"name": tool_name}))


@dataclass(frozen=True, slots=True)
class _LabelScope:
    label: str
    whole_gateway: bool = False
    server_ids: frozenset[str] = frozenset()
    toolset_tools: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    allowed_tools: frozenset[str] | None = None

    def serves(self, server: MCPServer | None, tool_name: str, listed_name: str) -> bool:
        if not self._allows(server, tool_name, listed_name):
            return False
        if self.whole_gateway:
            return True
        if server is None:
            return False
        toolset_names: Final = self.toolset_tools.get(server.server_id)
        return server.server_id in self.server_ids or (toolset_names is not None and tool_name in toolset_names)

    def _allows(self, server: MCPServer | None, tool_name: str, listed_name: str) -> bool:
        if self.allowed_tools is None or listed_name in self.allowed_tools:
            return True
        return server is not None and match_known_tool_name(tool_name, server, self.allowed_tools) is not None


_OBJECT_LIST: Final = TypeAdapter(list[object])


def _listed_allowed_tools(tool: Mapping[str, object]) -> frozenset[str] | None:
    try:
        allowed_tools: Final = _OBJECT_LIST.validate_python(tool.get("allowed_tools"))
    except ValidationError:
        return None
    return frozenset(name for name in allowed_tools if isinstance(name, str)) if allowed_tools else None


def _gateway_tool(name: str, server: MCPServer | None, tool_name: str, scopes: Iterable[_LabelScope]) -> GatewayTool:
    spellings: Final = () if server is None else tuple(iter_known_tool_name_spellings(tool_name, server))
    return GatewayTool(
        names=frozenset((name, tool_name, *spellings)),
        server_labels=frozenset(scope.label for scope in scopes if scope.serves(server, tool_name, name)),
    )


class LiteLLM_Proxy_MCP_Handler:
    """
    Helper class with static methods for MCP integration with Responses API.

    This handles when a user passes mcp server_url="litellm_proxy" in their tools.
    """

    @staticmethod
    def get_parent_request_tags(kwargs: dict[str, Any] | None) -> list[str]:
        """Tags from the parent LLM request, using the same extraction logic as standard logging (incl. User-Agent)."""
        if not kwargs:
            return []
        from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

        litellm_params: Final = kwargs.get("litellm_params") or kwargs
        proxy_server_request = litellm_params.get("proxy_server_request") or kwargs.get("proxy_server_request") or {}
        return StandardLoggingPayloadSetup.get_request_tags(
            litellm_params=litellm_params,
            proxy_server_request=proxy_server_request,
        )

    _get_parent_request_tags = get_parent_request_tags

    @staticmethod
    def should_use_litellm_mcp_gateway(tools: Iterable[ToolParam] | None) -> bool:
        return may_name_litellm_mcp_gateway(tools)

    _should_use_litellm_mcp_gateway = should_use_litellm_mcp_gateway

    @staticmethod
    def _parse_mcp_tools(tools: Iterable[Mapping[str, object]] | None) -> SplitTools:
        items: Final = tuple(tools or ())
        gateway_tools: Final[list[ToolParam]] = [tool for tool in items if names_gateway_explicitly(tool)]
        other_tools: Final[list[ToolParam]] = [tool for tool in items if not names_gateway_explicitly(tool)]
        return gateway_tools, other_tools

    @staticmethod
    async def split_mcp_tools(
        tools: Iterable[Mapping[str, object]] | None,
        served_names: Callable[[Collection[str]], Awaitable[frozenset[str]]] = _gateway_served_names,
    ) -> SplitTools:
        items: Final = tuple(tools or ())
        served: Final = await _served_mcp_path_names(items, served_names)
        return LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(
            [
                {**tool, "server_url": f"{LITELLM_PROXY_MCP_SERVER_URL_PREFIX}{name}"}
                if (name := proxy_path_mcp_name(tool)) in served
                else tool
                for tool in items
            ]
        )

    _split_mcp_tools = split_mcp_tools

    @staticmethod
    @catalog_operation(global_manager)
    async def routes_through_gateway(
        tools: Iterable[Mapping[str, object]] | None,
        served_names: Callable[[Collection[str]], Awaitable[frozenset[str]]] = _gateway_served_names,
    ) -> tuple[bool, ...]:
        items: Final = tuple(tools or ())
        served: Final = await _served_mcp_path_names(items, served_names)
        return tuple(names_gateway_explicitly(tool) or proxy_path_mcp_name(tool) in served for tool in items)

    @staticmethod
    async def _apply_toolset_permissions(
        resolved_toolset_ids: Sequence[str],
        resolved_mcp_servers: Sequence[str],
        user_api_key_auth: "UserAPIKeyAuth",
    ) -> "UserAPIKeyAuth":
        """Apply resolved toolset permissions to user_api_key_auth and return updated auth."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        try:
            from litellm.proxy.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            tool_permissions: Final = await global_mcp_server_manager.resolve_toolset_tool_permissions(
                toolset_ids=list(resolved_toolset_ids)
            )
            all_server_ids: Final = list(set(tool_permissions.keys()) | set(resolved_mcp_servers))
            existing_op: Final = user_api_key_auth.object_permission
            if existing_op is not None:
                merged_tool_perms: Final = dict(existing_op.mcp_tool_permissions or {})
                for server_id, tool_names in tool_permissions.items():
                    existing_tools = merged_tool_perms.get(server_id, [])
                    merged_tool_perms[server_id] = list(set(existing_tools) | set(tool_names))
                updated_op = existing_op.model_copy(
                    update={
                        "mcp_servers": all_server_ids,
                        "mcp_tool_permissions": merged_tool_perms,
                        "mcp_toolsets": [],
                    }
                )
            else:
                updated_op = LiteLLM_ObjectPermissionTable(
                    object_permission_id="toolset-scope",
                    mcp_servers=all_server_ids,
                    mcp_tool_permissions=tool_permissions,
                )
            return user_api_key_auth.model_copy(
                update={"object_permission": updated_op, "mcp_explicit_grants_only": True}
            )
        except Exception as _e:
            verbose_logger.debug("Could not apply toolset permissions: %s", _e)
            return user_api_key_auth

    @staticmethod
    async def _resolve_proxy_mcp_name(
        name: str,
        user_api_key_auth: "UserAPIKeyAuth | None",
        granted_toolsets: "GrantedToolsetIds | None",
    ) -> tuple[Literal["server", "toolset", "denied_toolset"], str]:
        from litellm.proxy.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        if global_mcp_server_manager.get_mcp_server_by_name(name):
            return ("server", name)
        try:
            from litellm.proxy.proxy_server import prisma_client

            if prisma_client is None:
                return ("server", name)
            toolset: Final = await global_mcp_server_manager.get_toolset_by_name_cached(prisma_client, name)
            if toolset is None:
                return ("server", name)
            from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import (
                has_team_unbound_mcp_admin_view,
            )
            from litellm.proxy.mcp_server.ui_session_utils import (
                granted_toolset_ids,
            )

            denied: Final = (
                user_api_key_auth is not None
                and not has_team_unbound_mcp_admin_view(user_api_key_auth)
                and toolset.toolset_id not in await (granted_toolsets or granted_toolset_ids)(user_api_key_auth)
            )
        except Exception as _e:
            verbose_logger.debug("Could not resolve '%s' as toolset: %s", name, _e)
            return ("server", name)
        if not denied:
            return ("toolset", toolset.toolset_id)
        if await LiteLLM_Proxy_MCP_Handler._holds_access_group_named(name, user_api_key_auth):
            return ("server", name)
        verbose_logger.debug("Key does not have access to toolset '%s', denying it.", name)
        return ("denied_toolset", name)

    @staticmethod
    async def _holds_access_group_named(name: str, user_api_key_auth: "UserAPIKeyAuth | None") -> bool:
        from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        group_server_ids: Final = await MCPRequestHandler.get_mcp_servers_from_access_groups([name])
        return bool(group_server_ids) and not frozenset(group_server_ids).isdisjoint(
            await global_mcp_server_manager.get_allowed_mcp_servers(user_api_key_auth)
        )

    @staticmethod
    @catalog_operation(global_manager)
    async def _get_mcp_tools_from_manager(
        user_api_key_auth: "UserAPIKeyAuth | None",
        mcp_tools_with_litellm_proxy: Iterable[Mapping[str, object]] | None,
        litellm_trace_id: str | None = None,
        mcp_auth_header: str | None = None,
        mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
        request_tags: list[str] | None = None,
        raw_headers: dict[str, str] | None = None,
        granted_toolsets: "GrantedToolsetIds | None" = None,
        request_trace_id: str | None = None,
    ) -> tuple[list[MCPTool], list[str]]:
        """
        Get available tools from the MCP server manager.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_tools_with_litellm_proxy: ToolParam objects with server_url starting with "litellm_proxy"
            mcp_auth_header: Optional deprecated auth header for MCP servers
            mcp_server_auth_headers: Optional server-specific auth headers (e.g. from x-mcp-{alias}-*)

        Returns:
            List of MCP tools
            List names of allowed MCP servers
        """
        from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import is_mcp_admitted_user_subject
        from litellm.proxy.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy.mcp_server.operations import shared_mcp_listing
        from litellm.proxy.mcp_server.server import (
            _get_allowed_mcp_servers_from_mcp_server_names,
            _get_tools_from_mcp_servers,
            raise_generic_mcp_access_denial,
        )

        mcp_servers: Final = [
            server_url.split("/")[-1]
            for _tool in (mcp_tools_with_litellm_proxy or ())
            for server_url in (_tool.get("server_url", "") if isinstance(_tool, dict) else "",)
            if isinstance(server_url, str) and server_url.startswith(LITELLM_PROXY_MCP_SERVER_URL_PREFIX)
        ]

        # Resolve toolset names: collect all toolset IDs first, then apply their
        # combined permissions in a single pass so multiple toolsets are unioned
        # rather than the last one overwriting the others.
        resolutions: Final = tuple(
            [
                await LiteLLM_Proxy_MCP_Handler._resolve_proxy_mcp_name(name, user_api_key_auth, granted_toolsets)
                for name in mcp_servers
            ]
        )
        resolved_mcp_servers: Final = tuple(value for kind, value in resolutions if kind == "server")
        resolved_toolset_ids: Final = tuple(value for kind, value in resolutions if kind == "toolset")
        denied_toolset_names: Final = tuple(value for kind, value in resolutions if kind == "denied_toolset")

        asks_whole_gateway: Final = any(map(names_whole_gateway, mcp_tools_with_litellm_proxy or ()))
        if denied_toolset_names and not (asks_whole_gateway or resolved_mcp_servers or resolved_toolset_ids):
            raise_generic_mcp_access_denial(denied_toolset_names)

        async def scope_servers(
            auth: "UserAPIKeyAuth | None", server_filter: Sequence[str] | None
        ) -> Sequence[MCPServer]:
            server_ids: Final = await global_mcp_server_manager.get_allowed_mcp_servers(auth)
            servers: Final = await _get_allowed_mcp_servers_from_mcp_server_names(
                mcp_servers=server_filter,
                allowed_mcp_servers=global_mcp_server_manager.get_mcp_servers_from_ids(server_ids),
            )
            return tuple(server for server in servers if server is not None)

        def server_names(servers: Sequence[MCPServer]) -> Sequence[str]:
            names: Final = (
                getattr(server, "server_name", None) or getattr(server, "alias", None) or getattr(server, "name", None)
                for server in servers
            )
            return tuple(dict.fromkeys(name for name in names if isinstance(name, str)))

        async def list_scope(
            auth: "UserAPIKeyAuth | None", server_filter: Sequence[str] | None
        ) -> tuple[Sequence[MCPTool], Sequence[str]]:
            listing: Final = await _get_tools_from_mcp_servers(
                user_api_key_auth=None if auth is None else auth.model_copy(update={"budget_reservation": None}),
                mcp_auth_header=mcp_auth_header,
                mcp_servers=list(server_filter) if server_filter is not None else None,
                mcp_server_auth_headers=mcp_server_auth_headers,
                log_list_tools_to_spendlogs=True,
                list_tools_log_source="responses",
                litellm_trace_id=litellm_trace_id,
                request_trace_id=request_trace_id,
                request_tags=request_tags,
                raw_headers=raw_headers,
            )
            return listing.tools, server_names(await scope_servers(auth, server_filter))

        if resolved_toolset_ids and is_mcp_admitted_user_subject(user_api_key_auth) and user_api_key_auth is not None:
            # Rewriting object_permission does not constrain an admitted user's team sources.
            scopes: Final = (
                *(((user_api_key_auth, resolved_mcp_servers),) if resolved_mcp_servers else ()),
                *(
                    (user_api_key_auth.model_copy(update={"mcp_toolset_id": toolset_id}), None)
                    for toolset_id in dict.fromkeys(resolved_toolset_ids)
                ),
            )
            servers: Final = tuple(
                chain.from_iterable([await scope_servers(auth, server_filter) for auth, server_filter in scopes])
            )
            with shared_mcp_listing(servers):
                listings: Final = [await list_scope(auth, server_filter) for auth, server_filter in scopes]
            return list(chain.from_iterable(tools for tools, _ in listings)), list(server_names(servers))

        if resolved_toolset_ids and user_api_key_auth is not None:
            scoped_auth: Final = await LiteLLM_Proxy_MCP_Handler._apply_toolset_permissions(
                resolved_toolset_ids=resolved_toolset_ids,
                resolved_mcp_servers=resolved_mcp_servers,
                user_api_key_auth=user_api_key_auth,
            )
            tools, names = await list_scope(scoped_auth, None)
        else:
            server_filter: Final = None if resolved_toolset_ids else (resolved_mcp_servers or None)
            tools, names = await list_scope(user_api_key_auth, server_filter)
        return list(tools), list(names)

    @staticmethod
    def _deduplicate_mcp_tools(
        mcp_tools: list[MCPTool], allowed_mcp_servers: list[str]
    ) -> tuple[list[MCPTool], dict[str, str]]:
        """
        Deduplicate MCP tools by name, keeping the first occurrence of each tool.

        Args:
            mcp_tools: List of MCP tools that may contain duplicates

        Returns:
            List of deduplicated MCP tools
            The returned dictionary maps each tool_name to the server_name
        """
        seen_names: Final[set[str]] = set()
        deduplicated_tools: Final[list[MCPTool]] = []
        tool_server_map: Final[dict[str, str]] = {}

        for tool in mcp_tools:
            if isinstance(tool, dict):
                tool_name = tool.get("name")
            else:
                tool_name = getattr(tool, "name", None)

            if tool_name and tool_name not in seen_names:
                seen_names.add(tool_name)
                deduplicated_tools.append(tool)
                if len(allowed_mcp_servers) == 1:
                    tool_server_map[tool_name] = allowed_mcp_servers[0]
                else:
                    _, tool_server_map[tool_name] = split_server_prefix_from_name(tool_name)

        return deduplicated_tools, tool_server_map

    @staticmethod
    def _filter_mcp_tools_by_allowed_tools(
        mcp_tools: list[MCPTool], mcp_tools_with_litellm_proxy: Sequence[Mapping[str, object]]
    ) -> list[MCPTool]:
        """Filter MCP tools based on allowed_tools parameter from the original tool configs."""
        # Collect all allowed tool names from all MCP tool configs
        allowed_tool_names: Final[set[str]] = set()
        for tool_config in mcp_tools_with_litellm_proxy:
            if isinstance(tool_config, dict) and "allowed_tools" in tool_config:
                allowed_tools = tool_config.get("allowed_tools", [])
                if isinstance(allowed_tools, list):
                    allowed_tool_names.update(allowed_tools)

        # If no allowed_tools specified, return all tools
        if not allowed_tool_names:
            return mcp_tools

        # Filter tools based on allowed names
        filtered_tools: Final = [
            mcp_tool
            for mcp_tool in mcp_tools
            for tool_name in (mcp_tool.get("name") if isinstance(mcp_tool, dict) else getattr(mcp_tool, "name", None),)
            if tool_name
            and (tool_name in allowed_tool_names or split_server_prefix_from_name(tool_name)[0] in allowed_tool_names)
        ]

        return filtered_tools

    @staticmethod
    async def process_mcp_tools_to_openai_format(
        user_api_key_auth: "UserAPIKeyAuth | None",
        mcp_tools_with_litellm_proxy: Sequence[Mapping[str, object]],
        litellm_trace_id: str | None = None,
        request_tags: list[str] | None = None,
    ) -> tuple[list[FunctionToolParam], dict[str, str]]:
        """
        Centralized method to process MCP tools through the complete pipeline.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_tools_with_litellm_proxy: ToolParam objects with server_url starting with "litellm_proxy"
            litellm_trace_id: Optional trace ID for linking list_mcp_tools spend logs to parent request

        Returns:
            List of tools in OpenAI format ready to be sent to the LLM
            The returned dictionary maps each tool_name to the server_name
        """
        (
            deduplicated_mcp_tools,
            tool_server_map,
        ) = await LiteLLM_Proxy_MCP_Handler.process_mcp_tools_without_openai_transform(
            user_api_key_auth,
            mcp_tools_with_litellm_proxy,
            litellm_trace_id=litellm_trace_id,
            request_tags=request_tags,
        )

        openai_tools: Final = LiteLLM_Proxy_MCP_Handler.transform_mcp_tools_to_openai(deduplicated_mcp_tools)

        return openai_tools, tool_server_map

    _process_mcp_tools_to_openai_format = process_mcp_tools_to_openai_format

    @staticmethod
    async def process_mcp_tools_without_openai_transform(
        user_api_key_auth: "UserAPIKeyAuth | None",
        mcp_tools_with_litellm_proxy: Sequence[Mapping[str, object]],
        litellm_trace_id: str | None = None,
        mcp_auth_header: str | None = None,
        mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
        request_tags: list[str] | None = None,
        raw_headers: dict[str, str] | None = None,
        request_trace_id: str | None = None,
    ) -> tuple[list[MCPTool], dict[str, str]]:
        """
        Process MCP tools through filtering and deduplication pipeline without OpenAI transformation.
        This is useful for cases where we need the original MCP tool objects (e.g., for events).

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_tools_with_litellm_proxy: ToolParam objects with server_url starting with "litellm_proxy"
            mcp_auth_header: Optional deprecated auth header for MCP servers
            mcp_server_auth_headers: Optional server-specific auth headers (e.g. from x-mcp-{alias}-*)

        Returns:
            List of filtered and deduplicated MCP tools in their original format
        """
        if not mcp_tools_with_litellm_proxy:
            return [], {}

        typed_user_api_key_auth: Final[UserAPIKeyAuth | None] = user_api_key_auth

        # Step 1: Fetch MCP tools from manager
        (
            mcp_tools_fetched,
            allowed_mcp_servers,
        ) = await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
            user_api_key_auth=typed_user_api_key_auth,
            mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
            litellm_trace_id=litellm_trace_id,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            request_tags=request_tags,
            raw_headers=raw_headers,
            request_trace_id=request_trace_id,
        )

        # Step 2: Filter tools based on allowed_tools parameter
        filtered_mcp_tools: Final = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
            mcp_tools=mcp_tools_fetched,
            mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
        )

        # Step 3: Deduplicate tools after filtering
        (
            deduplicated_mcp_tools,
            tool_server_map,
        ) = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(filtered_mcp_tools, allowed_mcp_servers)

        return deduplicated_mcp_tools, tool_server_map

    _process_mcp_tools_without_openai_transform = process_mcp_tools_without_openai_transform

    @overload
    @staticmethod
    def transform_mcp_tools_to_openai(
        mcp_tools: Sequence[MCPTool],
        target_format: Literal["responses"] = ...,
    ) -> list[FunctionToolParam]: ...

    @overload
    @staticmethod
    def transform_mcp_tools_to_openai(
        mcp_tools: Sequence[MCPTool],
        target_format: Literal["chat"],
    ) -> list[ChatCompletionToolParam]: ...

    @staticmethod
    def transform_mcp_tools_to_openai(
        mcp_tools: Sequence[MCPTool],
        target_format: Literal["responses", "chat"] = "responses",
    ) -> Sequence[FunctionToolParam | ChatCompletionToolParam]:
        """Transform MCP tools to OpenAI-compatible format."""
        from litellm.mcp_client.tools import (
            transform_mcp_tool_to_openai_responses_api_tool,
            transform_mcp_tool_to_openai_tool,
        )

        openai_tools: Final = [
            transform_mcp_tool_to_openai_tool(mcp_tool)
            if target_format == "chat"
            else transform_mcp_tool_to_openai_responses_api_tool(mcp_tool)
            for mcp_tool in mcp_tools
        ]

        return openai_tools

    _transform_mcp_tools_to_openai = transform_mcp_tools_to_openai

    @staticmethod
    def should_auto_execute_tools(
        mcp_tools_with_litellm_proxy: Sequence[Mapping[str, object]],
    ) -> bool:
        """Check if we should auto-execute tool calls.

        Auto-execution requires EVERY MCP reference to opt in with
        ``require_approval="never"``. A single reference that requires approval
        ("always", "manual", the object form, or an unset value) disables
        auto-execution for the whole request. This fails closed: when an
        approval-required reference shares a request with a "never" one, the
        model's tool calls are returned to the caller instead of being run, so
        an approval-gated tool can never be invoked without approval. Returns
        False for an empty list.
        """
        references: Final = list(mcp_tools_with_litellm_proxy or [])
        if not references:
            return False
        for tool in references:
            approval = (
                tool.get("require_approval") if isinstance(tool, dict) else getattr(tool, "require_approval", None)
            )
            if approval != "never":
                return False
        return True

    _should_auto_execute_tools = should_auto_execute_tools

    @staticmethod
    async def _label_scope(
        tool: Mapping[str, object],
        servers: Sequence[MCPServer],
        user_api_key_auth: "UserAPIKeyAuth | None",
        granted_toolsets: "GrantedToolsetIds | None",
    ) -> _LabelScope | None:
        from litellm.proxy.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy.mcp_server.server import (
            _get_allowed_mcp_servers_from_mcp_server_names,
        )

        label: Final = tool.get("server_label")
        server_url: Final = tool.get("server_url")
        if not isinstance(label, str) or not isinstance(server_url, str):
            return None
        allowed_tools: Final = _listed_allowed_tools(tool)
        if names_whole_gateway(tool):
            return _LabelScope(label=label, whole_gateway=True, allowed_tools=allowed_tools)
        kind, value = await LiteLLM_Proxy_MCP_Handler._resolve_proxy_mcp_name(
            server_url.split("/")[-1], user_api_key_auth, granted_toolsets
        )
        match kind:
            case "server":
                resolved: Final = await _get_allowed_mcp_servers_from_mcp_server_names([value], list(servers))
                return _LabelScope(
                    label=label,
                    server_ids=frozenset(server.server_id for server in resolved),
                    allowed_tools=allowed_tools,
                )
            case "toolset":
                permissions: Final = await global_mcp_server_manager.resolve_toolset_tool_permissions(
                    toolset_ids=[value]
                )
                return _LabelScope(
                    label=label,
                    toolset_tools=MappingProxyType({key: tuple(names) for key, names in permissions.items()}),
                    allowed_tools=allowed_tools,
                )
            case "denied_toolset":
                return _LabelScope(label=label)

    @staticmethod
    async def gateway_tools(
        tool_choice: object,
        tool_server_map: Mapping[str, str],
        mcp_tools_with_litellm_proxy: Iterable[Mapping[str, object]],
        user_api_key_auth: "UserAPIKeyAuth | None",
        granted_toolsets: "GrantedToolsetIds | None" = None,
    ) -> GatewayTools:
        if parse_allowed_tools_tool_choice(tool_choice) is None:
            return NO_GATEWAY_TOOLS
        served: Final = {name: _served_tool(name, server_name) for name, server_name in tool_server_map.items()}
        servers: Final = {server.server_id: server for server, _ in served.values() if server is not None}
        scopes: Final = tuple(
            [
                await LiteLLM_Proxy_MCP_Handler._label_scope(
                    tool, tuple(servers.values()), user_api_key_auth, granted_toolsets
                )
                for tool in mcp_tools_with_litellm_proxy
            ]
        )
        label_scopes: Final = tuple(scope for scope in scopes if scope is not None)
        return GatewayTools(
            server_labels=frozenset(scope.label for scope in label_scopes),
            tools=MappingProxyType(
                {
                    name: _gateway_tool(name, server, tool_name, label_scopes)
                    for name, (server, tool_name) in served.items()
                }
            ),
        )

    @staticmethod
    def tool_calls_to_run(tool_calls: Sequence[object], tool_choice: object, gateway: GatewayTools) -> Sequence[object]:
        names: Final = tuple(
            LiteLLM_Proxy_MCP_Handler.extract_tool_call_details(tool_call)[0] or "" for tool_call in tool_calls
        )
        return () if tool_calls_the_choice_excludes(tool_choice, names, gateway) else tool_calls

    @staticmethod
    def extract_tool_calls_from_response(response: ResponsesAPIResponse) -> list[object]:
        """Extract tool calls from the response output."""
        tool_calls: Final[list[object]] = []
        for output_item in response.output:
            # Check if this is a function call output item
            if isinstance(output_item, dict) and output_item.get("type") == "function_call":
                tool_calls.append(output_item)
            elif hasattr(output_item, "type") and getattr(output_item, "type") == "function_call":
                # Handle pydantic model case
                tool_calls.append(output_item)

        return tool_calls

    _extract_tool_calls_from_response = extract_tool_calls_from_response

    @staticmethod
    def _continued_choice(response: ModelResponse) -> Choices | None:
        choices: Final = tuple(choice for choice in response.choices if isinstance(choice, Choices))
        return next((choice for choice in choices if choice.message.tool_calls), next(iter(choices), None))

    @staticmethod
    def extract_tool_calls_from_chat_response(response: ModelResponse) -> list[object]:
        """Extract the tool calls of the first choice that made any, since the follow-up turn continues it alone."""
        continued_choice: Final = LiteLLM_Proxy_MCP_Handler._continued_choice(response)
        if continued_choice is None:
            return []
        return [tool_call.model_dump() for tool_call in continued_choice.message.tool_calls or ()]

    _extract_tool_calls_from_chat_response = extract_tool_calls_from_chat_response

    @staticmethod
    def extract_tool_call_details(
        tool_call: object,
    ) -> tuple[str | None, str | None, str | None]:
        """Extract tool name, arguments, and call_id from a tool call."""
        if isinstance(tool_call, dict):
            tool_call_id = tool_call.get("call_id") or tool_call.get("id")

            # OpenAI chat completions wrap tool info under a `function` block
            function_block: Final = tool_call.get("function")
            if isinstance(function_block, dict):
                tool_name = function_block.get("name")
                tool_arguments = function_block.get("arguments")
            else:
                tool_name = tool_call.get("name")
                # Anthropic tool_use blocks carry the arguments under `input`
                tool_arguments = tool_call.get("arguments")
                if tool_arguments is None:
                    tool_arguments = tool_call.get("input")
        else:
            tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", None)

            function_obj: Final[object] = getattr(tool_call, "function", None)
            if function_obj is not None:
                tool_name = getattr(function_obj, "name", None)
                tool_arguments = getattr(function_obj, "arguments", None)
            else:
                tool_name = getattr(tool_call, "name", None)
                tool_arguments = getattr(tool_call, "arguments", None)
                if tool_arguments is None:
                    tool_arguments = getattr(tool_call, "input", None)

        return tool_name, tool_arguments, tool_call_id

    _extract_tool_call_details = extract_tool_call_details

    @staticmethod
    def _parse_tool_arguments(tool_arguments: str | None) -> dict[str, object]:
        """Parse tool arguments, handling both string and dict formats."""
        import json

        if isinstance(tool_arguments, str):
            try:
                return json.loads(tool_arguments)
            except json.JSONDecodeError:
                return {}
        else:
            return tool_arguments or {}

    @staticmethod
    def _parse_mcp_result(result: "CallToolResult") -> str:
        """Parse MCP tool call result and extract meaningful content."""
        if not result or not hasattr(result, "content") or not result.content:
            return "Tool executed successfully"

        # Import MCP content types for isinstance checks
        try:
            from mcp.types import EmbeddedResource, ImageContent, TextContent
        except ImportError:
            # Fallback to generic handling if MCP types not available
            return "Tool executed successfully"

        text_parts: Final = [
            str(content_item.text) for content_item in result.content if isinstance(content_item, TextContent)
        ]
        other_content_types: Final = [
            "Image"
            if isinstance(content_item, ImageContent)
            else "EmbeddedResource"
            if isinstance(content_item, EmbeddedResource)
            else type(content_item).__name__
            for content_item in result.content
            if not isinstance(content_item, TextContent)
        ]

        # Combine text parts if any
        result_text = " ".join(text_parts) if text_parts else ""

        # Add info about other content types
        if other_content_types:
            other_info: Final = f"[Generated {', '.join(other_content_types)}]"
            result_text = f"{result_text} {other_info}".strip()

        return result_text or "Tool executed successfully"

    @staticmethod
    @catalog_operation(global_manager)
    async def execute_tool_calls(
        tool_server_map: dict[str, str],
        tool_calls: Sequence[object],
        user_api_key_auth: "UserAPIKeyAuth | None",
        mcp_auth_header: str | None = None,
        mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
        oauth2_headers: dict[str, str] | None = None,
        raw_headers: dict[str, str] | None = None,
        litellm_call_id: str | None = None,
        litellm_trace_id: str | None = None,
        request_tags: list[str] | None = None,
        guardrail_context: Mapping[str, object] | None = None,
        served_tools: Sequence[MCPTool] | None = None,
        request_trace_id: str | None = None,
        direct_tools: "Mapping[str, DirectTool] | None" = None,
        reports_errors: bool = False,
    ) -> list[MCPToolResult]:
        """Execute tool calls and return results."""
        from fastapi import HTTPException

        from litellm._uuid import uuid
        from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
        from litellm.proxy.hooks.parallel_request_limiter_v3 import mark_parallel_slot_reused_by_tool_calls
        from litellm.proxy.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy.proxy_server import proxy_logging_obj
        from litellm.responses.mcp.direct_url_claims import unregistered_mcp_server

        mark_parallel_slot_reused_by_tool_calls()
        tool_results: Final[list[MCPToolResult]] = []
        tool_call_id: str | None = None
        rules_obj: Final = Rules()
        logging_safe_headers: Final = logging_safe_mcp_headers(raw_headers)
        typed_user_api_key_auth: Final[UserAPIKeyAuth | None] = user_api_key_auth
        for tool_call in tool_calls:
            logging_request_data: dict[str, object] = {}
            tool_name: str | None = None
            try:
                (
                    tool_name,
                    tool_arguments,
                    tool_call_id,
                ) = LiteLLM_Proxy_MCP_Handler.extract_tool_call_details(tool_call)

                if not tool_name:
                    verbose_logger.warning("Tool call missing name: %s", tool_call)
                    continue

                parsed_arguments = LiteLLM_Proxy_MCP_Handler._parse_tool_arguments(tool_arguments)

                # Import here to avoid circular import
                from litellm.proxy.proxy_server import proxy_logging_obj

                direct_tool = (direct_tools or {}).get(tool_name)
                server_name = direct_tool.server.server_label if direct_tool else tool_server_map[tool_name]

                mcp_server, sanitized_tool_name = (
                    (None, direct_tool.tool.name) if direct_tool else _served_tool(tool_name, server_name)
                )

                start_time = datetime.now()
                logging_input: Sequence[Mapping[str, object]] = [
                    {
                        "role": "tool",
                        "content": {
                            "tool_name": sanitized_tool_name,
                            "arguments": parsed_arguments,
                        },
                    }
                ]
                tool_logging_call_id = (
                    f"{litellm_call_id}_{tool_call_id or uuid.uuid4()}" if litellm_call_id else str(uuid.uuid4())
                )
                logging_metadata: dict[str, object] = {
                    "tool_call_id": tool_call_id,
                    "tool_name": sanitized_tool_name,
                    "server_name": server_name,
                    "headers": logging_safe_headers,
                }
                logging_request_data = {
                    "model": f"MCP: {tool_name}",
                    "metadata": logging_metadata,
                    "input": logging_input,
                    "call_type": CallTypes.call_mcp_tool.value,
                    "litellm_call_id": tool_logging_call_id,
                    # Add proxy_server_request with arguments for callback logging
                    "proxy_server_request": {
                        "url": "/mcp/tools/call",
                        "method": "POST",
                        "headers": logging_safe_headers,
                        "body": {
                            "name": sanitized_tool_name,
                            "arguments": parsed_arguments,
                        },
                    },
                }
                if litellm_trace_id:
                    logging_request_data["litellm_trace_id"] = litellm_trace_id
                if request_trace_id:
                    logging_metadata["existing_trace_id"] = request_trace_id
                if request_tags:
                    logging_metadata["tags"] = request_tags
                if typed_user_api_key_auth is not None:
                    from litellm.proxy.litellm_pre_call_utils import (
                        LiteLLMProxyRequestSetup,
                    )

                    LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
                        data=logging_request_data,
                        user_api_key_dict=typed_user_api_key_auth.model_copy(update={"budget_reservation": None}),
                        _metadata_variable_name="metadata",
                    )
                    user_identifier = getattr(typed_user_api_key_auth, "end_user_id", None) or getattr(
                        typed_user_api_key_auth, "user_id", None
                    )
                    if user_identifier:
                        logging_request_data["user"] = user_identifier

                logging_metadata["guardrails"] = TypeAdapter(list[object]).validate_python(
                    TypeAdapter(Mapping[str, object])
                    .validate_python((guardrail_context or {}).get("metadata") or {})
                    .get("guardrails")
                    or ()
                )

                litellm_logging_obj: LiteLLMLoggingObj | None = None
                try:
                    litellm_logging_obj, _ = function_setup(
                        original_function="call_mcp_tool",
                        rules_obj=rules_obj,
                        start_time=start_time,
                        **logging_request_data,
                    )
                except Exception as logging_error:
                    verbose_logger.debug(
                        "Failed to initialize logging for MCP tool call %s: %s",
                        tool_name,
                        logging_error,
                    )
                    litellm_logging_obj = None

                logging_request_data["litellm_logging_obj"] = litellm_logging_obj
                logging_request_data["arguments"] = parsed_arguments

                if litellm_logging_obj:
                    try:
                        litellm_logging_obj.pre_call(
                            input=logging_input,
                            api_key="",
                        )
                    except Exception:
                        verbose_logger.exception("Failed to run pre_call for MCP tool logging")

                standard_logging_mcp_tool_call: StandardLoggingMCPToolCall = {
                    "name": sanitized_tool_name,
                    "arguments": parsed_arguments,
                    "namespaced_tool_name": tool_name,
                }
                if mcp_server:
                    mcp_info = mcp_server.mcp_info or {}
                    standard_logging_mcp_tool_call["mcp_server_name"] = (
                        mcp_info.get("server_name") or getattr(mcp_server, "server_name", None) or server_name
                    )
                    logo_url = mcp_info.get("logo_url")
                    if logo_url:
                        standard_logging_mcp_tool_call["mcp_server_logo_url"] = logo_url
                    cost_info = mcp_info.get("mcp_server_cost_info")
                    if cost_info:
                        standard_logging_mcp_tool_call["mcp_server_cost_info"] = cost_info

                if litellm_logging_obj:
                    litellm_logging_obj.model_call_details["mcp_tool_call_metadata"] = standard_logging_mcp_tool_call
                    litellm_logging_obj.model = f"MCP: {tool_name}"
                    litellm_logging_obj.call_type = CallTypes.call_mcp_tool.value

                result = await url_elicitation_as_error_result(
                    global_mcp_server_manager.call_unregistered_server_tool(
                        server=unregistered_mcp_server(direct_tool.server),
                        tool=direct_tool.tool,
                        arguments=parsed_arguments,
                        call=direct_tool.client.call_tool,
                        user_api_key_auth=typed_user_api_key_auth,
                        proxy_logging_obj=proxy_logging_obj,
                        raw_headers=raw_headers,
                        litellm_logging_obj=litellm_logging_obj,
                        guardrail_context=guardrail_context,
                    )
                    if direct_tool
                    else global_mcp_server_manager.call_tool(
                        server_name=server_name,
                        name=sanitized_tool_name,
                        arguments=parsed_arguments,
                        user_api_key_auth=typed_user_api_key_auth,
                        mcp_auth_header=mcp_auth_header,
                        mcp_server_auth_headers=mcp_server_auth_headers,
                        oauth2_headers=oauth2_headers,
                        raw_headers=raw_headers,
                        proxy_logging_obj=proxy_logging_obj,
                        litellm_logging_obj=litellm_logging_obj,
                        guardrail_context=guardrail_context,
                        listed_tool=(
                            next((tool for tool in served_tools if tool.name == tool_name), None)
                            if served_tools is not None
                            else ...
                        ),
                    )
                )

                if proxy_logging_obj:
                    result = await proxy_logging_obj.post_mcp_call_hook(
                        response=result,
                        request_data={
                            **(
                                litellm_logging_obj.model_call_details
                                if litellm_logging_obj
                                else {"mcp_tool_name": tool_name}
                            ),
                            "metadata": logging_metadata,
                        },
                        user_api_key_dict=typed_user_api_key_auth,
                    )

                if litellm_logging_obj:
                    try:
                        litellm_logging_obj.post_call(original_response=result)
                        result = await litellm_logging_obj.async_post_mcp_tool_call_hook(
                            kwargs=litellm_logging_obj.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=datetime.now(),
                        )
                    except Exception:
                        verbose_logger.exception("Failed to run post-call logging for MCP tool call %s", tool_name)
                    try:
                        await litellm_logging_obj.async_success_handler(
                            result=result,
                            start_time=start_time,
                            end_time=datetime.now(),
                        )
                    except Exception:
                        verbose_logger.exception("Failed to log MCP tool call success for %s", tool_name)

                # Format result for inclusion in response
                result_text = LiteLLM_Proxy_MCP_Handler._parse_mcp_result(result)
                tool_results.append(
                    _failed_tool_result(tool_call_id, result_text, tool_name, reports_errors)
                    if getattr(result, "is_error", False) is True
                    else MCPToolResult(tool_call_id=tool_call_id, result=result_text, name=tool_name)
                )

            except BlockedPiiEntityError as e:
                await LiteLLM_Proxy_MCP_Handler._log_mcp_tool_failure(
                    proxy_logging_obj=proxy_logging_obj,
                    user_api_key_auth=typed_user_api_key_auth,
                    request_data=logging_request_data,
                    error=e,
                )
                verbose_logger.error("BlockedPiiEntityError in MCP tool call: %s", e)
                if LiteLLM_Proxy_MCP_Handler.should_raise_mcp_guardrail_block(e):
                    raise
                error_message = f"Tool call blocked: PII entity '{getattr(e, 'entity_type', 'unknown')}' detected by guardrail '{getattr(e, 'guardrail_name', 'unknown')}'. {e}"
                tool_results.append(_failed_tool_result(tool_call_id, error_message, tool_name, reports_errors))
            except GuardrailRaisedException as e:
                await LiteLLM_Proxy_MCP_Handler._log_mcp_tool_failure(
                    proxy_logging_obj=proxy_logging_obj,
                    user_api_key_auth=typed_user_api_key_auth,
                    request_data=logging_request_data,
                    error=e,
                )
                verbose_logger.error("GuardrailRaisedException in MCP tool call: %s", e)
                if LiteLLM_Proxy_MCP_Handler.should_raise_mcp_guardrail_block(e):
                    raise
                error_message = (
                    f"Tool call blocked: Guardrail '{getattr(e, 'guardrail_name', 'unknown')}' violation. {e}"
                )
                tool_results.append(_failed_tool_result(tool_call_id, error_message, tool_name, reports_errors))
            except HTTPException as e:
                await LiteLLM_Proxy_MCP_Handler._log_mcp_tool_failure(
                    proxy_logging_obj=proxy_logging_obj,
                    user_api_key_auth=typed_user_api_key_auth,
                    request_data=logging_request_data,
                    error=e,
                )
                verbose_logger.error("HTTPException in MCP tool call: %s", e)
                if LiteLLM_Proxy_MCP_Handler.should_raise_mcp_guardrail_block(e):
                    raise
                error_message = f"Tool call failed: {str(e.detail) if hasattr(e, 'detail') else str(e)}"
                tool_results.append(_failed_tool_result(tool_call_id, error_message, tool_name, reports_errors))
            except Exception as e:
                await LiteLLM_Proxy_MCP_Handler._log_mcp_tool_failure(
                    proxy_logging_obj=proxy_logging_obj,
                    user_api_key_auth=typed_user_api_key_auth,
                    request_data=logging_request_data,
                    error=e,
                )
                verbose_logger.exception("Error executing MCP tool call: %s", e)
                tool_results.append(
                    _failed_tool_result(tool_call_id, f"Error executing tool: {e}", tool_name, reports_errors)
                )

        return tool_results

    _execute_tool_calls = execute_tool_calls

    @staticmethod
    def should_raise_mcp_guardrail_block(error: Exception) -> bool:
        return attribute_or_none(error, "mcp_block_behavior") == "raise"

    @staticmethod
    def create_follow_up_messages_for_chat(
        original_messages: Sequence[BaseModel | Mapping[str, object]],
        response: ModelResponse,
        tool_results: Sequence[Mapping[str, object]],
    ) -> Sequence[Mapping[str, object]]:
        """Create follow-up chat messages that include tool execution results."""
        from copy import deepcopy

        from litellm.utils import convert_list_message_to_dict

        follow_up_messages: list[dict[str, object]] = convert_list_message_to_dict(deepcopy(original_messages))

        if not follow_up_messages:
            follow_up_messages = []

        message_to_append: dict | None = None
        try:
            continued_choice: Final = LiteLLM_Proxy_MCP_Handler._continued_choice(response)
            if continued_choice is not None:
                message_to_append = continued_choice.message.model_dump(exclude_none=True)
                # Ensure tool_calls have arguments field (required by OpenAI API)
                if message_to_append.get("tool_calls"):
                    for tool_call in message_to_append["tool_calls"]:
                        if isinstance(tool_call, dict) and "function" in tool_call:
                            if "arguments" not in tool_call["function"]:
                                tool_call["function"]["arguments"] = "{}"
        except Exception:
            verbose_logger.exception("Failed to convert assistant message for MCP flow")

        if message_to_append:
            follow_up_messages.append(message_to_append)

        for tool_result in tool_results:
            follow_up_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result.get("tool_call_id"),
                    "name": tool_result.get("name"),
                    "content": tool_result.get("result", ""),
                }
            )

        return follow_up_messages

    _create_follow_up_messages_for_chat = create_follow_up_messages_for_chat

    @staticmethod
    def is_persistence_disabled(call_params: Mapping[str, object]) -> bool:
        """store=false means the provider kept nothing, so the follow-up call cannot chain on a response id."""
        return call_params.get("store") is False

    _is_persistence_disabled = is_persistence_disabled

    @staticmethod
    def create_follow_up_input(
        response: ResponsesAPIResponse,
        tool_results: Sequence[Mapping[str, object]],
        original_input: str | ResponseInputParam | None = None,
        preserve_reasoning: bool = False,
    ) -> list[object]:
        """Replay the round's output items in their order, so one turn's reasoning, text, and tool calls go back as
        one assistant message, then append the tool results."""
        replayed: Final = (_replayed_output_item(item, preserve_reasoning) for item in response.output)
        tool_outputs: Final = (
            {"type": "function_call_output", "call_id": tool_result["tool_call_id"], "output": tool_result["result"]}
            for tool_result in tool_results
        )
        return [*_history_items(original_input), *(item for item in replayed if item is not None), *tool_outputs]

    _create_follow_up_input = create_follow_up_input

    @staticmethod
    async def make_follow_up_call(
        follow_up_input: list[Any],
        model: str,
        all_tools: Sequence[ResponsesToolParam] | None,
        response_id: str | None,
        **call_params: Unpack[_AresponsesOptions],
    ) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
        """Make follow-up response API call with tool results."""
        return await aresponses(
            input=follow_up_input,
            model=model,
            tools=all_tools,  # Keep tools for potential future calls
            previous_response_id=response_id,  # Link to previous response
            **call_params,
        )

    _make_follow_up_call = make_follow_up_call

    @staticmethod
    async def _log_mcp_tool_failure(
        *,
        proxy_logging_obj: Optional["ProxyLogging"],
        user_api_key_auth: "UserAPIKeyAuth | None",
        request_data: dict[str, object],
        error: Exception,
    ) -> None:
        """Log MCP tool failures via proxy logging hooks."""

        if proxy_logging_obj is None or user_api_key_auth is None:
            return

        try:
            traceback_str: Final = traceback.format_exc(limit=MAXIMUM_TRACEBACK_LINES_TO_LOG)
            with nested_tool_call_failure():
                await proxy_logging_obj.post_call_failure_hook(
                    request_data=request_data,
                    original_exception=error,
                    user_api_key_dict=user_api_key_auth,
                    route="/responses/mcp/call_tool",
                    traceback_str=traceback_str,
                )
        except Exception:
            verbose_logger.exception("Failed to log MCP tool call failure")

    @staticmethod
    def _create_mcp_streaming_response(
        direct_tools: "Mapping[str, DirectTool]",
        stream_shaper: "OpenAIStreamShaper | None",
        /,
        input: str | ResponseInputParam,
        model: str,
        all_tools: Sequence[object] | None,
        mcp_tools_with_litellm_proxy: list[Mapping[str, object]],
        mcp_discovery_events: list[ResponsesAPIStreamingResponse],
        call_params: Mapping[str, object],
        previous_response_id: str | None,
        tool_server_map: dict[str, str],
        served_tools: Sequence[MCPTool] | None = None,
        tool_choice: object = None,
        gateway: GatewayTools = NO_GATEWAY_TOOLS,
        *,
        mcp_execution_auth: "UserAPIKeyAuth | None",
        **kwargs,
    ) -> "MCPEnhancedStreamingIterator":
        """
        Create MCP enhanced streaming response that handles the full MCP workflow.

        This creates a streaming iterator that:
        1. Immediately emits MCP discovery events
        2. Makes the LLM call and streams the response
        3. Handles tool execution and follow-up calls
        """
        from litellm.responses.mcp.mcp_streaming_iterator import (
            MCPEnhancedStreamingIterator,
        )

        # Build the complete request parameters by merging all sources
        request_params: Final = LiteLLM_Proxy_MCP_Handler._build_request_params(
            input=input,
            model=model,
            all_tools=all_tools,
            call_params=call_params,
            previous_response_id=previous_response_id,
            **kwargs,
        )

        # Create the enhanced streaming iterator that will handle everything
        return MCPEnhancedStreamingIterator(
            base_iterator=None,  # Will be created internally
            mcp_events=mcp_discovery_events,  # Pre-generated MCP discovery events
            tool_server_map=tool_server_map,
            served_tools=served_tools,
            mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
            user_api_key_auth=mcp_execution_auth,
            original_request_params=request_params,
            direct_tools=direct_tools,
            stream_shaper=stream_shaper,
            tool_choice=tool_choice,
            gateway=gateway,
        )

    @staticmethod
    def _build_request_params(
        input: str | ResponseInputParam,
        model: str,
        all_tools: Sequence[object] | None,
        call_params: Mapping[str, object],
        previous_response_id: str | None,
        **kwargs: object,
    ) -> dict[str, object]:
        """
        Build a clean request parameters dictionary for MCP streaming.

        Combines input, model, tools with call_params and additional kwargs
        in a clean, maintainable way.
        """
        # Start with the core required parameters
        request_params: Final[dict[str, object]] = {
            "input": input,
            "model": model,
            "tools": all_tools,
        }

        # Add previous_response_id if provided
        if previous_response_id is not None:
            request_params["previous_response_id"] = previous_response_id

        # Merge in all call_params (which contains most of the API parameters)
        request_params.update(call_params)

        # Merge in any additional kwargs
        request_params.update(kwargs)

        return request_params

    @staticmethod
    def create_tool_execution_events(
        tool_calls: Sequence[object], tool_results: Sequence[MCPToolResult]
    ) -> list[ResponsesAPIStreamingResponse]:
        """
        Create MCP tool execution events for streaming.

        Args:
            tool_calls: List of tool calls from the LLM response
            tool_results: List of tool execution results

        Returns:
            List of MCP tool execution events for streaming
        """
        from litellm._uuid import uuid
        from litellm.responses.mcp.mcp_streaming_iterator import create_mcp_call_events

        tool_execution_events: Final[list[ResponsesAPIStreamingResponse]] = []

        # Create events for each tool execution
        for tool_result in tool_results:
            tool_call_id = tool_result.get("tool_call_id", "unknown")
            result_text = tool_result.get("result", "")

            # Extract tool name and arguments from tool calls
            tool_name = "unknown"
            tool_arguments = "{}"
            for tool_call in tool_calls:
                (
                    name,
                    args,
                    call_id,
                ) = LiteLLM_Proxy_MCP_Handler.extract_tool_call_details(tool_call)
                if call_id == tool_call_id:
                    tool_name = name or "unknown"
                    tool_arguments = args or "{}"
                    break

            execution_events = create_mcp_call_events(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=tool_arguments,  # Use actual arguments
                result=result_text,
                base_item_id=f"mcp_{uuid.uuid4().hex[:8]}",  # Unique ID for each tool call
                sequence_start=len(tool_execution_events) + 1,
            )
            tool_execution_events.extend(execution_events)

        return tool_execution_events

    _create_tool_execution_events = create_tool_execution_events

    @staticmethod
    def prepare_initial_call_params(call_params: Mapping[str, object], should_auto_execute: bool) -> _AresponsesOptions:
        """
        Prepare call parameters for the initial LLM call.

        For auto-execute scenarios, we need to disable streaming for the initial call
        so we can process the tool calls before streaming the final response.
        """
        initial_params: Final = _AresponsesOptions(**call_params)

        if should_auto_execute:
            # Disable streaming for initial call when auto-executing tools
            initial_params["stream"] = False

        return initial_params

    _prepare_initial_call_params = prepare_initial_call_params

    @staticmethod
    def prepare_follow_up_call_params(
        call_params: Mapping[str, object], original_stream_setting: bool
    ) -> _AresponsesOptions:
        """Prepare call parameters for the follow-up LLM call after tool execution."""
        return _AresponsesOptions(
            **{key: value for key, value in call_params.items() if key not in ("stream", "tool_choice")},
            **allowed_tools_tool_choice_kept_on_follow_up(call_params.get("tool_choice")),
            stream=original_stream_setting,
        )

    _prepare_follow_up_call_params = prepare_follow_up_call_params

    @staticmethod
    def add_mcp_output_elements_to_response(
        response: ResponsesAPIResponse,
        mcp_tools_fetched: Sequence[object],
        tool_results: Sequence[Mapping[str, object]],
    ) -> ResponsesAPIResponse:
        """Add custom output elements to the final response for MCP tool execution."""
        # Import the required classes for creating output items
        import json

        from litellm._uuid import uuid
        from litellm.types.responses.main import GenericResponseOutputItem, OutputText

        # Create output element for initial MCP tools
        mcp_tools_output: Final = GenericResponseOutputItem(
            type="mcp_tools_fetched",
            id=f"mcp_tools_{uuid.uuid4().hex[:8]}",
            status="completed",
            role="system",
            content=[
                OutputText(
                    type="output_text",
                    text=json.dumps(mcp_tools_fetched, indent=2, default=str),
                    annotations=[],
                )
            ],
        )

        # Create output element for tool execution results
        tool_results_output: Final = GenericResponseOutputItem(
            type="tool_execution_results",
            id=f"tool_results_{uuid.uuid4().hex[:8]}",
            status="completed",
            role="system",
            content=[
                OutputText(
                    type="output_text",
                    text=json.dumps(tool_results, indent=2, default=str),
                    annotations=[],
                )
            ],
        )

        # Add the new output elements to the response
        response.output.append(mcp_tools_output.model_dump())
        response.output.append(tool_results_output.model_dump())

        return response

    _add_mcp_output_elements_to_response = add_mcp_output_elements_to_response
