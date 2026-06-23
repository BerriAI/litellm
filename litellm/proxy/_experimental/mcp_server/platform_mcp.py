import json
import weakref
from typing import Any, Iterable, Optional, Sequence

from litellm._logging import verbose_logger
from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPServer

try:
    from mcp.types import Tool as MCPTool
except ImportError:
    MCPTool = None  # type: ignore


DEFAULT_PLATFORM_MCP_TOOL_THRESHOLD = 10
PLATFORM_MCP_LIST_SERVERS_TOOL_NAME = "list_servers"
PLATFORM_MCP_ENABLE_SERVER_TOOL_NAME = "enable_server"
PLATFORM_MCP_TOOL_NAMES = frozenset(
    {
        PLATFORM_MCP_LIST_SERVERS_TOOL_NAME,
        PLATFORM_MCP_ENABLE_SERVER_TOOL_NAME,
    }
)

_enabled_servers_by_session: "weakref.WeakKeyDictionary[Any, frozenset[str]]" = weakref.WeakKeyDictionary()


async def get_platform_mcp_settings() -> tuple[bool, int]:
    from litellm.proxy.proxy_server import general_settings, prisma_client

    settings = dict(general_settings or {})
    if prisma_client is not None:
        from litellm.proxy.utils import get_config_param

        row = await get_config_param(prisma_client, "general_settings")
        param_value = getattr(row, "param_value", None) if row is not None else None
        if isinstance(param_value, dict):
            settings.update(param_value)

    enabled = _coerce_enabled(settings.get("platform_mcp_enabled"))
    threshold = _coerce_positive_threshold(settings.get("platform_mcp_tool_threshold"))
    return enabled, threshold


def build_platform_mcp_tools() -> list[Any]:
    if MCPTool is None:
        return []

    return [
        MCPTool(
            name=PLATFORM_MCP_LIST_SERVERS_TOOL_NAME,
            description=(
                "List the MCP servers this key can access, including the server "
                "name and description so you can choose which server to enable."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        MCPTool(
            name=PLATFORM_MCP_ENABLE_SERVER_TOOL_NAME,
            description=(
                "Return full tool definitions for one accessible MCP server. "
                "Use a server name returned by list_servers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "The MCP server name returned by list_servers.",
                    }
                },
                "required": ["server_name"],
                "additionalProperties": False,
            },
        ),
    ]


def should_compress_tools(
    *,
    platform_mcp_enabled: bool,
    threshold: int,
    tool_count: int,
    requested_mcp_servers: Optional[Sequence[str]],
    enabled_server_names: Sequence[str],
) -> bool:
    return (
        platform_mcp_enabled and requested_mcp_servers is None and not enabled_server_names and tool_count > threshold
    )


def should_include_platform_meta_tools(
    *,
    platform_mcp_enabled: bool,
    requested_mcp_servers: Optional[Sequence[str]],
    enabled_server_names: Sequence[str],
) -> bool:
    return platform_mcp_enabled and requested_mcp_servers is None and len(enabled_server_names) > 0


def get_enabled_server_names_for_session(session: Optional[Any]) -> tuple[str, ...]:
    if session is None:
        return ()
    try:
        return tuple(sorted(_enabled_servers_by_session.get(session, frozenset())))
    except TypeError:
        verbose_logger.debug(
            "Platform MCP session object cannot be used for enabled-server storage: %s",
            type(session).__name__,
        )
        return ()


def enable_server_for_session(session: Optional[Any], server: MCPServer) -> None:
    if session is None:
        return
    current = frozenset(get_enabled_server_names_for_session(session))
    next_value = current | frozenset([_server_match_name(server)])
    try:
        _enabled_servers_by_session[session] = next_value
    except TypeError:
        verbose_logger.debug(
            "Platform MCP could not store enabled server for session type: %s",
            type(session).__name__,
        )


def is_platform_mcp_tool(name: str) -> bool:
    return name in PLATFORM_MCP_TOOL_NAMES


def extract_enable_server_name(arguments: Optional[dict[str, Any]]) -> Optional[str]:
    if not arguments:
        return None
    value = arguments.get("server_name") or arguments.get("mcp_name") or arguments.get("name")
    return value if isinstance(value, str) and value.strip() else None


def serialize_server_summary(server: MCPServer) -> dict[str, str]:
    return {
        "name": _server_display_name(server),
        "description": _server_description(server),
    }


def serialize_server_tool_response(
    *,
    server: MCPServer,
    tools: Sequence[Any],
) -> str:
    return json.dumps(
        {
            "server": serialize_server_summary(server),
            "tools": [serialize_tool(tool) for tool in tools],
        }
    )


def serialize_servers_response(servers: Iterable[MCPServer]) -> str:
    return json.dumps(
        {"servers": [serialize_server_summary(server) for server in sorted(servers, key=_server_display_name)]}
    )


def serialize_tool(tool: Any) -> dict[str, Any]:
    if hasattr(tool, "model_dump"):
        try:
            dumped = tool.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )
        except TypeError:
            dumped = tool.model_dump()
        if isinstance(dumped, dict):
            return dumped

    input_schema = getattr(tool, "inputSchema", None)
    if input_schema is None:
        input_schema = getattr(tool, "input_schema", {})
    serialized_tool = {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", "") or "",
        "inputSchema": input_schema or {},
    }
    for attr_name, output_name in [
        ("title", "title"),
        ("outputSchema", "outputSchema"),
        ("icons", "icons"),
        ("annotations", "annotations"),
        ("meta", "_meta"),
        ("execution", "execution"),
    ]:
        value = getattr(tool, attr_name, None)
        if value is not None:
            serialized_tool[output_name] = value
    return serialized_tool


def _coerce_positive_threshold(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_PLATFORM_MCP_TOOL_THRESHOLD
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            threshold = int(value)
            if threshold > 0:
                return threshold
    return DEFAULT_PLATFORM_MCP_TOOL_THRESHOLD


def _coerce_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _server_match_name(server: MCPServer) -> str:
    return server.alias or server.server_name or server.name


def _server_display_name(server: MCPServer) -> str:
    return server.alias or server.server_name or server.name


def _server_description(server: MCPServer) -> str:
    mcp_info: Optional[MCPInfo] = server.mcp_info
    description = mcp_info.get("description") if mcp_info else None
    return description if isinstance(description, str) else ""
