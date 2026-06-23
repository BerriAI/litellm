import json
from typing import Any, Iterable, Optional, Sequence

from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPServer

try:
    from mcp.types import Tool as MCPTool
except ImportError:
    MCPTool = None  # type: ignore


PLATFORM_MCP_SERVER_ID = "platform_mcp"
PLATFORM_MCP_SERVER_NAME = "platform_mcp"
PLATFORM_MCP_SERVER_DESCRIPTION = (
    "Built-in LiteLLM Platform MCP server for discovering and invoking accessible "
    "downstream MCP servers."
)
PLATFORM_MCP_LIST_SERVERS_TOOL_NAME = "list_servers"
PLATFORM_MCP_GET_SERVER_TOOLS_TOOL_NAME = "get_server_tools"
PLATFORM_MCP_CALL_TOOL_NAME = "call_tool"
PLATFORM_MCP_TOOL_NAMES = frozenset(
    {
        PLATFORM_MCP_LIST_SERVERS_TOOL_NAME,
        PLATFORM_MCP_GET_SERVER_TOOLS_TOOL_NAME,
        PLATFORM_MCP_CALL_TOOL_NAME,
    }
)


async def get_platform_mcp_enabled() -> bool:
    from litellm.proxy.proxy_server import general_settings, prisma_client

    settings = dict(general_settings or {})
    if prisma_client is not None:
        from litellm.proxy.utils import get_config_param

        row = await get_config_param(prisma_client, "general_settings")
        param_value = getattr(row, "param_value", None) if row is not None else None
        if isinstance(param_value, dict):
            settings.update(param_value)

    return _coerce_enabled(settings.get("platform_mcp_enabled"))


def build_platform_mcp_server() -> MCPServer:
    from litellm.types.mcp import MCPTransport

    return MCPServer(
        server_id=PLATFORM_MCP_SERVER_ID,
        name=PLATFORM_MCP_SERVER_NAME,
        alias=PLATFORM_MCP_SERVER_NAME,
        server_name=PLATFORM_MCP_SERVER_NAME,
        url=None,
        transport=MCPTransport.http,
        auth_type=None,
        mcp_info={
            "description": PLATFORM_MCP_SERVER_DESCRIPTION,
            "server_name": PLATFORM_MCP_SERVER_NAME,
            "is_platform_mcp": True,
        },
        allow_all_keys=False,
        available_on_public_internet=True,
    )


def is_platform_mcp_server_identifier(value: Optional[str]) -> bool:
    if not value:
        return False
    return value in {
        PLATFORM_MCP_SERVER_ID,
        PLATFORM_MCP_SERVER_NAME,
    }


def is_platform_mcp_server(server: Optional[MCPServer]) -> bool:
    if server is None:
        return False
    return is_platform_mcp_server_identifier(server.server_id) or bool(
        server.mcp_info and server.mcp_info.get("is_platform_mcp") is True
    )


def without_platform_mcp_servers(servers: Iterable[MCPServer]) -> list[MCPServer]:
    return [server for server in servers if not is_platform_mcp_server(server)]


def build_platform_mcp_tools() -> list[Any]:
    if MCPTool is None:
        return []

    return [
        MCPTool(
            name=PLATFORM_MCP_LIST_SERVERS_TOOL_NAME,
            description=(
                "List the MCP servers this key can access, including the server "
                "name and description so you can choose which server to inspect."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        MCPTool(
            name=PLATFORM_MCP_GET_SERVER_TOOLS_TOOL_NAME,
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
        MCPTool(
            name=PLATFORM_MCP_CALL_TOOL_NAME,
            description=(
                "Call a tool on one accessible MCP server. Use tool names and input "
                "schemas returned by get_server_tools."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "The MCP server name returned by list_servers.",
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "The tool name returned by get_server_tools.",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the downstream MCP tool.",
                        "additionalProperties": True,
                    },
                },
                "required": ["server_name", "tool_name"],
                "additionalProperties": False,
            },
        ),
    ]


def is_platform_mcp_tool(name: str) -> bool:
    if name in PLATFORM_MCP_TOOL_NAMES:
        return True
    prefix = f"{PLATFORM_MCP_SERVER_NAME}-"
    return name.startswith(prefix) and name[len(prefix) :] in PLATFORM_MCP_TOOL_NAMES


def normalize_platform_mcp_tool_name(name: str) -> str:
    prefix = f"{PLATFORM_MCP_SERVER_NAME}-"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def extract_server_name(arguments: Optional[dict[str, Any]]) -> Optional[str]:
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


def _coerce_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _server_display_name(server: MCPServer) -> str:
    return server.alias or server.server_name or server.name


def _server_description(server: MCPServer) -> str:
    mcp_info: Optional[MCPInfo] = server.mcp_info
    description = mcp_info.get("description") if mcp_info else None
    return description if isinstance(description, str) else ""
