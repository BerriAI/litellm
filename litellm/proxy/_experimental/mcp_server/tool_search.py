from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth

MCP_TOOL_SEARCH_TOOL_NAME: str = "mcp_tool_search"
MCP_TOOL_CALL_TOOL_NAME: str = "mcp_tool_call"
DEFAULT_MCP_TOOL_SEARCH_TOP_K: int = 5


def coerce_top_k(value: Any, default: int = DEFAULT_MCP_TOOL_SEARCH_TOP_K) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_mcp_tool_search_default_top_k(
    litellm_settings: Optional[Mapping[str, object]],
) -> int:
    if litellm_settings is None:
        return DEFAULT_MCP_TOOL_SEARCH_TOP_K
    return coerce_top_k(
        litellm_settings.get("mcp_tool_search_default_top_k"),
        default=DEFAULT_MCP_TOOL_SEARCH_TOP_K,
    )


def search_tools(
    query: str,
    tools: list[dict[str, Any]],
    top_k: int = DEFAULT_MCP_TOOL_SEARCH_TOP_K,
) -> list[dict[str, Any]]:
    if not query:
        return []
    tokens = query.lower().split()

    def _score(tool: dict[str, Any]) -> int:
        haystack = (tool.get("name", "") + " " + tool.get("description", "")).lower()
        return sum(1 for t in tokens if t in haystack)

    scored = ((s, tool) for tool in tools if (s := _score(tool)) > 0)
    return [tool for _, tool in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]]


def get_virtual_tool_definitions(
    default_top_k: int = DEFAULT_MCP_TOOL_SEARCH_TOP_K,
) -> list[dict[str, Any]]:
    return [
        {
            "name": MCP_TOOL_SEARCH_TOOL_NAME,
            "description": "Search for MCP tools by keyword. Returns top matching tools with names, descriptions, and input schemas.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for in tool names and descriptions.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": default_top_k,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": MCP_TOOL_CALL_TOOL_NAME,
            "description": "Call an MCP tool by name with the given arguments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The exact name of the MCP tool to call.",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the tool.",
                    },
                },
                "required": ["tool_name"],
            },
        },
    ]


async def handle_mcp_tool_search(
    query: str,
    top_k: int,
    user_api_key_dict: UserAPIKeyAuth,
    client_ip: Optional[str] = None,
    mcp_servers: Optional[list[str]] = None,
    mcp_auth_header: Optional[str] = None,
    mcp_server_auth_headers: Optional[dict[str, dict[str, str]]] = None,
    oauth2_headers: Optional[dict[str, str]] = None,
    raw_headers: Optional[dict[str, str]] = None,
) -> CallToolResult:
    from mcp.types import CallToolResult, TextContent

    from litellm.proxy._experimental.mcp_server.server import _list_mcp_tools

    mcp_tools = await _list_mcp_tools(
        user_api_key_auth=user_api_key_dict,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
    )
    tools = [
        {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.inputSchema,
        }
        for t in mcp_tools
    ]
    results = search_tools(query, tools, top_k)
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(results))], isError=False)


async def handle_mcp_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    user_api_key_dict: UserAPIKeyAuth,
    client_ip: Optional[str] = None,
    mcp_servers: Optional[list[str]] = None,
    mcp_auth_header: Optional[str] = None,
    mcp_server_auth_headers: Optional[dict[str, dict[str, str]]] = None,
    oauth2_headers: Optional[dict[str, str]] = None,
    raw_headers: Optional[dict[str, str]] = None,
    litellm_logging_obj: Optional[LiteLLMLoggingObj] = None,
) -> CallToolResult:
    from litellm.proxy._experimental.mcp_server.server import (
        _get_allowed_mcp_servers,
        execute_mcp_tool,
    )

    allowed_mcp_servers = await _get_allowed_mcp_servers(
        user_api_key_auth=user_api_key_dict,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
    )

    # Reject before dispatch when the key has no accessible servers; otherwise an
    # unprefixed local tool name would fall through to the local registry in
    # execute_mcp_tool, which has no server permission check.
    if not allowed_mcp_servers:
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="User not allowed to call this tool.")

    return await execute_mcp_tool(
        name=tool_name,
        arguments=arguments,
        allowed_mcp_servers=allowed_mcp_servers,
        start_time=datetime.now(),
        user_api_key_auth=user_api_key_dict,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        litellm_logging_obj=litellm_logging_obj,
    )
