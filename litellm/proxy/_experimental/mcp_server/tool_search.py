from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, TypedDict, assert_never

from typing_extensions import ReadOnly, Required

import litellm
from litellm.proxy.agent_endpoints.agent_search import DEFAULT_AGENT_SEARCH_TOP_K

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth

MCP_TOOL_SEARCH_TOOL_NAME: Final[str] = "mcp_tool_search"
MCP_TOOL_CALL_TOOL_NAME: Final[str] = "mcp_tool_call"
AGENT_SEARCH_TOOL_NAME: Final[str] = "agent_search"
VIRTUAL_TOOL_NAMES: Final = frozenset((MCP_TOOL_SEARCH_TOOL_NAME, MCP_TOOL_CALL_TOOL_NAME, AGENT_SEARCH_TOOL_NAME))


def coerce_top_k(value: Any, default: int = 5) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def search_tools(query: str, tools: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    if not query:
        return []
    tokens: Final = query.lower().split()

    def _score(tool: dict[str, Any]) -> int:
        haystack: Final = (tool.get("name", "") + " " + tool.get("description", "")).lower()
        return sum(1 for t in tokens if t in haystack)

    scored: Final = ((s, tool) for tool in tools if (s := _score(tool)) > 0)
    return [tool for _, tool in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]]


class _ToolParamSchema(TypedDict, total=False):
    type: Required[ReadOnly[str]]
    description: Required[ReadOnly[str]]
    default: ReadOnly[int]


class _ToolInputSchema(TypedDict):
    type: ReadOnly[str]
    properties: ReadOnly[Mapping[str, _ToolParamSchema]]
    required: ReadOnly[Sequence[str]]


class VirtualToolDefinition(TypedDict):
    name: ReadOnly[str]
    description: ReadOnly[str]
    inputSchema: ReadOnly[_ToolInputSchema]


def _json_array(*items: str) -> Sequence[str]:
    return list(items)  # mutable-ok: jsonschema's metaschema only accepts a JSON array for required


_MCP_TOOL_SEARCH_DEFINITION: Final[VirtualToolDefinition] = {
    "name": MCP_TOOL_SEARCH_TOOL_NAME,
    "description": "Search for MCP tools by keyword. Returns top matching tools with names, descriptions, and input schemas.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for in tool names and descriptions."},
            "top_k": {"type": "integer", "description": "Maximum number of results to return.", "default": 5},
        },
        "required": _json_array("query"),
    },
}

_MCP_TOOL_CALL_DEFINITION: Final[VirtualToolDefinition] = {
    "name": MCP_TOOL_CALL_TOOL_NAME,
    "description": "Call an MCP tool by name with the given arguments.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string", "description": "The exact name of the MCP tool to call."},
            "arguments": {"type": "object", "description": "Arguments to pass to the tool."},
        },
        "required": _json_array("tool_name"),
    },
}

_AGENT_SEARCH_DEFINITION: Final[VirtualToolDefinition] = {
    "name": AGENT_SEARCH_TOOL_NAME,
    "description": "Find A2A agents by describing the task in natural language. Returns the best matching agents you can access, ranked by semantic similarity, each with its agent_id, name, description, skills, and score.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The task the agent should be able to do, in natural language."},
            "top_k": {
                "type": "integer",
                "description": "Maximum number of agents to return.",
                "default": DEFAULT_AGENT_SEARCH_TOP_K,
            },
        },
        "required": _json_array("query"),
    },
}


def get_virtual_tool_definitions() -> tuple[VirtualToolDefinition, ...]:
    return (_MCP_TOOL_SEARCH_DEFINITION, _MCP_TOOL_CALL_DEFINITION, _AGENT_SEARCH_DEFINITION)


def _text_tool_result(text: str, is_error: bool) -> CallToolResult:
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(
        content=[TextContent(type="text", text=text)],  # mutable-ok: CallToolResult accepts only list content
        isError=is_error,
    )


async def handle_agent_search(query: str, top_k: int, user_api_key_dict: UserAPIKeyAuth) -> CallToolResult:
    from litellm.proxy.agent_endpoints.agent_search import (
        AgentSearchEmbeddingFailed,
        AgentSearchHits,
        AgentSearchNotConfigured,
        agent_search_result,
        global_agent_search_index,
        search_agents,
    )
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import accessible_agents
    from litellm.proxy.common_utils.rbac_utils import check_feature_access_for_user
    from litellm.proxy.proxy_server import llm_router

    await check_feature_access_for_user(user_api_key_dict, "agents")
    outcome: Final = await search_agents(
        query=query,
        agents=await accessible_agents(user_api_key_dict),
        top_k=max(top_k, 1),
        router=llm_router,
        embedding_model=litellm.agent_search_embedding_model,
        index=global_agent_search_index,
        user_api_key_dict=user_api_key_dict,
    )
    match outcome:
        case AgentSearchHits(hits):
            results: Final = tuple(agent_search_result(hit).model_dump() for hit in hits)
            return _text_tool_result(json.dumps(results), is_error=False)
        case AgentSearchNotConfigured(reason) | AgentSearchEmbeddingFailed(reason):
            return _text_tool_result(reason, is_error=True)
        case _:
            assert_never(outcome)


async def handle_mcp_tool_search(
    query: str,
    top_k: int,
    user_api_key_dict: UserAPIKeyAuth,
    client_ip: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
) -> CallToolResult:
    from mcp.types import CallToolResult, TextContent

    from litellm.proxy._experimental.mcp_server.server import _list_mcp_tools

    mcp_listing: Final = await _list_mcp_tools(
        user_api_key_auth=user_api_key_dict,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
    )
    mcp_tools: Final = mcp_listing.tools
    tools: Final = [
        {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.inputSchema,
        }
        for t in mcp_tools
    ]
    results: Final = search_tools(query, tools, top_k)
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(results))], isError=False)


async def handle_mcp_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    user_api_key_dict: UserAPIKeyAuth,
    client_ip: str | None = None,
    mcp_servers: list[str] | None = None,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    oauth2_headers: dict[str, str] | None = None,
    raw_headers: dict[str, str] | None = None,
    litellm_logging_obj: LiteLLMLoggingObj | None = None,
) -> CallToolResult:
    from litellm.proxy._experimental.mcp_server.server import (
        _get_allowed_mcp_servers,
        execute_mcp_tool,
    )

    allowed_mcp_servers: Final = await _get_allowed_mcp_servers(
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
