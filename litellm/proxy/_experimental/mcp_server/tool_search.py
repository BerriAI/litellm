from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypedDict

from pydantic import ValidationError
from typing_extensions import ReadOnly, Required, assert_never

import litellm
from litellm.proxy.agent_endpoints.agent_search import DEFAULT_AGENT_SEARCH_TOP_K
from litellm.proxy.common_utils.semantic_text_index import (
    Embedder,
    EmbeddingFailed,
    SemanticTextIndex,
    router_embedder,
)
from litellm.types.mcp import MCPToolSearchSettings

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth

MCP_TOOL_SEARCH_SETTINGS_KEY: Final[str] = "mcp_tool_search"
MCP_TOOL_SEARCH_TOOL_NAME: Final[str] = "mcp_tool_search"
MCP_TOOL_CALL_TOOL_NAME: Final[str] = "mcp_tool_call"
AGENT_SEARCH_TOOL_NAME: Final[str] = "agent_search"
VIRTUAL_TOOL_NAMES: Final = frozenset((MCP_TOOL_SEARCH_TOOL_NAME, MCP_TOOL_CALL_TOOL_NAME, AGENT_SEARCH_TOOL_NAME))


def coerce_top_k(value: Any, default: int = 5) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ToolSearchResult(TypedDict, total=False):
    name: Required[ReadOnly[str]]
    description: Required[ReadOnly[str]]
    inputSchema: Required[ReadOnly[Mapping[str, object]]]
    score: ReadOnly[float]


@dataclass(frozen=True, slots=True)
class SemanticToolRanker:
    embed: Embedder
    embedding_model: str
    index: SemanticTextIndex


global_mcp_tool_search_index: Final = SemanticTextIndex()


def mcp_tool_search_settings() -> MCPToolSearchSettings | ValidationError:
    try:
        return MCPToolSearchSettings.model_validate(litellm.mcp_tool_search or {})
    except ValidationError as exc:
        return exc


def _tool_result(tool: Tool) -> ToolSearchResult:
    return {"name": tool.name, "description": tool.description or "", "inputSchema": tool.inputSchema}


def _scored_result(tool: Tool, score: float) -> ToolSearchResult:
    return {"name": tool.name, "description": tool.description or "", "inputSchema": tool.inputSchema, "score": score}


def _tool_text(tool: Tool) -> str:
    return "\n".join(part for part in (tool.name, tool.description or "") if part)


def _keyword_score(query: str, tool: Tool) -> float:
    haystack: Final = _tool_text(tool).lower()
    return float(sum(1 for token in query.lower().split() if token in haystack))


def _split_core_tools(tools: Sequence[Tool], core_tools: Sequence[str]) -> tuple[tuple[Tool, ...], tuple[Tool, ...]]:
    by_name: Final = MappingProxyType({tool.name: tool for tool in tools})
    core: Final = tuple(by_name[name] for name in dict.fromkeys(core_tools) if name in by_name)
    rest: Final = tuple(tool for tool in tools if tool.name not in frozenset(core_tools))
    return core, rest


def _top_hits(
    tools: Sequence[Tool], scores: Sequence[float], minimum: float, limit: int
) -> tuple[tuple[float, Tool], ...]:
    hits: Final = ((score, tool) for score, tool in zip(scores, tools, strict=True) if score >= minimum)
    return tuple(sorted(hits, key=lambda hit: hit[0], reverse=True)[:limit])


def search_tools(query: str, tools: Sequence[Tool], top_k: int = 5) -> tuple[ToolSearchResult, ...]:
    """Keyword fallback used when no embedding model is configured: one point per query token found in the tool."""
    if not query:
        return ()
    scores: Final = tuple(_keyword_score(query, tool) for tool in tools)
    return tuple(_tool_result(tool) for _, tool in _top_hits(tools, scores, minimum=1.0, limit=top_k))


async def search_mcp_tools(
    query: str,
    tools: Sequence[Tool],
    top_k: int,
    settings: MCPToolSearchSettings,
    ranker: SemanticToolRanker | None,
) -> tuple[ToolSearchResult, ...] | EmbeddingFailed:
    """Core tools the caller can access come first, then up to `top_k` ranked matches from the remaining tools."""
    core, rest = _split_core_tools(tools, settings.core_tools)
    limit: Final = min(top_k, settings.top_k)
    core_results: Final = tuple(_tool_result(tool) for tool in core)
    if ranker is None:
        return (*core_results, *search_tools(query, rest, limit))
    if not query:
        return core_results
    scores: Final = await ranker.index.scores(
        query, tuple(_tool_text(tool) for tool in rest), ranker.embed, ranker.embedding_model
    )
    if isinstance(scores, EmbeddingFailed):
        return scores
    hits: Final = _top_hits(rest, scores, minimum=settings.similarity_threshold, limit=limit)
    return (*core_results, *(_scored_result(tool, score) for score, tool in hits))


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
    "description": (
        "Search for MCP tools by describing what you need. "
        "Returns top matching tools with names, descriptions, and input schemas."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What the tool should do, matched against names and descriptions.",
            },
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
    from litellm.proxy._experimental.mcp_server.server import _list_mcp_tools
    from litellm.proxy.proxy_server import llm_router

    settings: Final = mcp_tool_search_settings()
    if isinstance(settings, ValidationError):
        return _text_tool_result(
            f"litellm_settings.{MCP_TOOL_SEARCH_SETTINGS_KEY} is invalid: {settings}", is_error=True
        )
    if settings.embedding_model is not None and llm_router is None:
        return _text_tool_result(
            f"litellm_settings.{MCP_TOOL_SEARCH_SETTINGS_KEY}.embedding_model needs a model_list so it can be called",
            is_error=True,
        )
    ranker: Final = (
        SemanticToolRanker(
            embed=router_embedder(llm_router, settings.embedding_model, user_api_key_dict),
            embedding_model=settings.embedding_model,
            index=global_mcp_tool_search_index,
        )
        if settings.embedding_model is not None and llm_router is not None
        else None
    )
    mcp_listing: Final = await _list_mcp_tools(
        user_api_key_auth=user_api_key_dict,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
    )
    results: Final = await search_mcp_tools(query, mcp_listing.tools, top_k, settings, ranker)
    if isinstance(results, EmbeddingFailed):
        return _text_tool_result(results.reason, is_error=True)
    return _text_tool_result(json.dumps(results), is_error=False)


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
