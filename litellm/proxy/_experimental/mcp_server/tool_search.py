from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypedDict

from pydantic import ValidationError
from typing_extensions import ReadOnly, Required, assert_never

import litellm
from litellm.llms.litellm_proxy.skills.skill_search import DEFAULT_SKILL_SEARCH_TOP_K
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
MCP_PROXY_SEARCH_TOOL_NAME: Final[str] = "search_tools"
MCP_PROXY_SCHEMA_TOOL_NAME: Final[str] = "get_tool_schema"
MCP_PROXY_CALL_TOOL_NAME: Final[str] = "call_tool"
MCP_PROXY_TOOL_NAMES: Final = frozenset(
    (MCP_PROXY_SEARCH_TOOL_NAME, MCP_PROXY_SCHEMA_TOOL_NAME, MCP_PROXY_CALL_TOOL_NAME)
)
AGENT_SEARCH_TOOL_NAME: Final[str] = "agent_search"
SKILL_SEARCH_TOOL_NAME: Final[str] = "skill_search"
VIRTUAL_TOOL_NAMES: Final = frozenset(
    (MCP_TOOL_SEARCH_TOOL_NAME, MCP_TOOL_CALL_TOOL_NAME, AGENT_SEARCH_TOOL_NAME, SKILL_SEARCH_TOOL_NAME)
)


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


class MCPProxySearchResult(TypedDict, total=False):
    tool_id: Required[ReadOnly[str]]
    name: Required[ReadOnly[str]]
    description: Required[ReadOnly[str]]
    score: ReadOnly[float]


class MCPProxySchemaResult(MCPProxySearchResult, total=False):
    inputSchema: Required[ReadOnly[Mapping[str, object]]]
    outputSchema: ReadOnly[Mapping[str, object]]


class MCPProxyToolIdentity(TypedDict):
    server_id: ReadOnly[str]
    tool_name: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class MCPToolSearchHit:
    tool: Tool
    score: float | None = None


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


_MCP_PROXY_IDENTITY_META_KEY: Final[str] = "litellm.ai/proxy_tool_identity"


def with_mcp_proxy_identity(tool: Tool, server_id: str) -> Tool:
    identity: Final[MCPProxyToolIdentity] = {"server_id": server_id, "tool_name": tool.name}
    return tool.model_copy(  # mutable-ok: Pydantic requires mutable update and metadata mappings
        update={  # mutable-ok: Pydantic update payload
            "meta": {**(tool.meta or {}), _MCP_PROXY_IDENTITY_META_KEY: identity}  # mutable-ok: metadata mapping
        }
    )


def _mcp_proxy_identity(tool: Tool) -> MCPProxyToolIdentity:
    identity: Final = (tool.meta or {}).get(_MCP_PROXY_IDENTITY_META_KEY)  # mutable-ok: absent metadata default
    if not isinstance(identity, Mapping):
        raise TypeError("MCP proxy tool identity is missing")
    server_id: Final = identity.get("server_id")
    tool_name: Final = identity.get("tool_name")
    if not isinstance(server_id, str) or not isinstance(tool_name, str):
        raise TypeError("MCP proxy tool identity is invalid")
    return {"server_id": server_id, "tool_name": tool_name}  # mutable-ok: TypedDict identity payload


def mcp_proxy_tool_id(tool: Tool) -> str:
    identity: Final = _mcp_proxy_identity(tool)
    return hashlib.sha256(f"{identity['server_id']}\0{identity['tool_name']}".encode()).hexdigest()[:32]


def _proxy_search_result(hit: MCPToolSearchHit) -> MCPProxySearchResult:
    base: Final[MCPProxySearchResult] = {
        "tool_id": mcp_proxy_tool_id(hit.tool),
        "name": hit.tool.name,
        "description": hit.tool.description or "",
    }
    return {**base, "score": hit.score} if hit.score is not None else base  # mutable-ok: wire result payload


def _proxy_schema_result(tool: Tool) -> MCPProxySchemaResult:
    base: Final[MCPProxySchemaResult] = {
        "tool_id": mcp_proxy_tool_id(tool),
        "name": tool.name,
        "description": tool.description or "",
        "inputSchema": tool.inputSchema,
    }
    if tool.outputSchema is None:
        return base
    return {**base, "outputSchema": tool.outputSchema}  # mutable-ok: wire schema payload


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


async def rank_mcp_tools(
    query: str,
    tools: Sequence[Tool],
    top_k: int,
    settings: MCPToolSearchSettings,
    ranker: SemanticToolRanker | None,
) -> tuple[MCPToolSearchHit, ...] | EmbeddingFailed:
    core, rest = _split_core_tools(tools, settings.core_tools)
    core_hits: Final = tuple(MCPToolSearchHit(tool) for tool in core)
    if not query:
        return core_hits
    limit: Final = min(top_k, settings.top_k)
    if ranker is None:
        scores: Final = tuple(_keyword_score(query, tool) for tool in rest)
        return (
            *core_hits,
            *(MCPToolSearchHit(tool) for _, tool in _top_hits(rest, scores, minimum=1.0, limit=limit)),
        )
    semantic_scores: Final = await ranker.index.scores(
        query, tuple(_tool_text(tool) for tool in rest), ranker.embed, ranker.embedding_model
    )
    if isinstance(semantic_scores, EmbeddingFailed):
        return semantic_scores
    return (
        *core_hits,
        *(
            MCPToolSearchHit(tool, score)
            for score, tool in _top_hits(rest, semantic_scores, settings.similarity_threshold, limit)
        ),
    )


async def search_mcp_tools(
    query: str,
    tools: Sequence[Tool],
    top_k: int,
    settings: MCPToolSearchSettings,
    ranker: SemanticToolRanker | None,
) -> tuple[ToolSearchResult, ...] | EmbeddingFailed:
    hits: Final = await rank_mcp_tools(query, tools, top_k, settings, ranker)
    if isinstance(hits, EmbeddingFailed):
        return hits
    return tuple(
        _scored_result(hit.tool, hit.score) if hit.score is not None else _tool_result(hit.tool) for hit in hits
    )


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


_SKILL_SEARCH_DEFINITION: Final[VirtualToolDefinition] = {
    "name": SKILL_SEARCH_TOOL_NAME,
    "description": "Find registered skills by describing what you need in natural language. Returns the best "
    "matching skills you can access, ranked by semantic similarity, each with its skill_id, display_title, "
    "description, and score.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What you need the skill to do, in natural language."},
            "top_k": {
                "type": "integer",
                "description": "Maximum number of skills to return.",
                "default": DEFAULT_SKILL_SEARCH_TOP_K,
            },
        },
        "required": _json_array("query"),
    },
}


_MCP_PROXY_SEARCH_DEFINITION: Final[VirtualToolDefinition] = {
    "name": MCP_PROXY_SEARCH_TOOL_NAME,
    "description": "Search accessible MCP tools by describing what you need. Returns opaque tool IDs.",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What the tool should do."}},
        "required": _json_array("query"),
    },
}

_MCP_PROXY_SCHEMA_DEFINITION: Final[VirtualToolDefinition] = {
    "name": MCP_PROXY_SCHEMA_TOOL_NAME,
    "description": "Return the complete schema for an accessible MCP tool ID.",
    "inputSchema": {
        "type": "object",
        "properties": {"tool_id": {"type": "string", "description": "Opaque ID from search_tools."}},
        "required": _json_array("tool_id"),
    },
}

_MCP_PROXY_CALL_DEFINITION: Final[VirtualToolDefinition] = {
    "name": MCP_PROXY_CALL_TOOL_NAME,
    "description": "Call an accessible MCP tool by opaque ID with schema-valid arguments.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "tool_id": {"type": "string", "description": "Opaque ID from search_tools."},
            "arguments": {"type": "object", "description": "Arguments validated against the selected tool schema."},
        },
        "required": _json_array("tool_id"),
    },
}


def get_virtual_tool_definitions() -> tuple[VirtualToolDefinition, ...]:
    return (_MCP_TOOL_SEARCH_DEFINITION, _MCP_TOOL_CALL_DEFINITION, _AGENT_SEARCH_DEFINITION, _SKILL_SEARCH_DEFINITION)


def get_mcp_proxy_tool_definitions() -> tuple[VirtualToolDefinition, ...]:
    return (_MCP_PROXY_SEARCH_DEFINITION, _MCP_PROXY_SCHEMA_DEFINITION, _MCP_PROXY_CALL_DEFINITION)


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
    from litellm.proxy.proxy_server import llm_router, proxy_logging_obj

    await check_feature_access_for_user(user_api_key_dict, "agents")
    outcome: Final = await search_agents(
        query=query,
        agents=await accessible_agents(user_api_key_dict),
        top_k=max(top_k, 1),
        router=llm_router,
        embedding_model=litellm.agent_search_embedding_model,
        index=global_agent_search_index,
        user_api_key_dict=user_api_key_dict,
        proxy_logging_obj=proxy_logging_obj,
    )
    match outcome:
        case AgentSearchHits(hits):
            results: Final = tuple(agent_search_result(hit).model_dump() for hit in hits)
            return _text_tool_result(json.dumps(results), is_error=False)
        case AgentSearchNotConfigured(reason) | AgentSearchEmbeddingFailed(reason):
            return _text_tool_result(reason, is_error=True)
        case _:
            assert_never(outcome)


async def handle_skill_search(query: str, top_k: int, user_api_key_dict: UserAPIKeyAuth) -> CallToolResult:
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
    from litellm.llms.litellm_proxy.skills.skill_search import (
        MAX_SKILL_SEARCH_TOP_K,
        SkillSearchEmbeddingFailed,
        SkillSearchHits,
        SkillSearchNotConfigured,
        global_skill_search_index,
        search_skills,
        skill_search_result,
    )
    from litellm.proxy.proxy_server import llm_router, proxy_logging_obj

    outcome: Final = await search_skills(
        query=query,
        skills=await LiteLLMSkillsHandler.list_skills_for_search(user_api_key_dict),
        top_k=min(max(top_k, 1), MAX_SKILL_SEARCH_TOP_K),
        router=llm_router,
        embedding_model=litellm.skill_search_embedding_model,
        index=global_skill_search_index,
        user_api_key_dict=user_api_key_dict,
        proxy_logging_obj=proxy_logging_obj,
    )
    match outcome:
        case SkillSearchHits(hits):
            results: Final = tuple(skill_search_result(hit).model_dump() for hit in hits)
            return _text_tool_result(json.dumps(results), is_error=False)
        case SkillSearchNotConfigured(reason) | SkillSearchEmbeddingFailed(reason):
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
    from litellm.proxy._experimental.mcp_server.server import (
        _list_mcp_tools,  # pyright: ignore[reportPrivateUsage]  # shared catalog owner
    )
    from litellm.proxy.proxy_server import llm_router, proxy_logging_obj

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
            embed=router_embedder(llm_router, settings.embedding_model, user_api_key_dict, proxy_logging_obj),
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


async def handle_mcp_proxy_tool(
    name: str,
    arguments: dict[str, object],  # mutable-ok: MCP dispatcher passes mutable call arguments
    user_api_key_dict: UserAPIKeyAuth,
    client_ip: str | None = None,
    mcp_servers: list[str] | None = None,  # mutable-ok: preserve MCP scope container for existing resolver
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,  # mutable-ok: preserve forwarded headers
    oauth2_headers: dict[str, str] | None = None,  # mutable-ok: preserve forwarded headers
    raw_headers: dict[str, str] | None = None,  # mutable-ok: preserve request headers
    litellm_logging_obj: LiteLLMLoggingObj | None = None,
) -> CallToolResult:
    from fastapi import HTTPException
    from jsonschema import ValidationError as JsonSchemaValidationError
    from jsonschema import validate

    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server.server import (  # pyright: ignore[reportPrivateUsage]  # shared catalog owner
        _list_mcp_tools,  # pyright: ignore[reportPrivateUsage]  # shared catalog owner
    )

    listing: Final = await _list_mcp_tools(
        user_api_key_auth=user_api_key_dict,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        mcp_proxy_mode=True,
    )
    tools_by_id: Final = {mcp_proxy_tool_id(tool): tool for tool in listing.tools}  # mutable-ok: lookup index

    if name == MCP_PROXY_SEARCH_TOOL_NAME:
        llm_router: Final = proxy_server.llm_router
        proxy_logging_obj: Final = proxy_server.proxy_logging_obj
        settings: Final = mcp_tool_search_settings()
        if isinstance(settings, ValidationError):
            return _text_tool_result(str(settings), is_error=True)
        if settings.embedding_model is not None and llm_router is None:
            return _text_tool_result(
                f"litellm_settings.{MCP_TOOL_SEARCH_SETTINGS_KEY}.embedding_model needs a model_list so it can be called",
                is_error=True,
            )
        ranker: Final = (
            SemanticToolRanker(
                embed=router_embedder(llm_router, settings.embedding_model, user_api_key_dict, proxy_logging_obj),
                embedding_model=settings.embedding_model,
                index=global_mcp_tool_search_index,
            )
            if settings.embedding_model is not None and llm_router is not None
            else None
        )
        results: Final = await rank_mcp_tools(str(arguments.get("query", "")), listing.tools, 5, settings, ranker)
        if isinstance(results, EmbeddingFailed):
            return _text_tool_result(results.reason, is_error=True)
        return _text_tool_result(json.dumps(tuple(_proxy_search_result(hit) for hit in results)), is_error=False)

    tool_id: Final = arguments.get("tool_id")
    tool: Final = tools_by_id.get(tool_id) if isinstance(tool_id, str) else None
    if tool is None:
        return _text_tool_result("Unknown or unauthorized tool_id", is_error=True)

    if name == MCP_PROXY_SCHEMA_TOOL_NAME:
        return _text_tool_result(json.dumps(_proxy_schema_result(tool)), is_error=False)
    if name != MCP_PROXY_CALL_TOOL_NAME:
        raise HTTPException(status_code=400, detail=f"Unknown MCP proxy tool: {name}")

    tool_arguments: Final = arguments.get("arguments", {})  # mutable-ok: JSON Schema validator consumes mapping
    if not isinstance(tool_arguments, dict):
        return _text_tool_result("arguments must be an object", is_error=True)
    try:
        validate(instance=tool_arguments, schema=tool.inputSchema)
    except JsonSchemaValidationError as exc:
        return _text_tool_result(f"Invalid arguments: {exc.message}", is_error=True)

    return await handle_mcp_tool_call(
        tool_name=_mcp_proxy_identity(tool)["tool_name"],
        arguments=tool_arguments,
        user_api_key_dict=user_api_key_dict,
        requested_server_id=_mcp_proxy_identity(tool)["server_id"],
        client_ip=client_ip,
        mcp_servers=mcp_servers,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        litellm_logging_obj=litellm_logging_obj,
    )


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
    requested_server_id: str | None = None,
) -> CallToolResult:
    from litellm.proxy._experimental.mcp_server.server import (
        _get_allowed_mcp_servers,
        execute_mcp_tool,
        raise_denied_scoped_mcp_access,
    )

    allowed_mcp_servers: Final = await _get_allowed_mcp_servers(
        user_api_key_auth=user_api_key_dict,
        mcp_servers=mcp_servers,
        client_ip=client_ip,
    )
    if mcp_servers and not allowed_mcp_servers:
        await raise_denied_scoped_mcp_access(
            requested_names=mcp_servers,
            user_api_key_auth=user_api_key_dict,
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
        requested_server_id=requested_server_id,
    )
