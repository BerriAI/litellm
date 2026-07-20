from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth

MCP_TOOL_SEARCH_TOOL_NAME: str = "mcp_tool_search"
MCP_TOOL_CALL_TOOL_NAME: str = "mcp_tool_call"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BM25_K1 = 1.5
_BM25_B = 0.75
_NAME_WEIGHT = 3.0
_PREFIX_WEIGHT = 0.3


def coerce_top_k(value: Any, default: int = 5) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.lower()))


def _field_tf(query_token: str, field_tokens: tuple[str, ...]) -> float:
    return sum(
        1.0 if token == query_token else _PREFIX_WEIGHT if token.startswith(query_token) else 0.0
        for token in field_tokens
    )


def search_tools(query: str, tools: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    query_tokens = tuple(dict.fromkeys(_tokenize(query)))
    if not query_tokens or not tools or top_k <= 0:
        return []
    docs = tuple(
        (tool, _tokenize(str(tool.get("name", ""))), _tokenize(str(tool.get("description") or ""))) for tool in tools
    )
    tf_rows = tuple(
        tuple(_NAME_WEIGHT * _field_tf(token, name_tokens) + _field_tf(token, desc_tokens) for token in query_tokens)
        for _, name_tokens, desc_tokens in docs
    )
    doc_lengths = tuple(_NAME_WEIGHT * len(name_tokens) + len(desc_tokens) for _, name_tokens, desc_tokens in docs)
    avg_doc_length = (sum(doc_lengths) / len(doc_lengths)) or 1.0
    doc_count = len(docs)
    idfs = tuple(
        math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
        for df in (sum(1 for row in tf_rows if row[i] > 0.0) for i in range(len(query_tokens)))
    )

    def _bm25(row: tuple[float, ...], doc_length: float) -> float:
        norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * doc_length / avg_doc_length)
        return sum(idf * tf * (_BM25_K1 + 1.0) / (tf + norm) for idf, tf in zip(idfs, row) if tf > 0.0)

    scored = tuple(
        (score, tool)
        for (tool, _, _), row, doc_length in zip(docs, tf_rows, doc_lengths)
        if (score := _bm25(row, doc_length)) > 0.0
    )
    ranked = sorted(scored, key=lambda item: (-item[0], str(item[1].get("name", ""))))
    return [tool for _, tool in ranked[:top_k]]


def get_virtual_tool_definitions() -> list[dict[str, Any]]:
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
                        "default": 5,
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
