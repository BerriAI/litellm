"""Responses-side adapter for the proxy's mcp_tool_search virtual tools.

``server.py`` owns the virtual tools; this module only supplies what the
Responses path needs on top of them: the request's ``allowed_tools`` (a
Responses-level concept ``execute_mcp_tool`` knows nothing about) and the inner
tool name, so the caller can spend-log the real tool rather than the wrapper.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from litellm.constants import MCP_VIRTUAL_TOOL_SEARCH_SERVER_NAME
from litellm.proxy._experimental.mcp_server.utils import split_server_prefix_from_name
from litellm.responses.mcp.mcp_streaming_iterator import MAX_MCP_TOOL_CALL_ROUNDS

if TYPE_CHECKING:
    from mcp.types import CallToolResult
    from mcp.types import Tool as MCPTool

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth

_SEARCH_RESULTS_ADAPTER: Final = TypeAdapter(tuple[Mapping[str, object], ...])


@dataclass(frozen=True, slots=True)
class VirtualToolScope:
    """Request-level scoping the virtual tools must reproduce, so a two-hop
    search + call sees exactly what a direct catalog listing would have."""

    mcp_servers: tuple[str, ...]
    allowed_tools: frozenset[str]

    @property
    def server_filter(self) -> list[str] | None:  # mutable-ok: handle_mcp_tool_* takes Optional[list[str]]
        return list(self.mcp_servers) or None  # mutable-ok: same


@dataclass(frozen=True, slots=True)
class VirtualToolCallPlan:
    """The real tool an ``mcp_tool_call`` unwraps to."""

    tool_name: str
    arguments: Mapping[str, object]


class _VirtualToolCallArguments(BaseModel):
    tool_name: str
    arguments: Mapping[str, object] = Field(default_factory=dict)


def max_tool_call_rounds(scope: VirtualToolScope | None) -> int:
    """Round budget for one auto-executed request. Doubled for the virtual
    tools, which spend two rounds per real tool use, so the flag does not halve
    a key's effective tool budget."""
    if scope is None:
        return MAX_MCP_TOOL_CALL_ROUNDS
    return MAX_MCP_TOOL_CALL_ROUNDS * 2


def is_mcp_tool_search_enabled(user_api_key_auth: object) -> bool:
    object_permission: Final = getattr(user_api_key_auth, "object_permission", None)
    return bool(getattr(object_permission, "mcp_tool_search_enabled", False))


def virtual_tool_definitions() -> tuple[MCPTool, ...]:
    """The two virtual tools, as MCP ``Tool`` objects. Any other definition the
    ``/mcp`` path may grow is excluded: this path only supports search + call."""
    from mcp.types import Tool

    from litellm.proxy._experimental.mcp_server.tool_search import (
        MCP_TOOL_CALL_TOOL_NAME,
        MCP_TOOL_SEARCH_TOOL_NAME,
        get_virtual_tool_definitions,
    )

    wanted: Final = (MCP_TOOL_SEARCH_TOOL_NAME, MCP_TOOL_CALL_TOOL_NAME)
    return tuple(
        Tool.model_validate(definition)
        for definition in get_virtual_tool_definitions()
        if definition.get("name") in wanted
    )


def virtual_tool_server_map() -> Mapping[str, str]:
    """``tool_server_map`` entries marking the two names as virtual, so a real
    upstream tool of the same name still dispatches normally."""
    return MappingProxyType({tool.name: MCP_VIRTUAL_TOOL_SEARCH_SERVER_NAME for tool in virtual_tool_definitions()})


def is_tool_name_allowed(tool_name: str, allowed_tools: frozenset[str]) -> bool:
    if not allowed_tools or tool_name in allowed_tools:
        return True
    unprefixed_name, _ = split_server_prefix_from_name(tool_name)
    return unprefixed_name in allowed_tools


def resolve_virtual_tool_call(
    arguments: Mapping[str, object],
    scope: VirtualToolScope,
) -> VirtualToolCallPlan | str:
    """The plan naming the real tool, or the rejection text to hand back as the tool result."""
    try:
        parsed: Final = _VirtualToolCallArguments.model_validate(arguments)
    except ValidationError:
        return "mcp_tool_call requires a 'tool_name' string argument"

    if not parsed.tool_name:
        return "mcp_tool_call requires a non-empty 'tool_name'"

    if not is_tool_name_allowed(parsed.tool_name, scope.allowed_tools):
        return f"Tool '{parsed.tool_name}' is not in allowed_tools for this request"

    return VirtualToolCallPlan(tool_name=parsed.tool_name, arguments=parsed.arguments)


async def run_virtual_tool_search(
    query: str,
    top_k: int,
    scope: VirtualToolScope,
    user_api_key_auth: UserAPIKeyAuth,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,  # mutable-ok: handle_mcp_tool_* takes dict
    oauth2_headers: dict[str, str] | None = None,  # mutable-ok: handle_mcp_tool_* takes dict
    raw_headers: dict[str, str] | None = None,  # mutable-ok: handle_mcp_tool_* takes dict
) -> CallToolResult:
    from litellm.proxy._experimental.mcp_server.tool_search import handle_mcp_tool_search

    result: Final = await handle_mcp_tool_search(
        query=query,
        top_k=top_k,
        user_api_key_dict=user_api_key_auth,
        mcp_servers=scope.server_filter,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
    )
    return _apply_allowed_tools_to_search_result(result, scope.allowed_tools)


async def run_virtual_tool_call(
    plan: VirtualToolCallPlan,
    scope: VirtualToolScope,
    user_api_key_auth: UserAPIKeyAuth,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,  # mutable-ok: handle_mcp_tool_* takes dict
    oauth2_headers: dict[str, str] | None = None,  # mutable-ok: handle_mcp_tool_* takes dict
    raw_headers: dict[str, str] | None = None,  # mutable-ok: handle_mcp_tool_* takes dict
    litellm_logging_obj: LiteLLMLoggingObj | None = None,
) -> CallToolResult:
    from litellm.proxy._experimental.mcp_server.tool_search import handle_mcp_tool_call

    return await handle_mcp_tool_call(
        tool_name=plan.tool_name,
        arguments=dict(plan.arguments),  # mutable-ok: handle_mcp_tool_call takes dict[str, Any]
        user_api_key_dict=user_api_key_auth,
        mcp_servers=scope.server_filter,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        litellm_logging_obj=litellm_logging_obj,
    )


def _apply_allowed_tools_to_search_result(
    result: CallToolResult,
    allowed_tools: frozenset[str],
) -> CallToolResult:
    if not allowed_tools or result.isError:
        return result

    from mcp.types import CallToolResult, TextContent

    return CallToolResult(
        content=[  # mutable-ok: CallToolResult.content is a list field
            TextContent(type="text", text=_filtered_search_text(item.text, allowed_tools))
            if isinstance(item, TextContent)
            else item
            for item in result.content
        ],
        isError=result.isError,
    )


def _filtered_search_text(text: str, allowed_tools: frozenset[str]) -> str:
    try:
        entries: Final[Sequence[Mapping[str, object]]] = _SEARCH_RESULTS_ADAPTER.validate_json(text)
    except ValidationError:
        return text

    return json.dumps(
        tuple(
            entry
            for entry in entries
            if isinstance(name := entry.get("name"), str) and is_tool_name_allowed(name, allowed_tools)
        )
    )
