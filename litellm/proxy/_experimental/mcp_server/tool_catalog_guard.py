"""Discovery-time guard for an MCP server's tool catalog.

Two checks run on the tools an upstream returns from ``tools/list`` before the gateway
serves them. The guardrail scan hands each tool's description and input schema to the
``pre_mcp_call`` guardrails as a ``list_mcp_tools`` payload: a blocked tool leaves the
listing and a masked description is what the client sees. A pinned catalog replaces the
scan for servers whose admin snapshotted the tool list: only pinned tools are served, with
their pinned descriptions, and any upstream drift is reported once per distinct diff.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from mcp.types import Tool as MCPTool
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._experimental.mcp_server.utils import logging_safe_mcp_headers
from litellm.types.mcp import MCPPreCallRequestObject
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging


class _ServedCatalogEntry(TypedDict, total=False):
    description: ReadOnly[str | None]
    input_schema: ReadOnly[Mapping[str, object]]


class _ScanRequest(TypedDict):
    tool_name: ReadOnly[str]
    arguments: ReadOnly[Mapping[str, object]]
    server_name: ReadOnly[str]


class _ScanKwargs(TypedDict):
    name: ReadOnly[str]
    arguments: ReadOnly[Mapping[str, object]]
    server_name: ReadOnly[str]
    mcp_rate_limit_server_name: ReadOnly[str]
    user_api_key_auth: ReadOnly[UserAPIKeyAuth | None]
    user_api_key_user_id: ReadOnly[object]
    user_api_key_team_id: ReadOnly[object]
    user_api_key_end_user_id: ReadOnly[object]
    user_api_key_hash: ReadOnly[object]
    headers: ReadOnly[Mapping[str, str]]
    mcp_tool_description: ReadOnly[str]
    mcp_input_schema: ReadOnly[Mapping[str, object]]


_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_OPTIONAL_GUARDED: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)
_ERROR_DETAIL: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_OPTIONAL_TEXT: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)


@dataclass(frozen=True, slots=True)
class CatalogAlert:
    signature: str
    message: str


@dataclass(frozen=True, slots=True)
class BlockedTool:
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class ToolDescriptionScan:
    served: tuple[MCPTool, ...]
    blocked: tuple[BlockedTool, ...]

    def alert(self, server: MCPServer) -> CatalogAlert | None:
        if not self.blocked:
            return None
        lines: Final = "\n".join(f"- `{tool.name}`: {tool.reason}" for tool in self.blocked)
        return CatalogAlert(
            signature=",".join(sorted(tool.name for tool in self.blocked)),
            message=(
                f"MCP server `{server.name}`: {len(self.blocked)} tool description(s) blocked by a guardrail "
                f"and hidden from tools/list\n{lines}"
            ),
        )


@dataclass(frozen=True, slots=True)
class PinnedCatalogDrift:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    def alert(self, server: MCPServer) -> CatalogAlert:
        parts: Final = tuple(
            f"{label}: {', '.join(f'`{name}`' for name in names)}"
            for label, names in (("added", self.added), ("removed", self.removed), ("changed", self.changed))
            if names
        )
        return CatalogAlert(
            signature="|".join(parts),
            message=(
                f"MCP server `{server.name}`: upstream tool list drifted from the pinned catalog; "
                f"serving the pinned tools and descriptions until an admin re-pins the server\n" + "\n".join(parts)
            ),
        )


def pin_tool_catalog(
    tools: Sequence[MCPTool], pinned_tools: Mapping[str, str]
) -> tuple[tuple[MCPTool, ...], PinnedCatalogDrift | None]:
    upstream: Final = MappingProxyType({tool.name: tool for tool in tools})
    added: Final = tuple(sorted(name for name in upstream if name not in pinned_tools))
    removed: Final = tuple(sorted(name for name in pinned_tools if name not in upstream))
    changed: Final = tuple(
        sorted(
            name
            for name, tool in upstream.items()
            if name in pinned_tools and (tool.description or "") != pinned_tools[name]
        )
    )
    served: Final = tuple(
        _pinned_tool(tool, pinned_tools[tool.name]) if tool.name in changed else tool
        for tool in tools
        if tool.name in pinned_tools
    )
    drift: Final = PinnedCatalogDrift(added, removed, changed) if added or removed or changed else None
    return served, drift


def _pinned_tool(tool: MCPTool, description: str) -> MCPTool:
    entry: Final[_ServedCatalogEntry] = {"description": description}
    return _with_served_entry(tool, entry)


async def scan_tool_descriptions(
    tools: Sequence[MCPTool],
    server: MCPServer,
    proxy_logging_obj: ProxyLogging,
    user_api_key_auth: UserAPIKeyAuth | None,
    raw_headers: Mapping[str, str] | None,
) -> ToolDescriptionScan:
    outcomes: Final = await asyncio.gather(
        *(_scan_tool(tool, server, proxy_logging_obj, user_api_key_auth, raw_headers) for tool in tools)
    )
    return ToolDescriptionScan(
        served=tuple(outcome for outcome in outcomes if isinstance(outcome, MCPTool)),
        blocked=tuple(outcome for outcome in outcomes if isinstance(outcome, BlockedTool)),
    )


def _has_scannable_text(tool: MCPTool) -> bool:
    return bool(tool.description) or bool(tool.input_schema)


async def _scan_tool(
    tool: MCPTool,
    server: MCPServer,
    proxy_logging_obj: ProxyLogging,
    user_api_key_auth: UserAPIKeyAuth | None,
    raw_headers: Mapping[str, str] | None,
) -> MCPTool | BlockedTool:
    if not _has_scannable_text(tool):
        return tool
    request: Final[_ScanRequest] = {"tool_name": tool.name, "arguments": {}, "server_name": server.name}
    request_obj: Final = MCPPreCallRequestObject.model_validate(request)
    kwargs: Final[_ScanKwargs] = {
        "name": tool.name,
        "arguments": {},
        "server_name": server.name,
        "mcp_rate_limit_server_name": server.alias or server.server_name or server.name,
        "user_api_key_auth": user_api_key_auth,
        "user_api_key_user_id": getattr(user_api_key_auth, "user_id", None),
        "user_api_key_team_id": getattr(user_api_key_auth, "team_id", None),
        "user_api_key_end_user_id": getattr(user_api_key_auth, "end_user_id", None),
        "user_api_key_hash": getattr(user_api_key_auth, "api_key", None),
        "headers": logging_safe_mcp_headers(raw_headers),
        "mcp_tool_description": tool.description or "",
        "mcp_input_schema": tool.input_schema,
    }
    data: Final = _JSON_OBJECT.validate_python(
        proxy_logging_obj._convert_mcp_to_llm_format(request_obj, kwargs)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the tool-call path builds its guardrail payload through this same untyped helper
    )
    try:
        guarded: Final = _OPTIONAL_GUARDED.validate_python(
            await proxy_logging_obj.pre_call_hook(  # pyright: ignore[reportUnknownMemberType, reportCallIssue, reportUnknownArgumentType]  # untyped hook; its overloads want an auth the MCP call types tolerate missing
                user_api_key_dict=user_api_key_auth,  # pyright: ignore[reportArgumentType]  # the tool-call path passes the same optional auth
                data=data,
                call_type=CallTypes.list_mcp_tools.value,
                guardrails_only=True,
            )
        )
        return tool if guarded is None else _masked_tool(tool, guarded)
    except Exception as e:  # noqa: BLE001  # any guardrail failure hides the tool: fail closed
        return BlockedTool(name=tool.name, reason=_block_reason(e))


def _block_reason(exc: Exception) -> str:
    detail: Final[object] = getattr(exc, "detail", None)
    error: Final = _ERROR_DETAIL.validate_python(detail).get("error") if isinstance(detail, Mapping) else None
    if error:
        return str(error)
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _masked_tool(tool: MCPTool, guarded: Mapping[str, object]) -> MCPTool:
    entry: Final[_ServedCatalogEntry] = {
        "description": _OPTIONAL_TEXT.validate_python(guarded.get("mcp_tool_description", tool.description)),
        "input_schema": _JSON_OBJECT.validate_python(guarded.get("mcp_input_schema", tool.input_schema)),
    }
    unchanged: Final = entry["description"] == tool.description and entry["input_schema"] == tool.input_schema
    return tool if unchanged else _with_served_entry(tool, entry)


def _with_served_entry(tool: MCPTool, update: _ServedCatalogEntry) -> MCPTool:
    return tool.model_copy(deep=True, update=update)
