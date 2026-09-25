"""Compatibility-aware conversion of upstream outcomes into MCP SDK results.

Every gateway surface that turns a tool outcome (text, JSON, an SDK result, an
interim result, an exception) into the ``CallToolResult`` it sends downstream
goes through ``to_call_tool_result`` so the per-revision wire rules live in one
place. SDK 2.x serializes ``structuredContent`` as object-only on the handshake
revisions (``2024-11-05`` .. ``2025-11-25``) and admits any JSON value, plus
``input_required`` interim results, only on ``2026-07-28``.
"""

from __future__ import annotations

import json
from typing import Final, TypeAlias

from mcp.types import CallToolResult, ContentBlock, InputRequiredResult, TextContent, Tool
from typing_extensions import ReadOnly, TypedDict, assert_never

from litellm.proxy._experimental.mcp_server.tool_outcome import (
    JsonResult,
    TextResult,
    WireCompat,
    handler_outcome,
    parse_http_body,
    wire_compat_for,
)

__all__ = (
    "INPUT_REQUIRED_UNSUPPORTED_MESSAGE",
    "JsonResult",
    "TextResult",
    "ToolOutcome",
    "WireCompat",
    "complete_call_tool_result",
    "error_text_result",
    "handler_outcome",
    "parse_http_body",
    "to_call_tool_result",
    "to_gateway_tool",
    "wire_compat_for",
)

ToolOutcome: TypeAlias = TextResult | JsonResult | CallToolResult | InputRequiredResult | Exception


class _Downgraded(TypedDict):
    structured_content: ReadOnly[None]
    content: ReadOnly[list[ContentBlock]]  # mutable-ok: SDK list field


class _Renamed(TypedDict):
    name: ReadOnly[str]


INPUT_REQUIRED_UNSUPPORTED_MESSAGE: Final = (
    "Error: upstream tool returned an input_required interim result, which this MCP protocol revision cannot carry"
)


def error_text_result(exc: Exception) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=f"{type(exc).__name__}: {exc}")],  # mutable-ok: SDK list field
        is_error=True,
    )


def to_call_tool_result(outcome: ToolOutcome, compat: WireCompat) -> CallToolResult | InputRequiredResult:
    match outcome:
        case TextResult():
            return CallToolResult(
                content=[TextContent(type="text", text=outcome.text)], is_error=False
            )  # mutable-ok: SDK
        case JsonResult():
            keep_structured: Final = compat is WireCompat.MODERN or isinstance(outcome.value, dict)
            return CallToolResult(
                content=[TextContent(type="text", text=outcome.original_text)],  # mutable-ok: SDK list field
                is_error=False,
                structured_content=outcome.value if keep_structured else None,
            )
        case CallToolResult():
            return _downgrade_structured_content(outcome) if compat is WireCompat.LEGACY else outcome
        case InputRequiredResult():
            if compat is WireCompat.MODERN:
                return outcome
            return CallToolResult(
                content=[TextContent(type="text", text=INPUT_REQUIRED_UNSUPPORTED_MESSAGE)],  # mutable-ok: SDK
                is_error=True,
            )
        case Exception():
            return error_text_result(outcome)
        case _:
            return assert_never(outcome)


def complete_call_tool_result(outcome: ToolOutcome, compat: WireCompat) -> CallToolResult:
    """``to_call_tool_result`` for callers that can never carry an interim result."""
    converted: Final = to_call_tool_result(outcome, compat)
    if isinstance(converted, InputRequiredResult):
        return CallToolResult(
            content=[TextContent(type="text", text=INPUT_REQUIRED_UNSUPPORTED_MESSAGE)],  # mutable-ok: SDK
            is_error=True,
        )
    return converted


def _downgrade_structured_content(result: CallToolResult) -> CallToolResult:
    structured: Final = result.structured_content
    if structured is None or isinstance(structured, dict):
        return result
    fallback: Final = TextContent(type="text", text=json.dumps(structured))
    update: Final[_Downgraded] = {
        "structured_content": None,
        "content": [*result.content, fallback],  # mutable-ok: SDK list field
    }
    return result.model_copy(update=update)


def to_gateway_tool(tool: Tool, name: str) -> Tool:
    update: Final[_Renamed] = {"name": name}
    return tool.model_copy(deep=True, update=update)
