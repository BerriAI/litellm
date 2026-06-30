"""
Utilities for handling OpenAI Responses API 'custom' tools (freeform/grammar tools)
when bridging to Chat Completions providers.

Custom tools are defined with ``type: "custom"`` and a grammar/format specification.
Since most Chat Completions providers only support standard ``function`` tools,
the bridge converts them to ``function`` tools with a single ``content`` string
parameter. When the model responds with a ``function_call`` for such a tool, this
module converts it back to the ``custom_tool_call`` format expected by clients like
Codex CLI.

The forward direction (custom -> function) and reverse direction (function_call ->
custom_tool_call) are both handled here so future custom tool types can be added by
extending this module without touching the streaming iterator or transformation
logic.
"""

import json
from typing import Any

_MAX_ARGUMENTS_LEN = 1_000_000


def extract_custom_tool_names(tools: list[Any] | None) -> set[str]:
    """Extract names of tools originally defined as ``type: "custom"``."""
    if not tools:
        return set()
    names: set[str] = set()
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "custom" and "name" in tool:
            names.add(tool["name"])
    return names


def is_custom_tool_call(tool_name: str, custom_tool_names: set[str]) -> bool:
    """Check if a tool call name corresponds to a custom tool."""
    return tool_name in custom_tool_names


def unwrap_custom_tool_arguments(arguments: str) -> str:
    """Extract the raw content string from JSON-wrapped arguments.

    The bridge converts custom tools to function tools with schema
    ``{"properties": {"content": {"type": "string"}}}``, so the model returns
    arguments like ``{"content": "*** Begin Patch\\n..."}``. This function
    extracts just the content string. If the arguments are not valid JSON or do
    not contain a ``content`` key, the original string is returned unchanged.
    """
    if not arguments:
        return ""
    if len(arguments) > _MAX_ARGUMENTS_LEN:
        return arguments
    try:
        parsed = json.loads(arguments)
        if isinstance(parsed, dict) and "content" in parsed:
            return str(parsed["content"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return arguments


def build_tool_call_item_kwargs(
    call_id: str,
    name: str,
    arguments_or_input: str,
    status: str,
    custom_tool_names: set[str],
) -> dict[str, Any]:
    """Build kwargs for an output item dict that is either a ``function_call``
    or a ``custom_tool_call`` depending on whether *name* is in
    *custom_tool_names*.

    For custom tools the ``arguments`` JSON is unwrapped into the ``input``
    field. For regular function tools the raw ``arguments`` string is kept.

    This centralises the branching logic so the streaming iterator and the
    non-streaming transformation share a single code path.
    """
    custom = is_custom_tool_call(name, custom_tool_names)
    item_type = "custom_tool_call" if custom else "function_call"
    kwargs: dict[str, Any] = {
        "type": item_type,
        "id": call_id,
        "call_id": call_id,
        "name": name,
        "status": status,
    }
    if custom:
        if status == "completed":
            kwargs["input"] = unwrap_custom_tool_arguments(arguments_or_input)
        else:
            kwargs["input"] = ""
    else:
        kwargs["arguments"] = arguments_or_input
    return kwargs


def build_custom_tool_call_item(
    call_id: str,
    name: str,
    input_str: str,
    status: str = "completed",
) -> dict[str, Any]:
    """Build a standalone ``custom_tool_call`` output item dict."""
    if not call_id:
        raise ValueError("call_id is required for custom_tool_call output item")
    return {
        "type": "custom_tool_call",
        "call_id": call_id,
        "id": call_id,
        "name": name,
        "input": input_str,
        "status": status,
    }


def convert_custom_tool_to_function_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Responses API ``custom`` tool to a Chat Completions ``function``
    tool.

    The grammar definition is embedded in the description so the model can
    produce correctly-formatted output. Returns ``None`` if the tool is not a
    valid custom tool.
    """
    if tool.get("type") != "custom":
        return None
    name = tool.get("name", "")
    desc = tool.get("description", "")
    fmt = tool.get("format", {})
    if isinstance(fmt, dict) and fmt.get("definition"):
        syntax = fmt.get("syntax", "")
        definition = fmt.get("definition", "")
        desc = desc + "\n\nFormat:\n```" + syntax + "\n" + definition + "\n```"
    allowed_callers = tool.get("allowed_callers")
    if allowed_callers is not None and not (
        isinstance(allowed_callers, list) and all(isinstance(item, str) for item in allowed_callers)
    ):
        raise ValueError("allowed_callers must be a list of strings")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": f"The {name} content following the specified format",
                    }
                },
                "required": ["content"],
            },
        },
        **({"allowed_callers": allowed_callers} if allowed_callers is not None else {}),
    }
