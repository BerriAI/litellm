"""Codex Responses API compatibility helpers.

Codex (code mode) does not declare its tools in the top-level ``tools`` array.
It sends an ``additional_tools`` input item instead, and leaves ``tools`` empty:

    {"tools": [],
     "input": [{"type": "additional_tools", "role": "developer",
                "tools": [{"type": "namespace", "name": "functions",
                           "tools": [...the real tools...]}]},
               ...messages...]}

``additional_tools`` is not part of the public Responses schema, so
OpenAI-compatible engines do not understand it:

- vLLM 0.25.x  -> 400 "cannot pickle 'pydantic_core...ValidatorIterator' object"
- SGLang       -> 400 "Unsupported Responses API input item type"
- vLLM 0.23.x / 0.26.x -> 200, but the item is ignored, so the model receives no
  tools at all and silently never calls one.

``hoist_codex_additional_tools`` normalises the payload for these providers:

1. Hoist the inner tools (unwrapping ``namespace`` entries) into ``tools``.
   Dropping the item is not sufficient - Codex sends an empty ``tools`` array,
   so dropping leaves the model with nothing to call.
2. Rewrite ``custom`` (freeform) tools as functions taking a single string
   argument, because engines skip any tool whose ``type`` is not ``function``.
   Note this discards ``format`` (lark/regex grammar), so such tools are no
   longer grammar-constrained.

BerriAI/litellm#33228 applies the same hoist for bedrock_mantle; this is the
provider-agnostic version.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_logger

ADDITIONAL_TOOLS_TYPE = "additional_tools"
NAMESPACE_TOOL_TYPE = "namespace"
CUSTOM_TOOL_TYPE = "custom"


def _shim_custom_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Represent a freeform ``custom`` tool as a single-string-arg function.

    The description is forwarded verbatim. Codex ships the whole code-mode API
    surface in it (~15KB for ``exec``), and the model cannot write valid calls
    against a truncated one.
    """
    return {
        "type": "function",
        "name": tool.get("name"),
        "description": tool.get("description") or "",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Raw tool input, passed through verbatim.",
                }
            },
            "required": ["input"],
        },
    }


def _expand_namespace(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten namespace entries into the tools they wrap.

    Recursive: Codex nests a namespace per MCP server, and leaving an inner
    ``namespace`` entry in place would put the very tool type we are removing
    back into ``tools``.
    """
    if entry.get("type") != NAMESPACE_TOOL_TYPE:
        return [entry]

    expanded: List[Dict[str, Any]] = []
    for tool in entry.get("tools") or []:
        if isinstance(tool, dict):
            expanded.extend(_expand_namespace(tool))
    return expanded


def hoist_codex_additional_tools(
    input: Union[str, List[Any]],
    tools: Optional[List[Any]],
) -> Tuple[Union[str, List[Any]], Optional[List[Any]], bool]:
    """Hoist Codex ``additional_tools`` items into a top-level tools array.

    Args:
        input: The Responses API ``input`` value.
        tools: The existing top-level ``tools`` value, if any.

    Returns:
        Tuple of (new_input, new_tools, changed). When no ``additional_tools``
        item is present the inputs are returned unchanged and ``changed`` is
        False, so callers can skip rewriting the request.
    """
    if not isinstance(input, list):
        return input, tools, False

    remaining_input: List[Any] = []
    hoisted: List[Dict[str, Any]] = []
    found = False

    for item in input:
        if isinstance(item, dict) and item.get("type") == ADDITIONAL_TOOLS_TYPE:
            found = True
            for entry in item.get("tools") or []:
                if not isinstance(entry, dict):
                    continue
                for tool in _expand_namespace(entry):
                    if tool.get("type") == CUSTOM_TOOL_TYPE:
                        hoisted.append(_shim_custom_tool(tool))
                    else:
                        hoisted.append(tool)
        else:
            remaining_input.append(item)

    if not found:
        return input, tools, False

    merged: List[Any] = list(tools or [])
    seen = {t.get("name") for t in merged if isinstance(t, dict) and t.get("name")}
    for tool in hoisted:
        name = tool.get("name")
        if not name:
            # Built-ins such as {"type": "web_search"} carry no name. Forward
            # them rather than dropping the capability on the floor.
            merged.append(tool)
            continue
        if name in seen:
            verbose_logger.warning(
                "codex_compat: dropping duplicate hoisted tool %r; a tool with "
                "that name is already declared. Namespaces are flattened, so "
                "same-named tools in different namespaces collide.",
                name,
            )
            continue
        merged.append(tool)
        seen.add(name)

    return remaining_input, merged, True
