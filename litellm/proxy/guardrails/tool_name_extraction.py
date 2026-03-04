"""
Extract tool names from request body by route/call type.

Used by auth (check_tools_allowlist) and ToolPolicyGuardrail so tool-format
knowledge lives in one place. Uses guardrail translation handlers where available,
with standalone extractors for generate_content and MCP.
"""

from typing import Any, Dict, List

from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.llms import load_guardrail_translation_mappings
from litellm.types.utils import CallTypes

# Call types that have no guardrail translation handler; we use standalone extractors
STANDALONE_EXTRACTORS: Dict[str, Any] = {}


def _extract_generate_content_tool_names(data: dict) -> List[str]:
    """Google generateContent: tools[].functionDeclarations[].name"""
    names: List[str] = []
    for tool in data.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        for decl in tool.get("functionDeclarations") or []:
            if isinstance(decl, dict) and decl.get("name"):
                names.append(str(decl["name"]))
    return names


def _extract_mcp_tool_names(data: dict) -> List[str]:
    """MCP call_tool: name or mcp_tool_name in body"""
    names: List[str] = []
    name = data.get("name") or data.get("mcp_tool_name")
    if name:
        names.append(str(name))
    return names


def _register_standalone_extractors() -> None:
    if STANDALONE_EXTRACTORS:
        return
    STANDALONE_EXTRACTORS[CallTypes.generate_content.value] = _extract_generate_content_tool_names
    STANDALONE_EXTRACTORS[CallTypes.agenerate_content.value] = _extract_generate_content_tool_names
    STANDALONE_EXTRACTORS[CallTypes.call_mcp_tool.value] = _extract_mcp_tool_names


# Tool-capable call types (routes that can send tools in the request)
TOOL_CAPABLE_CALL_TYPES = frozenset({
    CallTypes.completion.value,
    CallTypes.acompletion.value,
    CallTypes.responses.value,
    CallTypes.aresponses.value,
    CallTypes.anthropic_messages.value,
    CallTypes.generate_content.value,
    CallTypes.agenerate_content.value,
    CallTypes.call_mcp_tool.value,
})


def extract_request_tool_names(route: str, data: dict) -> List[str]:
    """
    Extract tool names from the request body for the given route.
    Uses guardrail translation handlers when available, else standalone extractors
    for generate_content and MCP. Returns [] for non-tool-capable routes or when
    no tools are present.
    """
    call_types = get_call_types_for_route(route)
    if not call_types:
        return []
    _register_standalone_extractors()
    mappings = load_guardrail_translation_mappings()
    for call_type in call_types:
        if not isinstance(call_type, CallTypes):
            continue
        if call_type.value not in TOOL_CAPABLE_CALL_TYPES:
            continue
        if call_type.value in STANDALONE_EXTRACTORS:
            return STANDALONE_EXTRACTORS[call_type.value](data)
        handler_cls = mappings.get(call_type)
        if handler_cls is not None:
            names = handler_cls().extract_request_tool_names(data)
            if names:
                return names
    return []
