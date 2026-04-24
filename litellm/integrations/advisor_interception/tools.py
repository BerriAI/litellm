"""
LiteLLM advisor tool definitions and detection helpers.
"""

from typing import Any, Dict, Optional

from litellm.constants import ADVISOR_TOOL_DESCRIPTION
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE

LITELLM_ADVISOR_TOOL_NAME = "litellm_advisor"
_ADVISOR_TOOL_NAMES = {LITELLM_ADVISOR_TOOL_NAME, "advisor"}


def get_litellm_advisor_tool(
    model: str,
    max_uses: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get the canonical advisor tool definition in Anthropic native format.
    """
    tool: Dict[str, Any] = {
        "type": ANTHROPIC_ADVISOR_TOOL_TYPE,
        "name": "advisor",
        "model": model,
    }
    if max_uses is not None:
        tool["max_uses"] = max_uses
    if api_key is not None:
        tool["api_key"] = api_key
    if api_base is not None:
        tool["api_base"] = api_base
    return tool


def get_litellm_advisor_tool_openai() -> Dict[str, Any]:
    """
    Get the canonical advisor tool definition in OpenAI function format.
    """
    return {
        "type": "function",
        "function": {
            "name": LITELLM_ADVISOR_TOOL_NAME,
            "description": ADVISOR_TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or challenge you want guidance on.",
                    }
                },
                "required": ["question"],
            },
        },
    }


def is_advisor_tool_chat_completion(tool: Any) -> bool:
    """
    Strict chat-completions advisor tool check.
    """
    if isinstance(tool, dict):
        tool_type = tool.get("type")
        function_def = tool.get("function", {}) or {}
        function_declarations = tool.get("function_declarations", [])
    else:
        tool_type = getattr(tool, "type", None)
        function_def = getattr(tool, "function", None) or {}
        function_declarations = getattr(tool, "function_declarations", None) or []

    if tool_type == "function":
        if not isinstance(function_def, dict):
            function_def = {
                "name": getattr(function_def, "name", None),
            }
        function_name = function_def.get("name")
        return function_name in _ADVISOR_TOOL_NAMES
    # Gemini tool schema: {"function_declarations": [{"name": "..."}]}
    if isinstance(function_declarations, list):
        for declaration in function_declarations:
            if isinstance(declaration, dict):
                name = declaration.get("name")
            else:
                name = getattr(declaration, "name", None)
            if name in _ADVISOR_TOOL_NAMES:
                return True
    return False


def is_advisor_tool(tool: Any) -> bool:
    """
    Check whether a tool is an advisor tool in any supported format.
    """
    tool_type = (
        tool.get("type") if isinstance(tool, dict) else getattr(tool, "type", None)
    )
    if tool_type == ANTHROPIC_ADVISOR_TOOL_TYPE:
        return True
    if is_advisor_tool_chat_completion(tool):
        return True
    tool_name = (
        tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
    )
    return tool_name in _ADVISOR_TOOL_NAMES
