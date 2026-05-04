"""
LiteLLM WebFetch Tool Definition

This module defines the standard web fetch tool used across LiteLLM.
Native provider tools (like Anthropic's web_fetch_20250305) are converted
to this format for consistent interception and execution.
"""

from typing import Any, Dict, Optional

from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME


# Native fetch tool formats that should be converted to LiteLLM standard
LITELLM_NATIVE_FETCH_TOOLS = [
    "web_fetch_20250305",  # Anthropic native format
    "web_fetch",  # Claude Code CLI
    "WebFetch",  # Legacy format
]


def get_litellm_web_fetch_tool() -> Dict[str, Any]:
    """
    Get the standard LiteLLM web fetch tool definition (Anthropic format).

    This is the canonical tool definition that all native web fetch tools
    (like Anthropic's web_fetch_20250305, Claude Code's web_fetch, etc.)
    are converted to for interception.

    Returns:
        Dict containing the Anthropic-style tool definition with:
        - name: Tool name
        - description: What the tool does
        - input_schema: JSON schema for tool parameters
    """
    return {
        "name": LITELLM_WEB_FETCH_TOOL_NAME,
        "description": (
            "Fetch and read the content of a specific URL. "
            "Use this when you need to read or analyze the content of a web page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and read content from",
                }
            },
            "required": ["url"],
        },
    }


def get_litellm_web_fetch_tool_openai() -> Dict[str, Any]:
    """
    Get the standard LiteLLM web fetch tool definition in OpenAI format.

    Used by async_pre_call_deployment_hook which runs in the chat completions
    path where tools must be in OpenAI format (type: "function" with
    function.parameters).

    Returns:
        Dict containing the OpenAI-style tool definition.
    """
    return {
        "type": "function",
        "function": {
            "name": LITELLM_WEB_FETCH_TOOL_NAME,
            "description": (
                "Fetch and read the content of a specific URL. "
                "Use this when you need to read or analyze the content of a web page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch and read content from",
                    }
                },
                "required": ["url"],
            },
        },
    }


def is_web_fetch_tool_chat_completion(tool: Dict[str, Any]) -> bool:
    """
    Check if a tool is a web fetch tool for Chat Completions API (strict check).

    This is a stricter version that ONLY checks for the exact LiteLLM web fetch tool name.
    Use this for Chat Completions API to avoid false positives with user-defined tools.

    Detects ONLY:
    - LiteLLM standard: name == "litellm_web_fetch" (Anthropic format)
    - OpenAI format: type == "function" with function.name == "litellm_web_fetch"

    Args:
        tool: Tool dictionary to check

    Returns:
        True if tool is exactly the LiteLLM web fetch tool
    """
    tool_name = tool.get("name", "")
    tool_type = tool.get("type", "")

    # Check for OpenAI format: {"type": "function", "function": {"name": "litellm_web_fetch"}}
    if tool_type == "function" and "function" in tool:
        function_def = tool.get("function", {})
        function_name = function_def.get("name", "")
        if function_name == LITELLM_WEB_FETCH_TOOL_NAME:
            return True

    # Check for LiteLLM standard tool (Anthropic format)
    if tool_name == LITELLM_WEB_FETCH_TOOL_NAME:
        return True

    return False


def is_web_fetch_tool(tool: Dict[str, Any]) -> bool:
    """
    Check if a tool is a web fetch tool (native or LiteLLM standard).

    Detects:
    - LiteLLM standard: name == "litellm_web_fetch"
    - OpenAI format: type == "function" with function.name == "litellm_web_fetch"
    - Anthropic native: type starts with "web_fetch_" (e.g., "web_fetch_20250305")
    - Claude Code: name == "web_fetch" with a type field
    - Custom: name == "WebFetch" (legacy format)

    Args:
        tool: Tool dictionary to check

    Returns:
        True if tool is a web fetch tool
    """
    tool_name = tool.get("name", "")
    tool_type = tool.get("type", "")

    # Check for OpenAI format: {"type": "function", "function": {"name": "..."}}
    if tool_type == "function" and "function" in tool:
        function_def = tool.get("function", {})
        function_name = function_def.get("name", "")
        if function_name == LITELLM_WEB_FETCH_TOOL_NAME:
            return True

    # Check for LiteLLM standard tool (Anthropic format)
    if tool_name == LITELLM_WEB_FETCH_TOOL_NAME:
        return True

    # Check for native Anthropic web_fetch_* types
    if tool_type and str(tool_type).startswith("web_fetch_"):
        return True

    # Check for Claude Code's web_fetch with a type field
    if tool_name == "web_fetch" and tool_type:
        return True

    # Check for legacy WebFetch format
    if tool_name == "WebFetch":
        return True

    return False


def is_native_fetch_tool(name: str, tool_type: Optional[str] = None) -> bool:
    """Check if a tool name/type is a native fetch tool (not LiteLLM standard)."""
    if name == LITELLM_WEB_FETCH_TOOL_NAME:
        return False  # Already standard
    if name in LITELLM_NATIVE_FETCH_TOOLS:
        return True
    if tool_type and str(tool_type).startswith("web_fetch_"):
        return True
    return False


def convert_native_fetch_to_litellm(native_tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a native fetch tool to LiteLLM standard format."""
    return get_litellm_web_fetch_tool()
