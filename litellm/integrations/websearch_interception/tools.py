"""
LiteLLM Web Search Tool Definition

This module defines the standard web search tool used across LiteLLM.
Native provider tools (like Anthropic's web_search_20250305) are converted
to this format for consistent interception and execution.
"""

from typing import Any, Dict

from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME


def get_litellm_web_search_tool() -> Dict[str, Any]:
    """
    Get the web search tool definition in Anthropic format.

    Uses the same name and schema that Claude Code expects so it appears
    as the native WebSearch tool in the client.

    Returns:
        Dict containing the Anthropic-style tool definition.
    """
    return {
        "name": LITELLM_WEB_SEARCH_TOOL_NAME,
        "description": (
            "Search the web for current information. Returns search results "
            "with titles, URLs, and snippets. Use this tool when you need "
            "up-to-date information beyond your knowledge cutoff."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to use",
                },
            },
            "required": ["query"],
        },
    }


def get_litellm_web_search_tool_openai() -> Dict[str, Any]:
    """
    Get the web search tool definition in OpenAI format.

    Used by async_pre_call_deployment_hook which runs in the chat completions
    path where tools must be in OpenAI format (type: "function" with
    function.parameters).

    Returns:
        Dict containing the OpenAI-style tool definition.
    """
    return {
        "type": "function",
        "function": {
            "name": LITELLM_WEB_SEARCH_TOOL_NAME,
            "description": (
                "Search the web for current information. Returns search results "
                "with titles, URLs, and snippets. Use this tool when you need "
                "up-to-date information beyond your knowledge cutoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use",
                    },
                },
                "required": ["query"],
            },
        },
    }


_WEB_SEARCH_NAMES = {LITELLM_WEB_SEARCH_TOOL_NAME, "WebSearch", "web_search", "litellm_web_search"}


def is_web_search_tool_chat_completion(tool: Dict[str, Any]) -> bool:
    """
    Check if a tool is a web search tool for Chat Completions API.

    Detects:
    - Anthropic format: name in {litellm_web_search, WebSearch, web_search}
    - OpenAI format: type == "function" with function.name in same set

    Args:
        tool: Tool dictionary to check

    Returns:
        True if tool is a web search tool
    """
    tool_name = tool.get("name", "")
    tool_type = tool.get("type", "")

    # Check for OpenAI format
    if tool_type == "function" and "function" in tool:
        function_def = tool.get("function", {})
        function_name = function_def.get("name", "")
        if function_name in _WEB_SEARCH_NAMES:
            return True

    # Check for Anthropic format
    if tool_name in _WEB_SEARCH_NAMES:
        return True

    return False


def is_web_search_tool(tool: Dict[str, Any]) -> bool:
    """
    Check if a tool is a web search tool (native or LiteLLM standard).

    Detects:
    - LiteLLM standard: name == "litellm_web_search"
    - OpenAI format: type == "function" with function.name == "litellm_web_search"
    - Anthropic native: type starts with "web_search_" (e.g., "web_search_20250305")
    - Claude Code: name == "web_search" with a type field
    - Custom: name == "WebSearch" (legacy format)

    Args:
        tool: Tool dictionary to check

    Returns:
        True if tool is a web search tool

    Example:
        >>> is_web_search_tool({"name": "litellm_web_search"})
        True
        >>> is_web_search_tool({"type": "function", "function": {"name": "litellm_web_search"}})
        True
        >>> is_web_search_tool({"type": "web_search_20250305", "name": "web_search"})
        True
        >>> is_web_search_tool({"name": "calculator"})
        False
    """
    tool_name = tool.get("name", "")
    tool_type = tool.get("type", "")

    # Check for OpenAI format: {"type": "function", "function": {"name": "..."}}
    if tool_type == "function" and "function" in tool:
        function_def = tool.get("function", {})
        function_name = function_def.get("name", "")
        if function_name == LITELLM_WEB_SEARCH_TOOL_NAME:
            return True

    # Check for LiteLLM standard tool (Anthropic format)
    if tool_name == LITELLM_WEB_SEARCH_TOOL_NAME:
        return True

    # Check for native Anthropic web_search_* types
    if tool_type.startswith("web_search_"):
        return True

    # Check for Claude Code's web_search with a type field
    if tool_name == "web_search" and tool_type:
        return True

    # Check for legacy names
    if tool_name in ("WebSearch", "litellm_web_search"):
        return True

    return False
