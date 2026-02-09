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
    Get the standard LiteLLM web search tool definition.

    This is the canonical tool definition that all native web search tools
    (like Anthropic's web_search_20250305, Claude Code's web_search, etc.)
    are converted to for interception.

    Returns:
        Dict containing the Anthropic-style tool definition with:
        - name: Tool name
        - description: What the tool does
        - input_schema: JSON schema for tool parameters

    Example:
        >>> tool = get_litellm_web_search_tool()
        >>> tool['name']
        'litellm_web_search'
    """
    return {
        "name": LITELLM_WEB_SEARCH_TOOL_NAME,
        "description": (
            "Search the web for information. Use this when you need current "
            "information or answers to questions that require up-to-date data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute"
                }
            },
            "required": ["query"]
        }
    }


def is_web_search_tool_chat_completion(tool: Dict[str, Any]) -> bool:
    """
    Check if a tool is a web search tool for Chat Completions API (strict check).
    
    This is a stricter version that ONLY checks for the exact LiteLLM web search tool name.
    Use this for Chat Completions API to avoid false positives with user-defined tools.

    Detects ONLY:
    - LiteLLM standard: name == "litellm_web_search" (Anthropic format)
    - OpenAI format: type == "function" with function.name == "litellm_web_search"

    Args:
        tool: Tool dictionary to check

    Returns:
        True if tool is exactly the LiteLLM web search tool

    Example:
        >>> is_web_search_tool_chat_completion({"name": "litellm_web_search"})
        True
        >>> is_web_search_tool_chat_completion({"type": "function", "function": {"name": "litellm_web_search"}})
        True
        >>> is_web_search_tool_chat_completion({"name": "web_search"})
        False
        >>> is_web_search_tool_chat_completion({"name": "WebSearch"})
        False
    """
    tool_name = tool.get("name", "")
    tool_type = tool.get("type", "")
    
    # Check for OpenAI format: {"type": "function", "function": {"name": "litellm_web_search"}}
    if tool_type == "function" and "function" in tool:
        function_def = tool.get("function", {})
        function_name = function_def.get("name", "")
        if function_name == LITELLM_WEB_SEARCH_TOOL_NAME:
            return True

    # Check for LiteLLM standard tool (Anthropic format)
    if tool_name == LITELLM_WEB_SEARCH_TOOL_NAME:
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

    # Check for legacy WebSearch format
    if tool_name == "WebSearch":
        return True

    return False
