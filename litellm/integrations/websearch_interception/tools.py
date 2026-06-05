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
                    "description": "The search query to execute",
                }
            },
            "required": ["query"],
        },
    }


def get_litellm_web_search_tool_openai() -> Dict[str, Any]:
    """
    Get the standard LiteLLM web search tool definition in OpenAI format.

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
                "Search the web for information. Use this when you need current "
                "information or answers to questions that require up-to-date data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute",
                    }
                },
                "required": ["query"],
            },
        },
    }


def get_litellm_web_search_tool_responses_api() -> Dict[str, Any]:
    """
    Get the standard LiteLLM web search tool definition in OpenAI Responses API format.

    Responses-API function tools are flat (no nested ``function`` key):
    ``{"type": "function", "name": "...", "description": "...", "parameters": {...}}``.

    Used by ``WebSearchInterceptionLogger.async_pre_call_deployment_hook`` for
    ``call_type == "aresponses"`` to convert server-hosted ``web_search`` tools
    (which providers like Bedrock Mantle reject) into a function-typed tool we
    can intercept and execute server-side.
    """
    return {
        "type": "function",
        "name": LITELLM_WEB_SEARCH_TOOL_NAME,
        "description": (
            "Search the web for information. Use this when you need current "
            "information or answers to questions that require up-to-date data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute",
                }
            },
            "required": ["query"],
        },
    }


def is_web_search_tool_responses_api(tool: Dict[str, Any]) -> bool:
    """
    Check if a tool is a web search tool in OpenAI Responses API shape.

    Detects:
    - Server-hosted web search:  ``{"type": "web_search"}`` /
      ``{"type": "web_search_preview"}`` (sent by Codex CLI and the OpenAI SDK
      when ``web_search`` is enabled).
    - LiteLLM standard, flat:    ``{"type": "function", "name": "litellm_web_search"}``
    - Anthropic-native variants: ``{"type": "web_search_*"}`` (forwarded
      verbatim by some clients).
    """
    tool_type = tool.get("type", "")
    if not isinstance(tool_type, str):
        return False
    if tool_type == "web_search" or tool_type.startswith("web_search_"):
        return True
    if tool_type == "function" and tool.get("name") == LITELLM_WEB_SEARCH_TOOL_NAME:
        return True
    return False


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


def is_anthropic_native_web_search_tool(tool: Dict[str, Any]) -> bool:
    """
    Check if a tool is an Anthropic-native ``web_search_*`` tool.

    Native clients (Anthropic SDK, Claude Desktop, Anthropic Console) send
    tools like ``{"type": "web_search_20250305", "name": "web_search"}`` and
    expect the response to contain ``web_search_tool_result`` content blocks
    so that citations can be rendered. This helper identifies that contract
    so the agentic loop can emit native-format blocks for those clients
    without affecting clients that send the LiteLLM standard tool.

    Returns False for the LiteLLM standard tool (``litellm_web_search``),
    the OpenAI-shaped variant, the bare ``WebSearch`` legacy name, and the
    bare ``web_search`` name (Claude Code style).
    """
    tool_type = tool.get("type", "")
    if not isinstance(tool_type, str):
        return False
    return tool_type.startswith("web_search_") and tool_type != "function"


def is_web_search_tool(tool: Dict[str, Any]) -> bool:
    """
    Check if a tool is a web search tool (native or LiteLLM standard).

    Detects:
    - LiteLLM standard: name == "litellm_web_search"
    - OpenAI format: type == "function" with function.name == "litellm_web_search"
    - Anthropic native: type starts with "web_search_" (e.g., "web_search_20250305")
    - Claude Code: name == "web_search" with a type field
    - Custom: name == "WebSearch" (legacy interception marker — only matched
      when input_schema is absent; see note below)

    Note on the legacy ``WebSearch`` name:
        Clients like Claude Desktop / Cowork ship a *client-side* tool called
        ``WebSearch`` (a fully-formed Anthropic client tool with its own
        ``input_schema``) that they handle themselves. Treating that as our
        interception marker hijacks it server-side and the client's own tool
        handler never fires — which means Cowork's separate native
        ``web_search_20250305`` sub-request (where citation data actually
        flows) never gets made.

        Real Anthropic client tools always carry an ``input_schema`` (the API
        rejects them otherwise), so a bare ``{name: "WebSearch"}`` with no
        schema is the only thing that could be a legacy interception marker.
        Gate the match on schema absence to keep both groups working.

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
        >>> is_web_search_tool({"name": "WebSearch"})  # legacy interception marker
        True
        >>> is_web_search_tool({"name": "WebSearch", "input_schema": {"type": "object"}})  # Cowork client tool
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

    # Legacy "WebSearch" interception marker — only when no schema is
    # present, so real client-side WebSearch tools (Cowork) pass through.
    if tool_name == "WebSearch" and "input_schema" not in tool:
        return True

    return False
