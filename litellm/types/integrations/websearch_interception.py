"""
Type definitions for WebSearch Interception integration.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel


class AnthropicSearchQuery(BaseModel):
    """``input`` of an Anthropic ``server_tool_use`` block for a web search."""

    query: str


class AnthropicServerToolUseBlock(BaseModel):
    """
    The ``server_tool_use`` block that must accompany a ``web_search_tool_result``.

    Anthropic requires the pair, with a ``srvtoolu_``-prefixed id shared by both.
    """

    type: Literal["server_tool_use"] = "server_tool_use"
    id: str
    name: Literal["web_search"] = "web_search"
    input: AnthropicSearchQuery


class WebSearchInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for WebSearchInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          websearch_interception_params:
            enabled_providers: ["bedrock"]
            search_tool_name: "my-perplexity-search"
    """

    enabled_providers: list[str]
    """List of LLM provider names to enable interception for (e.g., ['bedrock', 'vertex_ai'])"""

    search_tool_name: str | None
    """Name of search tool configured in router's search_tools. If None, uses first available."""
