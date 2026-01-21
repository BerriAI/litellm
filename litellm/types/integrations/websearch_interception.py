"""
Type definitions for WebSearch Interception integration.
"""

from typing import List, Optional, TypedDict


class WebSearchInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for WebSearchInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          websearch_interception_params:
            enabled_providers: ["bedrock"]
            search_tool_name: "my-perplexity-search"
    """

    enabled_providers: List[str]
    """List of LLM provider names to enable interception for (e.g., ['bedrock', 'vertex_ai'])"""

    search_tool_name: Optional[str]
    """Name of search tool configured in router's search_tools. If None, uses first available."""
