"""
Type definitions for WebFetch Interception integration.
"""

from typing import List, Optional, TypedDict


class WebFetchInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for WebFetchInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          webfetch_interception_params:
            enabled_providers: ["bedrock"]
            fetch_tool_name: "my-firecrawl-fetch"
    """

    enabled_providers: List[str]
    """List of LLM provider names to enable interception for (e.g., ['bedrock', 'vertex_ai'])"""

    fetch_tool_name: Optional[str]
    """Name of fetch tool configured in router's fetch_tools. If None, uses first available."""
