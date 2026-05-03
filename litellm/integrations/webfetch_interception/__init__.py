"""
WebFetch Interception Module

Provides server-side WebFetch tool execution for models that don't natively
support server-side tool calling (e.g., Bedrock/Claude).
"""

from litellm.integrations.webfetch_interception.handler import (
    WebFetchInterceptionLogger,
)
from litellm.integrations.webfetch_interception.tools import (
    get_litellm_web_fetch_tool,
    is_web_fetch_tool,
)

__all__ = [
    "WebFetchInterceptionLogger",
    "get_litellm_web_fetch_tool",
    "is_web_fetch_tool",
]
