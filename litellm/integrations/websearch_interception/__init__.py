"""
WebSearch Interception Module

Provides server-side WebSearch tool execution for models that don't natively
support server-side tool calling (e.g., Bedrock/Claude).
"""

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)

__all__ = ["WebSearchInterceptionLogger"]
