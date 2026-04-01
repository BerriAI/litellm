"""
MCP Semantic Tool Filter Hook

Semantic filtering for MCP tools to reduce context window size
and improve tool selection accuracy.
"""
from litellm.proxy.hooks.mcp_semantic_filter.hook import SemanticToolFilterHook

__all__ = ["SemanticToolFilterHook"]
