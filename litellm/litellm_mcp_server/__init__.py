"""
LiteLLM MCP Server - Expose LiteLLM functionality as MCP tools.

This module provides an MCP (Model Context Protocol) server that exposes
LiteLLM's core LLM operations (chat completions, embeddings, image generation,
transcription, text completion, reranking) as MCP tools.

Usage:
    # Standalone server (stdio transport for Claude Desktop, Cursor, etc.)
    litellm-mcp-server

    # With HTTP transport
    litellm-mcp-server --transport http --port 8000

    # Programmatic usage
    from litellm.litellm_mcp_server import create_mcp_server
    server = create_mcp_server()
"""

from litellm.litellm_mcp_server.server import create_mcp_server

__all__ = ["create_mcp_server"]
