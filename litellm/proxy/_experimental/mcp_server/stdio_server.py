from __future__ import annotations

"""
LiteLLM MCP (stdio) server entrypoint.

Provides a minimal stdio MCP surface for clients that speak the MCP stdio
transport. Reuses shared local tools (model.advice, llm.chat).
"""

import asyncio
from typing import Any, Dict, List

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.shared import (
    register_default_local_tools,
)
from litellm.proxy._experimental.mcp_server.tool_registry import (
    global_mcp_tool_registry,
)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "LiteLLM MCP stdio server requires the Python MCP package.\n"
        "Install with: pip install model-context-protocol\n"
        f"Import error: {e}"
    )


def build_server() -> Server:
    server = Server(name="litellm-mcp-stdio", version="1.0.0")

    @server.list_tools()
    async def list_tools() -> List[Dict[str, Any]]:
        try:
            register_default_local_tools(global_mcp_tool_registry)
        except Exception as e:
            verbose_logger.debug(f"stdio MCP: default tools registration skipped: {e}")
        # Return raw registry tools; stdio transport will serialize
        return [t for t in global_mcp_tool_registry.list_tools()]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any] | None):
        try:
            register_default_local_tools(global_mcp_tool_registry)
        except Exception as e:
            verbose_logger.debug(f"stdio MCP: default tools registration skipped: {e}")

        tool = global_mcp_tool_registry.get_tool(name)
        if not tool:
            return [TextContent(type="text", text=f"Error: tool '{name}' not found")]  # type: ignore
        args = arguments or {}
        try:
            res = tool.handler(**args)
            # If async handler
            if asyncio.iscoroutine(res):
                res = await res
            return [TextContent(type="text", text=str(res))]  # type: ignore
        except Exception as e:  # pragma: no cover
            return [TextContent(type="text", text=f"Error: {e}")]  # type: ignore

    return server


def main() -> None:  # pragma: no cover
    async def _run():
        server = build_server()
        async with stdio_server() as (read, write):
            await server.run(read, write)

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
