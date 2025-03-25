"""
LiteLLM MCP Server Routes
"""

import asyncio
from typing import Any, Dict, List, Union

from anyio import BrokenResourceError
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from litellm._logging import verbose_logger

# Check if MCP is available
# "mcp" requires python 3.10 or higher, but several litellm users use python 3.8
# We're making this conditional import to avoid breaking users who use python 3.8.
try:
    from mcp.server import Server

    MCP_AVAILABLE = True
except ImportError as e:
    verbose_logger.debug(f"MCP module not found: {e}")
    MCP_AVAILABLE = False
    router = APIRouter(
        prefix="/mcp",
        tags=["mcp"],
    )


if MCP_AVAILABLE:
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.types import EmbeddedResource as MCPEmbeddedResource
    from mcp.types import ImageContent as MCPImageContent
    from mcp.types import TextContent as MCPTextContent
    from mcp.types import Tool as MCPTool

    from .sse_transport import SseServerTransport
    from .tool_registry import global_mcp_tool_registry

    ########################################################
    ############ Initialize the MCP Server #################
    ########################################################
    router = APIRouter(
        prefix="/mcp",
        tags=["mcp"],
    )
    server: Server = Server("litellm-mcp-server")
    sse: SseServerTransport = SseServerTransport("/mcp/sse/messages")

    ########################################################
    ############### MCP Server Routes #######################
    ########################################################

    @server.list_tools()
    async def list_tools() -> list[MCPTool]:
        """
        List all available tools
        """
        tools = []
        for tool in global_mcp_tool_registry.list_tools():
            tools.append(
                MCPTool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,
                )
            )

        return tools

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any] | None
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Call a specific tool with the provided arguments
        """
        tool = global_mcp_tool_registry.get_tool(name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
        if arguments is None:
            raise HTTPException(
                status_code=400, detail="Request arguments are required"
            )

        try:
            result = tool.handler(**arguments)
            return [MCPTextContent(text=str(result), type="text")]
        except Exception as e:
            return [MCPTextContent(text=f"Error: {str(e)}", type="text")]

    @router.get("/", response_class=StreamingResponse)
    async def handle_sse(request: Request):
        verbose_logger.info("new incoming SSE connection established")
        async with sse.connect_sse(request) as streams:
            try:
                await server.run(streams[0], streams[1], options)
            except BrokenResourceError:
                pass
            except asyncio.CancelledError:
                pass
            except ValidationError:
                pass
            except Exception:
                raise
        await request.close()

    @router.post("/sse/messages")
    async def handle_messages(request: Request):
        verbose_logger.info("incoming SSE message received")
        await sse.handle_post_message(request.scope, request.receive, request._send)
        await request.close()

    options = InitializationOptions(
        server_name="litellm-mcp-server",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
