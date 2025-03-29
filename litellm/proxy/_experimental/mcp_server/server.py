"""
LiteLLM MCP Server Routes
"""

import asyncio
from typing import Any, Dict, List, Union

from anyio import BrokenResourceError
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from litellm._logging import verbose_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.mcp_server.mcp_server_manager import (
    ListMCPToolsRestAPIResponseObject,
)

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

    from .mcp_server_manager import global_mcp_server_manager
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
        return await _list_mcp_tools()

    async def _list_mcp_tools() -> List[MCPTool]:
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
        verbose_logger.debug(
            "GLOBAL MCP TOOLS: %s", global_mcp_tool_registry.list_tools()
        )
        sse_tools: List[MCPTool] = await global_mcp_server_manager.list_tools()
        verbose_logger.debug("SSE TOOLS: %s", sse_tools)
        if sse_tools is not None:
            tools.extend(sse_tools)
        return tools

    @server.call_tool()
    async def mcp_server_tool_call(
        name: str, arguments: Dict[str, Any] | None
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Call a specific tool with the provided arguments

        Args:
            name (str): Name of the tool to call
            arguments (Dict[str, Any] | None): Arguments to pass to the tool

        Returns:
            List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]: Tool execution results

        Raises:
            HTTPException: If tool not found or arguments missing
        """
        # Validate arguments
        response = await call_mcp_tool(
            name=name,
            arguments=arguments,
        )
        return response

    async def call_mcp_tool(
        name: str, arguments: Dict[str, Any] | None
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Call a specific tool with the provided arguments
        """
        if arguments is None:
            raise HTTPException(
                status_code=400, detail="Request arguments are required"
            )

        # Try managed server tool first
        if name in global_mcp_server_manager.tool_name_to_mcp_server_name_mapping:
            return await _handle_managed_mcp_tool(name, arguments)

        # Fall back to local tool registry
        return await _handle_local_mcp_tool(name, arguments)

    async def _handle_managed_mcp_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """Handle tool execution for managed server tools"""
        call_tool_result = await global_mcp_server_manager.call_tool(
            name=name,
            arguments=arguments,
        )
        verbose_logger.debug("CALL TOOL RESULT: %s", call_tool_result)
        return call_tool_result.content

    async def _handle_local_mcp_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """Handle tool execution for local registry tools"""
        tool = global_mcp_tool_registry.get_tool(name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

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

    ########################################################
    ############ MCP Server REST API Routes #################
    ########################################################
    @router.get("/tools/list", dependencies=[Depends(user_api_key_auth)])
    async def list_tool_rest_api() -> (
        List[Dict[str, ListMCPToolsRestAPIResponseObject]]
    ):
        """
        List all available tools with information about the server they belong to.

        Example response:
        Tools:
        [
            "zapier": {
                "tools": [
                    {
                        "name": "create_zap",
                        "description": "Create a new zap",
                        "inputSchema": "tool_input_schema",
                    }
                ],
                "mcp_info": {
                    "logo_url": "https://www.zapier.com/logo.png",
                }
            },
            "fetch": {
                "tools": [
                    {
                        "name": "fetch_data",
                        "description": "Fetch data from a URL",
                    }
                ],
                "mcp_info": {
                    "logo_url": "https://www.fetch.com/logo.png",
                }
            }
        """
        list_tools_result: List[Dict[str, ListMCPToolsRestAPIResponseObject]] = []
        for server in global_mcp_server_manager.mcp_servers:
            tools = await global_mcp_server_manager._get_tools_from_server(server)
            list_tools_result.append(
                {
                    server.name: ListMCPToolsRestAPIResponseObject(
                        tools=tools,
                        mcp_info=server.mcp_info,
                    )
                }
            )
        return list_tools_result

    @router.post("/tools/call", dependencies=[Depends(user_api_key_auth)])
    async def call_tool_rest_api(name: str, arguments: Dict[str, Any]):
        return await call_mcp_tool(
            name=name,
            arguments=arguments,
        )

    options = InitializationOptions(
        server_name="litellm-mcp-server",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
