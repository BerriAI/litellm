"""
LiteLLM MCP Server Routes
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from anyio import BrokenResourceError
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import ConfigDict, ValidationError

from litellm._logging import verbose_logger
from litellm.constants import MCP_TOOL_NAME_PREFIX
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import prisma_client
from litellm.types.mcp_server.mcp_server_manager import MCPInfo
from litellm.types.utils import StandardLoggingMCPToolCall
from litellm.utils import client

# Import database helpers for permission checks
from . import db as mcp_db

router = APIRouter(
    prefix="/mcp",
    tags=["mcp"],
)

# Check if MCP is available
# "mcp" requires python 3.10 or higher, but several litellm users use python 3.8
# We're making this conditional import to avoid breaking users who use python 3.8.
# TODO: Make this a util function for litellm client usage
MCP_AVAILABLE: bool = True
try:
    from mcp.server import Server
except ImportError as e:
    verbose_logger.debug(f"MCP module not found: {e}")
    MCP_AVAILABLE = False


# Routes
@router.get(
    "/enabled",
    description="Returns if the MCP server is enabled",
)
def get_mcp_server_enabled() -> Dict[str, bool]:
    """
    Returns if the MCP server is enabled
    """
    return {"enabled": MCP_AVAILABLE}


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

    ######################################################
    ############ MCP Tools List REST API Response Object #
    # Defined here because we don't want to add `mcp` as a
    # required dependency for `litellm` pip package
    ######################################################
    class ListMCPToolsRestAPIResponseObject(MCPTool):
        """
        Object returned by the /tools/list REST API route.
        """

        mcp_info: Optional[MCPInfo] = None
        model_config = ConfigDict(arbitrary_types_allowed=True)

    ########################################################
    ############ Initialize the MCP Server #################
    ########################################################
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

    @client
    async def call_mcp_tool(
        name: str, 
        arguments: Optional[Dict[str, Any]] = None, 
        user: Optional[UserAPIKeyAuth] = None,
        **kwargs: Any
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Call a specific tool with the provided arguments
        
        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            user: Optional user authentication info for permission checks
            **kwargs: Additional arguments
        """
        if arguments is None:
            raise HTTPException(
                status_code=400, detail="Request arguments are required"
            )

        # Check permissions if user is provided and prisma_client is available
        if user is not None and prisma_client is not None:
            mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)
            if mcp_server:
                # Get MCP servers the user has access to
                user_accessible_servers = await mcp_db.get_all_mcp_servers_for_user(
                    prisma_client=prisma_client,
                    user=user,
                )
                
                # Check if user has access to the server containing this tool
                accessible_server_ids = {server.server_id for server in user_accessible_servers}
                
                if mcp_server.server_id not in accessible_server_ids:
                    verbose_logger.warning(
                        f"User {user.user_id} attempted to call tool '{name}' "
                        f"from server {mcp_server.server_id} without permission"
                    )
                    raise HTTPException(
                        status_code=403, 
                        detail=f"Access denied: You don't have permission to use tools from server '{mcp_server.server_id}'"
                    )

        standard_logging_mcp_tool_call: StandardLoggingMCPToolCall = (
            _get_standard_logging_mcp_tool_call(
                name=name,
                arguments=arguments,
            )
        )
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
            "litellm_logging_obj", None
        )
        if litellm_logging_obj:
            litellm_logging_obj.model_call_details[
                "mcp_tool_call_metadata"
            ] = standard_logging_mcp_tool_call
            litellm_logging_obj.model_call_details[
                "model"
            ] = f"{MCP_TOOL_NAME_PREFIX}: {standard_logging_mcp_tool_call.get('name') or ''}"
            litellm_logging_obj.model_call_details[
                "custom_llm_provider"
            ] = standard_logging_mcp_tool_call.get("mcp_server_name")

        # Try managed server tool first
        if name in global_mcp_server_manager.tool_name_to_mcp_server_name_mapping:
            return await _handle_managed_mcp_tool(name, arguments)

        # Fall back to local tool registry
        return await _handle_local_mcp_tool(name, arguments)

    def _get_standard_logging_mcp_tool_call(
        name: str,
        arguments: Dict[str, Any],
    ) -> StandardLoggingMCPToolCall:
        mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)
        if mcp_server:
            mcp_info = mcp_server.mcp_info or {}
            return StandardLoggingMCPToolCall(
                name=name,
                arguments=arguments,
                mcp_server_name=mcp_info.get("server_name"),
                mcp_server_logo_url=mcp_info.get("logo_url"),
            )
        else:
            return StandardLoggingMCPToolCall(
                name=name,
                arguments=arguments,
            )

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
    async def list_tool_rest_api(
        server_id: Optional[str] = Query(
            None, description="The server id to list tools for"
        ),
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> List[ListMCPToolsRestAPIResponseObject]:
        """
        List all available tools with information about the server they belong to.

        Example response:
        Tools:
        [
            {
                "name": "create_zap",
                "description": "Create a new zap",
                "inputSchema": "tool_input_schema",
                "mcp_info": {
                    "server_name": "zapier",
                    "logo_url": "https://www.zapier.com/logo.png",
                }
            },
            {
                "name": "fetch_data",
                "description": "Fetch data from a URL",
                "inputSchema": "tool_input_schema",
                "mcp_info": {
                    "server_name": "fetch",
                    "logo_url": "https://www.fetch.com/logo.png",
                }
            }
        ]
        """
        if prisma_client is None:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        # Get MCP servers the user has access to
        user_accessible_servers = await mcp_db.get_all_mcp_servers_for_user(
            prisma_client=prisma_client,
            user=user_api_key_dict,
        )
        
        # Create a set of accessible server IDs for quick lookup
        accessible_server_ids = {server.server_id for server in user_accessible_servers}
        
        verbose_logger.debug(
            f"User {user_api_key_dict.user_id} has access to MCP servers: {accessible_server_ids}"
        )

        list_tools_result: List[ListMCPToolsRestAPIResponseObject] = []
        for server in global_mcp_server_manager.get_registry().values():
            # Check if user has access to this server
            if server.server_id not in accessible_server_ids:
                verbose_logger.debug(
                    f"User {user_api_key_dict.user_id} does not have access to server {server.server_id}"
                )
                continue
                
            if server_id and server.server_id != server_id:
                continue
            try:
                tools = await global_mcp_server_manager._get_tools_from_server(server)
                for tool in tools:
                    list_tools_result.append(
                        ListMCPToolsRestAPIResponseObject(
                            name=tool.name,
                            description=tool.description,
                            inputSchema=tool.inputSchema,
                            mcp_info=server.mcp_info,
                        )
                    )
            except Exception as e:
                verbose_logger.exception(f"Error getting tools from {server.name}: {e}")
                continue
        return list_tools_result

    @router.post("/tools/call", dependencies=[Depends(user_api_key_auth)])
    async def call_tool_rest_api(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        REST API to call a specific MCP tool with the provided arguments
        """
        from litellm.proxy.proxy_server import add_litellm_data_to_request, proxy_config

        if prisma_client is None:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        data = await request.json()
        
        # Check if the tool name is provided
        tool_name = data.get("name")
        if not tool_name:
            raise HTTPException(
                status_code=400, detail="Tool name is required"
            )
        
        # Get the MCP server for this tool
        mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(tool_name)
        if mcp_server:
            # Get MCP servers the user has access to
            user_accessible_servers = await mcp_db.get_all_mcp_servers_for_user(
                prisma_client=prisma_client,
                user=user_api_key_dict,
            )
            
            # Check if user has access to the server containing this tool
            accessible_server_ids = {server.server_id for server in user_accessible_servers}
            
            if mcp_server.server_id not in accessible_server_ids:
                verbose_logger.warning(
                    f"User {user_api_key_dict.user_id} attempted to call tool '{tool_name}' "
                    f"from server {mcp_server.server_id} without permission"
                )
                raise HTTPException(
                    status_code=403, 
                    detail=f"Access denied: You don't have permission to use tools from server '{mcp_server.server_id}'"
                )
        
        # If we reach here, user has permission to call the tool
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
        )
        # Pass user to call_mcp_tool for additional permission checks
        data["user"] = user_api_key_dict
        return await call_mcp_tool(**data)

    options = InitializationOptions(
        server_name="litellm-mcp-server",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
