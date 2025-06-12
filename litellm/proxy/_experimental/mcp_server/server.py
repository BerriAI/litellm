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
from litellm.types.mcp_server.mcp_server_manager import MCPInfo
from litellm.types.utils import StandardLoggingMCPToolCall
from litellm.utils import client

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
    from .db import get_all_mcp_servers_for_user

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
    ############ User Context Storage for SSE Sessions ####
    ########################################################
    # Store user context for each SSE session
    _session_user_context: dict[str, UserAPIKeyAuth] = {}

    def _store_user_context_for_session(session_id: str, user_context: UserAPIKeyAuth) -> None:
        """Store user context for a session ID"""
        _session_user_context[session_id] = user_context
        verbose_logger.debug(f"Stored user context for session {session_id}")

    def _get_user_context_for_session(session_id: str) -> Optional[UserAPIKeyAuth]:
        """Get user context for a session ID"""
        return _session_user_context.get(session_id)

    def _clear_user_context_for_session(session_id: str) -> None:
        """Clear user context for a session ID"""
        _session_user_context.pop(session_id, None)
        verbose_logger.debug(f"Cleared user context for session {session_id}")

    # Global variable to store current session context during server function calls
    _current_session_id: Optional[str] = None

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
        List all available tools with user permission filtering
        """
        return await _list_mcp_tools()

    async def _list_mcp_tools() -> List[MCPTool]:
        """
        List all available tools with user permission filtering
        """
        from litellm.proxy.proxy_server import prisma_client

        tools = []
        
        # Get user context for the current session
        user_context = None
        if _current_session_id:
            user_context = _get_user_context_for_session(_current_session_id)
        
        if user_context is None:
            verbose_logger.warning("No user context found for MCP list_tools request")
            return []

        verbose_logger.debug(f"Filtering MCP tools for user: {user_context.api_key}")

        # Get MCP servers the user has access to
        if prisma_client is None:
            verbose_logger.error("Prisma client is not available")
            return []
            
        allowed_mcp_servers = await get_all_mcp_servers_for_user(
            prisma_client=prisma_client,
            user=user_context,
        )
        allowed_server_ids = {server.server_id for server in allowed_mcp_servers}
        
        verbose_logger.debug(f"User has access to MCP servers: {allowed_server_ids}")

        # Add local tools (these are always allowed for authenticated users)
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
        
        # Add SSE tools from allowed servers only
        sse_tools: List[MCPTool] = await global_mcp_server_manager.list_tools()
        verbose_logger.debug("ALL SSE TOOLS: %s", sse_tools)
        
        filtered_sse_tools = []
        if sse_tools is not None:
            for tool in sse_tools:
                # Check if this tool belongs to an allowed server
                tool_server = global_mcp_server_manager._get_mcp_server_from_tool_name(tool.name)
                if tool_server and tool_server.server_id in allowed_server_ids:
                    filtered_sse_tools.append(tool)
                    verbose_logger.debug(f"Including tool {tool.name} from server {tool_server.server_id}")
                else:
                    verbose_logger.debug(f"Excluding tool {tool.name} - server not allowed")
            tools.extend(filtered_sse_tools)
            
        verbose_logger.debug(f"Returning {len(tools)} filtered tools for user")
        return tools

    @server.call_tool()
    async def mcp_server_tool_call(
        name: str, arguments: Dict[str, Any] | None
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Call a specific tool with the provided arguments, with user permission checking

        Args:
            name (str): Name of the tool to call
            arguments (Dict[str, Any] | None): Arguments to pass to the tool

        Returns:
            List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]: Tool execution results

        Raises:
            HTTPException: If tool not found, arguments missing, or user doesn't have permission
        """
        from litellm.proxy.proxy_server import prisma_client

        # Get user context for the current session
        user_context = None
        if _current_session_id:
            user_context = _get_user_context_for_session(_current_session_id)
        
        if user_context is None:
            verbose_logger.warning(f"No user context found for MCP tool call: {name}")
            raise HTTPException(
                status_code=401, 
                detail="Authentication required for MCP tool calls"
            )

        verbose_logger.debug(f"Checking permissions for tool {name} for user: {user_context.api_key}")

        # Check if this is a managed server tool and if user has access
        if name in global_mcp_server_manager.tool_name_to_mcp_server_name_mapping:
            tool_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)
            if tool_server and prisma_client is not None:
                # Get MCP servers the user has access to
                allowed_mcp_servers = await get_all_mcp_servers_for_user(
                    prisma_client=prisma_client,
                    user=user_context,
                )
                allowed_server_ids = {server.server_id for server in allowed_mcp_servers}
                
                if tool_server.server_id not in allowed_server_ids:
                    verbose_logger.warning(f"User {user_context.api_key} attempted to call tool {name} from server {tool_server.server_id} without permission")
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied: You don't have permission to use tools from server '{tool_server.server_id}'"
                    )

        # Validate arguments
        response = await call_mcp_tool(
            name=name,
            arguments=arguments,
        )
        return response

    @client
    async def call_mcp_tool(
        name: str, arguments: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Call a specific tool with the provided arguments
        """
        if arguments is None:
            raise HTTPException(
                status_code=400, detail="Request arguments are required"
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

    @router.get("/", response_class=StreamingResponse, dependencies=[Depends(user_api_key_auth)])
    async def handle_sse(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)
    ):
        verbose_logger.info("new incoming SSE connection established")
        verbose_logger.debug(f"SSE connection for user: {user_api_key_dict.api_key}")
        
        async with sse.connect_sse(request) as streams:
            # Store user context for this session using the current session ID
            session_id = sse.get_current_session_id()
            if session_id:
                _store_user_context_for_session(session_id, user_api_key_dict)
            
            try:
                # Set current session context for server function calls
                global _current_session_id
                _current_session_id = session_id
                
                await server.run(streams[0], streams[1], options)
            except BrokenResourceError:
                pass
            except asyncio.CancelledError:
                pass
            except ValidationError:
                pass
            except Exception:
                raise
            finally:
                # Clean up session context
                if session_id:
                    _clear_user_context_for_session(session_id)
                _current_session_id = None
                
        await request.close()

    @router.post("/sse/messages", dependencies=[Depends(user_api_key_auth)])
    async def handle_messages(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)
    ):
        verbose_logger.info("incoming SSE message received")
        verbose_logger.debug(f"SSE message from user: {user_api_key_dict.api_key}")
        
        # Set current session context for potential server function calls
        session_id_param = request.query_params.get("session_id")
        if session_id_param:
            global _current_session_id
            _current_session_id = session_id_param
            
            # Verify the user context matches the stored context for this session
            stored_context = _get_user_context_for_session(session_id_param)
            if stored_context and stored_context.api_key != user_api_key_dict.api_key:
                verbose_logger.warning(f"Session {session_id_param} user context mismatch")
                raise HTTPException(status_code=403, detail="Session user context mismatch")
        
        try:
            await sse.handle_post_message(request.scope, request.receive, request._send)
        finally:
            _current_session_id = None
            
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
        from litellm.proxy.proxy_server import prisma_client

        list_tools_result: List[ListMCPToolsRestAPIResponseObject] = []
        
        # Get MCP servers the user has access to
        if prisma_client is None:
            verbose_logger.error("Prisma client is not available")
            return list_tools_result
            
        allowed_mcp_servers = await get_all_mcp_servers_for_user(
            prisma_client=prisma_client,
            user=user_api_key_dict,
        )
        allowed_server_ids = {server.server_id for server in allowed_mcp_servers}
        
        for server in global_mcp_server_manager.get_registry().values():
            if server_id and server.server_id != server_id:
                continue
                
            # Check if user has access to this server
            if server.server_id not in allowed_server_ids:
                verbose_logger.debug(f"User {user_api_key_dict.api_key} doesn't have access to server {server.server_id}")
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
        from litellm.proxy.proxy_server import add_litellm_data_to_request, proxy_config, prisma_client

        data = await request.json()
        
        # Check if user has permission to call this tool
        tool_name = data.get("name")
        if tool_name and tool_name in global_mcp_server_manager.tool_name_to_mcp_server_name_mapping:
            tool_server = global_mcp_server_manager._get_mcp_server_from_tool_name(tool_name)
            if tool_server and prisma_client is not None:
                # Get MCP servers the user has access to
                allowed_mcp_servers = await get_all_mcp_servers_for_user(
                    prisma_client=prisma_client,
                    user=user_api_key_dict,
                )
                allowed_server_ids = {server.server_id for server in allowed_mcp_servers}
                
                if tool_server.server_id not in allowed_server_ids:
                    verbose_logger.warning(f"User {user_api_key_dict.api_key} attempted to call tool {tool_name} from server {tool_server.server_id} without permission")
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied: You don't have permission to use tools from server '{tool_server.server_id}'"
                    )
        
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
        )
        return await call_mcp_tool(**data)

    options = InitializationOptions(
        server_name="litellm-mcp-server",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
