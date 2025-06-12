"""
LiteLLM MCP Server Routes
"""

import asyncio
import contextlib
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from pydantic import ConfigDict
from starlette.types import Receive, Scope, Send

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

LITELLM_MCP_SERVER_NAME = "litellm-mcp-server"
LITELLM_MCP_SERVER_VERSION = "1.0.0"
LITELLM_MCP_SERVER_DESCRIPTION = "MCP Server for LiteLLM"

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


# Global variables to track initialization
_SESSION_MANAGERS_INITIALIZED = False
_SESSION_MANAGER_TASK = None

if MCP_AVAILABLE:
    from mcp.server import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.types import EmbeddedResource as MCPEmbeddedResource
    from mcp.types import ImageContent as MCPImageContent
    from mcp.types import TextContent as MCPTextContent
    from mcp.types import Tool as MCPTool

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.sse_transport import SseServerTransport
    from litellm.proxy._experimental.mcp_server.tool_registry import (
        global_mcp_tool_registry,
    )

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
    server: Server = Server(
        name=LITELLM_MCP_SERVER_NAME,
        version=LITELLM_MCP_SERVER_VERSION,
    )
    sse: SseServerTransport = SseServerTransport("/mcp/sse/messages")

    # Create session managers
    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=True,  # Use JSON responses instead of SSE by default
        stateless=True,
    )

    # Create SSE session manager
    sse_session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,  # Use SSE responses for this endpoint
        stateless=True,
    )

    async def initialize_session_managers():
        """Initialize the session managers. Can be called from main app lifespan."""
        global _SESSION_MANAGERS_INITIALIZED, _SESSION_MANAGER_TASK

        if _SESSION_MANAGERS_INITIALIZED:
            return

        verbose_logger.info("Initializing MCP session managers...")

        # Create a task to run the session managers
        async def run_session_managers():
            async with session_manager.run():
                async with sse_session_manager.run():
                    verbose_logger.info(
                        "MCP Server started with StreamableHTTP and SSE session managers!"
                    )
                    try:
                        # Keep running until cancelled
                        while True:
                            await asyncio.sleep(1)
                    except asyncio.CancelledError:
                        verbose_logger.info("MCP session managers shutting down...")
                        raise

        _SESSION_MANAGER_TASK = asyncio.create_task(run_session_managers())
        _SESSION_MANAGERS_INITIALIZED = True
        verbose_logger.info("MCP session managers initialization completed!")

    async def shutdown_session_managers():
        """Shutdown the session managers."""
        global _SESSION_MANAGERS_INITIALIZED, _SESSION_MANAGER_TASK

        if _SESSION_MANAGER_TASK and not _SESSION_MANAGER_TASK.done():
            verbose_logger.info("Shutting down MCP session managers...")
            _SESSION_MANAGER_TASK.cancel()
            try:
                await _SESSION_MANAGER_TASK
            except asyncio.CancelledError:
                pass

        _SESSION_MANAGERS_INITIALIZED = False
        _SESSION_MANAGER_TASK = None

    @contextlib.asynccontextmanager
    async def lifespan(app) -> AsyncIterator[None]:
        """Application lifespan context manager."""
        await initialize_session_managers()
        try:
            yield
        finally:
            await shutdown_session_managers()

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
        user_authorization_header: Optional[str] = None,
        **kwargs: Any
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Call a specific tool with the provided arguments
        
        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            user_authorization_header: User authorization header to pass through to MCP servers
            **kwargs: Additional arguments
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
            litellm_logging_obj.model_call_details["mcp_tool_call_metadata"] = (
                standard_logging_mcp_tool_call
            )
            litellm_logging_obj.model_call_details["model"] = (
                f"{MCP_TOOL_NAME_PREFIX}: {standard_logging_mcp_tool_call.get('name') or ''}"
            )
            litellm_logging_obj.model_call_details["custom_llm_provider"] = (
                standard_logging_mcp_tool_call.get("mcp_server_name")
            )

        # Try managed server tool first
        if name in global_mcp_server_manager.tool_name_to_mcp_server_name_mapping:
            return await _handle_managed_mcp_tool(name, arguments, user_authorization_header)

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
        name: str, arguments: Dict[str, Any], user_authorization_header: Optional[str] = None
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """Handle tool execution for managed server tools"""
        call_tool_result = await global_mcp_server_manager.call_tool(
            name=name,
            arguments=arguments,
            user_authorization_header=user_authorization_header,
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

    async def handle_streamable_http_mcp(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Handle MCP requests through StreamableHTTP."""
        try:
            # Ensure session managers are initialized
            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
                # Give it a moment to start up
                await asyncio.sleep(0.1)

            await session_manager.handle_request(scope, receive, send)
        except Exception as e:
            verbose_logger.exception(f"Error handling MCP request: {e}")
            raise e

    async def handle_sse_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP requests through SSE."""
        try:
            # Ensure session managers are initialized
            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
                # Give it a moment to start up
                await asyncio.sleep(0.1)

            await sse_session_manager.handle_request(scope, receive, send)
        except Exception as e:
            verbose_logger.exception(f"Error handling MCP request: {e}")
            raise e

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
        list_tools_result: List[ListMCPToolsRestAPIResponseObject] = []
        for server in global_mcp_server_manager.get_registry().values():
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
        
        Supports dual authentication:
        - x-litellm-key: for LiteLLM system authentication (handled by user_api_key_auth)
        - Authorization: for user authorization passed through to MCP servers
        """
        from litellm.proxy.proxy_server import add_litellm_data_to_request, proxy_config

        # Extract the Authorization header for pass-through to MCP servers
        authorization_header = request.headers.get("Authorization")
        
        data = await request.json()
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
        )
        
        # Pass the authorization header to the MCP tool call
        if authorization_header:
            data["user_authorization_header"] = authorization_header
        
        return await call_mcp_tool(**data)

    app = FastAPI(
        title=LITELLM_MCP_SERVER_NAME,
        description=LITELLM_MCP_SERVER_DESCRIPTION,
        version=LITELLM_MCP_SERVER_VERSION,
        lifespan=lifespan,
    )

    # Include the MCP router
    app.include_router(router)

    # Mount the MCP handlers
    app.mount("/", handle_streamable_http_mcp)
    app.mount("/sse", handle_sse_mcp)

else:
    app = FastAPI()
