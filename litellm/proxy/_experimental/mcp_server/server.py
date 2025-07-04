"""
LiteLLM MCP Server Routes
"""

import asyncio
import contextlib
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict
from starlette.types import Receive, Scope, Send

from litellm._logging import verbose_logger
from litellm.constants import MCP_TOOL_NAME_PREFIX
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.utils import (
    LITELLM_MCP_SERVER_NAME,
    LITELLM_MCP_SERVER_VERSION,
    LITELLM_MCP_SERVER_DESCRIPTION, normalize_server_name,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp_server.mcp_server_manager import MCPInfo
from litellm.types.utils import StandardLoggingMCPToolCall
from litellm.utils import client


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


# Global variables to track initialization
_SESSION_MANAGERS_INITIALIZED = False
_SESSION_MANAGER_TASK = None

if MCP_AVAILABLE:
    from mcp.server import Server

    # Import auth context variables and middleware
    from mcp.server.auth.middleware.auth_context import (
        AuthContextMiddleware,
        auth_context_var,
    )
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.types import EmbeddedResource as MCPEmbeddedResource
    from mcp.types import ImageContent as MCPImageContent
    from mcp.types import TextContent as MCPTextContent
    from mcp.types import Tool as MCPTool

    from litellm.proxy._experimental.mcp_server.auth.litellm_auth_handler import (
        MCPAuthenticatedUser,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.sse_transport import SseServerTransport
    from litellm.proxy._experimental.mcp_server.tool_registry import (
        global_mcp_tool_registry,
    )
    from litellm.proxy._experimental.mcp_server.utils import get_server_name_prefix_tool_mcp

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
        # Get user authentication from context variable
        user_api_key_auth, mcp_auth_header, mcp_servers = get_auth_context()
        verbose_logger.debug(
            f"MCP list_tools - User API Key Auth from context: {user_api_key_auth}"
        )
        # Get mcp_servers from context variable
        return await _list_mcp_tools(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
        )

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
        user_api_key_auth, mcp_auth_header, _ = get_auth_context()
        verbose_logger.debug(
            f"MCP mcp_server_tool_call - User API Key Auth from context: {user_api_key_auth}"
        )
        response = await call_mcp_tool(
            name=name,
            arguments=arguments,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
        )
        return response

    ########################################################
    ############ End of MCP Server Routes ##################
    ########################################################

    ########################################################
    ############ Helper Functions ##########################
    ########################################################

    async def _get_tools_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]]
    ) -> List[MCPTool]:
        """
        Helper method to fetch tools from MCP servers based on server filtering criteria.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server
            mcp_servers: Optional list of server names to filter by

        Returns:
            List[MCPTool]: List of tools from the specified or all allowed MCP servers
        """
        if mcp_servers:
            # If mcp_servers header is present, only get tools from specified servers
            tools = []
            for server_id in await global_mcp_server_manager.get_allowed_mcp_servers(user_api_key_auth):
                server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
                if server and any(normalize_server_name(server.name) == normalize_server_name(s) for s in mcp_servers):
                    server_tools = await global_mcp_server_manager._get_tools_from_server(
                        server=server,
                        mcp_auth_header=mcp_auth_header,
                    )
                    tools.extend(server_tools)
            return tools
        else:
            # If no mcp_servers header, get tools from all allowed servers
            return await global_mcp_server_manager.list_tools(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
            )

    async def _list_mcp_tools(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
    ) -> List[MCPTool]:
        """
        List all available tools

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server
            mcp_servers: Optional list of server names to filter by
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

        # Get tools from MCP servers
        tools_from_mcp_servers = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers
        )

        verbose_logger.debug("TOOLS FROM MCP SERVERS: %s", tools_from_mcp_servers)
        if tools_from_mcp_servers is not None:
            tools.extend(tools_from_mcp_servers)
        return tools

    @client
    async def call_mcp_tool(
            name: str,
            arguments: Optional[Dict[str, Any]] = None,
            user_api_key_auth: Optional[UserAPIKeyAuth] = None,
            mcp_auth_header: Optional[str] = None,
            **kwargs: Any
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Call a specific tool with the provided arguments (handles prefixed tool names)
        """
        if arguments is None:
            raise HTTPException(
                status_code=400, detail="Request arguments are required"
            )

        # Remove prefix from tool name for logging and processing
        original_tool_name, server_name_from_prefix = get_server_name_prefix_tool_mcp(
            name)

        standard_logging_mcp_tool_call: StandardLoggingMCPToolCall = (
            _get_standard_logging_mcp_tool_call(
                name=original_tool_name,  # Use original name for logging
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

        # Try managed server tool first (pass the full prefixed name)
        if name in global_mcp_server_manager.tool_name_to_mcp_server_name_mapping:
            return await _handle_managed_mcp_tool(
                name=name,  # Pass the full name (potentially prefixed)
                arguments=arguments,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
            )

        # Fall back to local tool registry (use original name)
        return await _handle_local_mcp_tool(original_tool_name, arguments)

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
        name: str, 
        arguments: Dict[str, Any],
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """Handle tool execution for managed server tools"""
        call_tool_result = await global_mcp_server_manager.call_tool(
            name=name,
            arguments=arguments,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
        )
        verbose_logger.debug("CALL TOOL RESULT: %s", call_tool_result)
        return call_tool_result.content  # type: ignore[return-value]

    async def _handle_local_mcp_tool(
            name: str, arguments: Dict[str, Any]
    ) -> List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]:
        """
        Handle tool execution for local registry tools
        Note: Local tools don't use prefixes, so we use the original name
        """
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
            # Validate headers and log request info
            user_api_key_auth, mcp_auth_header, mcp_servers = (
                await MCPRequestHandler.process_mcp_request(scope)
            )
            verbose_logger.debug(f"MCP request headers - mcp_servers: {mcp_servers}")
            # Set the auth context variable for easy access in MCP functions
            set_auth_context(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
            )

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
            # Validate headers and log request info
            user_api_key_auth, mcp_auth_header, mcp_servers = (
                await MCPRequestHandler.process_mcp_request(scope)
            )
            # Set the auth context variable for easy access in MCP functions
            set_auth_context(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
            )

            # Ensure session managers are initialized
            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
                # Give it a moment to start up
                await asyncio.sleep(0.1)

            await sse_session_manager.handle_request(scope, receive, send)
        except Exception as e:
            verbose_logger.exception(f"Error handling MCP request: {e}")
            raise e

    app = FastAPI(
        title=LITELLM_MCP_SERVER_NAME,
        description=LITELLM_MCP_SERVER_DESCRIPTION,
        version=LITELLM_MCP_SERVER_VERSION,
        lifespan=lifespan,
    )

    # Routes
    @app.get(
        "/enabled",
        description="Returns if the MCP server is enabled",
    )
    def get_mcp_server_enabled() -> Dict[str, bool]:
        """
        Returns if the MCP server is enabled
        """
        return {"enabled": MCP_AVAILABLE}

    # Mount the MCP handlers
    app.mount("/", handle_streamable_http_mcp)
    app.mount("/sse", handle_sse_mcp)
    app.add_middleware(AuthContextMiddleware)

    ########################################################
    ############ Auth Context Functions ####################
    ########################################################

    def set_auth_context(
        user_api_key_auth: UserAPIKeyAuth, 
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
    ) -> None:
        """
        Set the UserAPIKeyAuth in the auth context variable.

        Args:
            user_api_key_auth: UserAPIKeyAuth object
            mcp_auth_header: MCP auth header to be passed to the MCP server
            mcp_servers: Optional list of server names to filter by
        """
        auth_user = MCPAuthenticatedUser(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
        )
        auth_context_var.set(auth_user)

    def get_auth_context() -> Tuple[Optional[UserAPIKeyAuth], Optional[str], Optional[List[str]]]:
        """
        Get the UserAPIKeyAuth from the auth context variable.

        Returns:
            Tuple[Optional[UserAPIKeyAuth], Optional[str]]: UserAPIKeyAuth object and MCP auth header
        """
        auth_user = auth_context_var.get()
        if auth_user and isinstance(auth_user, MCPAuthenticatedUser):
            return auth_user.user_api_key_auth, auth_user.mcp_auth_header, auth_user.mcp_servers
        return None, None, None

    ########################################################
    ############ End of Auth Context Functions #############
    ########################################################

else:
    app = FastAPI()
