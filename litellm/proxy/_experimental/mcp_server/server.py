"""
LiteLLM MCP Server Routes
"""

import asyncio
import contextlib
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict
from starlette.types import Receive, Scope, Send

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.utils import (
    LITELLM_MCP_SERVER_DESCRIPTION,
    LITELLM_MCP_SERVER_NAME,
    LITELLM_MCP_SERVER_VERSION,
    normalize_server_name,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPServer
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

if MCP_AVAILABLE:
    from mcp.server import Server

    # Import auth context variables and middleware
    from mcp.server.auth.middleware.auth_context import (
        AuthContextMiddleware,
        auth_context_var,
    )
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.types import EmbeddedResource, ImageContent, TextContent
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
    from litellm.proxy._experimental.mcp_server.utils import (
        get_server_name_prefix_tool_mcp,
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

    # Context managers for proper lifecycle management
    _session_manager_cm = None
    _sse_session_manager_cm = None

    async def initialize_session_managers():
        """Initialize the session managers. Can be called from main app lifespan."""
        global _SESSION_MANAGERS_INITIALIZED, _session_manager_cm, _sse_session_manager_cm

        if _SESSION_MANAGERS_INITIALIZED:
            return

        verbose_logger.info("Initializing MCP session managers...")

        # Start the session managers with context managers
        _session_manager_cm = session_manager.run()
        _sse_session_manager_cm = sse_session_manager.run()

        # Enter the context managers
        await _session_manager_cm.__aenter__()
        await _sse_session_manager_cm.__aenter__()

        _SESSION_MANAGERS_INITIALIZED = True
        verbose_logger.info(
            "MCP Server started with StreamableHTTP and SSE session managers!"
        )

    async def shutdown_session_managers():
        """Shutdown the session managers."""
        global _SESSION_MANAGERS_INITIALIZED, _session_manager_cm, _sse_session_manager_cm

        if _SESSION_MANAGERS_INITIALIZED:
            verbose_logger.info("Shutting down MCP session managers...")

            try:
                if _session_manager_cm:
                    await _session_manager_cm.__aexit__(None, None, None)
                if _sse_session_manager_cm:
                    await _sse_session_manager_cm.__aexit__(None, None, None)
            except Exception as e:
                verbose_logger.exception(f"Error during session manager shutdown: {e}")

            _session_manager_cm = None
            _sse_session_manager_cm = None
            _SESSION_MANAGERS_INITIALIZED = False

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
    async def list_tools() -> List[MCPTool]:
        """
        List all available tools
        """
        # Get user authentication from context variable
        user_api_key_auth, mcp_auth_header, mcp_servers = get_auth_context()
        verbose_logger.debug(
            f"MCP list_tools - User API Key Auth from context: {user_api_key_auth}"
        )
        verbose_logger.debug(
            f"MCP list_tools - MCP servers from context: {mcp_servers}"
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
    ) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
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
        from fastapi import Request

        from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
        from litellm.proxy.proxy_server import proxy_config

        # Validate arguments
        user_api_key_auth, mcp_auth_header, _ = get_auth_context()

        verbose_logger.debug(
            f"MCP mcp_server_tool_call - User API Key Auth from context: {user_api_key_auth}"
        )
        try:
            # Create a body date for logging
            body_data = {"name": name, "arguments": arguments}

            request = Request(
                scope={
                    "type": "http",
                    "method": "POST",
                    "path": "/mcp/tools/call",
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            if user_api_key_auth is not None:
                data = await add_litellm_data_to_request(
                    data=body_data,
                    request=request,
                    user_api_key_dict=user_api_key_auth,
                    proxy_config=proxy_config,
                )
            else:
                data = body_data

            response = await call_mcp_tool(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                **data,  # for logging
            )
        except Exception as e:
            verbose_logger.exception(f"MCP mcp_server_tool_call - error: {e}")
            raise e

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
        mcp_servers: Optional[List[str]],
    ) -> List[MCPTool]:
        """
        Helper method to fetch tools from MCP servers based on server filtering criteria.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server
            mcp_servers: Optional list of server names and access groups to filter by

        Returns:
            List[MCPTool]: List of tools from the specified or all allowed MCP servers
        """
        # Get all allowed servers for the user
        allowed_server_ids = await global_mcp_server_manager.get_allowed_mcp_servers(
            user_api_key_auth
        )
        
        # If mcp_servers header is present, filter servers
        if mcp_servers:
            tools = []
            filtered_server_ids = set()
            
            # For each item in mcp_servers, try to match as server name first, then as access group
            for server_or_group in mcp_servers:
                # Try to match as server name first
                server_name_matched = False
                for server_id in allowed_server_ids:
                    server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
                    if server and normalize_server_name(server.name) == normalize_server_name(server_or_group):
                        filtered_server_ids.add(server_id)
                        server_name_matched = True
                
                # If no server name matched, try to treat it as an access group
                if not server_name_matched:
                    try:
                        access_group_server_ids = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                            [server_or_group]
                        )
                        # Only include servers that the user has access to
                        for server_id in access_group_server_ids:
                            if server_id in allowed_server_ids:
                                filtered_server_ids.add(server_id)
                    except Exception as e:
                        verbose_logger.debug(f"Could not resolve '{server_or_group}' as access group: {e}")
            
            # Get tools from filtered servers
            for server_id in filtered_server_ids:
                server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
                if server:
                    server_tools = (
                        await global_mcp_server_manager._get_tools_from_server(
                            server=server,
                            mcp_auth_header=mcp_auth_header,
                        )
                    )
                    tools.extend(server_tools)
            return tools
        else:
            # If no filtering headers, get tools from all allowed servers
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
            mcp_servers: Optional list of server names and access groups to filter by
        """
        tools = []

        # Get tools from MCP servers
        tools_from_mcp_servers = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
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
    ) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
        """
        Call a specific tool with the provided arguments (handles prefixed tool names)
        """
        start_time = datetime.now()
        if arguments is None:
            raise HTTPException(
                status_code=400, detail="Request arguments are required"
            )

        # Remove prefix from tool name for logging and processing
        original_tool_name, server_name_from_prefix = get_server_name_prefix_tool_mcp(
            name
        )

        standard_logging_mcp_tool_call: StandardLoggingMCPToolCall = (
            _get_standard_logging_mcp_tool_call(
                name=original_tool_name,  # Use original name for logging
                arguments=arguments,
                server_name=server_name_from_prefix,
            )
        )
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
            "litellm_logging_obj", None
        )
        if litellm_logging_obj:
            litellm_logging_obj.model_call_details["mcp_tool_call_metadata"] = (
                standard_logging_mcp_tool_call
            )
            litellm_logging_obj.model = f"MCP: {name}"
        # Try managed server tool first (pass the full prefixed name)
        # Primary and recommended way to use MCP servers
        #########################################################
        mcp_server: Optional[MCPServer] = (
            global_mcp_server_manager._get_mcp_server_from_tool_name(name)
        )
        if mcp_server:
            standard_logging_mcp_tool_call["mcp_server_cost_info"] = (
                mcp_server.mcp_info or {}
            ).get("mcp_server_cost_info")
            response =  await _handle_managed_mcp_tool(
                name=name,  # Pass the full name (potentially prefixed)
                arguments=arguments,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
            )

        # Fall back to local tool registry (use original name)
        #########################################################
        # Deprecated: Local MCP Server Tool
        #########################################################
        else:
            response = await _handle_local_mcp_tool(original_tool_name, arguments)
        
        #########################################################
        # Post MCP Tool Call Hook
        # Allow modifying the MCP tool call response before it is returned to the user
        #########################################################
        if litellm_logging_obj:
            end_time = datetime.now()
            await litellm_logging_obj.async_post_mcp_tool_call_hook(
                kwargs=litellm_logging_obj.model_call_details,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
            )
        return response

    def _get_standard_logging_mcp_tool_call(
        name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str],
    ) -> StandardLoggingMCPToolCall:
        mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)
        if mcp_server:
            mcp_info = mcp_server.mcp_info or {}
            return StandardLoggingMCPToolCall(
                name=name,
                arguments=arguments,
                mcp_server_name=mcp_info.get("server_name"),
                mcp_server_logo_url=mcp_info.get("logo_url"),
                namespaced_tool_name=f"{server_name}/{name}" if server_name else name,
            )
        else:
            return StandardLoggingMCPToolCall(
                name=name,
                arguments=arguments,
                namespaced_tool_name=f"{server_name}/{name}" if server_name else name,
            )

    async def _handle_managed_mcp_tool(
        name: str,
        arguments: Dict[str, Any],
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
    ) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
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
    ) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
        """
        Handle tool execution for local registry tools
        Note: Local tools don't use prefixes, so we use the original name
        """
        tool = global_mcp_tool_registry.get_tool(name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        try:
            result = tool.handler(**arguments)
            return [TextContent(text=str(result), type="text")]
        except Exception as e:
            return [TextContent(text=f"Error: {str(e)}", type="text")]

    async def extract_mcp_auth_context(scope, path):
        """
        Extracts mcp_servers from the path and processes the MCP request for auth context.
        Returns: (user_api_key_auth, mcp_auth_header, mcp_servers)
        """
        import re
        mcp_servers_from_path = None
        mcp_path_match = re.match(r"^/mcp/([^/]+)(/.*)?$", path)
        if mcp_path_match:
            mcp_servers_str = mcp_path_match.group(1)
            if mcp_servers_str:
                mcp_servers_from_path = [s.strip() for s in mcp_servers_str.split(",") if s.strip()]

        if mcp_servers_from_path is not None:
            user_api_key_auth, mcp_auth_header, _ = (
                await MCPRequestHandler.process_mcp_request(scope)
            )
            mcp_servers = mcp_servers_from_path
        else:
            user_api_key_auth, mcp_auth_header, mcp_servers = (
                await MCPRequestHandler.process_mcp_request(scope)
            )
        return user_api_key_auth, mcp_auth_header, mcp_servers

    async def handle_streamable_http_mcp(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Handle MCP requests through StreamableHTTP."""
        try:
            path = scope.get("path", "")
            user_api_key_auth, mcp_auth_header, mcp_servers = await extract_mcp_auth_context(scope, path)
            verbose_logger.debug(f"MCP request mcp_servers (header/path): {mcp_servers}")
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
            path = scope.get("path", "")
            user_api_key_auth, mcp_auth_header, mcp_servers = await extract_mcp_auth_context(scope, path)
            verbose_logger.debug(f"MCP request mcp_servers (header/path): {mcp_servers}")
            set_auth_context(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
            )

            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
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
            mcp_servers: Optional list of server names and access groups to filter by
        """
        auth_user = MCPAuthenticatedUser(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
        )
        auth_context_var.set(auth_user)

    def get_auth_context() -> (
        Tuple[Optional[UserAPIKeyAuth], Optional[str], Optional[List[str]]]
    ):
        """
        Get the UserAPIKeyAuth from the auth context variable.

        Returns:
            Tuple[Optional[UserAPIKeyAuth], Optional[str], Optional[List[str]]]: UserAPIKeyAuth object, MCP auth header, and MCP servers (can include access groups)
        """
        auth_user = auth_context_var.get()
        if auth_user and isinstance(auth_user, MCPAuthenticatedUser):
            return (
                auth_user.user_api_key_auth,
                auth_user.mcp_auth_header,
                auth_user.mcp_servers,
            )
        return None, None, None

    ########################################################
    ############ End of Auth Context Functions #############
    ########################################################

else:
    app = FastAPI()
