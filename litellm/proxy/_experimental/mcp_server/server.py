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
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp import MCPAuth
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
_INITIALIZATION_LOCK = asyncio.Lock()

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

        # Use async lock to prevent concurrent initialization
        async with _INITIALIZATION_LOCK:
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
        try:
            # Get user authentication from context variable
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = get_auth_context()
            verbose_logger.debug(
                f"MCP list_tools - User API Key Auth from context: {user_api_key_auth}"
            )
            verbose_logger.debug(
                f"MCP list_tools - MCP servers from context: {mcp_servers}"
            )
            verbose_logger.debug(
                f"MCP list_tools - MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )
            # Get mcp_servers from context variable
            verbose_logger.debug("MCP list_tools - Calling _list_mcp_tools")
            tools = await _list_mcp_tools(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.info(
                f"MCP list_tools - Successfully returned {len(tools)} tools"
            )
            return tools
        except Exception as e:
            verbose_logger.exception(f"Error in list_tools endpoint: {str(e)}")
            # Return empty list instead of failing completely
            # This prevents the HTTP stream from failing and allows the client to get a response
            return []

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

        from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
        from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
        from litellm.proxy.proxy_server import proxy_config

        # Validate arguments
        (
            user_api_key_auth,
            mcp_auth_header,
            _,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
        ) = get_auth_context()

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
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                **data,  # for logging
            )
        except BlockedPiiEntityError as e:
            verbose_logger.error(f"BlockedPiiEntityError in MCP tool call: {str(e)}")
            # Return error as text content for MCP protocol
            return [
                TextContent(
                    text=f"Error: Blocked PII entity detected - {str(e)}", type="text"
                )
            ]
        except GuardrailRaisedException as e:
            verbose_logger.error(f"GuardrailRaisedException in MCP tool call: {str(e)}")
            # Return error as text content for MCP protocol
            return [
                TextContent(text=f"Error: Guardrail violation - {str(e)}", type="text")
            ]
        except HTTPException as e:
            verbose_logger.error(f"HTTPException in MCP tool call: {str(e)}")
            # Return error as text content for MCP protocol
            return [TextContent(text=f"Error: {str(e.detail)}", type="text")]
        except Exception as e:
            verbose_logger.exception(f"MCP mcp_server_tool_call - error: {e}")
            # Return error as text content for MCP protocol
            return [TextContent(text=f"Error: {str(e)}", type="text")]

        return response

    ########################################################
    ############ End of MCP Server Routes ##################
    ########################################################

    ########################################################
    ############ Helper Functions ##########################
    ########################################################

    async def _get_allowed_mcp_servers_from_mcp_server_names(
        mcp_servers: Optional[List[str]],
        allowed_mcp_servers: List[str],
    ) -> List[str]:
        """
        Get the filtered MCP servers from the MCP server names
        """
        from typing import Set

        filtered_server_ids: Set[str] = set()
        # Filter servers based on mcp_servers parameter if provided
        if mcp_servers is not None:
            for server_or_group in mcp_servers:
                server_name_matched = False

                for server_id in allowed_mcp_servers:
                    server = global_mcp_server_manager.get_mcp_server_by_id(server_id)

                    if server:
                        match_list = [
                            s.lower()
                            for s in [server.alias, server.server_name, server_id]
                            if s is not None
                        ]

                        if server_or_group.lower() in match_list:
                            filtered_server_ids.add(server_id)
                            server_name_matched = True
                            break

                if not server_name_matched:
                    try:
                        access_group_server_ids = (
                            await MCPRequestHandler._get_mcp_servers_from_access_groups(
                                [server_or_group]
                            )
                        )
                        # Only include servers that the user has access to
                        for server_id in access_group_server_ids:
                            if server_id in allowed_mcp_servers:
                                filtered_server_ids.add(server_id)
                    except Exception as e:
                        verbose_logger.debug(
                            f"Could not resolve '{server_or_group}' as access group: {e}"
                        )

        if filtered_server_ids:
            allowed_mcp_servers = list(filtered_server_ids)

        return allowed_mcp_servers

    def _tool_name_matches(tool_name: str, filter_list: List[str]) -> bool:
        """
        Check if a tool name matches any name in the filter list.

        Checks both the full tool name and unprefixed version (without server prefix).
        This allows users to configure simple tool names regardless of prefixing.

        Args:
            tool_name: The tool name to check (may be prefixed like "server-tool_name")
            filter_list: List of tool names to match against

        Returns:
            True if the tool name (prefixed or unprefixed) is in the filter list
        """
        from litellm.proxy._experimental.mcp_server.utils import (
            get_server_name_prefix_tool_mcp,
        )

        # Check if the full name is in the list
        if tool_name in filter_list:
            return True

        # Check if the unprefixed name is in the list
        unprefixed_name, _ = get_server_name_prefix_tool_mcp(tool_name)
        return unprefixed_name in filter_list

    def filter_tools_by_allowed_tools(
        tools: List[MCPTool],
        mcp_server: MCPServer,
    ) -> List[MCPTool]:
        """
        Filter tools by allowed/disallowed tools configuration.

        If allowed_tools is set, only tools in that list are returned.
        If disallowed_tools is set, tools in that list are excluded.
        Tool names are matched with and without server prefixes for flexibility.

        Args:
            tools: List of tools to filter
            mcp_server: Server configuration with allowed_tools/disallowed_tools

        Returns:
            Filtered list of tools
        """
        tools_to_return = tools

        # Filter by allowed_tools (whitelist)
        if mcp_server.allowed_tools:
            tools_to_return = [
                tool
                for tool in tools
                if _tool_name_matches(tool.name, mcp_server.allowed_tools)
            ]

        # Filter by disallowed_tools (blacklist)
        if mcp_server.disallowed_tools:
            tools_to_return = [
                tool
                for tool in tools_to_return
                if not _tool_name_matches(tool.name, mcp_server.disallowed_tools)
            ]

        return tools_to_return

    async def _get_tools_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[MCPTool]:
        """
        Helper method to fetch tools from MCP servers based on server filtering criteria.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server (deprecated)
            mcp_servers: Optional list of server names/aliases to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers
            oauth2_headers: Optional dict of oauth2 headers

        Returns:
            List[MCPTool]: Combined list of tools from filtered servers
        """
        if not MCP_AVAILABLE:
            return []

        # Get allowed MCP servers based on user permissions
        allowed_mcp_servers = await global_mcp_server_manager.get_allowed_mcp_servers(
            user_api_key_auth
        )

        if mcp_servers is not None:
            allowed_mcp_servers = await _get_allowed_mcp_servers_from_mcp_server_names(
                mcp_servers=mcp_servers,
                allowed_mcp_servers=allowed_mcp_servers,
            )

        # Decide whether to add prefix based on number of allowed servers
        add_prefix = not (len(allowed_mcp_servers) == 1)

        # Get tools from each allowed server
        all_tools = []
        for server_id in allowed_mcp_servers:
            server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
            if server is None:
                continue

            # Get server-specific auth header if available
            server_auth_header: Optional[Union[Dict[str, str], str]] = None
            if mcp_server_auth_headers and server.alias is not None:
                server_auth_header = mcp_server_auth_headers.get(server.alias)
            elif mcp_server_auth_headers and server.server_name is not None:
                server_auth_header = mcp_server_auth_headers.get(server.server_name)

            extra_headers: Optional[Dict[str, str]] = None
            if server.auth_type == MCPAuth.oauth2:
                extra_headers = oauth2_headers

            if server.extra_headers and raw_headers:
                if extra_headers is None:
                    extra_headers = {}
                for header in server.extra_headers:
                    if header in raw_headers:
                        extra_headers[header] = raw_headers[header]

            # Fall back to deprecated mcp_auth_header if no server-specific header found
            if server_auth_header is None:
                server_auth_header = mcp_auth_header

            try:
                tools = await global_mcp_server_manager._get_tools_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    extra_headers=extra_headers,
                    add_prefix=add_prefix,
                )

                filtered_tools = filter_tools_by_allowed_tools(tools, server)

                filtered_tools = await filter_tools_by_key_team_permissions(
                    tools=filtered_tools,
                    server_id=server_id,
                    user_api_key_auth=user_api_key_auth,
                )

                all_tools.extend(filtered_tools)

                verbose_logger.debug(
                    f"Successfully fetched {len(tools)} tools from server {server.name}, {len(filtered_tools)} after filtering"
                )
            except Exception as e:
                verbose_logger.exception(
                    f"Error getting tools from server {server.name}: {str(e)}"
                )
                # Continue with other servers instead of failing completely

        verbose_logger.info(
            f"Successfully fetched {len(all_tools)} tools total from all MCP servers"
        )

        return all_tools

    async def filter_tools_by_key_team_permissions(
        tools: List[MCPTool],
        server_id: str,
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> List[MCPTool]:
        """
        Filter tools based on key/team mcp_tool_permissions.

        Note: Tool names in the DB are stored without server prefixes,
        but tool names from MCP servers are prefixed. We need to strip
        the prefix before comparing.
        """
        # Filter by key/team tool-level permissions
        allowed_tool_names = await MCPRequestHandler.get_allowed_tools_for_server(
            server_id=server_id,
            user_api_key_auth=user_api_key_auth,
        )
        if allowed_tool_names is not None:
            # Strip prefix from tool names before comparing
            # Tools are stored in DB without prefix, but come from MCP server with prefix
            filtered_tools = []
            for t in tools:
                # Get tool name without server prefix
                unprefixed_tool_name, _ = get_server_name_prefix_tool_mcp(t.name)
                if unprefixed_tool_name in allowed_tool_names:
                    filtered_tools.append(t)
        else:
            # No restrictions, return all tools
            filtered_tools = tools

        return filtered_tools

    async def _list_mcp_tools(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[MCPTool]:
        """
        List all available MCP tools.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server (deprecated)
            mcp_servers: Optional list of server names/aliases to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}

        Returns:
            List[MCPTool]: Combined list of tools from all accessible servers
        """
        if not MCP_AVAILABLE:
            return []
        # Get tools from managed MCP servers with error handling
        managed_tools = []
        try:
            managed_tools = await _get_tools_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.debug(
                f"Successfully fetched {len(managed_tools)} tools from managed MCP servers"
            )
        except Exception as e:
            verbose_logger.exception(
                f"Error getting tools from managed MCP servers: {str(e)}"
            )
            # Continue with empty managed tools list instead of failing completely

        return managed_tools

    @client
    async def call_mcp_tool(
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
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

        ## CHECK IF USER IS ALLOWED TO CALL THIS TOOL
        allowed_mcp_server_ids = await MCPRequestHandler.get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
        )

        allowed_mcp_servers = global_mcp_server_manager.get_mcp_server_names_from_ids(
            allowed_mcp_server_ids
        )

        if not MCPRequestHandler.is_tool_allowed(
            allowed_mcp_servers=allowed_mcp_servers,
            server_name=server_name_from_prefix,
        ):

            raise HTTPException(
                status_code=403,
                detail=f"User not allowed to call this tool. Allowed MCP servers: {allowed_mcp_servers}",
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
        # Check if tool exists in local registry first (for OpenAPI-based tools)
        # These tools are registered with their prefixed names
        #########################################################
        local_tool = global_mcp_tool_registry.get_tool(name)
        if local_tool:
            verbose_logger.debug(f"Executing local registry tool: {name}")
            response = await _handle_local_mcp_tool(name, arguments)

        # Try managed MCP server tool (pass the full prefixed name)
        # Primary and recommended way to use external MCP servers
        #########################################################
        else:
            mcp_server: Optional[MCPServer] = (
                global_mcp_server_manager._get_mcp_server_from_tool_name(name)
            )
            if mcp_server:
                standard_logging_mcp_tool_call["mcp_server_cost_info"] = (
                    mcp_server.mcp_info or {}
                ).get("mcp_server_cost_info")
                response = await _handle_managed_mcp_tool(
                    name=name,  # Pass the full name (potentially prefixed)
                    arguments=arguments,
                    user_api_key_auth=user_api_key_auth,
                    mcp_auth_header=mcp_auth_header,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                    litellm_logging_obj=litellm_logging_obj,
                )

            # Fall back to local tool registry with original name (legacy support)
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
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        litellm_logging_obj: Optional[Any] = None,
    ) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
        """Handle tool execution for managed server tools"""
        # Import here to avoid circular import
        from litellm.proxy.proxy_server import proxy_logging_obj

        call_tool_result = await global_mcp_server_manager.call_tool(
            name=name,
            arguments=arguments,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            proxy_logging_obj=proxy_logging_obj,
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
        import inspect

        tool = global_mcp_tool_registry.get_tool(name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        try:
            # Check if handler is async or sync
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
            return [TextContent(text=str(result), type="text")]
        except Exception as e:
            verbose_logger.exception(f"Error executing local tool {name}: {str(e)}")
            return [TextContent(text=f"Error: {str(e)}", type="text")]

    def _get_mcp_servers_in_path(path: str) -> Optional[List[str]]:
        """
        Get the MCP servers from the path
        """
        import re

        mcp_servers_from_path: Optional[List[str]] = None
        # Match /mcp/<servers_and_maybe_path>
        # Where servers can be comma-separated list of server names
        # Server names can contain slashes (e.g., "custom_solutions/user_123")
        mcp_path_match = re.match(r"^/mcp/([^?#]+)(?:\?.*)?(?:#.*)?$", path)
        if mcp_path_match:
            servers_and_path = mcp_path_match.group(1)

            if servers_and_path:
                # Check if it contains commas (comma-separated servers)
                if "," in servers_and_path:
                    # For comma-separated, look for a path at the end
                    # Common patterns: /tools, /chat/completions, etc.
                    path_match = re.search(r"/([^/,]+(?:/[^/,]+)*)$", servers_and_path)
                    if path_match:
                        # Path found at the end, remove it from servers
                        path_part = "/" + path_match.group(1)
                        servers_part = servers_and_path[: -len(path_part)]
                        mcp_servers_from_path = [
                            s.strip() for s in servers_part.split(",") if s.strip()
                        ]
                    else:
                        # No path, just comma-separated servers
                        mcp_servers_from_path = [
                            s.strip() for s in servers_and_path.split(",") if s.strip()
                        ]
                else:
                    # Single server case - use regex approach for server/path separation
                    # This handles cases like "custom_solutions/user_123/chat/completions"
                    # where we want to extract "custom_solutions/user_123" as the server name
                    single_server_match = re.match(
                        r"^([^/]+(?:/[^/]+)?)(?:/.*)?$", servers_and_path
                    )
                    if single_server_match:
                        server_name = single_server_match.group(1)
                        mcp_servers_from_path = [server_name]
                    else:
                        mcp_servers_from_path = [servers_and_path]
        return mcp_servers_from_path

    async def extract_mcp_auth_context(scope, path):
        """
        Extracts mcp_servers from the path and processes the MCP request for auth context.
        Returns: (user_api_key_auth, mcp_auth_header, mcp_servers, mcp_server_auth_headers)
        """
        mcp_servers_from_path = _get_mcp_servers_in_path(path)
        if mcp_servers_from_path is not None:
            (
                user_api_key_auth,
                mcp_auth_header,
                _,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)
            mcp_servers = mcp_servers_from_path
        else:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)
        return (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
        )

    async def handle_streamable_http_mcp(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Handle MCP requests through StreamableHTTP."""
        try:
            path = scope.get("path", "")
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await extract_mcp_auth_context(scope, path)
            verbose_logger.debug(
                f"MCP request mcp_servers (header/path): {mcp_servers}"
            )
            verbose_logger.debug(
                f"MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )
            # Set the auth context variable for easy access in MCP functions
            set_auth_context(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )

            # Ensure session managers are initialized
            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
                # Give it a moment to start up
                await asyncio.sleep(0.1)

            await session_manager.handle_request(scope, receive, send)
        except Exception as e:
            raise e
            verbose_logger.exception(f"Error handling MCP request: {e}")
            # Instead of re-raising, try to send a graceful error response
            try:
                # Send a proper HTTP error response instead of letting the exception bubble up
                from starlette.responses import JSONResponse
                from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

                error_response = JSONResponse(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"error": "MCP request failed", "details": str(e)},
                )
                await error_response(scope, receive, send)
            except Exception as response_error:
                verbose_logger.exception(
                    f"Failed to send error response: {response_error}"
                )
                # If we can't send a proper response, re-raise the original error
                raise e

    async def handle_sse_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP requests through SSE."""
        try:
            path = scope.get("path", "")
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await extract_mcp_auth_context(scope, path)
            verbose_logger.debug(
                f"MCP request mcp_servers (header/path): {mcp_servers}"
            )
            verbose_logger.debug(
                f"MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )
            set_auth_context(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )

            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
                await asyncio.sleep(0.1)

            await sse_session_manager.handle_request(scope, receive, send)
        except Exception as e:
            verbose_logger.exception(f"Error handling MCP request: {e}")
            # Instead of re-raising, try to send a graceful error response
            try:
                # Send a proper HTTP error response instead of letting the exception bubble up
                from starlette.responses import JSONResponse
                from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

                error_response = JSONResponse(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"error": "MCP request failed", "details": str(e)},
                )
                await error_response(scope, receive, send)
            except Exception as response_error:
                verbose_logger.exception(
                    f"Failed to send error response: {response_error}"
                )
                # If we can't send a proper response, re-raise the original error
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
    app.mount("/mcp", handle_streamable_http_mcp)
    app.mount("/{mcp_server_name}/mcp", handle_streamable_http_mcp)
    app.mount("/sse", handle_sse_mcp)
    app.add_middleware(AuthContextMiddleware)

    ########################################################
    ############ Auth Context Functions ####################
    ########################################################

    def set_auth_context(
        user_api_key_auth: UserAPIKeyAuth,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set the UserAPIKeyAuth in the auth context variable.

        Args:
            user_api_key_auth: UserAPIKeyAuth object
            mcp_auth_header: MCP auth header to be passed to the MCP server (deprecated)
            mcp_servers: Optional list of server names and access groups to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
        """
        auth_user = MCPAuthenticatedUser(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
        )
        auth_context_var.set(auth_user)

    def get_auth_context() -> Tuple[
        Optional[UserAPIKeyAuth],
        Optional[str],
        Optional[List[str]],
        Optional[Dict[str, Dict[str, str]]],
        Optional[Dict[str, str]],
        Optional[Dict[str, str]],
    ]:
        """
        Get the UserAPIKeyAuth from the auth context variable.

        Returns:
            Tuple[Optional[UserAPIKeyAuth], Optional[str], Optional[List[str]], Optional[Dict[str, str]]]:
            UserAPIKeyAuth object, MCP auth header (deprecated), MCP servers (can include access groups), and server-specific auth headers
        """
        auth_user = auth_context_var.get()
        if auth_user and isinstance(auth_user, MCPAuthenticatedUser):
            return (
                auth_user.user_api_key_auth,
                auth_user.mcp_auth_header,
                auth_user.mcp_servers,
                auth_user.mcp_server_auth_headers,
                auth_user.oauth2_headers,
                auth_user.raw_headers,
            )
        return None, None, None, None, None, None

    ########################################################
    ############ End of Auth Context Functions #############
    ########################################################

else:
    app = FastAPI()
