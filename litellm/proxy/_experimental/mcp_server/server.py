"""
LiteLLM MCP Server Routes
"""
# pyright: reportInvalidTypeForm=false, reportArgumentType=false, reportOptionalCall=false

import asyncio
import contextlib
import traceback
import uuid
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from fastapi import FastAPI, HTTPException
from pydantic import AnyUrl, ConfigDict
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from litellm._logging import verbose_logger
from litellm.constants import MAXIMUM_TRACEBACK_LINES_TO_LOG
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    get_request_base_url,
)
from litellm.proxy._experimental.mcp_server.mcp_debug import MCPDebug
from litellm.proxy._experimental.mcp_server.utils import (
    LITELLM_MCP_SERVER_DESCRIPTION,
    LITELLM_MCP_SERVER_NAME,
    LITELLM_MCP_SERVER_VERSION,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.litellm_pre_call_utils import (
    LiteLLMProxyRequestSetup,
)
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPServer
from litellm.types.utils import CallTypes, StandardLoggingMCPToolCall
from litellm.utils import Rules, client, function_setup

# Check if MCP is available
# "mcp" requires python 3.10 or higher, but several litellm users use python 3.8
# We're making this conditional import to avoid breaking users who use python 3.8.
# TODO: Make this a util function for litellm client usage
MCP_AVAILABLE: bool = True
try:
    from mcp import ReadResourceResult, Resource
    from mcp.server import Server
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.types import (
        BlobResourceContents,
        GetPromptResult,
        ResourceTemplate,
        TextResourceContents,
    )
except ImportError as e:
    verbose_logger.debug(f"MCP module not found: {e}")
    MCP_AVAILABLE = False
    # When MCP is not available, we set these to None at module level
    # All code using these types is inside `if MCP_AVAILABLE:` blocks
    # so they will never be accessed at runtime
    BlobResourceContents = None  # type: ignore
    GetPromptResult = None  # type: ignore
    ReadResourceContents = None  # type: ignore
    ReadResourceResult = None  # type: ignore
    Resource = None  # type: ignore
    ResourceTemplate = None  # type: ignore
    Server = None  # type: ignore
    TextResourceContents = None  # type: ignore


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

    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    except ImportError:
        StreamableHTTPSessionManager = None  # type: ignore
    from mcp.types import (
        CallToolResult,
        EmbeddedResource,
        ImageContent,
        Prompt,
        TextContent,
    )
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
        split_server_prefix_from_name,
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
        json_response=False, # enables SSE streaming
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
                _client_ip,
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
                log_list_tools_to_spendlogs=True,
                list_tools_log_source="mcp_protocol",
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
    ) -> CallToolResult:
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
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        ) = get_auth_context()

        verbose_logger.debug(
            f"MCP mcp_server_tool_call - User API Key Auth from context: {user_api_key_auth}"
        )
        host_progress_callback = None
        try:
            host_ctx = server.request_context
            if host_ctx and hasattr(host_ctx, 'meta') and host_ctx.meta:
                host_token = getattr(host_ctx.meta, 'progressToken', None)
                if host_token and hasattr(host_ctx, 'session') and host_ctx.session:
                    host_session = host_ctx.session

                    async def forward_progress(progress: float, total: float | None):
                        """Forward progress notifications from external MCP to Host"""
                        try:
                            await host_session.send_progress_notification(
                                progress_token=host_token,
                                progress=progress,
                                total=total
                            )
                            verbose_logger.debug(f"Forwarded progress {progress}/{total} to Host")
                        except Exception as e:
                            verbose_logger.error(f"Failed to forward progress to Host: {e}")

                    host_progress_callback = forward_progress
                    verbose_logger.debug(f"Host progressToken captured: {host_token[:8]}...")
        except Exception as e:
            verbose_logger.warning(f"Could not capture host progress context: {e}")
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
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                host_progress_callback=host_progress_callback,
                **data,  # for logging
            )
        except BlockedPiiEntityError as e:
            verbose_logger.error(f"BlockedPiiEntityError in MCP tool call: {str(e)}")
            return CallToolResult(
                content=[
                    TextContent(
                        text=f"Error: Blocked PII entity detected - {str(e)}",
                        type="text",
                    )
                ],
                isError=True,
            )
        except GuardrailRaisedException as e:
            verbose_logger.error(f"GuardrailRaisedException in MCP tool call: {str(e)}")
            return CallToolResult(
                content=[
                    TextContent(
                        text=f"Error: Guardrail violation - {str(e)}", type="text"
                    )
                ],
                isError=True,
            )
        except HTTPException as e:
            verbose_logger.error(f"HTTPException in MCP tool call: {str(e)}")
            return CallToolResult(
                content=[TextContent(text=f"Error: {str(e.detail)}", type="text")],
                isError=True,
            )
        except Exception as e:
            verbose_logger.exception(f"MCP mcp_server_tool_call - error: {e}")
            return CallToolResult(
                content=[TextContent(text=f"Error: {str(e)}", type="text")],
                isError=True,
            )

        return response

    @server.list_prompts()
    async def list_prompts() -> List[Prompt]:
        """
        List all available prompts
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
                _client_ip,
            ) = get_auth_context()
            verbose_logger.debug(
                f"MCP list_prompts - User API Key Auth from context: {user_api_key_auth}"
            )
            verbose_logger.debug(
                f"MCP list_prompts - MCP servers from context: {mcp_servers}"
            )
            verbose_logger.debug(
                f"MCP list_prompts - MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )
            # Get mcp_servers from context variable
            verbose_logger.debug("MCP list_prompts - Calling _list_prompts")
            prompts = await _list_mcp_prompts(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.info(
                f"MCP list_prompts - Successfully returned {len(prompts)} prompts"
            )
            return prompts
        except Exception as e:
            verbose_logger.exception(f"Error in list_prompts endpoint: {str(e)}")
            # Return empty list instead of failing completely
            # This prevents the HTTP stream from failing and allows the client to get a response
            return []

    @server.get_prompt()
    async def get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> GetPromptResult:
        """
        Get a specific prompt with the provided arguments

        Args:
            name (str): Name of the prompt to get
            arguments (Dict[str, Any] | None): Arguments to pass to the prompt

        Returns:
            GetPromptResult: Getting prompt execution results
        """

        # Validate arguments
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        ) = get_auth_context()

        verbose_logger.debug(
            f"MCP mcp_server_tool_call - User API Key Auth from context: {user_api_key_auth}"
        )
        return await mcp_get_prompt(
            name=name,
            arguments=arguments,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
        )

    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """List all available resources."""
        try:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = get_auth_context()
            verbose_logger.debug(
                f"MCP list_resources - User API Key Auth from context: {user_api_key_auth}"
            )
            verbose_logger.debug(
                f"MCP list_resources - MCP servers from context: {mcp_servers}"
            )
            verbose_logger.debug(
                f"MCP list_resources - MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )

            resources = await _list_mcp_resources(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.info(
                f"MCP list_resources - Successfully returned {len(resources)} resources"
            )
            return resources
        except Exception as e:
            verbose_logger.exception(f"Error in list_resources endpoint: {str(e)}")
            return []

    @server.list_resource_templates()
    async def list_resource_templates() -> List[ResourceTemplate]:
        """List all available resource templates."""
        try:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = get_auth_context()
            verbose_logger.debug(
                f"MCP list_resource_templates - User API Key Auth from context: {user_api_key_auth}"
            )
            verbose_logger.debug(
                f"MCP list_resource_templates - MCP servers from context: {mcp_servers}"
            )
            verbose_logger.debug(
                f"MCP list_resource_templates - MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )

            resource_templates = await _list_mcp_resource_templates(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.info(
                "MCP list_resource_templates - Successfully returned "
                f"{len(resource_templates)} resource templates"
            )
            return resource_templates
        except Exception as e:
            verbose_logger.exception(
                f"Error in list_resource_templates endpoint: {str(e)}"
            )
            return []

    @server.read_resource()
    async def read_resource(url: AnyUrl) -> list[ReadResourceContents]:
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        ) = get_auth_context()

        read_resource_result = await mcp_read_resource(
            url=url,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
        )

        normalized_contents: List[ReadResourceContents] = []
        for content in read_resource_result.contents:
            if isinstance(content, TextResourceContents):
                text_content: TextResourceContents = content
                normalized_contents.append(
                    ReadResourceContents(
                        content=text_content.text,
                        mime_type=text_content.mimeType,
                    )
                )
            elif isinstance(content, BlobResourceContents):
                blob_content: BlobResourceContents = content
                normalized_contents.append(
                    ReadResourceContents(
                        content=blob_content.blob,
                        mime_type=None,
                    )
                )

        return normalized_contents

    ########################################################
    ############ End of MCP Server Routes ##################
    ########################################################

    ########################################################
    ############ Helper Functions ##########################
    ########################################################

    async def _get_allowed_mcp_servers_from_mcp_server_names(
        mcp_servers: Optional[List[str]],
        allowed_mcp_servers: List[MCPServer],
    ) -> List[MCPServer]:
        """
        Get the filtered MCP servers from the MCP server names
        """

        filtered_server: dict[str, MCPServer] = {}
        # Filter servers based on mcp_servers parameter if provided
        if mcp_servers is not None:
            for server_or_group in mcp_servers:
                server_name_matched = False

                for server in allowed_mcp_servers:
                    if server:
                        match_list = [
                            s.lower()
                            for s in [
                                server.alias,
                                server.server_name,
                                server.server_id,
                            ]
                            if s is not None
                        ]

                        if server_or_group.lower() in match_list:
                            filtered_server[server.server_id] = server
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
                            for server in allowed_mcp_servers:
                                if server_id == server.server_id:
                                    filtered_server[server.server_id] = server
                    except Exception as e:
                        verbose_logger.debug(
                            f"Could not resolve '{server_or_group}' as access group: {e}"
                        )

        if filtered_server:
            return list(filtered_server.values())

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
            split_server_prefix_from_name,
        )

        # Check if the full name is in the list
        if tool_name in filter_list:
            return True

        # Check if the unprefixed name is in the list
        unprefixed_name, _ = split_server_prefix_from_name(tool_name)
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

    def _get_client_ip_from_context() -> Optional[str]:
        """
        Extract client_ip from auth context.
        Returns None if context not set (caller should handle this as "no IP filtering").
        """
        try:
            auth_user = auth_context_var.get()
            if auth_user and isinstance(auth_user, MCPAuthenticatedUser):
                return auth_user.client_ip
        except Exception:
            pass
        return None

    async def _get_allowed_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_servers: Optional[List[str]],
        client_ip: Optional[str] = None,
    ) -> List[MCPServer]:
        """Return allowed MCP servers for a request after applying filters.

        Args:
            user_api_key_auth: The authenticated user's API key info.
            mcp_servers: Optional list of server names to filter to.
            client_ip: Client IP for IP-based access control. If None, falls back to
                      auth context. Pass explicitly from request handlers for safety.
        Note: If client_ip is None and auth context is not set, IP filtering is skipped.
              This is intentional for internal callers but may indicate a bug if called
              from a request handler without proper context setup.
        """
        # Use explicit client_ip if provided, otherwise try auth context
        if client_ip is None:
            client_ip = _get_client_ip_from_context()
            if client_ip is None:
                verbose_logger.debug(
                    "MCP _get_allowed_mcp_servers called without client_ip and no auth context. "
                    "IP filtering will be skipped. This is expected for internal calls."
                )

        allowed_mcp_server_ids = (
            await global_mcp_server_manager.get_allowed_mcp_servers(
                user_api_key_auth
            )
        )
        allowed_mcp_server_ids = (
            global_mcp_server_manager.filter_server_ids_by_ip(
                allowed_mcp_server_ids, client_ip
            )
        )
        verbose_logger.debug(
            "MCP IP filter: client_ip=%s, allowed_server_ids=%s",
            client_ip, allowed_mcp_server_ids,
        )
        allowed_mcp_servers: List[MCPServer] = []
        for allowed_mcp_server_id in allowed_mcp_server_ids:
            mcp_server = global_mcp_server_manager.get_mcp_server_by_id(
                allowed_mcp_server_id
            )
            if mcp_server is not None:
                allowed_mcp_servers.append(mcp_server)

        if mcp_servers is not None:
            allowed_mcp_servers = await _get_allowed_mcp_servers_from_mcp_server_names(
                mcp_servers=mcp_servers,
                allowed_mcp_servers=allowed_mcp_servers,
            )

        return allowed_mcp_servers

    def _prepare_mcp_server_headers(
        server: MCPServer,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        mcp_auth_header: Optional[str],
        oauth2_headers: Optional[Dict[str, str]],
        raw_headers: Optional[Dict[str, str]],
    ) -> Tuple[Optional[Union[Dict[str, str], str]], Optional[Dict[str, str]]]:
        """Build auth and extra headers for a server."""
        server_auth_header: Optional[Union[Dict[str, str], str]] = None
        if mcp_server_auth_headers and server.alias is not None:
            server_auth_header = mcp_server_auth_headers.get(server.alias)
        elif mcp_server_auth_headers and server.server_name is not None:
            server_auth_header = mcp_server_auth_headers.get(server.server_name)

        extra_headers: Optional[Dict[str, str]] = None
        if server.auth_type == MCPAuth.oauth2:
            # Copy to avoid mutating the original dict (important for parallel fetching)
            extra_headers = oauth2_headers.copy() if oauth2_headers else None

        if server.extra_headers and raw_headers:
            if extra_headers is None:
                extra_headers = {}

            normalized_raw_headers = {
                str(k).lower(): v for k, v in raw_headers.items() if isinstance(k, str)
            }

            for header in server.extra_headers:
                if not isinstance(header, str):
                    continue
                header_value = normalized_raw_headers.get(header.lower())
                if header_value is None:
                    continue
                extra_headers[header] = header_value

        if server_auth_header is None:
            server_auth_header = mcp_auth_header

        return server_auth_header, extra_headers

    async def _get_tools_from_mcp_servers(  # noqa: PLR0915
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        log_list_tools_to_spendlogs: bool = False,
        list_tools_log_source: Optional[str] = None,
        litellm_trace_id: Optional[str] = None,
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

        list_tools_start_time = datetime.now()
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = None
        list_tools_request_data: Dict[str, Any] = {}

        if log_list_tools_to_spendlogs:
            # This is intentionally minimal: only async_success_handler / post_call_failure_hook
            rules_obj = Rules()
            list_tools_call_id = str(uuid.uuid4())
            spend_logs_metadata: Dict[str, Any] = {
                "mcp_operation": "list_tools",
            }
            if isinstance(list_tools_log_source, str):
                spend_logs_metadata["source"] = list_tools_log_source
            if isinstance(mcp_servers, list):
                spend_logs_metadata["requested_mcp_servers"] = mcp_servers

            list_tools_request_data = {
                "model": "MCP: list_tools",
                "call_type": CallTypes.list_mcp_tools.value,
                "litellm_call_id": list_tools_call_id,
                "litellm_trace_id": litellm_trace_id,
                "metadata": {
                    "spend_logs_metadata": spend_logs_metadata,
                },
                # Provide a small input payload for standard logging
                "input": [
                    {
                        "role": "system",
                        "content": {
                            "mcp_operation": "list_tools",
                            "requested_mcp_servers": mcp_servers,
                        },
                    }
                ],
            }

            # Attach user identifiers using the standard helper
            if user_api_key_auth is not None:

                LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
                    data=list_tools_request_data,
                    user_api_key_dict=user_api_key_auth,
                    _metadata_variable_name="metadata",
                )

                user_identifier = getattr(
                    user_api_key_auth, "end_user_id", None
                ) or getattr(user_api_key_auth, "user_id", None)
                if user_identifier:
                    list_tools_request_data["user"] = user_identifier

            try:
                litellm_logging_obj, _ = function_setup(
                    original_function="list_mcp_tools",
                    rules_obj=rules_obj,
                    start_time=list_tools_start_time,
                    **list_tools_request_data,
                )
                if litellm_logging_obj:
                    litellm_logging_obj.call_type = CallTypes.list_mcp_tools.value
                    litellm_logging_obj.model = "MCP: list_tools"
            except Exception as logging_error:
                verbose_logger.debug(
                    "Failed to initialize logging for MCP list_tools: %s", logging_error
                )
                litellm_logging_obj = None

        try:
            allowed_mcp_servers = await _get_allowed_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_servers=mcp_servers,
            )

            # Decide whether to add prefix based on number of allowed servers
            add_prefix = not (len(allowed_mcp_servers) == 1)

            async def _fetch_and_filter_server_tools(
                server: MCPServer,
            ) -> List[MCPTool]:
                """Fetch and filter tools from a single server with error handling."""
                if server is None:
                    return []

                server_auth_header, extra_headers = _prepare_mcp_server_headers(
                    server=server,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    mcp_auth_header=mcp_auth_header,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                )

                try:
                    tools = await global_mcp_server_manager._get_tools_from_server(
                        server=server,
                        mcp_auth_header=server_auth_header,
                        extra_headers=extra_headers,
                        add_prefix=add_prefix,
                        raw_headers=raw_headers,
                    )
                    filtered_tools = filter_tools_by_allowed_tools(tools, server)

                    filtered_tools = await filter_tools_by_key_team_permissions(
                        tools=filtered_tools,
                        server_id=server.server_id,
                        user_api_key_auth=user_api_key_auth,
                    )

                    verbose_logger.debug(
                        f"Successfully fetched {len(tools)} tools from server {server.name}, {len(filtered_tools)} after filtering"
                    )
                    return filtered_tools
                except Exception as e:
                    verbose_logger.exception(
                        f"Error getting tools from server {server.name}: {str(e)}"
                    )
                    return []

            # Fetch tools from all servers in parallel
            tasks = [
                _fetch_and_filter_server_tools(server) for server in allowed_mcp_servers
            ]
            results = await asyncio.gather(*tasks)

            # Flatten results into single list
            all_tools: List[MCPTool] = [tool for tools in results for tool in tools]

            # If logging is enabled, enrich spend_logs_metadata with counts
            if litellm_logging_obj:
                per_server_tool_counts: Dict[str, int] = {}
                for server, server_tools in zip(allowed_mcp_servers, results):
                    if server is None:
                        continue
                    server_key = (
                        getattr(server, "server_name", None)
                        or getattr(server, "alias", None)
                        or getattr(server, "name", None)
                        or "unknown"
                    )
                    per_server_tool_counts[str(server_key)] = len(server_tools)

                metadata_dict = litellm_logging_obj.model_call_details.get("metadata")
                if isinstance(metadata_dict, dict):
                    spend_meta = metadata_dict.get("spend_logs_metadata")
                    if not isinstance(spend_meta, dict):
                        spend_meta = {}
                        metadata_dict["spend_logs_metadata"] = spend_meta
                    spend_meta["allowed_server_count"] = len(allowed_mcp_servers)
                    spend_meta["tool_count_total"] = len(all_tools)
                    spend_meta["per_server_tool_counts"] = per_server_tool_counts

                end_time = datetime.now()
                await litellm_logging_obj.async_success_handler(
                    result=all_tools,
                    start_time=list_tools_start_time,
                    end_time=end_time,
                )

            verbose_logger.info(
                f"Successfully fetched {len(all_tools)} tools total from all MCP servers"
            )

            return all_tools
        except Exception as e:
            # Only fire failure hook if logging was requested for this list-tools execution
            if log_list_tools_to_spendlogs and user_api_key_auth is not None:
                try:
                    from litellm.proxy.proxy_server import proxy_logging_obj

                    if proxy_logging_obj:
                        traceback_str = traceback.format_exc(
                            limit=MAXIMUM_TRACEBACK_LINES_TO_LOG
                        )
                        await proxy_logging_obj.post_call_failure_hook(
                            request_data=list_tools_request_data or {},
                            original_exception=e,
                            user_api_key_dict=user_api_key_auth,
                            route="/mcp/list_tools",
                            traceback_str=traceback_str,
                        )
                except Exception:
                    verbose_logger.debug(
                        "Failed to log MCP list_tools failure via post_call_failure_hook"
                    )
            raise

    async def _get_prompts_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]:
        """
        Helper method to fetch prompt from MCP servers based on server filtering criteria.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server (deprecated)
            mcp_servers: Optional list of server names/aliases to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers
            oauth2_headers: Optional dict of oauth2 headers

        Returns:
            List[Prompt]: Combined list of prompts from filtered servers
        """
        if not MCP_AVAILABLE:
            return []

        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        # Decide whether to add prefix based on number of allowed servers
        add_prefix = not (len(allowed_mcp_servers) == 1)

        # Get prompts from each allowed server
        all_prompts = []
        for server in allowed_mcp_servers:
            if server is None:
                continue

            server_auth_header, extra_headers = _prepare_mcp_server_headers(
                server=server,
                mcp_server_auth_headers=mcp_server_auth_headers,
                mcp_auth_header=mcp_auth_header,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )

            try:
                prompts = await global_mcp_server_manager.get_prompts_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    extra_headers=extra_headers,
                    add_prefix=add_prefix,
                    raw_headers=raw_headers,
                )

                all_prompts.extend(prompts)

                verbose_logger.debug(
                    f"Successfully fetched {len(prompts)} prompts from server {server.name}"
                )
            except Exception as e:
                verbose_logger.exception(
                    f"Error getting prompts from server {server.name}: {str(e)}"
                )
                # Continue with other servers instead of failing completely

        verbose_logger.info(
            f"Successfully fetched {len(all_prompts)} prompts total from all MCP servers"
        )

        return all_prompts

    async def _get_resources_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Resource]:
        """Fetch resources from allowed MCP servers."""

        if not MCP_AVAILABLE:
            return []

        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        add_prefix = not (len(allowed_mcp_servers) == 1)

        all_resources: List[Resource] = []
        for server in allowed_mcp_servers:
            if server is None:
                continue

            server_auth_header, extra_headers = _prepare_mcp_server_headers(
                server=server,
                mcp_server_auth_headers=mcp_server_auth_headers,
                mcp_auth_header=mcp_auth_header,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )

            try:
                resources = await global_mcp_server_manager.get_resources_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    extra_headers=extra_headers,
                    add_prefix=add_prefix,
                    raw_headers=raw_headers,
                )
                all_resources.extend(resources)

                verbose_logger.debug(
                    f"Successfully fetched {len(resources)} resources from server {server.name}"
                )
            except Exception as e:
                verbose_logger.exception(
                    f"Error getting resources from server {server.name}: {str(e)}"
                )

        verbose_logger.info(
            f"Successfully fetched {len(all_resources)} resources total from all MCP servers"
        )

        return all_resources

    async def _get_resource_templates_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[ResourceTemplate]:
        """Fetch resource templates from allowed MCP servers."""

        if not MCP_AVAILABLE:
            return []

        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        add_prefix = not (len(allowed_mcp_servers) == 1)

        all_resource_templates: List[ResourceTemplate] = []
        for server in allowed_mcp_servers:
            if server is None:
                continue

            server_auth_header, extra_headers = _prepare_mcp_server_headers(
                server=server,
                mcp_server_auth_headers=mcp_server_auth_headers,
                mcp_auth_header=mcp_auth_header,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )

            try:
                resource_templates = (
                    await global_mcp_server_manager.get_resource_templates_from_server(
                        server=server,
                        mcp_auth_header=server_auth_header,
                        extra_headers=extra_headers,
                        add_prefix=add_prefix,
                        raw_headers=raw_headers,
                    )
                )
                all_resource_templates.extend(resource_templates)
                verbose_logger.debug(
                    "Successfully fetched %s resource templates from server %s",
                    len(resource_templates),
                    server.name,
                )
            except Exception as e:
                verbose_logger.exception(
                    "Error getting resource templates from server %s: %s",
                    server.name,
                    str(e),
                )

        verbose_logger.info(
            "Successfully fetched %s resource templates total from all MCP servers",
            len(all_resource_templates),
        )

        return all_resource_templates

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
                unprefixed_tool_name, _ = split_server_prefix_from_name(t.name)
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
        log_list_tools_to_spendlogs: bool = False,
        list_tools_log_source: Optional[str] = None,
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
                log_list_tools_to_spendlogs=log_list_tools_to_spendlogs,
                list_tools_log_source=list_tools_log_source,
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

    async def _list_mcp_prompts(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]:
        """
        List all available MCP prompts.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server (deprecated)
            mcp_servers: Optional list of server names/aliases to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}

        Returns:
            List[Prompt]: Combined list of tools from all accessible servers
        """
        if not MCP_AVAILABLE:
            return []
        # Get tools from managed MCP servers with error handling
        managed_prompts = []
        try:
            managed_prompts = await _get_prompts_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.debug(
                f"Successfully fetched {len(managed_prompts)} prompts from managed MCP servers"
            )
        except Exception as e:
            verbose_logger.exception(
                f"Error getting tools from managed MCP servers: {str(e)}"
            )
            # Continue with empty managed tools list instead of failing completely

        return managed_prompts

    async def _list_mcp_resources(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Resource]:
        """List all available MCP resources."""

        if not MCP_AVAILABLE:
            return []

        managed_resources: List[Resource] = []
        try:
            managed_resources = await _get_resources_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.debug(
                f"Successfully fetched {len(managed_resources)} resources from managed MCP servers"
            )
        except Exception as e:
            verbose_logger.exception(
                f"Error getting resources from managed MCP servers: {str(e)}"
            )

        return managed_resources

    async def _list_mcp_resource_templates(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[ResourceTemplate]:
        """List all available MCP resource templates."""

        if not MCP_AVAILABLE:
            return []

        managed_resource_templates: List[ResourceTemplate] = []
        try:
            managed_resource_templates = await _get_resource_templates_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.debug(
                "Successfully fetched %s resource templates from managed MCP servers",
                len(managed_resource_templates),
            )
        except Exception as e:
            verbose_logger.exception(
                "Error getting resource templates from managed MCP servers: %s",
                str(e),
            )

        return managed_resource_templates

    async def execute_mcp_tool(
        name: str,
        arguments: Dict[str, Any],
        allowed_mcp_servers: List[MCPServer],
        start_time: datetime,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        host_progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> CallToolResult:
        """
        Execute MCP tool.

        This function assumes permission checks have already been performed.

        Args:
            name: Tool name (may include server prefix)
            arguments: Tool arguments
            allowed_mcp_servers: Pre-validated list of servers the user can access
            start_time: Start time for logging
            user_api_key_auth: Optional user API key auth for logging
            mcp_auth_header: Optional MCP auth header
            mcp_server_auth_headers: Optional server-specific auth headers
            oauth2_headers: Optional OAuth2 headers
            raw_headers: Optional raw HTTP headers
            **kwargs: Additional arguments (e.g., litellm_logging_obj)

        Returns:
            CallToolResult: Tool execution result
        """
        # Track resolved MCP server for both permission checks and dispatch
        mcp_server: Optional[MCPServer] = None

        # Remove prefix from tool name for logging and processing
        original_tool_name, server_name = split_server_prefix_from_name(name)

        # If tool name is unprefixed, resolve its server so we can enforce permissions
        if not server_name:
            mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)
            if mcp_server:
                server_name = mcp_server.name

        # Only enforce server-level permissions when we can resolve a server
        if server_name:
            if not MCPRequestHandler.is_tool_allowed(
                allowed_mcp_servers=[server.name for server in allowed_mcp_servers],
                server_name=server_name,
            ):
                raise HTTPException(
                    status_code=403,
                    detail=f"User not allowed to call this tool. Allowed MCP servers: {allowed_mcp_servers}",
                )

        standard_logging_mcp_tool_call: StandardLoggingMCPToolCall = (
            _get_standard_logging_mcp_tool_call(
                name=original_tool_name,  # Use original name for logging
                arguments=arguments,
                server_name=server_name,
            )
        )
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
            "litellm_logging_obj", None
        )
        if litellm_logging_obj:
            litellm_logging_obj.model_call_details[
                "mcp_tool_call_metadata"
            ] = standard_logging_mcp_tool_call
            litellm_logging_obj.model = f"MCP: {name}"
        # Check if tool exists in local registry first (for OpenAPI-based tools)
        # These tools are registered with their prefixed names
        #########################################################
        local_tool = global_mcp_tool_registry.get_tool(name)
        if local_tool:
            verbose_logger.debug(f"Executing local registry tool: {name}")
            local_content = await _handle_local_mcp_tool(name, arguments)
            response = CallToolResult(content=cast(Any, local_content), isError=False)

        # Try managed MCP server tool (pass the full prefixed name)
        # Primary and recommended way to use external MCP servers
        #########################################################
        else:
            # If we haven't already resolved the server, do it now for dispatch
            if mcp_server is None:
                mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(
                    name
                )
            if mcp_server:
                standard_logging_mcp_tool_call["mcp_server_cost_info"] = (
                    mcp_server.mcp_info or {}
                ).get("mcp_server_cost_info")
                # Update model_call_details with the cost info
                if litellm_logging_obj:
                    litellm_logging_obj.model_call_details[
                        "mcp_tool_call_metadata"
                    ] = standard_logging_mcp_tool_call
                response = await _handle_managed_mcp_tool(
                    server_name=server_name,
                    name=original_tool_name,  # Pass the full name (potentially prefixed)
                    arguments=arguments,
                    user_api_key_auth=user_api_key_auth,
                    mcp_auth_header=mcp_auth_header,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                    litellm_logging_obj=litellm_logging_obj,
                    host_progress_callback=host_progress_callback,
                )

            # Fall back to local tool registry with original name (legacy support)
            #########################################################
            # Deprecated: Local MCP Server Tool
            #########################################################
            else:
                local_content = await _handle_local_mcp_tool(
                    original_tool_name, arguments
                )
                response = CallToolResult(
                    content=cast(Any, local_content), isError=False
                )

        return response

    @client
    async def call_mcp_tool(
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> CallToolResult:
        """
        Call a specific tool with the provided arguments (handles prefixed tool names).
        """
        start_time = datetime.now()
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
            "litellm_logging_obj", None
        )

        try:
            if arguments is None:
                raise HTTPException(
                    status_code=400, detail="Request arguments are required"
                )

            ## CHECK IF USER IS ALLOWED TO CALL THIS TOOL
            allowed_mcp_server_ids = (
                await global_mcp_server_manager.get_allowed_mcp_servers(
                    user_api_key_auth=user_api_key_auth,
                )
            )

            allowed_mcp_servers: List[MCPServer] = []
            for allowed_mcp_server_id in allowed_mcp_server_ids:
                allowed_server = global_mcp_server_manager.get_mcp_server_by_id(
                    allowed_mcp_server_id
                )
                if allowed_server is not None:
                    allowed_mcp_servers.append(allowed_server)

            allowed_mcp_servers = await _get_allowed_mcp_servers_from_mcp_server_names(
                mcp_servers=mcp_servers,
                allowed_mcp_servers=allowed_mcp_servers,
            )
            if not allowed_mcp_servers:
                raise HTTPException(
                    status_code=403,
                    detail="User not allowed to call this tool.",
                )

            # Delegate to execute_mcp_tool for execution
            response = await execute_mcp_tool(
                name=name,
                arguments=arguments,
                allowed_mcp_servers=allowed_mcp_servers,
                start_time=start_time,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                **kwargs,
            )
        except Exception as e:
            traceback_str = traceback.format_exc(limit=MAXIMUM_TRACEBACK_LINES_TO_LOG)
            from litellm.proxy.proxy_server import proxy_logging_obj

            if proxy_logging_obj and user_api_key_auth:
                await proxy_logging_obj.post_call_failure_hook(
                    request_data=kwargs,
                    original_exception=e,
                    user_api_key_dict=user_api_key_auth,
                    route="/mcp/call_tool",
                    traceback_str=traceback_str,
                )
            raise

        if litellm_logging_obj:
            litellm_logging_obj.post_call(original_response=response)
            end_time = datetime.now()
            await litellm_logging_obj.async_post_mcp_tool_call_hook(
                kwargs=litellm_logging_obj.model_call_details,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
            )
            litellm_logging_obj.call_type = CallTypes.call_mcp_tool.value
            await litellm_logging_obj.async_success_handler(
                result=response, start_time=start_time, end_time=end_time
            )
        return response

    async def mcp_get_prompt(
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> GetPromptResult:
        """
        Fetch a specific MCP prompt, handling both prefixed and unprefixed names.
        """
        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        if not allowed_mcp_servers:
            raise HTTPException(
                status_code=403,
                detail="User not allowed to get this prompt.",
            )

        # Decide whether to add prefix based on number of allowed servers
        add_prefix = not (len(allowed_mcp_servers) == 1)

        if add_prefix:
            original_prompt_name, server_name = split_server_prefix_from_name(name)
        else:
            original_prompt_name = name
            server_name = allowed_mcp_servers[0].name

        server = next((s for s in allowed_mcp_servers if s.name == server_name), None)
        if server is None:
            raise HTTPException(
                status_code=403,
                detail="User not allowed to get this prompt.",
            )

        server_auth_header, extra_headers = _prepare_mcp_server_headers(
            server=server,
            mcp_server_auth_headers=mcp_server_auth_headers,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
        )

        return await global_mcp_server_manager.get_prompt_from_server(
            server=server,
            prompt_name=original_prompt_name,
            arguments=arguments,
            mcp_auth_header=server_auth_header,
            extra_headers=extra_headers,
            raw_headers=raw_headers,
        )

    async def mcp_read_resource(
        url: AnyUrl,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> ReadResourceResult:
        """Read resource contents from upstream MCP servers."""

        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        if not allowed_mcp_servers:
            raise HTTPException(
                status_code=403,
                detail="User not allowed to read this resource.",
            )

        if len(allowed_mcp_servers) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multiple MCP servers configured; read_resource currently "
                    "supports exactly one allowed server."
                ),
            )

        server = allowed_mcp_servers[0]

        server_auth_header, extra_headers = _prepare_mcp_server_headers(
            server=server,
            mcp_server_auth_headers=mcp_server_auth_headers,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
        )

        return await global_mcp_server_manager.read_resource_from_server(
            server=server,
            url=url,
            mcp_auth_header=server_auth_header,
            extra_headers=extra_headers,
            raw_headers=raw_headers,
        )

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
        server_name: str,
        name: str,
        arguments: Dict[str, Any],
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        litellm_logging_obj: Optional[Any] = None,
        host_progress_callback: Optional[Callable] = None,
    ) -> CallToolResult:
        """Handle tool execution for managed server tools"""
        # Import here to avoid circular import
        from litellm.proxy.proxy_server import proxy_logging_obj

        call_tool_result = await global_mcp_server_manager.call_tool(
            server_name=server_name,
            name=name,
            arguments=arguments,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            proxy_logging_obj=proxy_logging_obj,
            host_progress_callback=host_progress_callback,
        )
        verbose_logger.debug("CALL TOOL RESULT: %s", call_tool_result)
        return call_tool_result

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

    async def _handle_stale_mcp_session(
        scope: Scope,
        receive: Receive,
        send: Send,
        mgr: "StreamableHTTPSessionManager",
    ) -> bool:
        """
        Handle stale MCP session IDs to prevent "Session not found" errors.

        When clients reconnect after a server restart or session cleanup, they may
        send a session ID that no longer exists. This function handles two scenarios:

        1. Non-DELETE requests: Strip the stale session ID header so the session
           manager creates a fresh session transparently.

        2. DELETE requests: Return success (200) immediately for idempotent behavior,
           since the desired state (session doesn't exist) is already achieved.

        Returns:
            True if the request was handled (DELETE on non-existent session)
            False if the request should continue to the session manager

        Fixes https://github.com/BerriAI/litellm/issues/20292
        """
        _mcp_session_header = b"mcp-session-id"
        _session_id: Optional[str] = None
        for header_name, header_value in scope.get("headers", []):
            if header_name == _mcp_session_header:
                _session_id = header_value.decode("utf-8", errors="replace")
                break

        if _session_id is None:
            return False

        known_sessions = getattr(mgr, "_server_instances", None)
        if known_sessions is None or _session_id in known_sessions:
            # Session exists or we can't check - let the session manager handle it
            return False

        # Session doesn't exist - handle based on request method
        method = scope.get("method", "").upper()
        
        if method == "DELETE":
            # Idempotent DELETE: session doesn't exist, return success
            verbose_logger.info(
                f"DELETE request for non-existent MCP session '{_session_id}'. "
                "Returning success (idempotent DELETE)."
            )
            success_response = JSONResponse(
                status_code=200,
                content={"message": "Session terminated successfully"}
            )
            await success_response(scope, receive, send)
            return True
        else:
            # Non-DELETE: strip stale session ID to allow new session creation
            verbose_logger.warning(
                "MCP session ID '%s' not found in active sessions. "
                "Stripping stale header to force new session creation.",
                _session_id,
            )
            scope["headers"] = [
                (k, v) for k, v in scope["headers"]
                if k != _mcp_session_header
            ]
            return False

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

            # Extract client IP for MCP access control
            _client_ip = IPAddressUtils.get_mcp_client_ip(StarletteRequest(scope))

            verbose_logger.debug(
                f"MCP request mcp_servers (header/path): {mcp_servers}"
            )
            verbose_logger.debug(
                f"MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )
            # https://datatracker.ietf.org/doc/html/rfc9728#name-www-authenticate-response
            for server_name in mcp_servers or []:
                server = global_mcp_server_manager.get_mcp_server_by_name(
                    server_name, client_ip=_client_ip
                )
                if server and server.auth_type == MCPAuth.oauth2 and not oauth2_headers:
                    request = StarletteRequest(scope)
                    base_url = get_request_base_url(request)

                    authorization_uri = (
                        f"Bearer authorization_uri="
                        f"{base_url}/.well-known/oauth-authorization-server/{server_name}"
                    )

                    raise HTTPException(
                        status_code=401,
                        detail="Unauthorized",
                        headers={"www-authenticate": authorization_uri},
                    )

            # Inject masked debug headers when client sends x-litellm-mcp-debug: true
            _debug_headers = MCPDebug.maybe_build_debug_headers(
                raw_headers=raw_headers,
                scope=dict(scope),
                mcp_servers=mcp_servers,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                client_ip=_client_ip,
            )
            if _debug_headers:
                send = MCPDebug.wrap_send_with_debug_headers(send, _debug_headers)

            # Set the auth context variable for easy access in MCP functions
            set_auth_context(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                client_ip=_client_ip,
            )

            # Ensure session managers are initialized
            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
                # Give it a moment to start up
                await asyncio.sleep(0.1)

            # Handle stale session IDs - either strip them for reconnection
            # or return success for idempotent DELETE operations
            handled = await _handle_stale_mcp_session(scope, receive, send, session_manager)
            if handled:
                # Request was fully handled (e.g., DELETE on non-existent session)
                return

            await session_manager.handle_request(scope, receive, send)
        except HTTPException:
            # Re-raise HTTP exceptions to preserve status codes and details
            raise
        except Exception as e:
            verbose_logger.exception(f"Error handling MCP request: {e}")
            # Try to send a graceful error response for non-HTTP exceptions
            try:
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

            # Extract client IP for MCP access control
            _sse_client_ip = IPAddressUtils.get_mcp_client_ip(StarletteRequest(scope))

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
                client_ip=_sse_client_ip,
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
        client_ip: Optional[str] = None,
    ) -> None:
        """
        Set the UserAPIKeyAuth in the auth context variable.

        Args:
            user_api_key_auth: UserAPIKeyAuth object
            mcp_auth_header: MCP auth header to be passed to the MCP server (deprecated)
            mcp_servers: Optional list of server names and access groups to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
            client_ip: Client IP address for MCP access control
        """
        auth_user = MCPAuthenticatedUser(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )
        auth_context_var.set(auth_user)

    def get_auth_context() -> (
        Tuple[
            Optional[UserAPIKeyAuth],
            Optional[str],
            Optional[List[str]],
            Optional[Dict[str, Dict[str, str]]],
            Optional[Dict[str, str]],
            Optional[Dict[str, str]],
            Optional[str],
        ]
    ):
        """
        Get the UserAPIKeyAuth from the auth context variable.

        Returns:
            Tuple containing: UserAPIKeyAuth, MCP auth header (deprecated),
            MCP servers, server-specific auth headers, OAuth2 headers, raw headers, client IP
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
                auth_user.client_ip,
            )
        return None, None, None, None, None, None, None

    ########################################################
    ############ End of Auth Context Functions #############
    ########################################################

else:
    app = FastAPI()
