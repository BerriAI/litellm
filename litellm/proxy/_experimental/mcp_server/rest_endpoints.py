import importlib
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.ui_session_utils import (
    build_effective_auth_contexts,
)
from litellm.proxy._experimental.mcp_server.utils import merge_mcp_headers
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.mcp import MCPAuth
from litellm.types.utils import CallTypes

MCP_AVAILABLE: bool = True
try:
    importlib.import_module("mcp")
except ImportError as e:
    verbose_logger.debug(f"MCP module not found: {e}")
    MCP_AVAILABLE = False


router = APIRouter(
    prefix="/mcp-rest",
    tags=["mcp"],
)

if MCP_AVAILABLE:
    from mcp.types import Tool as MCPTool

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.server import (
        ListMCPToolsRestAPIResponseObject,
        MCPServer,
        _tool_name_matches,
        execute_mcp_tool,
        filter_tools_by_allowed_tools,
    )

    ########################################################
    ############ MCP Server REST API Routes #################
    def _get_server_auth_header(
        server,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        mcp_auth_header: Optional[str],
    ) -> Optional[Union[Dict[str, str], str]]:
        """Helper function to get server-specific auth header with case-insensitive matching."""
        if mcp_server_auth_headers and server.alias:
            normalized_server_alias = server.alias.lower()
            normalized_headers = {
                k.lower(): v for k, v in mcp_server_auth_headers.items()
            }
            server_auth = normalized_headers.get(normalized_server_alias)
            if server_auth is not None:
                return server_auth
        elif mcp_server_auth_headers and server.server_name:
            normalized_server_name = server.server_name.lower()
            normalized_headers = {
                k.lower(): v for k, v in mcp_server_auth_headers.items()
            }
            server_auth = normalized_headers.get(normalized_server_name)
            if server_auth is not None:
                return server_auth
        return mcp_auth_header

    def _create_tool_response_objects(tools, server_mcp_info):
        """Helper function to create tool response objects."""
        return [
            ListMCPToolsRestAPIResponseObject(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.inputSchema,
                mcp_info=server_mcp_info,
            )
            for tool in tools
        ]

    def _extract_mcp_headers_from_request(
        request: Request,
        mcp_request_handler_cls,
    ) -> tuple:
        """
        Extract MCP auth headers from HTTP request.

        Returns:
            Tuple of (mcp_auth_header, mcp_server_auth_headers, raw_headers)
        """
        headers = request.headers
        raw_headers = dict(headers)
        mcp_auth_header = mcp_request_handler_cls._get_mcp_auth_header_from_headers(
            headers
        )
        mcp_server_auth_headers = (
            mcp_request_handler_cls._get_mcp_server_auth_headers_from_headers(headers)
        )
        return mcp_auth_header, mcp_server_auth_headers, raw_headers

    async def _resolve_allowed_mcp_servers_with_ip_filter(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth,
        server_id: str,
    ) -> List[MCPServer]:
        """
        Resolve allowed MCP servers for a tool call with IP filtering.

        Args:
            request: The HTTP request object
            user_api_key_dict: The user's API key auth object
            server_id: The server ID to validate access for

        Returns:
            List of allowed MCPServer objects

        Raises:
            HTTPException: If the server_id is not allowed
        """
        # Get all auth contexts
        auth_contexts = await build_effective_auth_contexts(user_api_key_dict)

        # Collect allowed server IDs from all contexts, then apply IP filtering
        _rest_client_ip = IPAddressUtils.get_mcp_client_ip(request)
        allowed_server_ids_set = set()
        for auth_context in auth_contexts:
            servers = await global_mcp_server_manager.get_allowed_mcp_servers(
                user_api_key_auth=auth_context,
            )
            allowed_server_ids_set.update(servers)

        allowed_server_ids_set = set(
            global_mcp_server_manager.filter_server_ids_by_ip(
                list(allowed_server_ids_set), _rest_client_ip
            )
        )

        # Check if the specified server_id is allowed
        if server_id not in allowed_server_ids_set:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "access_denied",
                    "message": f"The key is not allowed to access server {server_id}",
                },
            )

        # Build allowed_mcp_servers list (only include allowed servers)
        allowed_mcp_servers: List[MCPServer] = []
        for allowed_server_id in allowed_server_ids_set:
            server = global_mcp_server_manager.get_mcp_server_by_id(allowed_server_id)
            if server is not None:
                allowed_mcp_servers.append(server)

        return allowed_mcp_servers

    async def _get_tools_for_single_server(
        server,
        server_auth_header,
        raw_headers: Optional[Dict[str, str]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ):
        """Helper function to get tools for a single server."""
        tools = await global_mcp_server_manager._get_tools_from_server(
            server=server,
            mcp_auth_header=server_auth_header,
            add_prefix=False,
            raw_headers=raw_headers,
        )

        # Filter tools based on allowed_tools configuration
        # Only filter if allowed_tools is explicitly configured (not None and not empty)
        if server.allowed_tools is not None and len(server.allowed_tools) > 0:
            tools = filter_tools_by_allowed_tools(tools, server)

        # Filter tools based on user_api_key_auth.object_permission.mcp_tool_permissions
        # This provides per-key/team/org control over which tools can be accessed
        if (
            user_api_key_auth
            and user_api_key_auth.object_permission
            and user_api_key_auth.object_permission.mcp_tool_permissions
        ):
            allowed_tools_for_server = (
                user_api_key_auth.object_permission.mcp_tool_permissions.get(
                    server.server_id
                )
            )
            if (
                allowed_tools_for_server is not None
                and len(allowed_tools_for_server) > 0
            ):
                # Filter tools to only include those in the allowed list
                tools = [
                    tool
                    for tool in tools
                    if _tool_name_matches(tool.name, allowed_tools_for_server)
                ]

        return _create_tool_response_objects(tools, server.mcp_info)

    async def _resolve_allowed_mcp_servers_for_tool_call(
        user_api_key_dict: UserAPIKeyAuth,
        server_id: str,
    ) -> List[MCPServer]:
        """Resolve allowed MCP servers for the given user and validate server_id access."""
        auth_contexts = await build_effective_auth_contexts(user_api_key_dict)
        allowed_server_ids_set = set()
        for auth_context in auth_contexts:
            servers = await global_mcp_server_manager.get_allowed_mcp_servers(
                user_api_key_auth=auth_context
            )
            allowed_server_ids_set.update(servers)
        if server_id not in allowed_server_ids_set:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "access_denied",
                    "message": f"The key is not allowed to access server {server_id}",
                },
            )
        allowed_mcp_servers: List[MCPServer] = []
        for allowed_server_id in allowed_server_ids_set:
            server = global_mcp_server_manager.get_mcp_server_by_id(allowed_server_id)
            if server is not None:
                allowed_mcp_servers.append(server)
        return allowed_mcp_servers

    ########################################################
    @router.get("/tools/list", dependencies=[Depends(user_api_key_auth)])
    async def list_tool_rest_api(
        request: Request,
        server_id: Optional[str] = Query(
            None, description="The server id to list tools for"
        ),
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> dict:
        """
        List all available tools with information about the server they belong to.

        Example response:
        {
            "tools": [
                {
                    "name": "create_zap",
                    "description": "Create a new zap",
                    "inputSchema": "tool_input_schema",
                    "mcp_info": {
                        "server_name": "zapier",
                        "logo_url": "https://www.zapier.com/logo.png",
                    }
                }
            ],
            "error": null,
            "message": "Successfully retrieved tools"
        }
        """
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        try:
            # Extract auth headers from request
            headers = request.headers
            raw_headers_from_request = dict(headers)
            mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(
                headers
            )
            mcp_server_auth_headers = (
                MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
            )

            auth_contexts = await build_effective_auth_contexts(user_api_key_dict)

            _rest_client_ip = IPAddressUtils.get_mcp_client_ip(request)

            allowed_server_ids_set = set()
            for auth_context in auth_contexts:
                servers = await global_mcp_server_manager.get_allowed_mcp_servers(
                    user_api_key_auth=auth_context,
                )
                allowed_server_ids_set.update(servers)

            allowed_server_ids = global_mcp_server_manager.filter_server_ids_by_ip(
                list(allowed_server_ids_set), _rest_client_ip
            )

            list_tools_result = []
            error_message = None

            # If server_id is specified, only query that specific server
            if server_id:
                if server_id not in allowed_server_ids:
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "access_denied",
                            "message": f"The key is not allowed to access server {server_id}",
                        },
                    )
                server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
                if server is None:
                    return {
                        "tools": [],
                        "error": "server_not_found",
                        "message": f"Server with id {server_id} not found",
                    }

                server_auth_header = _get_server_auth_header(
                    server, mcp_server_auth_headers, mcp_auth_header
                )

                try:
                    list_tools_result = await _get_tools_for_single_server(
                        server,
                        server_auth_header,
                        raw_headers_from_request,
                        user_api_key_dict,
                    )
                except Exception as e:
                    verbose_logger.exception(
                        f"Error getting tools from {server.name}: {e}"
                    )
                    return {
                        "tools": [],
                        "error": "server_error",
                        "message": f"Failed to get tools from server {server.name}: {str(e)}",
                    }
            else:
                if not allowed_server_ids:
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "access_denied",
                            "message": "The key is not allowed to access any MCP servers.",
                        },
                    )

                # Query all servers the user has access to
                errors = []
                for allowed_server_id in allowed_server_ids:
                    server = global_mcp_server_manager.get_mcp_server_by_id(
                        allowed_server_id
                    )
                    if server is None:
                        continue

                    server_auth_header = _get_server_auth_header(
                        server, mcp_server_auth_headers, mcp_auth_header
                    )

                    try:
                        tools_result = await _get_tools_for_single_server(
                            server,
                            server_auth_header,
                            raw_headers_from_request,
                            user_api_key_dict,
                        )
                        list_tools_result.extend(tools_result)
                    except Exception as e:
                        verbose_logger.exception(
                            f"Error getting tools from {server.name}: {e}"
                        )
                        errors.append(f"{server.name}: {str(e)}")
                        continue

                if errors and not list_tools_result:
                    error_message = "Failed to get tools from servers: " + "; ".join(
                        errors
                    )

            return {
                "tools": list_tools_result,
                "error": "partial_failure" if error_message else None,
                "message": (
                    error_message if error_message else "Successfully retrieved tools"
                ),
            }

        except Exception as e:
            verbose_logger.exception(
                "Unexpected error in list_tool_rest_api: %s", str(e)
            )
            return {
                "tools": [],
                "error": "unexpected_error",
                "message": f"An unexpected error occurred: {str(e)}",
            }

    @router.post("/tools/call", dependencies=[Depends(user_api_key_auth)])
    async def call_tool_rest_api(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        REST API to call a specific MCP tool with the provided arguments
        """
        from fastapi import HTTPException

        from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )
        from litellm.proxy.proxy_server import (
            general_settings,
            proxy_config,
            proxy_logging_obj,
        )

        try:
            data = await request.json()

            # Validate required parameters early
            server_id = data.get("server_id")
            if not server_id:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_parameter",
                        "message": "server_id is required in request body",
                    },
                )

            tool_name = data.get("name")
            if not tool_name:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_parameter",
                        "message": "name is required in request body",
                    },
                )

            tool_arguments = data.get("arguments")

            proxy_base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
            data, logging_obj = (
                await proxy_base_llm_response_processor.common_processing_pre_call_logic(
                    request=request,
                    user_api_key_dict=user_api_key_dict,
                    proxy_config=proxy_config,
                    route_type=CallTypes.call_mcp_tool.value,
                    proxy_logging_obj=proxy_logging_obj,
                    general_settings=general_settings,
                )
            )

            # Extract MCP auth headers from request and add to data dict
            mcp_auth_header, mcp_server_auth_headers, raw_headers_from_request = (
                _extract_mcp_headers_from_request(request, MCPRequestHandler)
            )
            if mcp_auth_header:
                data["mcp_auth_header"] = mcp_auth_header
            if mcp_server_auth_headers:
                data["mcp_server_auth_headers"] = mcp_server_auth_headers
            data["raw_headers"] = raw_headers_from_request

            # Extract user_api_key_auth from metadata and add to top level
            # call_mcp_tool expects user_api_key_auth as a top-level parameter
            if "metadata" in data and "user_api_key_auth" in data["metadata"]:
                data["user_api_key_auth"] = data["metadata"]["user_api_key_auth"]

            # Resolve allowed MCP servers with IP filtering
            allowed_mcp_servers = await _resolve_allowed_mcp_servers_with_ip_filter(
                request, user_api_key_dict, server_id
            )

            # Call execute_mcp_tool directly (permission checks already done)
            result = await execute_mcp_tool(
                name=tool_name,
                arguments=tool_arguments,
                allowed_mcp_servers=allowed_mcp_servers,
                start_time=datetime.now(),
                user_api_key_auth=data.get("user_api_key_auth"),
                mcp_auth_header=data.get("mcp_auth_header"),
                mcp_server_auth_headers=data.get("mcp_server_auth_headers"),
                oauth2_headers=data.get("oauth2_headers"),
                raw_headers=data.get("raw_headers"),
                litellm_logging_obj=data.get("litellm_logging_obj"),
            )
            return result
        except BlockedPiiEntityError as e:
            verbose_logger.error(f"BlockedPiiEntityError in MCP tool call: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "blocked_pii_entity",
                    "message": str(e),
                    "entity_type": getattr(e, "entity_type", None),
                    "guardrail_name": getattr(e, "guardrail_name", None),
                },
            )
        except GuardrailRaisedException as e:
            verbose_logger.error(f"GuardrailRaisedException in MCP tool call: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "guardrail_violation",
                    "message": str(e),
                    "guardrail_name": getattr(e, "guardrail_name", None),
                },
            )
        except HTTPException as e:
            # Re-raise HTTPException as-is to preserve status code and detail
            verbose_logger.error(f"HTTPException in MCP tool call: {str(e)}")
            raise e
        except Exception as e:
            verbose_logger.exception(f"Unexpected error in MCP tool call: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_server_error",
                    "message": f"An unexpected error occurred: {str(e)}",
                },
            )

    ########################################################
    # MCP Connection testing routes
    # /health -> Test if we can connect to the MCP server
    # /health/tools/list -> List tools from MCP server
    # For these routes users will dynamically pass the MCP connection params, they don't need to be on the MCP registry
    ########################################################
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        NewMCPServerRequest,
    )

    def _extract_credentials(
        request: NewMCPServerRequest,
    ) -> tuple:
        """
        Extract OAuth credentials from the nested ``request.credentials`` dict.

        Returns:
            (client_id, client_secret, scopes) — any value may be ``None``.
        """
        creds = request.credentials if isinstance(request.credentials, dict) else {}
        client_id: Optional[str] = creds.get("client_id")
        client_secret: Optional[str] = creds.get("client_secret")
        scopes_raw = creds.get("scopes")
        scopes: Optional[List[str]] = scopes_raw if isinstance(scopes_raw, list) else None
        return client_id, client_secret, scopes

    async def _execute_with_mcp_client(
        request: NewMCPServerRequest,
        operation: Callable[..., Awaitable[Any]],
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Create a temporary MCP client from *request*, run *operation*, and return the result.

        For M2M OAuth servers (those with ``client_id``, ``client_secret``, and
        ``token_url``), the incoming ``oauth2_headers`` are dropped so that
        ``resolve_mcp_auth`` can auto-fetch a token via ``client_credentials``.

        Args:
            request: MCP server configuration submitted by the UI.
            operation: Async callable that receives the created client and returns a result dict.
            mcp_auth_header: Pre-resolved credential header (API-key / bearer token).
            oauth2_headers: Headers extracted from the incoming request (may contain the
                litellm API key — must NOT be forwarded for M2M servers).
            raw_headers: Raw request headers forwarded for stdio env construction.

        Returns:
            The dict returned by *operation*, or an error dict on failure.
        """
        try:
            client_id, client_secret, scopes = _extract_credentials(request)

            server_model = MCPServer(
                server_id=request.server_id or "",
                name=request.alias or request.server_name or "",
                url=request.url,
                transport=request.transport,
                auth_type=request.auth_type,
                mcp_info=request.mcp_info,
                command=request.command,
                args=request.args,
                env=request.env,
                static_headers=request.static_headers,
                client_id=client_id,
                client_secret=client_secret,
                token_url=request.token_url,
                scopes=scopes,
                authorization_url=request.authorization_url,
                registration_url=request.registration_url,
            )

            stdio_env = global_mcp_server_manager._build_stdio_env(
                server_model, raw_headers
            )

            # For M2M OAuth servers, drop the incoming Authorization header so that
            # resolve_mcp_auth can auto-fetch a token via client_credentials.
            effective_oauth2_headers = (
                None if server_model.has_client_credentials else oauth2_headers
            )

            merged_headers = merge_mcp_headers(
                extra_headers=effective_oauth2_headers,
                static_headers=request.static_headers,
            )

            client = await global_mcp_server_manager._create_mcp_client(
                server=server_model,
                mcp_auth_header=mcp_auth_header,
                extra_headers=merged_headers,
                stdio_env=stdio_env,
            )

            return await operation(client)

        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            verbose_logger.error("Error in MCP operation: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": True,
                "message": "Failed to connect to MCP server. Check proxy logs for details.",
            }

    @router.post("/test/connection", dependencies=[Depends(user_api_key_auth)])
    async def test_connection(
        request: Request,
        new_mcp_server_request: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Test if we can connect to the provided MCP server before adding it
        """

        async def _test_connection_operation(client):
            async def _noop(session):
                return "ok"

            await client.run_with_session(_noop)
            return {"status": "ok"}

        return await _execute_with_mcp_client(
            new_mcp_server_request,
            _test_connection_operation,
            raw_headers=dict(request.headers),
        )

    @router.post("/test/tools/list")
    async def test_tools_list(
        request: Request,
        new_mcp_server_request: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Preview tools available from MCP server before adding it
        """
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        headers = request.headers

        mcp_auth_header: Optional[str] = None
        if new_mcp_server_request.auth_type in {
            MCPAuth.api_key,
            MCPAuth.bearer_token,
            MCPAuth.basic,
            MCPAuth.authorization,
        }:
            credentials = getattr(new_mcp_server_request, "credentials", None)
            if isinstance(credentials, dict):
                mcp_auth_header = credentials.get("auth_value")

        oauth2_headers: Optional[Dict[str, str]] = None
        if new_mcp_server_request.auth_type == MCPAuth.oauth2:
            oauth2_headers = MCPRequestHandler._get_oauth2_headers_from_headers(headers)

        async def _list_tools_operation(client):
            async def _list_tools_session_operation(session):
                return await session.list_tools()

            list_tools_response = await client.run_with_session(
                _list_tools_session_operation
            )
            list_tools_result: List[MCPTool] = list_tools_response.tools
            model_dumped_tools: List[dict] = [
                tool.model_dump() for tool in list_tools_result
            ]
            return {
                "tools": model_dumped_tools,
                "error": None,
                "message": "Successfully retrieved tools",
            }

        return await _execute_with_mcp_client(
            new_mcp_server_request,
            _list_tools_operation,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=dict(request.headers),
        )
