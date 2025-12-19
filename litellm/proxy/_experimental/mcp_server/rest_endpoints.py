import importlib
import traceback
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Query, Request

from litellm._logging import verbose_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.mcp import MCPAuth

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
    from litellm.experimental_mcp_client.client import MCPTool
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.server import (
        ListMCPToolsRestAPIResponseObject,
        call_mcp_tool,
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

    async def _get_tools_for_single_server(server, server_auth_header):
        """Helper function to get tools for a single server."""
        tools = await global_mcp_server_manager._get_tools_from_server(
            server=server,
            mcp_auth_header=server_auth_header,
            add_prefix=False,
        )

        # Filter tools based on allowed_tools configuration
        # Only filter if allowed_tools is explicitly configured (not None and not empty)
        if server.allowed_tools is not None and len(server.allowed_tools) > 0:
            tools = filter_tools_by_allowed_tools(tools, server)

        return _create_tool_response_objects(tools, server.mcp_info)

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
            mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(
                headers
            )
            mcp_server_auth_headers = (
                MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
            )

            list_tools_result = []
            error_message = None

            # If server_id is specified, only query that specific server
            if server_id:
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
                        server, server_auth_header
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
                # Query all servers
                errors = []
                for server in global_mcp_server_manager.get_registry().values():
                    server_auth_header = _get_server_auth_header(
                        server, mcp_server_auth_headers, mcp_auth_header
                    )

                    try:
                        tools_result = await _get_tools_for_single_server(
                            server, server_auth_header
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
        from litellm.proxy.proxy_server import add_litellm_data_to_request, proxy_config
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        try:
            data = await request.json()
            data = await add_litellm_data_to_request(
                data=data,
                request=request,
                user_api_key_dict=user_api_key_dict,
                proxy_config=proxy_config,
            )

            # FIX: Extract MCP auth headers from request
            # The UI sends bearer token in x-mcp-auth header and server-specific headers,
            # but they weren't being extracted and passed to call_mcp_tool.
            # This fix ensures auth headers are properly extracted from the HTTP request
            # and passed through to the MCP server for authentication.
            mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(
                request.headers
            )
            mcp_server_auth_headers = (
                MCPRequestHandler._get_mcp_server_auth_headers_from_headers(
                    request.headers
                )
            )

            # Add extracted headers to data dict to pass to call_mcp_tool
            if mcp_auth_header:
                data["mcp_auth_header"] = mcp_auth_header
            if mcp_server_auth_headers:
                data["mcp_server_auth_headers"] = mcp_server_auth_headers

            result = await call_mcp_tool(**data)
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
    from litellm.proxy._experimental.mcp_server.server import MCPServer
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        NewMCPServerRequest,
    )

    async def _execute_with_mcp_client(
        request: NewMCPServerRequest,
        operation,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Common helper to create MCP client, execute operation, and ensure proper cleanup.

        Args:
            request: MCP server configuration
            operation: Async function that takes a client and returns the operation result

        Returns:
            Operation result or error response
        """
        try:
            client = global_mcp_server_manager._create_mcp_client(
                server=MCPServer(
                    server_id=request.server_id or "",
                    name=request.alias or request.server_name or "",
                    url=request.url,
                    transport=request.transport,
                    auth_type=request.auth_type,
                    mcp_info=request.mcp_info,
                ),
                mcp_auth_header=mcp_auth_header,
                extra_headers=oauth2_headers,
            )

            return await operation(client)

        except Exception as e:
            verbose_logger.error(f"Error in MCP operation: {e}", exc_info=True)
            stack_trace = traceback.format_exc()
            return {
                "status": "error",
                "message": f"An internal error has occurred: {str(e)}",
                "stack_trace": stack_trace,
            }

    @router.post("/test/connection")
    async def test_connection(
        request: NewMCPServerRequest,
    ):
        """
        Test if we can connect to the provided MCP server before adding it
        """

        async def _test_connection_operation(client):
            async def _noop(session):
                return "ok"

            await client.run_with_session(_noop)
            return {"status": "ok"}

        return await _execute_with_mcp_client(request, _test_connection_operation)

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
        )
