import importlib
from typing import Optional, Dict

from fastapi import APIRouter, Depends, Query, Request

from litellm._logging import verbose_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

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
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
        _convert_protocol_version_to_enum,
    )
    from litellm.proxy._experimental.mcp_server.server import (
        ListMCPToolsRestAPIResponseObject,
        call_mcp_tool,
    )

    ########################################################
    ############ MCP Server REST API Routes #################
    def _get_server_auth_header(
        server, mcp_server_auth_headers: Optional[Dict[str, str]], mcp_auth_header: Optional[str]
    ) -> Optional[str]:
        """Helper function to get server-specific auth header with case-insensitive matching."""
        if mcp_server_auth_headers and server.alias:
            normalized_server_alias = server.alias.lower()
            normalized_headers = {k.lower(): v for k, v in mcp_server_auth_headers.items()}
            server_auth = normalized_headers.get(normalized_server_alias)
            if server_auth is not None:
                return server_auth
        elif mcp_server_auth_headers and server.server_name:
            normalized_server_name = server.server_name.lower()
            normalized_headers = {k.lower(): v for k, v in mcp_server_auth_headers.items()}
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

    async def _get_tools_for_single_server(server, server_auth_header, mcp_protocol_version):
        """Helper function to get tools for a single server."""
        tools = await global_mcp_server_manager._get_tools_from_server(
            server=server,
            mcp_auth_header=server_auth_header,
            mcp_protocol_version=mcp_protocol_version,
        )
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
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
        
        try:
            # Extract auth headers from request
            headers = request.headers
            mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(headers)
            mcp_server_auth_headers = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
            mcp_protocol_version = headers.get(MCPRequestHandler.MCP_PROTOCOL_VERSION_HEADER_NAME)
            
            list_tools_result = []
            error_message = None
            
            # If server_id is specified, only query that specific server
            if server_id:
                server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
                if server is None:
                    return {
                        "tools": [],
                        "error": "server_not_found",
                        "message": f"Server with id {server_id} not found"
                    }
                
                server_auth_header = _get_server_auth_header(server, mcp_server_auth_headers, mcp_auth_header)
                
                try:
                    list_tools_result = await _get_tools_for_single_server(server, server_auth_header, mcp_protocol_version)
                except Exception as e:
                    verbose_logger.exception(f"Error getting tools from {server.name}: {e}")
                    return {
                        "tools": [],
                        "error": "server_error",
                        "message": f"Failed to get tools from server {server.name}: {str(e)}"
                    }
            else:
                # Query all servers
                errors = []
                for server in global_mcp_server_manager.get_registry().values():
                    server_auth_header = _get_server_auth_header(server, mcp_server_auth_headers, mcp_auth_header)
                    
                    try:
                        tools_result = await _get_tools_for_single_server(server, server_auth_header, mcp_protocol_version)
                        list_tools_result.extend(tools_result)
                    except Exception as e:
                        verbose_logger.exception(f"Error getting tools from {server.name}: {e}")
                        errors.append(f"{server.name}: {str(e)}")
                        continue
                
                if errors and not list_tools_result:
                    error_message = "Failed to get tools from servers: " + "; ".join(errors)
            
            return {
                "tools": list_tools_result,
                "error": "partial_failure" if error_message else None,
                "message": error_message if error_message else "Successfully retrieved tools"
            }
            
        except Exception as e:
            verbose_logger.exception("Unexpected error in list_tool_rest_api: %s", str(e))
            return {
                "tools": [],
                "error": "unexpected_error",
                "message": f"An unexpected error occurred: {str(e)}"
            }

    @router.post("/tools/call", dependencies=[Depends(user_api_key_auth)])
    async def call_tool_rest_api(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        REST API to call a specific MCP tool with the provided arguments
        """
        from litellm.proxy.proxy_server import add_litellm_data_to_request, proxy_config
        from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
        from fastapi import HTTPException

        try:
            data = await request.json()
            data = await add_litellm_data_to_request(
                data=data,
                request=request,
                user_api_key_dict=user_api_key_dict,
                proxy_config=proxy_config,
            )
            return await call_mcp_tool(**data)
        except BlockedPiiEntityError as e:
            verbose_logger.error(f"BlockedPiiEntityError in MCP tool call: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "blocked_pii_entity",
                    "message": str(e),
                    "entity_type": getattr(e, 'entity_type', None),
                    "guardrail_name": getattr(e, 'guardrail_name', None)
                }
            )
        except GuardrailRaisedException as e:
            verbose_logger.error(f"GuardrailRaisedException in MCP tool call: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "guardrail_violation",
                    "message": str(e),
                    "guardrail_name": getattr(e, 'guardrail_name', None)
                }
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
                    "message": f"An unexpected error occurred: {str(e)}"
                }
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
    @router.post("/test/connection")
    async def test_connection(
        request: NewMCPServerRequest,
    ):
        """
        Test if we can connect to the provided MCP server before adding it
        """
        try:
            client = global_mcp_server_manager._create_mcp_client(
                server=MCPServer(
                    server_id=request.server_id or "",
                    name=request.alias or request.server_name or "",
                    url=request.url,
                    transport=request.transport,
                    spec_version=_convert_protocol_version_to_enum(request.spec_version),
                    auth_type=request.auth_type,
                    mcp_info=request.mcp_info,
                ),
                mcp_auth_header=None,
            )

            await client.connect()
        except Exception as e:
            verbose_logger.error(f"Error in test_connection: {e}", exc_info=True)
            return {"status": "error", "message": "An internal error has occurred."}
        return {"status": "ok"}
        
    
    @router.post("/test/tools/list")
    async def test_tools_list(
        request: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Preview tools available from MCP server before adding it
        """
        try:
            client = global_mcp_server_manager._create_mcp_client(
                server=MCPServer(
                    server_id=request.server_id or "",
                    name=request.alias or request.server_name or "",
                    url=request.url,
                    transport=request.transport,
                    spec_version=_convert_protocol_version_to_enum(request.spec_version),
                    auth_type=request.auth_type,
                    mcp_info=request.mcp_info,
                ),
                mcp_auth_header=None,
            )
            list_tools_result = await client.list_tools()
        except Exception as e:
            verbose_logger.error(f"Error in test_tools_list: {e}", exc_info=True)
            return {"status": "error", "message": "An internal error has occurred."}
        return {
            "tools": list_tools_result,
            "error": None,
            "message": "Successfully retrieved tools"
        }
