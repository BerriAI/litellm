import importlib
from typing import Optional

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
    )
    from litellm.proxy._experimental.mcp_server.server import (
        ListMCPToolsRestAPIResponseObject,
        call_mcp_tool,
    )

    ########################################################
    ############ MCP Server REST API Routes #################
    ########################################################
    @router.get("/tools/list", dependencies=[Depends(user_api_key_auth)])
    async def list_tool_rest_api(
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
        try:
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
                try:
                    tools = await global_mcp_server_manager._get_tools_from_server(
                        server=server,
                    )
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
                    return {
                        "tools": [],
                        "error": "server_error",
                        "message": f"Failed to get tools from server {server.name}: {str(e)}"
                    }
            else:
                # Query all servers
                errors = []
                for server in global_mcp_server_manager.get_registry().values():
                    try:
                        tools = await global_mcp_server_manager._get_tools_from_server(
                            server=server,
                        )
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

        data = await request.json()
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
        )
        return await call_mcp_tool(**data)
