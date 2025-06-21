import importlib
from typing import List, Optional

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
        from litellm.proxy.proxy_server import add_litellm_data_to_request, proxy_config

        data = await request.json()
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
        )
        return await call_mcp_tool(**data)
