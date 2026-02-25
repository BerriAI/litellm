"""
TOOL POLICY MANAGEMENT

All /tool management endpoints

GET  /v1/tool/list              - List all discovered tools and their policies
GET  /v1/tool/{tool_name}       - Get a single tool's details
POST /v1/tool/policy            - Update the call_policy for a tool
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.tool_management import (
    LiteLLM_ToolTableRow,
    ToolCallPolicy,
    ToolListResponse,
    ToolPolicyUpdateRequest,
    ToolPolicyUpdateResponse,
)

router = APIRouter()


@router.get(
    "/v1/tool/list",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ToolListResponse,
)
async def list_tools(
    call_policy: Optional[ToolCallPolicy] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all auto-discovered tools and their call policies.

    Parameters:
    - call_policy: Optional filter — one of "trusted", "untrusted", "dual_llm", "blocked"
    """
    from litellm.proxy.db.tool_registry_writer import list_tools as db_list_tools
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        tools = await db_list_tools(prisma_client=prisma_client, call_policy=call_policy)
        return ToolListResponse(tools=tools, total=len(tools))
    except Exception as e:
        verbose_proxy_logger.exception("Error listing tools: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/tool/{tool_name:path}",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ToolTableRow,
)
async def get_tool(
    tool_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get details for a single tool.

    Parameters:
    - tool_name: The tool name (supports namespaced names with slashes)
    """
    from litellm.proxy.db.tool_registry_writer import get_tool as db_get_tool
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        tool = await db_get_tool(prisma_client=prisma_client, tool_name=tool_name)
        if tool is None:
            raise HTTPException(
                status_code=404, detail=f"Tool '{tool_name}' not found"
            )
        return tool
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting tool: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/v1/tool/policy",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ToolPolicyUpdateResponse,
)
async def update_tool_policy(
    data: ToolPolicyUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Set the call policy for a tool.

    Parameters:
    - tool_name: str - The tool to update
    - call_policy: "trusted" | "untrusted" | "dual_llm" | "blocked"

    Setting a tool to "blocked" will cause the ToolPolicyGuardrail to remove
    that tool_call from LLM responses before returning them to the client.
    """
    from litellm.proxy.db.tool_registry_writer import (
        update_tool_policy as db_update_tool_policy,
    )
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        updated = await db_update_tool_policy(
            prisma_client=prisma_client,
            tool_name=data.tool_name,
            call_policy=data.call_policy,
            updated_by=user_api_key_dict.user_id,
        )
        if updated is None:
            raise HTTPException(
                status_code=500, detail=f"Failed to update policy for tool '{data.tool_name}'"
            )
        return ToolPolicyUpdateResponse(
            tool_name=updated.tool_name,
            call_policy=updated.call_policy,
            updated=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error updating tool policy: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
