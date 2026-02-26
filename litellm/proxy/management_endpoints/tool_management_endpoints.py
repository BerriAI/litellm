"""
TOOL POLICY MANAGEMENT

All /tool management endpoints

GET  /v1/tool/list              - List all discovered tools and their policies
GET  /v1/tool/{tool_name}       - Get a single tool's details
POST /v1/tool/policy            - Update the call_policy for a tool
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.tool_management import (
    LiteLLM_ToolTableRow,
    ToolCallPolicy,
    ToolDetailResponse,
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
    "/v1/tool/{tool_name:path}/detail",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ToolDetailResponse,
)
async def get_tool_detail(
    tool_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a single tool with its policy overrides (for UI detail view).

    Parameters:
    - tool_name: The tool name (supports namespaced names with slashes)
    """
    from litellm.proxy.db.tool_registry_writer import get_tool as db_get_tool
    from litellm.proxy.db.tool_registry_writer import list_overrides_for_tool
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
        overrides = await list_overrides_for_tool(
            prisma_client=prisma_client, tool_name=tool_name
        )
        return ToolDetailResponse(tool=tool, overrides=overrides)
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting tool detail: %s", e)
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
    Set the call policy for a tool (global) or for a specific team/key (override).

    Parameters:
    - tool_name: str - The tool to update
    - call_policy: "trusted" | "untrusted" | "dual_llm" | "blocked"
    - team_id: optional - if set, create/update override for this team only
    - key_hash: optional - if set, create/update override for this key only
    - key_alias: optional - human-readable key alias for UI

    If both team_id and key_hash are omitted, updates the global tool policy.
    Setting a tool to "blocked" will cause the ToolPolicyGuardrail to reject
    that tool_call for the relevant scope.
    """
    from litellm.proxy.db.tool_registry_writer import (
        update_tool_policy as db_update_tool_policy,
    )
    from litellm.proxy.db.tool_registry_writer import upsert_tool_policy_override
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        if data.team_id is not None or data.key_hash is not None:
            override = await upsert_tool_policy_override(
                prisma_client=prisma_client,
                tool_name=data.tool_name,
                call_policy=data.call_policy,
                team_id=data.team_id,
                key_hash=data.key_hash,
                key_alias=data.key_alias,
                updated_by=user_api_key_dict.user_id,
            )
            if override is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to update policy override for tool '{data.tool_name}'",
                )
            return ToolPolicyUpdateResponse(
                tool_name=override.tool_name,
                call_policy=override.call_policy,
                updated=True,
                team_id=override.team_id,
                key_hash=override.key_hash,
            )
        updated = await db_update_tool_policy(
            prisma_client=prisma_client,
            tool_name=data.tool_name,
            call_policy=data.call_policy,
            updated_by=user_api_key_dict.user_id,
        )
        if updated is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update policy for tool '{data.tool_name}'",
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


@router.delete(
    "/v1/tool/{tool_name:path}/overrides",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_tool_policy_override(
    tool_name: str,
    team_id: Optional[str] = Query(None, description="Team ID of the override to remove"),
    key_hash: Optional[str] = Query(None, description="Key hash of the override to remove"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Remove a policy override for a tool. Specify the override by team_id and/or key_hash
    (must match the override that was created; use empty string for unscoped dimension).
    """
    from litellm.proxy.db.tool_registry_writer import delete_tool_policy_override
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )
    if team_id is None and key_hash is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of team_id or key_hash is required to identify the override",
        )
    try:
        deleted = await delete_tool_policy_override(
            prisma_client=prisma_client,
            tool_name=tool_name,
            team_id=team_id,
            key_hash=key_hash,
        )
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"No override found for tool '{tool_name}' with the given scope",
            )
        return {"deleted": True, "tool_name": tool_name}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting tool policy override: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
