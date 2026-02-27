"""
TOOL POLICY MANAGEMENT

All /tool management endpoints

GET  /v1/tool/list              - List all discovered tools and their policies
GET  /v1/tool/{tool_name}       - Get a single tool's details
POST /v1/tool/policy            - Update the call_policy for a tool
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

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
    ToolUsageLogEntry,
    ToolUsageLogsResponse,
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
        tools = await db_list_tools(
            prisma_client=prisma_client, call_policy=call_policy
        )
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
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        overrides = await list_overrides_for_tool(
            prisma_client=prisma_client, tool_name=tool_name
        )
        return ToolDetailResponse(tool=tool, overrides=overrides)
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting tool detail: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _input_snippet_for_tool_log(sl: Any, max_len: int = 200) -> Optional[str]:
    """Short snippet from messages or proxy_server_request for tool usage log row."""
    if sl is None:
        return None
    messages = getattr(sl, "messages", None)
    if messages is not None:
        s = _snippet_str(messages, max_len)
        if s:
            return s
    psr = getattr(sl, "proxy_server_request", None)
    if not psr:
        return None
    if isinstance(psr, str):
        import json

        try:
            psr = json.loads(psr)
        except Exception:
            return _snippet_str(psr, max_len)
    if isinstance(psr, dict):
        msgs = psr.get("messages")
        if msgs is None and isinstance(psr.get("body"), dict):
            msgs = psr["body"].get("messages")
        s = _snippet_str(msgs, max_len)
        if s:
            return s
    return _snippet_str(psr, max_len)


def _snippet_str(text: Any, max_len: int = 200) -> Optional[str]:
    if text is None:
        return None
    if isinstance(text, str):
        s = text
    elif isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict) and "content" in item:
                c = item["content"]
                parts.append(c if isinstance(c, str) else str(c))
            else:
                parts.append(str(item))
        s = " ".join(parts)
    else:
        s = str(text)
    if not s or s == "{}":
        return None
    return (s[:max_len] + "...") if len(s) > max_len else s


@router.get(
    "/v1/tool/{tool_name:path}/logs",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ToolUsageLogsResponse,
)
async def get_tool_usage_logs(
    tool_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Return paginated spend logs for requests that used this tool (from SpendLogToolIndex).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        where: dict = {"tool_name": tool_name}
        if start_date or end_date:
            start_time_filter: Optional[datetime] = None
            end_time_filter: Optional[datetime] = None
            if start_date:
                try:
                    start_time_filter = datetime.strptime(
                        start_date + "T00:00:00", "%Y-%m-%dT%H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            if end_date:
                try:
                    end_time_filter = datetime.strptime(
                        end_date + "T23:59:59", "%Y-%m-%dT%H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            if start_time_filter is not None or end_time_filter is not None:
                where["start_time"] = {}
                if start_time_filter is not None:
                    where["start_time"]["gte"] = start_time_filter
                if end_time_filter is not None:
                    where["start_time"]["lte"] = end_time_filter

        total = await prisma_client.db.litellm_spendlogtoolindex.count(where=where)
        index_rows = await prisma_client.db.litellm_spendlogtoolindex.find_many(
            where=where,
            order={"start_time": "desc"},
            skip=(page - 1) * page_size,
            take=page_size,
        )
        request_ids = [r.request_id for r in index_rows]
        if not request_ids:
            return ToolUsageLogsResponse(
                logs=[], total=total, page=page, page_size=page_size
            )

        spend_logs = await prisma_client.db.litellm_spendlogs.find_many(
            where={"request_id": {"in": request_ids}}
        )
        log_by_id = {s.request_id: s for s in spend_logs}

        logs_out: List[ToolUsageLogEntry] = []
        for r in index_rows:
            sl = log_by_id.get(r.request_id)
            if not sl:
                continue
            ts = (
                sl.startTime.isoformat()
                if hasattr(sl.startTime, "isoformat")
                else str(sl.startTime)
            )
            logs_out.append(
                ToolUsageLogEntry(
                    id=sl.request_id,
                    timestamp=ts,
                    model=getattr(sl, "model", None) or None,
                    spend=getattr(sl, "spend", None),
                    total_tokens=getattr(sl, "total_tokens", None),
                    input_snippet=_input_snippet_for_tool_log(sl),
                )
            )

        return ToolUsageLogsResponse(
            logs=logs_out, total=total, page=page, page_size=page_size
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting tool usage logs: %s", e)
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
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        return tool
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting tool: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _resolve_key_hash_to_object_permission_id(
    prisma_client: "PrismaClient",
    key_hash: str,
) -> Optional[str]:
    """Resolve key (hash or raw) to object_permission_id; create permission if key has none."""
    from litellm.proxy.proxy_server import hash_token

    hashed = key_hash if "sk-" not in (key_hash or "") else hash_token(key_hash)
    if not hashed:
        return None
    row = await prisma_client.db.litellm_verificationtoken.find_unique(
        where={"token": hashed}
    )
    if row is None:
        return None
    op_id = getattr(row, "object_permission_id", None)
    if op_id:
        return op_id
    # Create new object permission and atomically assign to key.
    # Uses update_many with object_permission_id=None to prevent race conditions:
    # only one concurrent request wins; the loser cleans up its orphaned row.
    new_id = str(uuid.uuid4())
    await prisma_client.db.litellm_objectpermissiontable.create(
        data={"object_permission_id": new_id, "blocked_tools": []}
    )
    updated_count = await prisma_client.db.litellm_verificationtoken.update_many(
        where={"token": hashed, "object_permission_id": None},
        data={"object_permission_id": new_id},
    )
    if updated_count == 0:
        # Another request already assigned a permission; clean up orphan
        await prisma_client.db.litellm_objectpermissiontable.delete(
            where={"object_permission_id": new_id}
        )
        row = await prisma_client.db.litellm_verificationtoken.find_unique(
            where={"token": hashed}
        )
        return getattr(row, "object_permission_id", None) if row else None
    return new_id


async def _resolve_team_id_to_object_permission_id(
    prisma_client: "PrismaClient",
    team_id: str,
) -> Optional[str]:
    """Resolve team_id to object_permission_id; create permission if team has none."""
    if not team_id or not team_id.strip():
        return None
    team_id_clean = team_id.strip()
    row = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id_clean},
        select={"object_permission_id": True},
    )
    if row is None:
        return None
    op_id = getattr(row, "object_permission_id", None)
    if op_id:
        return op_id
    # Same atomic pattern as _resolve_key_hash_to_object_permission_id
    new_id = str(uuid.uuid4())
    await prisma_client.db.litellm_objectpermissiontable.create(
        data={"object_permission_id": new_id, "blocked_tools": []}
    )
    updated_count = await prisma_client.db.litellm_teamtable.update_many(
        where={"team_id": team_id_clean, "object_permission_id": None},
        data={"object_permission_id": new_id},
    )
    if updated_count == 0:
        await prisma_client.db.litellm_objectpermissiontable.delete(
            where={"object_permission_id": new_id}
        )
        row = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": team_id_clean},
            select={"object_permission_id": True},
        )
        return getattr(row, "object_permission_id", None) if row else None
    return new_id


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
        add_tool_to_object_permission_blocked,
        get_tool_policy_registry,
        remove_tool_from_object_permission_blocked,
    )
    from litellm.proxy.db.tool_registry_writer import (
        update_tool_policy as db_update_tool_policy,
    )
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        if data.team_id is not None or data.key_hash is not None:
            if data.team_id is not None and data.key_hash is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Provide either team_id or key_hash, not both",
                )
            if data.key_hash is not None:
                op_id = await _resolve_key_hash_to_object_permission_id(
                    prisma_client, data.key_hash
                )
            else:
                op_id = await _resolve_team_id_to_object_permission_id(
                    prisma_client, data.team_id or ""
                )
            if op_id is None:
                raise HTTPException(
                    status_code=404,
                    detail="Key or team not found for the given identifier",
                )
            if data.call_policy == "blocked":
                ok = await add_tool_to_object_permission_blocked(
                    prisma_client=prisma_client,
                    object_permission_id=op_id,
                    tool_name=data.tool_name,
                )
            else:
                ok = await remove_tool_from_object_permission_blocked(
                    prisma_client=prisma_client,
                    object_permission_id=op_id,
                    tool_name=data.tool_name,
                )
            if not ok:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to update policy override for tool '{data.tool_name}'",
                )
            registry = get_tool_policy_registry()
            if registry.is_initialized():
                await registry.sync_tool_policy_from_db(prisma_client)
            return ToolPolicyUpdateResponse(
                tool_name=data.tool_name,
                call_policy=data.call_policy,
                updated=True,
                team_id=data.team_id,
                key_hash=data.key_hash,
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
        registry = get_tool_policy_registry()
        if registry.is_initialized():
            await registry.sync_tool_policy_from_db(prisma_client)
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
    team_id: Optional[str] = Query(
        None, description="Team ID of the override to remove"
    ),
    key_hash: Optional[str] = Query(
        None, description="Key hash of the override to remove"
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Remove a policy override for a tool. Specify the override by team_id or key_hash
    (exactly one required).
    """
    from litellm.proxy.db.tool_registry_writer import (
        get_tool_policy_registry,
        remove_tool_from_object_permission_blocked,
    )
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
    if team_id is not None and key_hash is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either team_id or key_hash, not both",
        )
    try:
        if key_hash is not None:
            op_id = await _resolve_key_hash_to_object_permission_id(
                prisma_client, key_hash
            )
        else:
            op_id = await _resolve_team_id_to_object_permission_id(
                prisma_client, team_id or ""
            )
        if op_id is None:
            raise HTTPException(
                status_code=404,
                detail="Key or team not found for the given identifier",
            )
        deleted = await remove_tool_from_object_permission_blocked(
            prisma_client=prisma_client,
            object_permission_id=op_id,
            tool_name=tool_name,
        )
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"No override found for tool '{tool_name}' with the given scope",
            )
        registry = get_tool_policy_registry()
        if registry.is_initialized():
            await registry.sync_tool_policy_from_db(prisma_client)
        return {"deleted": True, "tool_name": tool_name}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting tool policy override: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
