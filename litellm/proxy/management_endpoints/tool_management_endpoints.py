"""
TOOL POLICY MANAGEMENT

All /tool management endpoints

GET  /v1/tool/list              - List all discovered tools and their policies
GET  /v1/tool/policy/options    - List available input/output policy options with descriptions
GET  /v1/tool/{tool_name}       - Get a single tool's details
POST /v1/tool/policy            - Update the input_policy / output_policy for a tool
"""

import uuid
from datetime import datetime, timedelta, timezone
from itertools import groupby
from typing import TYPE_CHECKING, Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, TypeAdapter

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.object_permission_repository import ObjectPermissionRepository
from litellm.repositories.table_repositories import (
    SpendLogsRepository,
    SpendLogToolIndexRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
from litellm.types.tool_management import (
    LiteLLM_ToolTableRow,
    ToolDetailResponse,
    ToolInputPolicy,
    ToolListResponse,
    ToolPolicyOption,
    ToolPolicyOptionsResponse,
    ToolPolicyUpdateRequest,
    ToolPolicyUpdateResponse,
    ToolSpendDailyEntry,
    ToolSpendEntry,
    ToolSpendResponse,
    ToolUsageLogEntry,
    ToolUsageLogsResponse,
)

router = APIRouter()

TOOL_POLICY_OPTIONS = ToolPolicyOptionsResponse(
    input_policies=[
        ToolPolicyOption(
            value="untrusted",
            label="Untrusted",
            description="Tool accepts any input, including data from untrusted tool outputs. Default for newly discovered tools.",
        ),
        ToolPolicyOption(
            value="trusted",
            label="Trusted",
            description="Tool requires trusted input. Blocked if the conversation contains output from any tool with output_policy=untrusted.",
        ),
        ToolPolicyOption(
            value="blocked",
            label="Blocked",
            description="Tool is completely prohibited. Any attempt to call it is rejected.",
        ),
    ],
    output_policies=[
        ToolPolicyOption(
            value="untrusted",
            label="Untrusted",
            description="Tool output may contain unsafe content (prompt injection, risky code). Downstream tools with input_policy=trusted will be blocked.",
        ),
        ToolPolicyOption(
            value="trusted",
            label="Trusted",
            description="Tool output is verified safe. Will not trigger trust-chain blocks on downstream tools.",
        ),
    ],
)


@router.get(
    "/v1/tool/policy/options",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ToolPolicyOptionsResponse,
)
async def get_tool_policy_options(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Return the available input and output policy options with descriptions.
    Static data — no DB call.
    """
    return TOOL_POLICY_OPTIONS


@router.get(
    "/v1/tool/list",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ToolListResponse,
)
async def list_tools(
    input_policy: Optional[ToolInputPolicy] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all auto-discovered tools and their policies.

    Parameters:
    - input_policy: Optional filter — one of "trusted", "untrusted", "blocked"
    """
    from litellm.proxy.db.tool_registry_writer import list_tools as db_list_tools
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    try:
        tools = await db_list_tools(prisma_client=prisma_client, input_policy=input_policy)
        return ToolListResponse(tools=tools, total=len(tools))
    except Exception as e:
        verbose_proxy_logger.exception("Error listing tools: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _parse_day_start(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {value}. Expected: 'YYYY-MM-DD'",
        )


class _ToolSpendRow(BaseModel):
    date: str
    tool_name: str
    call_count: int
    spend: float
    total_tokens: int


class _RequestTotalRow(BaseModel):
    total_spend: float


_TOOL_SPEND_ROWS = TypeAdapter(list[_ToolSpendRow])
_REQUEST_TOTAL_ROWS = TypeAdapter(list[_RequestTotalRow])


def _summarize_tool(name: str, grp: tuple[_ToolSpendRow, ...]) -> ToolSpendEntry:
    return ToolSpendEntry(
        tool_name=name,
        spend=sum(r.spend for r in grp),
        call_count=sum(r.call_count for r in grp),
        total_tokens=sum(r.total_tokens for r in grp),
    )


def _build_tool_spend_response(
    rows: list[_ToolSpendRow],
    total_spend: float,
    start_date: str,
    end_date: str,
) -> ToolSpendResponse:
    daily = [
        ToolSpendDailyEntry(date=r.date, tool_name=r.tool_name, spend=r.spend, call_count=r.call_count) for r in rows
    ]
    grouped = groupby(sorted(rows, key=lambda r: r.tool_name), key=lambda r: r.tool_name)
    by_tool = sorted(
        (_summarize_tool(name, tuple(grp)) for name, grp in grouped),
        key=lambda e: e.spend,
        reverse=True,
    )
    return ToolSpendResponse(
        by_tool=by_tool,
        daily=daily,
        total_spend=total_spend,
        start_date=start_date,
        end_date=end_date,
    )


@router.get(
    "/v1/tool/spend",
    tags=["tool management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ToolSpendResponse,
)
async def get_tool_spend(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str | None, Query(description="YYYY-MM-DD (defaults to 30 days ago)")] = None,
    end_date: Annotated[str | None, Query(description="YYYY-MM-DD (defaults to today)")] = None,
):
    """
    Spend attributed to each tool over a date range, for the Cost Optimization dashboard.

    Joins ``LiteLLM_SpendLogToolIndex`` (which tool names ran on which request) to
    ``LiteLLM_SpendLogs`` (what the request cost). A request that used multiple tools
    counts its full spend toward each of those tools, so per-tool numbers are
    attributions. ``total_spend`` is the deduplicated spend of every request that
    called at least one tool in the window, so it never double counts.
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Only proxy admin roles can view tool spend across the deployment",
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    now = datetime.now(timezone.utc)
    end_day = _parse_day_start(end_date)
    start_dt = _parse_day_start(start_date) or ((end_day or now) - timedelta(days=30))
    end_exclusive = (end_day + timedelta(days=1)) if end_day else now

    rows = await prisma_client.db.query_raw(
        """
        SELECT to_char(ti.start_time, 'YYYY-MM-DD') AS date,
               ti.tool_name AS tool_name,
               COUNT(*)::int AS call_count,
               COALESCE(SUM(sl.spend), 0)::double precision AS spend,
               COALESCE(SUM(sl.total_tokens), 0)::bigint AS total_tokens
        FROM "LiteLLM_SpendLogToolIndex" ti
        JOIN "LiteLLM_SpendLogs" sl ON sl.request_id = ti.request_id
        WHERE ti.start_time >= ($1::timestamptz AT TIME ZONE 'UTC')
          AND ti.start_time < ($2::timestamptz AT TIME ZONE 'UTC')
        GROUP BY date, ti.tool_name
        ORDER BY date ASC, spend DESC
        """,
        start_dt.isoformat(),
        end_exclusive.isoformat(),
    )
    totals = await prisma_client.db.query_raw(
        """
        SELECT COALESCE(SUM(sl.spend), 0)::double precision AS total_spend
        FROM "LiteLLM_SpendLogs" sl
        WHERE EXISTS (
            SELECT 1
            FROM "LiteLLM_SpendLogToolIndex" ti
            WHERE ti.request_id = sl.request_id
              AND ti.start_time >= ($1::timestamptz AT TIME ZONE 'UTC')
              AND ti.start_time < ($2::timestamptz AT TIME ZONE 'UTC')
        )
        """,
        start_dt.isoformat(),
        end_exclusive.isoformat(),
    )
    total_rows = _REQUEST_TOTAL_ROWS.validate_python(totals or [])
    return _build_tool_spend_response(
        rows=_TOOL_SPEND_ROWS.validate_python(rows or []),
        total_spend=total_rows[0].total_spend if total_rows else 0.0,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=(end_day or now).strftime("%Y-%m-%d"),
    )


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
    """
    from litellm.proxy.db.tool_registry_writer import get_tool as db_get_tool
    from litellm.proxy.db.tool_registry_writer import list_overrides_for_tool
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    try:
        tool = await db_get_tool(prisma_client=prisma_client, tool_name=tool_name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        overrides = await list_overrides_for_tool(prisma_client=prisma_client, tool_name=tool_name)
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
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    try:
        where: dict = {"tool_name": tool_name}
        if start_date or end_date:
            start_time_filter: Optional[datetime] = None
            end_time_filter: Optional[datetime] = None
            if start_date:
                try:
                    start_time_filter = datetime.strptime(start_date + "T00:00:00", "%Y-%m-%dT%H:%M:%S").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    pass
            if end_date:
                try:
                    end_time_filter = datetime.strptime(end_date + "T23:59:59", "%Y-%m-%dT%H:%M:%S").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    pass
            if start_time_filter is not None or end_time_filter is not None:
                where["start_time"] = {}
                if start_time_filter is not None:
                    where["start_time"]["gte"] = start_time_filter
                if end_time_filter is not None:
                    where["start_time"]["lte"] = end_time_filter

        total = await SpendLogToolIndexRepository(prisma_client).table.count(where=where)
        index_rows = await SpendLogToolIndexRepository(prisma_client).table.find_many(
            where=where,
            order={"start_time": "desc"},
            skip=(page - 1) * page_size,
            take=page_size,
        )
        request_ids = [r.request_id for r in index_rows]
        if not request_ids:
            return ToolUsageLogsResponse(logs=[], total=total, page=page, page_size=page_size)

        spend_logs = await SpendLogsRepository(prisma_client).table.find_many(where={"request_id": {"in": request_ids}})
        log_by_id = {s.request_id: s for s in spend_logs}

        logs_out: List[ToolUsageLogEntry] = []
        for r in index_rows:
            sl = log_by_id.get(r.request_id)
            if not sl:
                continue
            ts = sl.startTime.isoformat() if hasattr(sl.startTime, "isoformat") else str(sl.startTime)
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

        return ToolUsageLogsResponse(logs=logs_out, total=total, page=page, page_size=page_size)
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
    """
    from litellm.proxy.db.tool_registry_writer import get_tool as db_get_tool
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

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
    row = await VerificationTokenRepository(prisma_client).table.find_unique(where={"token": hashed})
    if row is None:
        return None
    op_id = getattr(row, "object_permission_id", None)
    if op_id:
        return op_id
    new_id = str(uuid.uuid4())
    await ObjectPermissionRepository(prisma_client).table.create(
        data={"object_permission_id": new_id, "blocked_tools": []}
    )
    updated_count = await VerificationTokenRepository(prisma_client).table.update_many(
        where={"token": hashed, "object_permission_id": None},
        data={"object_permission_id": new_id},
    )
    if updated_count == 0:
        await ObjectPermissionRepository(prisma_client).table.delete(where={"object_permission_id": new_id})
        row = await VerificationTokenRepository(prisma_client).table.find_unique(where={"token": hashed})
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
    row = await TeamRepository(prisma_client).table.find_unique(
        where={"team_id": team_id_clean},
        select={"object_permission_id": True},
    )
    if row is None:
        return None
    op_id = getattr(row, "object_permission_id", None)
    if op_id:
        return op_id
    new_id = str(uuid.uuid4())
    await ObjectPermissionRepository(prisma_client).table.create(
        data={"object_permission_id": new_id, "blocked_tools": []}
    )
    updated_count = await TeamRepository(prisma_client).table.update_many(
        where={"team_id": team_id_clean, "object_permission_id": None},
        data={"object_permission_id": new_id},
    )
    if updated_count == 0:
        await ObjectPermissionRepository(prisma_client).table.delete(where={"object_permission_id": new_id})
        row = await TeamRepository(prisma_client).table.find_unique(
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
    Set the input_policy and/or output_policy for a tool (global), or block for a specific team/key (override).

    Parameters:
    - tool_name: str - The tool to update
    - input_policy: optional - "trusted" | "untrusted" | "blocked"
    - output_policy: optional - "trusted" | "untrusted"
    - team_id: optional - if set, create/update override for this team only
    - key_hash: optional - if set, create/update override for this key only
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
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    try:
        if data.team_id is not None or data.key_hash is not None:
            if data.team_id is not None and data.key_hash is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Provide either team_id or key_hash, not both",
                )
            if data.key_hash is not None:
                op_id = await _resolve_key_hash_to_object_permission_id(prisma_client, data.key_hash)
            else:
                op_id = await _resolve_team_id_to_object_permission_id(prisma_client, data.team_id or "")
            if op_id is None:
                raise HTTPException(
                    status_code=404,
                    detail="Key or team not found for the given identifier",
                )
            is_blocking = data.input_policy == "blocked"
            if is_blocking:
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
                input_policy=data.input_policy,
                output_policy=data.output_policy,
                updated=True,
                team_id=data.team_id,
                key_hash=data.key_hash,
            )

        if data.input_policy is None and data.output_policy is None:
            raise HTTPException(
                status_code=400,
                detail="At least one of input_policy or output_policy must be provided",
            )

        updated = await db_update_tool_policy(
            prisma_client=prisma_client,
            tool_name=data.tool_name,
            updated_by=user_api_key_dict.user_id,
            input_policy=data.input_policy,
            output_policy=data.output_policy,
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
            input_policy=updated.input_policy,
            output_policy=updated.output_policy,
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
    Remove a policy override for a tool. Specify the override by team_id or key_hash
    (exactly one required).
    """
    from litellm.proxy.db.tool_registry_writer import (
        get_tool_policy_registry,
        remove_tool_from_object_permission_blocked,
    )
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
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
            op_id = await _resolve_key_hash_to_object_permission_id(prisma_client, key_hash)
        else:
            op_id = await _resolve_team_id_to_object_permission_id(prisma_client, team_id or "")
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
