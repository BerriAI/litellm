"""
TOOL POLICY MANAGEMENT

All /tool management endpoints

GET  /v1/tool/list              - List all discovered tools and their policies
GET  /v1/tool/policy/options    - List available input/output policy options with descriptions
GET  /v1/tool/{tool_name}       - Get a single tool's details
POST /v1/tool/policy            - Update the input_policy / output_policy for a tool
"""

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated, Final, Protocol, TypeAlias, TypeVar, overload

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, TypeAdapter

if TYPE_CHECKING:
    from prisma.models import LiteLLM_DailyToolSpend as PrismaDailyToolSpendRow
    from prisma.models import LiteLLM_ObjectPermissionTable as PrismaObjectPermissionRow
    from prisma.models import LiteLLM_SpendLogs as PrismaSpendLogRow
    from prisma.models import LiteLLM_SpendLogToolIndex as PrismaSpendLogToolIndexRow
    from prisma.models import LiteLLM_TeamTable as PrismaTeamRow
    from prisma.models import LiteLLM_VerificationToken as PrismaVerificationTokenRow

    from litellm.proxy.utils import PrismaClient

from litellm._logging import verbose_proxy_logger
from litellm.constants import TOOL_SPEND_TOP_TOOLS
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.object_permission_repository import ObjectPermissionRepository
from litellm.repositories.table_repositories import (
    DailyToolSpendRepository,
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

_RowT_co: Final = TypeVar("_RowT_co", covariant=True)

if TYPE_CHECKING:

    class _TableOps(Protocol[_RowT_co]):
        async def find_many(
            self,
            where: Mapping[str, object] | None = None,
            order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
            skip: int | None = None,
            take: int | None = None,
        ) -> Sequence[_RowT_co]: ...

        async def find_unique(self, where: Mapping[str, object]) -> _RowT_co | None: ...

        async def count(self, where: Mapping[str, object] | None = None) -> int: ...

        async def create(self, data: Mapping[str, object]) -> _RowT_co: ...

        async def update_many(
            self,
            where: Mapping[str, object],
            data: Mapping[str, object],
        ) -> int: ...

        async def delete(self, where: Mapping[str, object]) -> _RowT_co | None: ...

        async def group_by(
            self,
            by: Sequence[str],
            sum: Mapping[str, bool] | None = None,
            where: Mapping[str, object] | None = None,
            order: Mapping[str, object] | None = None,
            take: int | None = None,
        ) -> Sequence[Mapping[str, object]]: ...

    class _SpendLogRow(Protocol):
        @property
        def messages(self) -> object: ...
        @property
        def proxy_server_request(self) -> str | Mapping[str, object] | None: ...


@overload
def _typed_table(repo: DailyToolSpendRepository) -> "_TableOps[PrismaDailyToolSpendRow]": ...
@overload
def _typed_table(repo: SpendLogToolIndexRepository) -> "_TableOps[PrismaSpendLogToolIndexRow]": ...
@overload
def _typed_table(repo: SpendLogsRepository) -> "_TableOps[PrismaSpendLogRow]": ...
@overload
def _typed_table(repo: VerificationTokenRepository) -> "_TableOps[PrismaVerificationTokenRow]": ...
@overload
def _typed_table(repo: TeamRepository) -> "_TableOps[PrismaTeamRow]": ...
@overload
def _typed_table(repo: ObjectPermissionRepository) -> "_TableOps[PrismaObjectPermissionRow]": ...
def _typed_table(
    repo: DailyToolSpendRepository
    | SpendLogToolIndexRepository
    | SpendLogsRepository
    | VerificationTokenRepository
    | TeamRepository
    | ObjectPermissionRepository,
) -> object:
    return repo.table


router: Final = APIRouter()

TOOL_POLICY_OPTIONS: Final = ToolPolicyOptionsResponse(
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
    input_policy: ToolInputPolicy | None = None,
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
        tools: Final = await db_list_tools(prisma_client=prisma_client, input_policy=input_policy)
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


class _ToolSpendSums(BaseModel):
    spend: float = 0.0
    total_tokens: int = 0
    request_count: int = 0


class _TopToolRow(BaseModel):
    tool_name: str
    sums: _ToolSpendSums = Field(alias="_sum")


_TOP_TOOL_ROWS: Final = TypeAdapter(list[_TopToolRow])


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

    Reads the ``LiteLLM_DailyToolSpend`` rollup, written at request time from invoked
    tools only (MCP tool calls and response tool_calls; declaring a tool without
    invoking it does not count). A request that invoked multiple tools counts its
    full spend toward each of them, so per-tool numbers are attributions and do not
    sum to a deduplicated total.

    ``by_tool`` is the top ``TOOL_SPEND_TOP_TOOLS`` tools by spend, aggregated in
    SQL, and ``daily`` covers only those tools, so the response is bounded by
    days x TOOL_SPEND_TOP_TOOLS regardless of the requested range or how many
    distinct tool names exist.
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

    end_day: Final = _parse_day_start(end_date) or datetime.now(timezone.utc)
    start_day: Final = _parse_day_start(start_date) or end_day - timedelta(days=30)
    start_str: Final = start_day.strftime("%Y-%m-%d")
    end_str: Final = end_day.strftime("%Y-%m-%d")
    date_window: Final = {"date": {"gte": start_str, "lte": end_str}}

    table: Final = _typed_table(DailyToolSpendRepository(prisma_client))
    top_tools: Final = _TOP_TOOL_ROWS.validate_python(
        await table.group_by(
            by=["tool_name"],
            sum={"spend": True, "total_tokens": True, "request_count": True},
            where=date_window,
            order={"_sum": {"spend": "desc"}},
            take=TOOL_SPEND_TOP_TOOLS,
        )
        or []
    )
    by_tool: Final = [
        ToolSpendEntry(
            tool_name=row.tool_name,
            spend=row.sums.spend,
            call_count=row.sums.request_count,
            total_tokens=row.sums.total_tokens,
        )
        for row in top_tools
    ]

    daily_rows: Final[Sequence[PrismaDailyToolSpendRow]] = (
        await table.find_many(
            where={**date_window, "tool_name": {"in": [row.tool_name for row in top_tools]}},
            order=[{"date": "asc"}, {"spend": "desc"}],
        )
        if top_tools
        else []
    )
    daily: Final = [
        ToolSpendDailyEntry(date=row.date, tool_name=row.tool_name, spend=row.spend, call_count=row.request_count)
        for row in daily_rows
    ]
    return ToolSpendResponse(by_tool=by_tool, daily=daily, start_date=start_str, end_date=end_str)


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
        tool: Final = await db_get_tool(prisma_client=prisma_client, tool_name=tool_name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        overrides: Final = await list_overrides_for_tool(prisma_client=prisma_client, tool_name=tool_name)
        return ToolDetailResponse(tool=tool, overrides=overrides)
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting tool detail: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


_ParsedJson: TypeAlias = dict[str, object] | list[object] | str | int | float | bool | None
_PARSED_JSON: Final[TypeAdapter[_ParsedJson]] = TypeAdapter(_ParsedJson)
_STR_OBJECT_DICT: Final = TypeAdapter(dict[str, object])


def _input_snippet_for_tool_log(sl: "_SpendLogRow | None", max_len: int = 200) -> str | None:
    """Short snippet from messages or proxy_server_request for tool usage log row."""
    if sl is None:
        return None
    messages: Final = sl.messages
    if messages is not None:
        s = _snippet_str(messages, max_len)
        if s:
            return s
    psr = sl.proxy_server_request
    if not psr:
        return None
    if isinstance(psr, str):
        import json

        try:
            psr = _PARSED_JSON.validate_python(json.loads(psr))
        except Exception:
            return _snippet_str(psr, max_len)
    if isinstance(psr, dict):
        msgs = psr.get("messages")
        if msgs is None:
            body: Final = psr.get("body")
            if isinstance(body, dict):
                msgs = _STR_OBJECT_DICT.validate_python(body).get("messages")
        s = _snippet_str(msgs, max_len)
        if s:
            return s
    return _snippet_str(psr, max_len)


def _snippet_str(text: object, max_len: int = 200) -> str | None:
    if text is None:
        return None
    if isinstance(text, str):
        s = text
    elif isinstance(text, list):
        parts: Final = []
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
    start_date: str | None = Query(None, description="YYYY-MM-DD"),
    end_date: str | None = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Return paginated spend logs for requests that invoked this tool (from SpendLogToolIndex).
    Declaring a tool in a request body without the model invoking it does not create an entry.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    try:
        where: Final[dict[str, object]] = {"tool_name": tool_name}
        if start_date or end_date:
            start_time_filter: datetime | None = None
            end_time_filter: datetime | None = None
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
                where["start_time"] = {
                    key: value
                    for key, value in (("gte", start_time_filter), ("lte", end_time_filter))
                    if value is not None
                }

        total: Final = await _typed_table(SpendLogToolIndexRepository(prisma_client)).count(where=where)
        index_rows: Final = await _typed_table(SpendLogToolIndexRepository(prisma_client)).find_many(
            where=where,
            order={"start_time": "desc"},
            skip=(page - 1) * page_size,
            take=page_size,
        )
        request_ids: Final = [r.request_id for r in index_rows]
        if not request_ids:
            return ToolUsageLogsResponse(logs=[], total=total, page=page, page_size=page_size)

        spend_logs = await _typed_table(SpendLogsRepository(prisma_client)).find_many(
            where={"request_id": {"in": request_ids}}
        )
        log_by_id: Final = {s.request_id: s for s in spend_logs}

        logs_out: Final[list[ToolUsageLogEntry]] = []
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
        tool: Final = await db_get_tool(prisma_client=prisma_client, tool_name=tool_name)
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
) -> str | None:
    """Resolve key (hash or raw) to object_permission_id; create permission if key has none."""
    from litellm.proxy.proxy_server import hash_token

    hashed: Final = key_hash if "sk-" not in (key_hash or "") else hash_token(key_hash)
    if not hashed:
        return None
    row = await _typed_table(VerificationTokenRepository(prisma_client)).find_unique(where={"token": hashed})
    if row is None:
        return None
    op_id: Final = row.object_permission_id
    if op_id:
        return op_id
    new_id: Final = str(uuid.uuid4())
    await _typed_table(ObjectPermissionRepository(prisma_client)).create(
        data={"object_permission_id": new_id, "blocked_tools": []}
    )
    updated_count: Final = await _typed_table(VerificationTokenRepository(prisma_client)).update_many(
        where={"token": hashed, "object_permission_id": None},
        data={"object_permission_id": new_id},
    )
    if updated_count == 0:
        await _typed_table(ObjectPermissionRepository(prisma_client)).delete(where={"object_permission_id": new_id})
        row = await _typed_table(VerificationTokenRepository(prisma_client)).find_unique(where={"token": hashed})
        return row.object_permission_id if row else None
    return new_id


async def _resolve_team_id_to_object_permission_id(
    prisma_client: "PrismaClient",
    team_id: str,
) -> str | None:
    """Resolve team_id to object_permission_id; create permission if team has none."""
    if not team_id or not team_id.strip():
        return None
    team_id_clean: Final = team_id.strip()
    row = await _typed_table(TeamRepository(prisma_client)).find_unique(where={"team_id": team_id_clean})
    if row is None:
        return None
    op_id: Final = row.object_permission_id
    if op_id:
        return op_id
    new_id: Final = str(uuid.uuid4())
    await _typed_table(ObjectPermissionRepository(prisma_client)).create(
        data={"object_permission_id": new_id, "blocked_tools": []}
    )
    updated_count: Final = await _typed_table(TeamRepository(prisma_client)).update_many(
        where={"team_id": team_id_clean, "object_permission_id": None},
        data={"object_permission_id": new_id},
    )
    if updated_count == 0:
        await _typed_table(ObjectPermissionRepository(prisma_client)).delete(where={"object_permission_id": new_id})
        row = await _typed_table(TeamRepository(prisma_client)).find_unique(where={"team_id": team_id_clean})
        return row.object_permission_id if row else None
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
            is_blocking: Final = data.input_policy == "blocked"
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

        updated: Final = await db_update_tool_policy(
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
    team_id: str | None = Query(None, description="Team ID of the override to remove"),
    key_hash: str | None = Query(None, description="Key hash of the override to remove"),
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
        deleted: Final = await remove_tool_from_object_permission_blocked(
            prisma_client=prisma_client,
            object_permission_id=op_id,
            tool_name=tool_name,
        )
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"No override found for tool '{tool_name}' with the given scope",
            )
        registry: Final = get_tool_policy_registry()
        if registry.is_initialized():
            await registry.sync_tool_policy_from_db(prisma_client)
        return {"deleted": True, "tool_name": tool_name}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting tool policy override: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
