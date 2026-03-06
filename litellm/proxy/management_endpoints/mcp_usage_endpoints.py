"""
MCP server operational visibility endpoints.

- GET /v1/mcp/usage/logs       – paginated request logs per MCP server
- GET /v1/mcp/usage/overview   – per-server request counts and top tools
- GET /v1/mcp/usage/tools      – which users/keys called which tools on a server
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter(prefix="/v1/mcp", tags=["MCP Usage"])


# --- Response models ---


class MCPUsageLogEntry(BaseModel):
    id: str
    timestamp: str
    mcp_server_name: str
    tool_name: Optional[str] = None
    api_key_hash: Optional[str] = None
    api_key_alias: Optional[str] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None
    spend: Optional[float] = None
    input_snippet: Optional[str] = None
    output_snippet: Optional[str] = None


class MCPUsageLogsResponse(BaseModel):
    logs: List[MCPUsageLogEntry]
    total: int
    page: int
    page_size: int


class MCPServerOverviewRow(BaseModel):
    mcp_server_name: str
    server_id: Optional[str] = None
    description: Optional[str] = None
    total_requests: int
    top_tools: List[Dict[str, Any]]
    unique_users: int
    unique_keys: int


class MCPUsageOverviewResponse(BaseModel):
    servers: List[MCPServerOverviewRow]
    total_requests: int


class MCPToolUserEntry(BaseModel):
    tool_name: str
    api_key_hash: Optional[str] = None
    api_key_alias: Optional[str] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    call_count: int
    last_called: str


class MCPToolUsersResponse(BaseModel):
    entries: List[MCPToolUserEntry]
    total: int


# --- Helpers ---


def _snippet(text: Any, max_len: int = 200) -> Optional[str]:
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
    result = (s[:max_len] + "...") if len(s) > max_len else s
    if result == "{}":
        return None
    return result


def _input_snippet_for_log(sl: Any) -> Optional[str]:
    out = _snippet(sl.messages)
    if out:
        return out
    psr = getattr(sl, "proxy_server_request", None)
    if not psr:
        return None
    if isinstance(psr, str):
        try:
            psr = json.loads(psr)
        except Exception:
            return _snippet(psr)
    if isinstance(psr, dict):
        msgs = psr.get("messages")
        if msgs is None and isinstance(psr.get("body"), dict):
            msgs = psr["body"].get("messages")
        out = _snippet(msgs)
        if out:
            return out
        return _snippet(psr)
    return _snippet(psr)


def _build_index_where(
    mcp_server_name: Optional[str],
    tool_name: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, Any]:
    where: Dict[str, Any] = {}
    if mcp_server_name:
        where["mcp_server_name"] = mcp_server_name
    if tool_name:
        where["tool_name"] = tool_name
    if start_date or end_date:
        st_filter: Dict[str, Any] = {}
        if start_date:
            sd = start_date.replace("Z", "+00:00").strip()
            if "T" not in sd:
                sd += "T00:00:00+00:00"
            st_filter["gte"] = datetime.fromisoformat(sd)
        if end_date:
            ed = end_date.replace("Z", "+00:00").strip()
            if "T" not in ed:
                ed += "T23:59:59+00:00"
            st_filter["lte"] = datetime.fromisoformat(ed)
        where["start_time"] = st_filter
    return where


def _extract_api_key_alias(sl: Any) -> Optional[str]:
    meta = getattr(sl, "metadata", None)
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return None
    if isinstance(meta, dict):
        return meta.get("user_api_key_alias")
    return None


# --- Endpoints ---


@router.get(
    "/usage/logs",
    dependencies=[Depends(user_api_key_auth)],
    response_model=MCPUsageLogsResponse,
)
async def mcp_usage_logs(
    mcp_server_name: Optional[str] = Query(None),
    tool_name: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return paginated MCP server request logs from SpendLogs via index."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return MCPUsageLogsResponse(
            logs=[], total=0, page=page, page_size=page_size
        )

    try:
        where = _build_index_where(mcp_server_name, tool_name, start_date, end_date)

        index_rows = await prisma_client.db.litellm_spendlogmcpserverindex.find_many(
            where=where,
            order={"start_time": "desc"},
            skip=(page - 1) * page_size,
            take=page_size,
        )
        total = await prisma_client.db.litellm_spendlogmcpserverindex.count(
            where=where
        )
        request_ids = [r.request_id for r in index_rows]
        if not request_ids:
            return MCPUsageLogsResponse(
                logs=[], total=total, page=page, page_size=page_size
            )

        spend_logs = await prisma_client.db.litellm_spendlogs.find_many(
            where={"request_id": {"in": request_ids}}
        )
        log_by_id = {s.request_id: s for s in spend_logs}

        logs_out: List[MCPUsageLogEntry] = []
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
                MCPUsageLogEntry(
                    id=r.request_id,
                    timestamp=ts,
                    mcp_server_name=r.mcp_server_name,
                    tool_name=r.tool_name,
                    api_key_hash=r.api_key_hash,
                    api_key_alias=_extract_api_key_alias(sl),
                    user_id=r.user_id,
                    team_id=r.team_id,
                    model=sl.model,
                    status=sl.status,
                    spend=float(sl.spend) if sl.spend else None,
                    input_snippet=_input_snippet_for_log(sl),
                    output_snippet=_snippet(sl.response),
                )
            )
        return MCPUsageLogsResponse(
            logs=logs_out, total=total, page=page, page_size=page_size
        )
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)


@router.get(
    "/usage/overview",
    dependencies=[Depends(user_api_key_auth)],
    response_model=MCPUsageOverviewResponse,
)
async def mcp_usage_overview(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return per-MCP-server request counts and top tools for the dashboard."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return MCPUsageOverviewResponse(servers=[], total_requests=0)

    try:
        where = _build_index_where(None, None, start_date, end_date)

        index_rows = await prisma_client.db.litellm_spendlogmcpserverindex.find_many(
            where=where if where else {},
        )

        server_info_by_name: Dict[str, Dict[str, Optional[str]]] = {}
        try:
            mcp_servers_db = (
                await prisma_client.db.litellm_mcpservertable.find_many()
            )
            for srv in mcp_servers_db:
                name = getattr(srv, "server_name", None) or getattr(
                    srv, "alias", None
                )
                if name:
                    server_info_by_name[name] = {
                        "server_id": srv.server_id,
                        "description": getattr(srv, "description", None),
                    }
        except Exception:
            pass

        server_data: Dict[str, Dict[str, Any]] = {}
        for r in index_rows:
            name = r.mcp_server_name
            if name not in server_data:
                server_data[name] = {
                    "total_requests": 0,
                    "tools": {},
                    "users": set(),
                    "keys": set(),
                }
            sd = server_data[name]
            sd["total_requests"] += 1
            tool = r.tool_name or "unknown"
            sd["tools"][tool] = sd["tools"].get(tool, 0) + 1
            if r.user_id:
                sd["users"].add(r.user_id)
            if r.api_key_hash:
                sd["keys"].add(r.api_key_hash)

        servers: List[MCPServerOverviewRow] = []
        for name, sd in sorted(
            server_data.items(), key=lambda x: x[1]["total_requests"], reverse=True
        ):
            top_tools = sorted(
                sd["tools"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            info = server_info_by_name.get(name, {})
            servers.append(
                MCPServerOverviewRow(
                    mcp_server_name=name,
                    server_id=info.get("server_id"),
                    description=info.get("description"),
                    total_requests=sd["total_requests"],
                    top_tools=[
                        {"name": t, "count": c} for t, c in top_tools
                    ],
                    unique_users=len(sd["users"]),
                    unique_keys=len(sd["keys"]),
                )
            )

        return MCPUsageOverviewResponse(
            servers=servers,
            total_requests=sum(sd["total_requests"] for sd in server_data.values()),
        )
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)


@router.get(
    "/usage/tools",
    dependencies=[Depends(user_api_key_auth)],
    response_model=MCPToolUsersResponse,
)
async def mcp_usage_tools(
    mcp_server_name: str = Query(..., description="MCP server name"),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return which users/keys called which tools on a specific MCP server."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return MCPToolUsersResponse(entries=[], total=0)

    try:
        where = _build_index_where(mcp_server_name, None, start_date, end_date)

        index_rows = await prisma_client.db.litellm_spendlogmcpserverindex.find_many(
            where=where,
            order={"start_time": "desc"},
        )

        aggregation: Dict[tuple, Dict[str, Any]] = {}
        for r in index_rows:
            key = (
                r.tool_name or "unknown",
                r.api_key_hash or "",
                r.user_id or "",
                r.team_id or "",
            )
            if key not in aggregation:
                ts = (
                    r.start_time.isoformat()
                    if hasattr(r.start_time, "isoformat")
                    else str(r.start_time)
                )
                aggregation[key] = {"count": 0, "last_called": ts}
            aggregation[key]["count"] += 1

        entries: List[MCPToolUserEntry] = []
        for (tool, api_key, user, team), data in sorted(
            aggregation.items(), key=lambda x: x[1]["count"], reverse=True
        ):
            entries.append(
                MCPToolUserEntry(
                    tool_name=tool,
                    api_key_hash=api_key or None,
                    user_id=user or None,
                    team_id=team or None,
                    call_count=data["count"],
                    last_called=data["last_called"],
                )
            )

        return MCPToolUsersResponse(entries=entries, total=len(entries))
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)


# --- Alert Rule Models ---


class MCPAlertRuleCreate(BaseModel):
    mcp_server_name: Optional[str] = None
    tool_name_pattern: str
    webhook_url: str
    alert_name: str
    description: Optional[str] = None
    enabled: bool = True


class MCPAlertRuleResponse(BaseModel):
    id: str
    mcp_server_name: Optional[str] = None
    tool_name_pattern: str
    webhook_url: str
    alert_name: str
    description: Optional[str] = None
    enabled: bool
    created_at: str
    updated_at: str


class MCPAlertRulesListResponse(BaseModel):
    rules: List[MCPAlertRuleResponse]
    total: int


# --- Alert Rule Endpoints ---


@router.get(
    "/alert-rules",
    dependencies=[Depends(user_api_key_auth)],
    response_model=MCPAlertRulesListResponse,
)
async def list_mcp_alert_rules(
    mcp_server_name: Optional[str] = Query(None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List all MCP alert rules, optionally filtered by server name."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return MCPAlertRulesListResponse(rules=[], total=0)

    try:
        where: Dict[str, Any] = {}
        if mcp_server_name:
            where["mcp_server_name"] = mcp_server_name
        rules = await prisma_client.db.litellm_mcpalertrule.find_many(
            where=where if where else {},
            order={"created_at": "desc"},
        )
        result = [
            MCPAlertRuleResponse(
                id=r.id,
                mcp_server_name=r.mcp_server_name,
                tool_name_pattern=r.tool_name_pattern,
                webhook_url=r.webhook_url,
                alert_name=r.alert_name,
                description=r.description,
                enabled=r.enabled,
                created_at=r.created_at.isoformat()
                if hasattr(r.created_at, "isoformat")
                else str(r.created_at),
                updated_at=r.updated_at.isoformat()
                if hasattr(r.updated_at, "isoformat")
                else str(r.updated_at),
            )
            for r in rules
        ]
        return MCPAlertRulesListResponse(rules=result, total=len(result))
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)


@router.post(
    "/alert-rules",
    dependencies=[Depends(user_api_key_auth)],
    response_model=MCPAlertRuleResponse,
)
async def create_mcp_alert_rule(
    data: MCPAlertRuleCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Create a new MCP alert rule."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        rule = await prisma_client.db.litellm_mcpalertrule.create(
            data={
                "mcp_server_name": data.mcp_server_name,
                "tool_name_pattern": data.tool_name_pattern,
                "webhook_url": data.webhook_url,
                "alert_name": data.alert_name,
                "description": data.description,
                "enabled": data.enabled,
            }
        )
        return MCPAlertRuleResponse(
            id=rule.id,
            mcp_server_name=rule.mcp_server_name,
            tool_name_pattern=rule.tool_name_pattern,
            webhook_url=rule.webhook_url,
            alert_name=rule.alert_name,
            description=rule.description,
            enabled=rule.enabled,
            created_at=rule.created_at.isoformat()
            if hasattr(rule.created_at, "isoformat")
            else str(rule.created_at),
            updated_at=rule.updated_at.isoformat()
            if hasattr(rule.updated_at, "isoformat")
            else str(rule.updated_at),
        )
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)


@router.delete(
    "/alert-rules/{rule_id}",
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_mcp_alert_rule(
    rule_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Delete an MCP alert rule."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        await prisma_client.db.litellm_mcpalertrule.delete(
            where={"id": rule_id}
        )
        return {"status": "ok", "deleted": rule_id}
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)


@router.put(
    "/alert-rules/{rule_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_model=MCPAlertRuleResponse,
)
async def update_mcp_alert_rule(
    rule_id: str,
    data: MCPAlertRuleCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Update an existing MCP alert rule."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        rule = await prisma_client.db.litellm_mcpalertrule.update(
            where={"id": rule_id},
            data={
                "mcp_server_name": data.mcp_server_name,
                "tool_name_pattern": data.tool_name_pattern,
                "webhook_url": data.webhook_url,
                "alert_name": data.alert_name,
                "description": data.description,
                "enabled": data.enabled,
            },
        )
        return MCPAlertRuleResponse(
            id=rule.id,
            mcp_server_name=rule.mcp_server_name,
            tool_name_pattern=rule.tool_name_pattern,
            webhook_url=rule.webhook_url,
            alert_name=rule.alert_name,
            description=rule.description,
            enabled=rule.enabled,
            created_at=rule.created_at.isoformat()
            if hasattr(rule.created_at, "isoformat")
            else str(rule.created_at),
            updated_at=rule.updated_at.isoformat()
            if hasattr(rule.updated_at, "isoformat")
            else str(rule.updated_at),
        )
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)
