"""
Guardrails and policies usage endpoints for the dashboard.
GET /guardrails/usage/overview, /guardrails/usage/detail/:id, /guardrails/usage/logs
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


# --- Response models ---


class UsageOverviewRow(BaseModel):
    id: str
    name: str
    type: str
    provider: str
    requestsEvaluated: int
    failRate: float
    avgScore: Optional[float]
    avgLatency: Optional[float]
    status: str  # healthy | warning | critical
    trend: str  # up | down | stable


class UsageOverviewResponse(BaseModel):
    rows: List[UsageOverviewRow]
    chart: List[Dict[str, Any]]  # [{ date, passed, blocked }]
    totalRequests: int
    totalBlocked: int
    passRate: float


class UsageDetailResponse(BaseModel):
    guardrail_id: str
    guardrail_name: str
    type: str
    provider: str
    requestsEvaluated: int
    failRate: float
    avgScore: Optional[float]
    avgLatency: Optional[float]
    status: str
    trend: str
    description: Optional[str]
    time_series: List[Dict[str, Any]]


class UsageLogEntry(BaseModel):
    id: str
    timestamp: str
    action: str  # blocked | passed | flagged
    score: Optional[float]
    latency_ms: Optional[float]
    model: Optional[str]
    input_snippet: Optional[str]
    output_snippet: Optional[str]
    reason: Optional[str]


class UsageLogsResponse(BaseModel):
    logs: List[UsageLogEntry]
    total: int
    page: int
    page_size: int


def _status_from_fail_rate(fail_rate: float) -> str:
    if fail_rate > 15:
        return "critical"
    if fail_rate > 5:
        return "warning"
    return "healthy"


def _trend_from_comparison(current_fail: float, previous_fail: float) -> str:
    if previous_fail <= 0:
        return "stable"
    diff = current_fail - previous_fail
    if diff > 0.5:
        return "up"
    if diff < -0.5:
        return "down"
    return "stable"


def _aggregate_daily_metrics(metrics: Any, id_attr: str) -> Dict[str, Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        gid = getattr(m, id_attr)
        if gid not in agg:
            agg[gid] = {"requests": 0, "passed": 0, "blocked": 0, "flagged": 0}
        agg[gid]["requests"] += int(m.requests_evaluated or 0)
        agg[gid]["passed"] += int(m.passed_count or 0)
        agg[gid]["blocked"] += int(m.blocked_count or 0)
        agg[gid]["flagged"] += int(m.flagged_count or 0)
    return agg


def _prev_fail_rates(metrics_prev: Any, id_attr: str) -> Dict[str, float]:
    prev_agg_raw: Dict[str, Dict[str, int]] = {}
    for m in metrics_prev:
        gid = getattr(m, id_attr)
        r, b = int(m.requests_evaluated or 0), int(m.blocked_count or 0)
        if gid not in prev_agg_raw:
            prev_agg_raw[gid] = {"req": 0, "blocked": 0}
        prev_agg_raw[gid]["req"] += r
        prev_agg_raw[gid]["blocked"] += b
    return {
        gid: (100.0 * v["blocked"] / v["req"]) if v["req"] else 0.0
        for gid, v in prev_agg_raw.items()
    }


def _chart_from_metrics(metrics: Any) -> List[Dict[str, Any]]:
    chart_by_date: Dict[str, Dict[str, int]] = {}
    for m in metrics:
        d = m.date
        if d not in chart_by_date:
            chart_by_date[d] = {"passed": 0, "blocked": 0}
        chart_by_date[d]["passed"] += int(m.passed_count or 0)
        chart_by_date[d]["blocked"] += int(m.blocked_count or 0)
    return [
        {"date": d, "passed": v["passed"], "blocked": v["blocked"]}
        for d, v in sorted(chart_by_date.items())
    ]


def _get_guardrail_attrs(g: Any) -> tuple[Any, str]:
    """Get (guardrail_id, display_name) from guardrail - handles Prisma model or dict."""
    gid = getattr(g, "guardrail_id", None) or (
        g.get("guardrail_id") if isinstance(g, dict) else None
    )
    name = getattr(g, "guardrail_name", None) or (
        g.get("guardrail_name") if isinstance(g, dict) else None
    )
    return gid, (name or gid or "")


def _guardrail_overview_rows(
    guardrails: Any,
    agg: Dict[str, Dict[str, Any]],
    prev_agg: Dict[str, float],
) -> List[UsageOverviewRow]:
    rows: List[UsageOverviewRow] = []
    covered_keys: set = set()
    for g in guardrails:
        gid, display_name = _get_guardrail_attrs(g)
        # Metrics are keyed by logical name from spend log metadata; guardrails table uses UUID
        lookup_keys = [k for k in (display_name, gid) if k]
        covered_keys.update(lookup_keys)
        a = {"requests": 0, "passed": 0, "blocked": 0, "flagged": 0}
        for k in lookup_keys:
            if k in agg:
                a = agg[k]
                break
        req, blocked = a["requests"], a["blocked"]
        fail_rate = (100.0 * blocked / req) if req else 0.0
        litellm_params = (
            (g.litellm_params or {}) if isinstance(g.litellm_params, dict) else {}
        )
        provider = str(litellm_params.get("guardrail", "Unknown"))
        guardrail_info = (
            (g.guardrail_info or {}) if isinstance(g.guardrail_info, dict) else {}
        )
        gtype = str(guardrail_info.get("type", "Guardrail"))
        prev_fail = 0.0
        for k in lookup_keys:
            if k in prev_agg:
                prev_fail = float(prev_agg.get(k, 0.0) or 0.0)
                break
        trend = _trend_from_comparison(fail_rate, prev_fail)
        rows.append(
            UsageOverviewRow(
                id=gid,
                name=display_name or str(gid),
                type=gtype,
                provider=provider,
                requestsEvaluated=req,
                failRate=round(fail_rate, 1),
                avgScore=None,
                avgLatency=None,
                status=_status_from_fail_rate(fail_rate),
                trend=trend,
            )
        )
    # Add rows for guardrails with metrics but not in guardrails table (e.g. MCP, config)
    for agg_key, a in agg.items():
        if agg_key in covered_keys or a["requests"] == 0:
            continue
        req, blocked = a["requests"], a["blocked"]
        fail_rate = (100.0 * blocked / req) if req else 0.0
        prev_fail = float(prev_agg.get(agg_key, 0.0) or 0.0)
        trend = _trend_from_comparison(fail_rate, prev_fail)
        rows.append(
            UsageOverviewRow(
                id=agg_key,
                name=agg_key,
                type="Guardrail",
                provider="Custom",
                requestsEvaluated=req,
                failRate=round(fail_rate, 1),
                avgScore=None,
                avgLatency=None,
                status=_status_from_fail_rate(fail_rate),
                trend=trend,
            )
        )
    return rows


def _policy_overview_rows(
    policies: Any,
    agg: Dict[str, Dict[str, Any]],
    prev_agg: Dict[str, float],
) -> List[UsageOverviewRow]:
    rows: List[UsageOverviewRow] = []
    for p in policies:
        pid = p.policy_id
        a = agg.get(pid, {"requests": 0, "passed": 0, "blocked": 0, "flagged": 0})
        req, blocked = a["requests"], a["blocked"]
        fail_rate = (100.0 * blocked / req) if req else 0.0
        trend = _trend_from_comparison(fail_rate, prev_agg.get(pid, 0.0))
        rows.append(
            UsageOverviewRow(
                id=pid,
                name=p.policy_name or pid,
                type="Policy",
                provider="LiteLLM",
                requestsEvaluated=req,
                failRate=round(fail_rate, 1),
                avgScore=None,
                avgLatency=None,
                status=_status_from_fail_rate(fail_rate),
                trend=trend,
            )
        )
    return rows


@router.get(
    "/guardrails/usage/overview",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UsageOverviewResponse,
)
async def guardrails_usage_overview(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return guardrail performance overview for the dashboard."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return UsageOverviewResponse(
            rows=[], chart=[], totalRequests=0, totalBlocked=0, passRate=100.0
        )

    now = datetime.now(timezone.utc)
    end = end_date or now.strftime("%Y-%m-%d")
    start = start_date or (now - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        # Guardrails from DB
        guardrails = await prisma_client.db.litellm_guardrailstable.find_many()

        # Daily metrics in range
        metrics = await prisma_client.db.litellm_dailyguardrailmetrics.find_many(
            where={"date": {"gte": start, "lte": end}}
        )

        # Previous period for trend
        start_prev = (
            datetime.strptime(start, "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")
        metrics_prev = await prisma_client.db.litellm_dailyguardrailmetrics.find_many(
            where={"date": {"gte": start_prev, "lt": start}}
        )

        agg = _aggregate_daily_metrics(metrics, "guardrail_id")
        prev_agg = _prev_fail_rates(metrics_prev, "guardrail_id")
        chart = _chart_from_metrics(metrics)
        total_requests = sum(a["requests"] for a in agg.values())
        total_blocked = sum(a["blocked"] for a in agg.values())
        pass_rate = (
            (100.0 * (total_requests - total_blocked) / total_requests)
            if total_requests
            else 100.0
        )
        rows = _guardrail_overview_rows(guardrails, agg, prev_agg)
        return UsageOverviewResponse(
            rows=rows,
            chart=chart,
            totalRequests=total_requests,
            totalBlocked=total_blocked,
            passRate=round(pass_rate, 1),
        )
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)


@router.get(
    "/guardrails/usage/detail/{guardrail_id}",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UsageDetailResponse,
)
async def guardrails_usage_detail(
    guardrail_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return single guardrail usage metrics and time series."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    now = datetime.now(timezone.utc)
    end = end_date or now.strftime("%Y-%m-%d")
    start = start_date or (now - timedelta(days=7)).strftime("%Y-%m-%d")

    guardrail = await prisma_client.db.litellm_guardrailstable.find_unique(
        where={"guardrail_id": guardrail_id}
    )
    if not guardrail:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Guardrail not found")

    # Metrics are keyed by logical name (from spend log metadata), not UUID
    logical_id = getattr(guardrail, "guardrail_name", None) or (
        guardrail.get("guardrail_name") if isinstance(guardrail, dict) else None
    )
    metric_ids = [i for i in (logical_id, guardrail_id) if i]

    metrics = await prisma_client.db.litellm_dailyguardrailmetrics.find_many(
        where={
            "guardrail_id": {"in": metric_ids},
            "date": {"gte": start, "lte": end},
        }
    )
    metrics_prev = await prisma_client.db.litellm_dailyguardrailmetrics.find_many(
        where={
            "guardrail_id": {"in": metric_ids},
            "date": {"lt": start},
        }
    )

    requests = sum(int(m.requests_evaluated or 0) for m in metrics)
    blocked = sum(int(m.blocked_count or 0) for m in metrics)
    fail_rate = (100.0 * blocked / requests) if requests else 0.0

    prev_blocked = sum(int(m.blocked_count or 0) for m in metrics_prev)
    prev_req = sum(int(m.requests_evaluated or 0) for m in metrics_prev)
    prev_fail = (100.0 * prev_blocked / prev_req) if prev_req else 0.0
    trend = _trend_from_comparison(fail_rate, prev_fail)

    # Aggregate by date in case metrics exist under both UUID and logical name
    ts_by_date: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        d = m.date
        if d not in ts_by_date:
            ts_by_date[d] = {"passed": 0, "blocked": 0}
        ts_by_date[d]["passed"] += int(m.passed_count or 0)
        ts_by_date[d]["blocked"] += int(m.blocked_count or 0)
    time_series = [
        {"date": d, "passed": v["passed"], "blocked": v["blocked"], "score": None}
        for d, v in sorted(ts_by_date.items())
    ]
    _litellm_params = getattr(guardrail, "litellm_params", None) or (
        guardrail.get("litellm_params") if isinstance(guardrail, dict) else None
    )
    litellm_params = (
        _litellm_params
        if isinstance(_litellm_params, dict)
        else {}
    )
    _guardrail_info = getattr(guardrail, "guardrail_info", None) or (
        guardrail.get("guardrail_info") if isinstance(guardrail, dict) else None
    )
    guardrail_info = (
        _guardrail_info
        if isinstance(_guardrail_info, dict)
        else {}
    )
    _guardrail_name = getattr(guardrail, "guardrail_name", None) or (
        guardrail.get("guardrail_name") if isinstance(guardrail, dict) else None
    )

    return UsageDetailResponse(
        guardrail_id=guardrail_id,
        guardrail_name=_guardrail_name or guardrail_id,
        type=str(guardrail_info.get("type", "Guardrail")),
        provider=str(litellm_params.get("guardrail", "Unknown")),
        requestsEvaluated=requests,
        failRate=round(fail_rate, 1),
        avgScore=None,
        avgLatency=None,
        status=_status_from_fail_rate(fail_rate),
        trend=trend,
        description=guardrail_info.get("description"),
        time_series=time_series,
    )


def _build_usage_logs_where(
    guardrail_ids: Optional[List[str]],
    policy_id: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, Any]:
    where: Dict[str, Any] = {}
    if guardrail_ids:
        where["guardrail_id"] = (
            {"in": guardrail_ids} if len(guardrail_ids) > 1 else guardrail_ids[0]
        )
    if policy_id:
        where["policy_id"] = policy_id
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


def _usage_log_entry_from_row(
    r: Any, sl: Any, action_filter: Optional[str]
) -> Optional[UsageLogEntry]:
    meta = sl.metadata
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    guardrail_info_list = (meta or {}).get("guardrail_information") or []
    entry_for_guardrail = None
    for gi in guardrail_info_list:
        if (gi.get("guardrail_id") or gi.get("guardrail_name")) == r.guardrail_id:
            entry_for_guardrail = gi
            break
    action_val = "passed"
    score_val = None
    latency_val = None
    reason_val = None
    if entry_for_guardrail:
        st = (entry_for_guardrail.get("guardrail_status") or "").lower()
        if "intervened" in st or "block" in st:
            action_val = "blocked"
        elif "fail" in st or "error" in st:
            action_val = "flagged"
        duration = entry_for_guardrail.get("duration")
        if duration is not None:
            latency_val = round(float(duration) * 1000, 0)
        score_val = entry_for_guardrail.get(
            "confidence_score"
        ) or entry_for_guardrail.get("risk_score")
        if score_val is not None:
            score_val = round(float(score_val), 2)
        resp = entry_for_guardrail.get("guardrail_response")
        if isinstance(resp, str):
            reason_val = resp[:500]
        elif isinstance(resp, dict):
            reason_val = str(resp)[:500]
    if action_filter and action_val != action_filter:
        return None
    ts = (
        sl.startTime.isoformat()
        if hasattr(sl.startTime, "isoformat")
        else str(sl.startTime)
    )
    return UsageLogEntry(
        id=r.request_id,
        timestamp=ts,
        action=action_val,
        score=score_val,
        latency_ms=latency_val,
        model=sl.model,
        input_snippet=_input_snippet_for_log(sl),
        output_snippet=_snippet(sl.response),
        reason=reason_val,
    )


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
    """Snippet for request input: prefer messages, fall back to proxy_server_request (same as drawer)."""
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


@router.get(
    "/guardrails/usage/logs",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UsageLogsResponse,
)
async def guardrails_usage_logs(
    guardrail_id: Optional[str] = Query(None),
    policy_id: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    action: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return paginated run logs for a guardrail (or policy) from SpendLogs via index."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return UsageLogsResponse(logs=[], total=0, page=page, page_size=page_size)

    if not guardrail_id and not policy_id:
        return UsageLogsResponse(logs=[], total=0, page=page, page_size=page_size)

    try:
        # Index rows may store either guardrail_id (UUID) or guardrail_name from metadata.
        # Query by both so we match regardless of which was written.
        effective_guardrail_ids: List[str] = [guardrail_id] if guardrail_id else []
        if guardrail_id:
            guardrail = await prisma_client.db.litellm_guardrailstable.find_unique(
                where={"guardrail_id": guardrail_id}
            )
            if guardrail:
                logical_name = getattr(guardrail, "guardrail_name", None)
                if logical_name and logical_name not in effective_guardrail_ids:
                    effective_guardrail_ids.append(logical_name)

        where = _build_usage_logs_where(
            effective_guardrail_ids or None, policy_id, start_date, end_date
        )
        index_rows = await prisma_client.db.litellm_spendlogguardrailindex.find_many(
            where=where,
            order={"start_time": "desc"},
            skip=(page - 1) * page_size,
            take=page_size + 1,
        )
        total = await prisma_client.db.litellm_spendlogguardrailindex.count(where=where)
        request_ids = [r.request_id for r in index_rows[:page_size]]
        if not request_ids:
            return UsageLogsResponse(
                logs=[], total=total, page=page, page_size=page_size
            )
        spend_logs = await prisma_client.db.litellm_spendlogs.find_many(
            where={"request_id": {"in": request_ids}}
        )
        log_by_id = {s.request_id: s for s in spend_logs}
        logs_out: List[UsageLogEntry] = []
        for r in index_rows[:page_size]:
            sl = log_by_id.get(r.request_id)
            if not sl:
                continue
            entry = _usage_log_entry_from_row(r, sl, action)
            if entry is not None:
                logs_out.append(entry)
        return UsageLogsResponse(
            logs=logs_out, total=total, page=page, page_size=page_size
        )
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)


# --- Policy usage (same shape as guardrails; policy metrics populated when policy_run is in metadata) ---


@router.get(
    "/policies/usage/overview",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UsageOverviewResponse,
)
async def policies_usage_overview(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return policy performance overview for the dashboard."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return UsageOverviewResponse(
            rows=[], chart=[], totalRequests=0, totalBlocked=0, passRate=100.0
        )

    now = datetime.now(timezone.utc)
    end = end_date or now.strftime("%Y-%m-%d")
    start = start_date or (now - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        policies = await prisma_client.db.litellm_policytable.find_many()
        metrics = await prisma_client.db.litellm_dailypolicymetrics.find_many(
            where={"date": {"gte": start, "lte": end}}
        )
        metrics_prev = await prisma_client.db.litellm_dailypolicymetrics.find_many(
            where={
                "date": {
                    "gte": (
                        datetime.strptime(start, "%Y-%m-%d") - timedelta(days=7)
                    ).strftime("%Y-%m-%d"),
                    "lt": start,
                }
            }
        )
        agg = _aggregate_daily_metrics(metrics, "policy_id")
        prev_agg = _prev_fail_rates(metrics_prev, "policy_id")
        chart = _chart_from_metrics(metrics)
        total_requests = sum(a["requests"] for a in agg.values())
        total_blocked = sum(a["blocked"] for a in agg.values())
        pass_rate = (
            (100.0 * (total_requests - total_blocked) / total_requests)
            if total_requests
            else 100.0
        )
        rows = _policy_overview_rows(policies, agg, prev_agg)
        return UsageOverviewResponse(
            rows=rows,
            chart=chart,
            totalRequests=total_requests,
            totalBlocked=total_blocked,
            passRate=round(pass_rate, 1),
        )
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)
