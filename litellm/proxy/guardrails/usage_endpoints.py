"""
Guardrails and policies usage endpoints for the dashboard.
GET /guardrails/usage/overview, /guardrails/usage/detail/:id, /guardrails/usage/logs
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, overload

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import (
    DailyGuardrailMetricsRepository,
    DailyGuardrailUsageUnitsRepository,
    DailyPolicyMetricsRepository,
    GuardrailsRepository,
    PolicyRepository,
    SpendLogGuardrailIndexRepository,
    SpendLogsRepository,
)

if TYPE_CHECKING:
    from prisma import models as prisma_models
    from prisma import types as prisma_types

    from litellm.proxy.utils import PrismaClient
    from litellm.types.guardrails import Guardrail

    _DbOrConfigGuardrail = prisma_models.LiteLLM_GuardrailsTable | Guardrail
    _DailyMetricsRow = prisma_models.LiteLLM_DailyGuardrailMetrics | prisma_models.LiteLLM_DailyPolicyMetrics

router: Final = APIRouter()

_EMPTY_UNITS: Final[Mapping[str, int]] = MappingProxyType({})

_USAGE_MAX_RANGE_DAYS: Final = 366


def _resolve_usage_window(start_date: str | None, end_date: str | None) -> tuple[str, str]:
    from fastapi import HTTPException, status

    now: Final = datetime.now(timezone.utc)
    end: Final = end_date or now.strftime("%Y-%m-%d")
    start: Final = start_date or (now - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        parsed: Final = (date.fromisoformat(start), date.fromisoformat(end))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date and end_date must be in YYYY-MM-DD format",
        )
    start_obj, end_obj = parsed
    if (start_obj.isoformat(), end_obj.isoformat()) != (start, end):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date and end_date must be in YYYY-MM-DD format",
        )
    if end_obj < start_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be on or before end_date",
        )
    if end_obj - start_obj > timedelta(days=_USAGE_MAX_RANGE_DAYS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Date range too large; maximum is {_USAGE_MAX_RANGE_DAYS} days",
        )
    return start, end


def _guardrails_table(
    prisma_client: "PrismaClient",
) -> "TableActions[prisma_models.LiteLLM_GuardrailsTable]":
    guardrails_table: Final[TableActions[prisma_models.LiteLLM_GuardrailsTable]] = GuardrailsRepository(
        prisma_client
    ).table
    return guardrails_table


def _policies_table(
    prisma_client: "PrismaClient",
) -> "TableActions[prisma_models.LiteLLM_PolicyTable]":
    policies_table: Final[TableActions[prisma_models.LiteLLM_PolicyTable]] = PolicyRepository(prisma_client).table
    return policies_table


def _daily_guardrail_metrics_table(
    prisma_client: "PrismaClient",
) -> "TableActions[prisma_models.LiteLLM_DailyGuardrailMetrics]":
    metrics_table: Final[TableActions[prisma_models.LiteLLM_DailyGuardrailMetrics]] = DailyGuardrailMetricsRepository(
        prisma_client
    ).table
    return metrics_table


def _daily_policy_metrics_table(
    prisma_client: "PrismaClient",
) -> "TableActions[prisma_models.LiteLLM_DailyPolicyMetrics]":
    metrics_table: Final[TableActions[prisma_models.LiteLLM_DailyPolicyMetrics]] = DailyPolicyMetricsRepository(
        prisma_client
    ).table
    return metrics_table


async def _find_daily_guardrail_metrics(
    prisma_client: "PrismaClient",
    where: "prisma_types.LiteLLM_DailyGuardrailMetricsWhereInput",
) -> "Sequence[prisma_models.LiteLLM_DailyGuardrailMetrics]":
    return await _daily_guardrail_metrics_table(prisma_client).find_many(where=where)


async def _find_daily_policy_metrics(
    prisma_client: "PrismaClient",
    where: "prisma_types.LiteLLM_DailyPolicyMetricsWhereInput",
) -> "Sequence[prisma_models.LiteLLM_DailyPolicyMetrics]":
    return await _daily_policy_metrics_table(prisma_client).find_many(where=where)


def _daily_guardrail_usage_units_table(
    prisma_client: "PrismaClient",
) -> "TableActions[prisma_models.LiteLLM_DailyGuardrailUsageUnits]":
    units_table: Final[TableActions[prisma_models.LiteLLM_DailyGuardrailUsageUnits]] = (
        DailyGuardrailUsageUnitsRepository(prisma_client).table
    )
    return units_table


async def _find_daily_guardrail_usage_units(
    prisma_client: "PrismaClient",
    where: "prisma_types.LiteLLM_DailyGuardrailUsageUnitsWhereInput",
) -> "Sequence[prisma_models.LiteLLM_DailyGuardrailUsageUnits]":
    from prisma.errors import TableNotFoundError

    try:
        return await _daily_guardrail_usage_units_table(prisma_client).find_many(where=where)
    except TableNotFoundError as e:
        verbose_proxy_logger.warning(
            "Guardrail usage units are unavailable until the LiteLLM_DailyGuardrailUsageUnits migration is applied: %s",
            e,
        )
        return ()


def _counter_name(row: "prisma_models.LiteLLM_DailyGuardrailUsageUnits") -> str:
    return row.usage_unit


def _sum_counter_units(rows: "Iterable[prisma_models.LiteLLM_DailyGuardrailUsageUnits]") -> Mapping[str, int]:
    ordered: Final = sorted(rows, key=_counter_name)
    return MappingProxyType(
        {name: sum(int(r.units) for r in group) for name, group in groupby(ordered, key=_counter_name)}
    )


def _units_by(
    rows: "Sequence[prisma_models.LiteLLM_DailyGuardrailUsageUnits]",
    key_of: "Callable[[prisma_models.LiteLLM_DailyGuardrailUsageUnits], str]",
) -> Mapping[str, Mapping[str, int]]:
    ordered: Final = sorted(rows, key=key_of)
    return MappingProxyType({key: _sum_counter_units(group) for key, group in groupby(ordered, key=key_of)})


# --- Response models ---


class _GuardrailRunInfo(TypedDict, total=False):
    guardrail_id: ReadOnly[str | None]
    guardrail_name: ReadOnly[str | None]
    guardrail_status: ReadOnly[str | None]
    duration: ReadOnly[float | None]
    confidence_score: ReadOnly[float | None]
    risk_score: ReadOnly[float | None]
    guardrail_response: ReadOnly[str | Mapping[str, object] | Sequence[Mapping[str, object]] | None]


class UsageChartPoint(TypedDict):
    date: str
    passed: int
    blocked: int
    score: NotRequired[float | None]


class _MetricTotals(TypedDict):
    requests: int
    passed: int
    blocked: int
    flagged: int


class _PrevPeriodCounts(TypedDict):
    req: int
    blocked: int


class _DailyPassBlocked(TypedDict):
    passed: int
    blocked: int


class UsageOverviewRow(BaseModel):
    id: str
    name: str
    type: str
    provider: str
    requestsEvaluated: int
    failRate: float
    avgScore: float | None
    avgLatency: float | None
    status: str  # healthy | warning | critical
    trend: str  # up | down | stable
    usageUnits: Mapping[str, int]


class UsageOverviewResponse(BaseModel):
    rows: list[UsageOverviewRow]
    chart: list[UsageChartPoint]  # [{ date, passed, blocked }]
    totalRequests: int
    totalBlocked: int
    passRate: float
    totalUsageUnits: Mapping[str, int]


class UsageUnitsDailyPoint(BaseModel):
    date: str
    units: Mapping[str, int]


class UsageDetailResponse(BaseModel):
    guardrail_id: str
    guardrail_name: str
    type: str
    provider: str
    requestsEvaluated: int
    failRate: float
    avgScore: float | None
    avgLatency: float | None
    status: str
    trend: str
    description: str | None
    time_series: list[UsageChartPoint]
    usage_units: Mapping[str, int]
    usage_units_daily: Sequence[UsageUnitsDailyPoint]
    usage_units_by_team: Mapping[str, Mapping[str, int]]
    usage_units_by_key: Mapping[str, Mapping[str, int]]


class UsageLogEntry(BaseModel):
    id: str
    timestamp: str
    action: str  # blocked | passed | flagged
    score: float | None
    latency_ms: float | None
    model: str | None
    input_snippet: str | None
    output_snippet: str | None
    reason: str | None


class UsageLogsResponse(BaseModel):
    logs: list[UsageLogEntry]
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
    diff: Final = current_fail - previous_fail
    if diff > 0.5:
        return "up"
    if diff < -0.5:
        return "down"
    return "stable"


def _aggregate_daily_metrics(metrics: "Sequence[_DailyMetricsRow]", id_attr: str) -> Mapping[str, _MetricTotals]:
    agg: Final[dict[str, _MetricTotals]] = {}
    for m in metrics:
        gid: str = getattr(m, id_attr)
        if gid not in agg:
            agg[gid] = {"requests": 0, "passed": 0, "blocked": 0, "flagged": 0}
        agg[gid]["requests"] += int(m.requests_evaluated or 0)
        agg[gid]["passed"] += int(m.passed_count or 0)
        agg[gid]["blocked"] += int(m.blocked_count or 0)
        agg[gid]["flagged"] += int(m.flagged_count or 0)
    return agg


def _prev_fail_rates(metrics_prev: "Sequence[_DailyMetricsRow]", id_attr: str) -> Mapping[str, float]:
    prev_agg_raw: Final[dict[str, _PrevPeriodCounts]] = {}
    for m in metrics_prev:
        gid: str = getattr(m, id_attr)
        r, b = int(m.requests_evaluated or 0), int(m.blocked_count or 0)
        if gid not in prev_agg_raw:
            prev_agg_raw[gid] = {"req": 0, "blocked": 0}
        prev_agg_raw[gid]["req"] += r
        prev_agg_raw[gid]["blocked"] += b
    return {gid: (100.0 * v["blocked"] / v["req"]) if v["req"] else 0.0 for gid, v in prev_agg_raw.items()}


def _chart_from_metrics(metrics: "Sequence[_DailyMetricsRow]") -> list[UsageChartPoint]:
    chart_by_date: Final[dict[str, _DailyPassBlocked]] = {}
    for m in metrics:
        d = m.date
        if d not in chart_by_date:
            chart_by_date[d] = {"passed": 0, "blocked": 0}
        chart_by_date[d]["passed"] += int(m.passed_count or 0)
        chart_by_date[d]["blocked"] += int(m.blocked_count or 0)
    return [{"date": d, "passed": v["passed"], "blocked": v["blocked"]} for d, v in sorted(chart_by_date.items())]


_GuardrailStrField = Literal["guardrail_id", "guardrail_name"]
_GuardrailObjectField = Literal["litellm_params", "guardrail_info"]


@overload
def _get_guardrail_field(g: "_DbOrConfigGuardrail", field: _GuardrailStrField) -> str | None: ...


@overload
def _get_guardrail_field(g: "_DbOrConfigGuardrail", field: _GuardrailObjectField) -> object: ...


def _get_guardrail_field(g: "_DbOrConfigGuardrail", field: _GuardrailStrField | _GuardrailObjectField) -> object:
    """Read `field` off a guardrail whether it's a Prisma row (attr) or a dict/TypedDict (key)."""
    if isinstance(g, dict):
        return g.get(field)
    return getattr(g, field, None)


def _to_dict(value: object) -> dict[str, Any]:
    """Coerce a pydantic model (e.g. LitellmParams) / dict value into a plain dict."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return value
    return {}


def _get_guardrail_attrs(g: "_DbOrConfigGuardrail") -> tuple[Any, str]:
    """Get (guardrail_id, display_name) from guardrail - handles Prisma model or dict."""
    gid: Final = _get_guardrail_field(g, "guardrail_id")
    name: Final = _get_guardrail_field(g, "guardrail_name")
    return gid, (name or gid or "")


def _guardrail_overview_rows(
    guardrails: "Sequence[_DbOrConfigGuardrail]",
    agg: Mapping[str, _MetricTotals],
    prev_agg: Mapping[str, float],
    units_agg: Mapping[str, Mapping[str, int]],
) -> list[UsageOverviewRow]:
    rows: Final[list[UsageOverviewRow]] = []
    covered_keys: Final[set[str]] = set()
    for g in guardrails:
        gid, display_name = _get_guardrail_attrs(g)
        # Metrics are keyed by logical name from spend log metadata; guardrails table uses UUID
        lookup_keys: Sequence[str] = [k for k in (display_name, gid) if k]
        covered_keys.update(lookup_keys)
        a: _MetricTotals = {"requests": 0, "passed": 0, "blocked": 0, "flagged": 0}
        for k in lookup_keys:
            if k in agg:
                a = agg[k]
                break
        req, blocked = a["requests"], a["blocked"]
        fail_rate = (100.0 * blocked / req) if req else 0.0
        litellm_params = _to_dict(_get_guardrail_field(g, "litellm_params"))
        provider = str(litellm_params.get("guardrail", "Unknown"))
        guardrail_info = _to_dict(_get_guardrail_field(g, "guardrail_info"))
        gtype = str(guardrail_info.get("type", "Guardrail"))
        prev_fail = 0.0
        for k in lookup_keys:
            if k in prev_agg:
                prev_fail = float(prev_agg.get(k, 0.0) or 0.0)
                break
        trend = _trend_from_comparison(fail_rate, prev_fail)
        row_units: Mapping[str, int] = next((units_agg[k] for k in lookup_keys if k in units_agg), _EMPTY_UNITS)
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
                usageUnits=row_units,
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
                usageUnits=units_agg.get(agg_key, _EMPTY_UNITS),
            )
        )
    return rows


def _policy_overview_rows(
    policies: "Sequence[prisma_models.LiteLLM_PolicyTable]",
    agg: Mapping[str, _MetricTotals],
    prev_agg: Mapping[str, float],
) -> list[UsageOverviewRow]:
    rows: Final[list[UsageOverviewRow]] = []
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
                usageUnits=_EMPTY_UNITS,
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
    start_date: str | None = Query(None, description="YYYY-MM-DD"),
    end_date: str | None = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return guardrail performance overview for the dashboard."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return UsageOverviewResponse(
            rows=[], chart=[], totalRequests=0, totalBlocked=0, passRate=100.0, totalUsageUnits=_EMPTY_UNITS
        )

    start, end = _resolve_usage_window(start_date, end_date)

    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    try:
        db_guardrails: Final = await _guardrails_table(prisma_client).find_many()
        seen_ids: Final = {gid for g in db_guardrails if (gid := _get_guardrail_field(g, "guardrail_id")) is not None}
        config_guardrails: Final = [
            g for g in IN_MEMORY_GUARDRAIL_HANDLER.list_config_guardrails() if g.get("guardrail_id") not in seen_ids
        ]
        guardrails: Final[Sequence[_DbOrConfigGuardrail]] = [*db_guardrails, *config_guardrails]

        # Daily metrics in range
        metrics: Final[Sequence[prisma_models.LiteLLM_DailyGuardrailMetrics]] = await _find_daily_guardrail_metrics(
            prisma_client, where={"date": {"gte": start, "lte": end}}
        )

        # Previous period for trend
        start_prev: Final = (date.fromisoformat(start) - timedelta(days=7)).isoformat()
        metrics_prev: Sequence[prisma_models.LiteLLM_DailyGuardrailMetrics] = await _find_daily_guardrail_metrics(
            prisma_client, where={"date": {"gte": start_prev, "lt": start}}
        )

        units_where: Final[prisma_types.LiteLLM_DailyGuardrailUsageUnitsWhereInput] = {
            "date": {"gte": start, "lte": end}
        }
        units_rows: Final[
            Sequence[prisma_models.LiteLLM_DailyGuardrailUsageUnits]
        ] = await _find_daily_guardrail_usage_units(prisma_client, where=units_where)

        agg: Final = _aggregate_daily_metrics(metrics, "guardrail_id")
        prev_agg: Final = _prev_fail_rates(metrics_prev, "guardrail_id")
        units_agg: Final = _units_by(units_rows, lambda r: r.guardrail_id)
        chart: Final = _chart_from_metrics(metrics)
        total_requests: Final = sum(a["requests"] for a in agg.values())
        total_blocked: Final = sum(a["blocked"] for a in agg.values())
        pass_rate: Final = (100.0 * (total_requests - total_blocked) / total_requests) if total_requests else 100.0
        rows: Final = _guardrail_overview_rows(guardrails, agg, prev_agg, units_agg)
        return UsageOverviewResponse(
            rows=rows,
            chart=chart,
            totalRequests=total_requests,
            totalBlocked=total_blocked,
            passRate=round(pass_rate, 1),
            totalUsageUnits=_sum_counter_units(units_rows),
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
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return single guardrail usage metrics and time series."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    start, end = _resolve_usage_window(start_date, end_date)

    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    guardrail = await _guardrails_table(prisma_client).find_unique(where={"guardrail_id": guardrail_id})
    if guardrail is None:
        guardrail = IN_MEMORY_GUARDRAIL_HANDLER.get_config_guardrail_by_id(guardrail_id=guardrail_id)
    if guardrail is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Guardrail not found")

    # Metrics are keyed by logical name (from spend log metadata), not UUID
    logical_id: Final = _get_guardrail_field(guardrail, "guardrail_name")
    metric_ids: Final = [i for i in (logical_id, guardrail_id) if i]

    metrics: Final[Sequence[prisma_models.LiteLLM_DailyGuardrailMetrics]] = await _find_daily_guardrail_metrics(
        prisma_client,
        where={
            "guardrail_id": {"in": metric_ids},
            "date": {"gte": start, "lte": end},
        },
    )
    start_prev: Final = (date.fromisoformat(start) - timedelta(days=7)).isoformat()
    metrics_prev: Final[Sequence[prisma_models.LiteLLM_DailyGuardrailMetrics]] = await _find_daily_guardrail_metrics(
        prisma_client,
        where={
            "guardrail_id": {"in": metric_ids},
            "date": {"gte": start_prev, "lt": start},
        },
    )
    units_where: Final[prisma_types.LiteLLM_DailyGuardrailUsageUnitsWhereInput] = {
        "guardrail_id": {"in": metric_ids},
        "date": {"gte": start, "lte": end},
    }
    units_rows: Final[
        Sequence[prisma_models.LiteLLM_DailyGuardrailUsageUnits]
    ] = await _find_daily_guardrail_usage_units(prisma_client, where=units_where)

    requests: Final = sum(int(m.requests_evaluated or 0) for m in metrics)
    blocked: Final = sum(int(m.blocked_count or 0) for m in metrics)
    fail_rate: Final = (100.0 * blocked / requests) if requests else 0.0

    prev_blocked: Final = sum(int(m.blocked_count or 0) for m in metrics_prev)
    prev_req: Final = sum(int(m.requests_evaluated or 0) for m in metrics_prev)
    prev_fail: Final = (100.0 * prev_blocked / prev_req) if prev_req else 0.0
    trend: Final = _trend_from_comparison(fail_rate, prev_fail)

    # Aggregate by date in case metrics exist under both UUID and logical name
    ts_by_date: Final[dict[str, _DailyPassBlocked]] = {}
    for m in metrics:
        d = m.date
        if d not in ts_by_date:
            ts_by_date[d] = {"passed": 0, "blocked": 0}
        ts_by_date[d]["passed"] += int(m.passed_count or 0)
        ts_by_date[d]["blocked"] += int(m.blocked_count or 0)
    time_series: Final[list[UsageChartPoint]] = [
        {"date": d, "passed": v["passed"], "blocked": v["blocked"], "score": None}
        for d, v in sorted(ts_by_date.items())
    ]
    litellm_params: Final = _to_dict(_get_guardrail_field(guardrail, "litellm_params"))
    guardrail_info: Final = _to_dict(_get_guardrail_field(guardrail, "guardrail_info"))
    _guardrail_name: Final = _get_guardrail_field(guardrail, "guardrail_name")
    daily_unit_sums: Final = sorted(_units_by(units_rows, lambda r: r.date).items())
    units_daily: Final = tuple(UsageUnitsDailyPoint(date=d, units=units) for d, units in daily_unit_sums)

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
        usage_units=_sum_counter_units(units_rows),
        usage_units_daily=units_daily,
        usage_units_by_team=_units_by(units_rows, lambda r: r.team_id),
        usage_units_by_key=_units_by(units_rows, lambda r: r.api_key),
    )


def _build_usage_logs_where(
    guardrail_ids: list[str] | None,
    policy_id: str | None,
    start_date: str | None,
    end_date: str | None,
) -> "prisma_types.LiteLLM_SpendLogGuardrailIndexWhereInput":
    where: Final[prisma_types.LiteLLM_SpendLogGuardrailIndexWhereInput] = {}
    if guardrail_ids:
        where["guardrail_id"] = {"in": guardrail_ids} if len(guardrail_ids) > 1 else guardrail_ids[0]
    if policy_id:
        where["policy_id"] = policy_id
    if start_date or end_date:
        st_filter: Final[prisma_types.DateTimeFilter] = {}
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
    r: "prisma_models.LiteLLM_SpendLogGuardrailIndex",
    sl: "prisma_models.LiteLLM_SpendLogs",
    action_filter: str | None,
) -> UsageLogEntry | None:
    meta = sl.metadata
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    guardrail_info_list: Final[Sequence[_GuardrailRunInfo]] = (meta or {}).get("guardrail_information") or []
    entry_for_guardrail: _GuardrailRunInfo | None = None
    for gi in guardrail_info_list:
        if (gi.get("guardrail_id") or gi.get("guardrail_name")) == r.guardrail_id:
            entry_for_guardrail = gi
            break
    action_val = "passed"
    score_val = None
    latency_val = None
    reason_val = None
    if entry_for_guardrail:
        st: Final = (entry_for_guardrail.get("guardrail_status") or "").lower()
        if "intervened" in st or "block" in st:
            action_val = "blocked"
        elif "fail" in st or "error" in st:
            action_val = "flagged"
        duration: Final = entry_for_guardrail.get("duration")
        if duration is not None:
            latency_val = round(float(duration) * 1000, 0)
        score_val = entry_for_guardrail.get("confidence_score") or entry_for_guardrail.get("risk_score")
        if score_val is not None:
            score_val = round(float(score_val), 2)
        resp: Final = entry_for_guardrail.get("guardrail_response")
        if isinstance(resp, str):
            reason_val = resp[:500]
        elif isinstance(resp, dict):
            reason_val = str(resp)[:500]
    if action_filter and action_val != action_filter:
        return None
    ts: Final = sl.startTime.isoformat() if hasattr(sl.startTime, "isoformat") else str(sl.startTime)
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


def _snippet(text: Any, max_len: int = 200) -> str | None:
    if text is None:
        return None
    if isinstance(text, str):
        s = text
    elif isinstance(text, list):
        parts: Final[Sequence[str]] = [
            (c if isinstance(c := item["content"], str) else str(c))
            if isinstance(item, dict) and "content" in item
            else str(item)
            for item in text
        ]
        s = " ".join(parts)
    else:
        s = str(text)
    result: Final = (s[:max_len] + "...") if len(s) > max_len else s
    if result == "{}":
        return None
    return result


def _input_snippet_for_log(sl: "prisma_models.LiteLLM_SpendLogs") -> str | None:
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
    guardrail_id: str | None = Query(None),
    policy_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    action: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
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
        effective_guardrail_ids: Final[list[str]] = [guardrail_id] if guardrail_id else []
        if guardrail_id:
            from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

            guardrail = await _guardrails_table(prisma_client).find_unique(where={"guardrail_id": guardrail_id})
            if guardrail is None:
                guardrail = IN_MEMORY_GUARDRAIL_HANDLER.get_config_guardrail_by_id(guardrail_id=guardrail_id)
            if guardrail:
                logical_name: Final = _get_guardrail_field(guardrail, "guardrail_name")
                if logical_name and logical_name not in effective_guardrail_ids:
                    effective_guardrail_ids.append(logical_name)

        where: Final = _build_usage_logs_where(effective_guardrail_ids or None, policy_id, start_date, end_date)
        index_rows: Sequence[prisma_models.LiteLLM_SpendLogGuardrailIndex] = await SpendLogGuardrailIndexRepository(
            prisma_client
        ).table.find_many(
            where=where,
            order={"start_time": "desc"},
            skip=(page - 1) * page_size,
            take=page_size + 1,
        )
        total: Final[int] = await SpendLogGuardrailIndexRepository(prisma_client).table.count(where=where)
        request_ids: Final = [r.request_id for r in index_rows[:page_size]]
        if not request_ids:
            return UsageLogsResponse(logs=[], total=total, page=page, page_size=page_size)
        spend_logs: Final[Sequence[prisma_models.LiteLLM_SpendLogs]] = await SpendLogsRepository(
            prisma_client
        ).table.find_many(where={"request_id": {"in": request_ids}})
        log_by_id: Final = {s.request_id: s for s in spend_logs}
        logs_out: Final[list[UsageLogEntry]] = []
        for r in index_rows[:page_size]:
            sl = log_by_id.get(r.request_id)
            if not sl:
                continue
            entry = _usage_log_entry_from_row(r, sl, action)
            if entry is not None:
                logs_out.append(entry)
        return UsageLogsResponse(logs=logs_out, total=total, page=page, page_size=page_size)
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
    start_date: str | None = Query(None, description="YYYY-MM-DD"),
    end_date: str | None = Query(None, description="YYYY-MM-DD"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return policy performance overview for the dashboard."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return UsageOverviewResponse(
            rows=[], chart=[], totalRequests=0, totalBlocked=0, passRate=100.0, totalUsageUnits=_EMPTY_UNITS
        )

    start, end = _resolve_usage_window(start_date, end_date)

    try:
        policies: Final = await _policies_table(prisma_client).find_many()
        metrics: Final[Sequence[prisma_models.LiteLLM_DailyPolicyMetrics]] = await _find_daily_policy_metrics(
            prisma_client, where={"date": {"gte": start, "lte": end}}
        )
        metrics_prev: Final[Sequence[prisma_models.LiteLLM_DailyPolicyMetrics]] = await _find_daily_policy_metrics(
            prisma_client,
            where={
                "date": {
                    "gte": (date.fromisoformat(start) - timedelta(days=7)).isoformat(),
                    "lt": start,
                }
            },
        )
        agg: Final = _aggregate_daily_metrics(metrics, "policy_id")
        prev_agg: Final = _prev_fail_rates(metrics_prev, "policy_id")
        chart: Final = _chart_from_metrics(metrics)
        total_requests: Final = sum(a["requests"] for a in agg.values())
        total_blocked: Final = sum(a["blocked"] for a in agg.values())
        pass_rate: Final = (100.0 * (total_requests - total_blocked) / total_requests) if total_requests else 100.0
        rows: Final = _policy_overview_rows(policies, agg, prev_agg)
        return UsageOverviewResponse(
            rows=rows,
            chart=chart,
            totalRequests=total_requests,
            totalBlocked=total_blocked,
            passRate=round(pass_rate, 1),
            totalUsageUnits=_EMPTY_UNITS,
        )
    except Exception as e:
        from litellm.proxy.utils import handle_exception_on_proxy

        raise handle_exception_on_proxy(e)
