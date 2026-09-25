"""
GATEWAY REQUEST COUNTS (SGR)

GET /gateway/daily/activity - successful/failed gateway requests by date and route

Source of truth is LiteLLM_DailyGatewayRequests, written at the ASGI edge by
BillableRequestMetricsMiddleware. This counts what the proxy answered, so it is
independent of whether a request reached litellm's logging callbacks.

The table carries no key/user/team dimension, so these totals are deployment-wide
and the endpoint is restricted to proxy admin roles.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.gateway_requests import (
    GatewayRequestActivityResponse,
    GatewayRequestBreakdownEntry,
    GatewayRequestDailyEntry,
)

router: Final = APIRouter()

_DEFAULT_LOOKBACK_DAYS: Final = 30

_AGGREGATE_SQL: Final = """
    SELECT
        date,
        category,
        route,
        SUM(successful_requests)::bigint AS successful_requests,
        SUM(failed_requests)::bigint AS failed_requests
    FROM "LiteLLM_DailyGatewayRequests"
    WHERE date >= $1 AND date <= $2
    GROUP BY date, category, route
"""


class _AggregateRow(BaseModel):
    """Validates one query_raw row so the handler works with typed values, not Any."""

    date: str
    category: str
    route: str
    successful_requests: int
    failed_requests: int


_ROWS_ADAPTER: Final = TypeAdapter(tuple[_AggregateRow, ...])


def _default_range() -> tuple[str, str]:
    end: Final = datetime.now(timezone.utc)
    start: Final = end - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _fold_by_date(rows: Sequence[_AggregateRow]) -> tuple[GatewayRequestDailyEntry, ...]:
    dates: Final = sorted(frozenset(row.date for row in rows))
    return tuple(
        GatewayRequestDailyEntry(
            date=date,
            successful_requests=sum(row.successful_requests for row in rows if row.date == date),
            failed_requests=sum(row.failed_requests for row in rows if row.date == date),
        )
        for date in dates
    )


def _fold_by_route(rows: Sequence[_AggregateRow]) -> tuple[GatewayRequestBreakdownEntry, ...]:
    pairs: Final = sorted(frozenset((row.category, row.route) for row in rows))
    entries: Final = tuple(
        GatewayRequestBreakdownEntry(
            category=category,
            route=route,
            successful_requests=sum(
                row.successful_requests for row in rows if row.category == category and row.route == route
            ),
            failed_requests=sum(row.failed_requests for row in rows if row.category == category and row.route == route),
        )
        for category, route in pairs
    )
    return tuple(sorted(entries, key=lambda entry: entry.successful_requests, reverse=True))


@router.get(
    "/gateway/daily/activity",
    tags=["Budget & Spend Tracking"],  # mutable-ok: fastapi's decorator signature types tags as a list
    response_model=GatewayRequestActivityResponse,
)
async def get_gateway_daily_activity(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: str | None = Query(default=None, description="Start date in YYYY-MM-DD format"),
    end_date: str | None = Query(default=None, description="End date in YYYY-MM-DD format"),
) -> GatewayRequestActivityResponse:
    """
    Successful and failed gateway requests, counted at the ASGI edge.

    Deployment-wide: the underlying table has no per-key or per-user dimension,
    so this is admin-only.
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Only proxy admin roles can view gateway request counts across the deployment",
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    default_start, default_end = _default_range()
    raw_rows: Final = await prisma_client.db.query_raw(  # pyright: ignore[reportAny]  # untyped prisma client
        _AGGREGATE_SQL,
        start_date or default_start,
        end_date or default_end,
    )
    # Every downstream use is typed: the adapter returns _AggregateRow or raises.
    rows: Final = _ROWS_ADAPTER.validate_python(raw_rows or ())
    verbose_proxy_logger.debug("/gateway/daily/activity - aggregated %d rows", len(rows))

    return GatewayRequestActivityResponse(
        total_successful_requests=sum(row.successful_requests for row in rows),
        total_failed_requests=sum(row.failed_requests for row in rows),
        by_date=_fold_by_date(rows),
        by_route=_fold_by_route(rows),
    )
