"""
FAILED REQUEST ANALYTICS

GET /gateway/errors/activity - failure rate over time, failures by HTTP status, and
the keys, teams, users and model groups the failures land on.

Totals and the per-status time series come from the edge counters
(LiteLLM_DailyGatewayRequests, LiteLLM_DailyGatewayFailedRequests), which count what
the proxy answered. The caller breakdown reads the daily spend rollups for request
volume and LiteLLM_DailyRequestErrors for the status each caller failed with.
Deployment-wide, so admin-only.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Annotated, Final, Literal, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.request_errors import (
    RequestErrorActivityResponse,
    RequestErrorDailyEntry,
    RequestErrorEntityEntry,
    RequestErrorStatusCodeEntry,
)

router: Final = APIRouter()

_DEFAULT_LOOKBACK_DAYS: Final = 30
_ENTITY_LIMIT: Final = 25

EntityKind: TypeAlias = Literal["key", "team", "user", "model"]

_DATE_SQL: Final = """
    SELECT date, NULL::integer AS status_code,
        SUM(successful_requests)::bigint AS successful_requests,
        SUM(failed_requests)::bigint AS failed_requests
    FROM "LiteLLM_DailyGatewayRequests"
    WHERE date >= $1 AND date <= $2
    GROUP BY date
    UNION ALL
    SELECT date, status_code, 0::bigint, SUM(failed_requests)::bigint
    FROM "LiteLLM_DailyGatewayFailedRequests"
    WHERE date >= $1 AND date <= $2
    GROUP BY date, status_code
"""

_ENTITY_SQL: Final = f"""
    SELECT * FROM (
        SELECT 'key' AS kind, u.api_key AS id, MAX(v.key_alias) AS label,
            SUM(u.api_requests)::bigint AS api_requests, SUM(u.failed_requests)::bigint AS failed_requests
        FROM "LiteLLM_DailyUserSpend" u
        LEFT JOIN "LiteLLM_VerificationToken" v ON v.token = u.api_key
        WHERE u.date >= $1 AND u.date <= $2
        GROUP BY u.api_key HAVING SUM(u.failed_requests) > 0
        ORDER BY failed_requests DESC, id LIMIT {_ENTITY_LIMIT}
    ) keys
    UNION ALL
    SELECT * FROM (
        SELECT 'team' AS kind, t.team_id AS id, MAX(tt.team_alias) AS label,
            SUM(t.api_requests)::bigint, SUM(t.failed_requests)::bigint AS failed_requests
        FROM "LiteLLM_DailyTeamSpend" t
        LEFT JOIN "LiteLLM_TeamTable" tt ON tt.team_id = t.team_id
        WHERE t.date >= $1 AND t.date <= $2 AND t.team_id IS NOT NULL AND t.team_id <> ''
        GROUP BY t.team_id HAVING SUM(t.failed_requests) > 0
        ORDER BY failed_requests DESC, id LIMIT {_ENTITY_LIMIT}
    ) teams
    UNION ALL
    SELECT * FROM (
        SELECT 'user' AS kind, u.user_id AS id, MAX(ut.user_email) AS label,
            SUM(u.api_requests)::bigint, SUM(u.failed_requests)::bigint AS failed_requests
        FROM "LiteLLM_DailyUserSpend" u
        LEFT JOIN "LiteLLM_UserTable" ut ON ut.user_id = u.user_id
        WHERE u.date >= $1 AND u.date <= $2 AND u.user_id IS NOT NULL AND u.user_id <> ''
        GROUP BY u.user_id HAVING SUM(u.failed_requests) > 0
        ORDER BY failed_requests DESC, id LIMIT {_ENTITY_LIMIT}
    ) users
    UNION ALL
    SELECT * FROM (
        SELECT 'model' AS kind, u.model_group AS id, NULL::text AS label,
            SUM(u.api_requests)::bigint, SUM(u.failed_requests)::bigint AS failed_requests
        FROM "LiteLLM_DailyUserSpend" u
        WHERE u.date >= $1 AND u.date <= $2 AND u.model_group IS NOT NULL AND u.model_group <> ''
        GROUP BY u.model_group HAVING SUM(u.failed_requests) > 0
        ORDER BY failed_requests DESC, id LIMIT {_ENTITY_LIMIT}
    ) models
"""

_ENTITY_STATUS_SQL: Final = f"""
    SELECT * FROM (
        SELECT DISTINCT ON (api_key) 'key' AS kind, api_key AS id, status_code, SUM(failed_requests)::bigint AS failed_requests
        FROM "LiteLLM_DailyRequestErrors" WHERE date >= $1 AND date <= $2
            AND api_key IN (SELECT u.api_key FROM "LiteLLM_DailyUserSpend" u WHERE u.date >= $1 AND u.date <= $2 GROUP BY u.api_key HAVING SUM(u.failed_requests) > 0 ORDER BY SUM(u.failed_requests) DESC, u.api_key LIMIT {_ENTITY_LIMIT})
        GROUP BY api_key, status_code ORDER BY api_key, failed_requests DESC, status_code
    ) key_status
    UNION ALL
    SELECT * FROM (
        SELECT DISTINCT ON (team_id) 'team' AS kind, team_id AS id, status_code, SUM(failed_requests)::bigint AS failed_requests
        FROM "LiteLLM_DailyRequestErrors" WHERE date >= $1 AND date <= $2 AND team_id <> ''
            AND team_id IN (SELECT t.team_id FROM "LiteLLM_DailyTeamSpend" t WHERE t.date >= $1 AND t.date <= $2 AND t.team_id IS NOT NULL AND t.team_id <> '' GROUP BY t.team_id HAVING SUM(t.failed_requests) > 0 ORDER BY SUM(t.failed_requests) DESC, t.team_id LIMIT {_ENTITY_LIMIT})
        GROUP BY team_id, status_code ORDER BY team_id, failed_requests DESC, status_code
    ) team_status
    UNION ALL
    SELECT * FROM (
        SELECT DISTINCT ON (user_id) 'user' AS kind, user_id AS id, status_code, SUM(failed_requests)::bigint AS failed_requests
        FROM "LiteLLM_DailyRequestErrors" WHERE date >= $1 AND date <= $2 AND user_id <> ''
            AND user_id IN (SELECT u.user_id FROM "LiteLLM_DailyUserSpend" u WHERE u.date >= $1 AND u.date <= $2 AND u.user_id IS NOT NULL AND u.user_id <> '' GROUP BY u.user_id HAVING SUM(u.failed_requests) > 0 ORDER BY SUM(u.failed_requests) DESC, u.user_id LIMIT {_ENTITY_LIMIT})
        GROUP BY user_id, status_code ORDER BY user_id, failed_requests DESC, status_code
    ) user_status
    UNION ALL
    SELECT * FROM (
        SELECT DISTINCT ON (model_group) 'model' AS kind, model_group AS id, status_code, SUM(failed_requests)::bigint AS failed_requests
        FROM "LiteLLM_DailyRequestErrors" WHERE date >= $1 AND date <= $2 AND model_group <> ''
            AND model_group IN (SELECT u.model_group FROM "LiteLLM_DailyUserSpend" u WHERE u.date >= $1 AND u.date <= $2 AND u.model_group IS NOT NULL AND u.model_group <> '' GROUP BY u.model_group HAVING SUM(u.failed_requests) > 0 ORDER BY SUM(u.failed_requests) DESC, u.model_group LIMIT {_ENTITY_LIMIT})
        GROUP BY model_group, status_code ORDER BY model_group, failed_requests DESC, status_code
    ) model_status
"""


class _DateRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    date: str
    status_code: int | None = None
    successful_requests: int = 0
    failed_requests: int = 0


class _EntityRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    kind: EntityKind
    id: str
    label: str | None = None
    api_requests: int = 0
    failed_requests: int = 0


class _EntityStatusRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    kind: EntityKind
    id: str
    status_code: int
    failed_requests: int = 0


_DATE_ROWS: Final = TypeAdapter(tuple[_DateRow, ...])
_ENTITY_ROWS: Final = TypeAdapter(tuple[_EntityRow, ...])
_ENTITY_STATUS_ROWS: Final = TypeAdapter(tuple[_EntityStatusRow, ...])


def _default_range() -> tuple[str, str]:
    end: Final = datetime.now(timezone.utc)
    start: Final = end - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _is_client_error(status_code: int | None) -> bool:
    return status_code is not None and 400 <= status_code <= 499


def _is_server_error(status_code: int | None) -> bool:
    return status_code is not None and 500 <= status_code <= 599


def _day_entry(date: str, rows: Sequence[_DateRow]) -> RequestErrorDailyEntry:
    day_rows: Final = tuple(row for row in rows if row.date == date)
    return RequestErrorDailyEntry(
        date=date,
        successful_requests=sum(row.successful_requests for row in day_rows),
        failed_requests=sum(row.failed_requests for row in day_rows if row.status_code is None),
        client_errors=sum(row.failed_requests for row in day_rows if _is_client_error(row.status_code)),
        server_errors=sum(row.failed_requests for row in day_rows if _is_server_error(row.status_code)),
        by_status_code=fold_by_status_code(day_rows),
    )


def fold_by_date(rows: Sequence[_DateRow]) -> tuple[RequestErrorDailyEntry, ...]:
    """Edge totals per day with the 4xx and 5xx share of the failures and the per-status counts."""
    return tuple(_day_entry(date, rows) for date in sorted(frozenset(row.date for row in rows)))


def fold_by_status_code(rows: Sequence[_DateRow]) -> tuple[RequestErrorStatusCodeEntry, ...]:
    codes: Final = frozenset(row.status_code for row in rows if row.status_code is not None)
    entries: Final = tuple(
        RequestErrorStatusCodeEntry(
            status_code=code, failed_requests=sum(row.failed_requests for row in rows if row.status_code == code)
        )
        for code in codes
    )
    return tuple(sorted(entries, key=lambda entry: (-entry.failed_requests, entry.status_code)))


def _entity_entry(row: _EntityRow, top_statuses: Mapping[tuple[str, str], _EntityStatusRow]) -> RequestErrorEntityEntry:
    top: Final = top_statuses.get((row.kind, row.id))
    return RequestErrorEntityEntry(
        id=row.id,
        label=row.label,
        api_requests=row.api_requests,
        failed_requests=row.failed_requests,
        top_status_code=top.status_code if top is not None else None,
        top_status_code_requests=top.failed_requests if top is not None else 0,
    )


def fold_entities(
    kind: EntityKind, rows: Sequence[_EntityRow], status_rows: Sequence[_EntityStatusRow]
) -> tuple[RequestErrorEntityEntry, ...]:
    """Callers of one kind ranked by failures, each with the status they failed with most (one row per caller)."""
    top_statuses: Final = {(status.kind, status.id): status for status in status_rows}
    entries: Final = tuple(_entity_entry(row, top_statuses) for row in filter(lambda row: row.kind == kind, rows))
    return tuple(sorted(entries, key=lambda entry: (-entry.failed_requests, entry.id)))


@router.get(
    "/gateway/errors/activity",
    tags=["Budget & Spend Tracking"],
    response_model=RequestErrorActivityResponse,
)
async def get_request_error_activity(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: str | None = Query(default=None, description="Start date in YYYY-MM-DD format"),
    end_date: str | None = Query(default=None, description="End date in YYYY-MM-DD format"),
) -> RequestErrorActivityResponse:
    """
    Failed requests over time by HTTP status, and the keys, teams, users and
    model groups they land on. Deployment-wide, so admin-only.
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise HTTPException(
            status_code=403, detail="Only proxy admin roles can view failed requests across the deployment"
        )
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    default_start, default_end = _default_range()
    selected_start: Final = start_date or default_start
    selected_end: Final = end_date or default_end
    db: Final = prisma_client.db
    date_rows: Final = _DATE_ROWS.validate_python(
        await db.query_raw(_DATE_SQL, selected_start, selected_end) or ()  # pyright: ignore[reportAny]  # untyped prisma client
    )
    entity_rows: Final = _ENTITY_ROWS.validate_python(
        await db.query_raw(_ENTITY_SQL, selected_start, selected_end) or ()  # pyright: ignore[reportAny]  # untyped prisma client
    )
    status_rows: Final = _ENTITY_STATUS_ROWS.validate_python(
        await db.query_raw(_ENTITY_STATUS_SQL, selected_start, selected_end) or ()  # pyright: ignore[reportAny]  # untyped prisma client
    )
    verbose_proxy_logger.debug(
        "/gateway/errors/activity - %d date rows, %d entity rows, %d status rows",
        len(date_rows),
        len(entity_rows),
        len(status_rows),
    )
    totals: Final = tuple(row for row in date_rows if row.status_code is None)
    return RequestErrorActivityResponse(
        total_successful_requests=sum(row.successful_requests for row in totals),
        total_failed_requests=sum(row.failed_requests for row in totals),
        by_date=fold_by_date(date_rows),
        by_status_code=fold_by_status_code(date_rows),
        by_key=fold_entities("key", entity_rows, status_rows),
        by_team=fold_entities("team", entity_rows, status_rows),
        by_user=fold_entities("user", entity_rows, status_rows),
        by_model=fold_entities("model", entity_rows, status_rows),
    )
