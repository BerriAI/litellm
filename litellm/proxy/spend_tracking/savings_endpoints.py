"""
COST SAVINGS READ ENDPOINTS

GET /v1/savings/hourly - prompt-caching and compression savings bucketed by hour

The daily rollup tables (`LiteLLM_DailyUserSpend` and friends) are keyed by a
`YYYY-MM-DD` string, so a one-day range renders as a single point. This endpoint
recomputes the same two savings drivers straight from `LiteLLM_SpendLogs` at
hour granularity, using the same extraction rules the daily writer uses, so the
hourly buckets for a day sum to that day's rollup row.

Buckets land on the caller's clock. The caller passes an IANA timezone name
(e.g. ``America/Los_Angeles``) rather than a fixed UTC offset, so Postgres
resolves the offset that was actually in effect at each request's own instant.
A fixed offset would misplace any historical day whose daylight-saving offset
differs from today's, and no single offset can be right for a range that
straddles a DST transition.
"""

from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, TypeAdapter

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.spend_tracking.cache_savings import CACHE_READ_INPUT_TOKENS_SQL
from litellm.proxy.spend_tracking.compression_savings import COMPRESSION_SAVED_TOKENS_SQL
from litellm.proxy.spend_tracking.savings import SavingsSpend, compute_savings_spend
from litellm.types.savings import HourlySavingsBucket, HourlySavingsResponse

router = APIRouter()

MAX_HOURLY_SPAN_DAYS = 7

# Postgres is the single timezone authority: it converts each UTC-stored
# `startTime` to the caller's local wall clock with `AT TIME ZONE $3` (which
# honours DST per instant), derives the UTC scan window from the local day
# bounds the same way (kept sargable — the tz expressions fold to constants, so
# the `startTime` index still drives the scan), and emits one zero row per hour
# in the window via `generate_series` so quiet hours plot as $0 instead of a
# gap. Walking the spine in UTC and labelling in local time is what makes it
# DST-correct: a spring-forward day yields 23 labels, a fall-back day 24.
_HOURLY_SAVINGS_QUERY = f"""
SELECT
    bucket_start,
    model,
    custom_llm_provider,
    SUM(cache_read_input_tokens)::bigint AS cache_read_input_tokens,
    SUM(compression_saved_tokens)::bigint AS compression_saved_tokens
FROM (
    SELECT
        to_char(
            date_trunc('hour', (sl."startTime" AT TIME ZONE 'UTC') AT TIME ZONE $3::text),
            'YYYY-MM-DD"T"HH24:00'
        ) AS bucket_start,
        sl.model AS model,
        sl.custom_llm_provider AS custom_llm_provider,
        {CACHE_READ_INPUT_TOKENS_SQL} AS cache_read_input_tokens,
        {COMPRESSION_SAVED_TOKENS_SQL} AS compression_saved_tokens
    FROM "LiteLLM_SpendLogs" sl
    WHERE sl."startTime" >= (($1::timestamp AT TIME ZONE $3::text) AT TIME ZONE 'UTC')
      AND sl."startTime" <  (($2::timestamp AT TIME ZONE $3::text) AT TIME ZONE 'UTC')

    UNION ALL

    SELECT
        to_char(
            date_trunc('hour', (hour_start AT TIME ZONE 'UTC') AT TIME ZONE $3::text),
            'YYYY-MM-DD"T"HH24:00'
        ) AS bucket_start,
        NULL AS model,
        NULL AS custom_llm_provider,
        0 AS cache_read_input_tokens,
        0 AS compression_saved_tokens
    FROM generate_series(
        ($1::timestamp AT TIME ZONE $3::text) AT TIME ZONE 'UTC',
        (($2::timestamp AT TIME ZONE $3::text) AT TIME ZONE 'UTC') - interval '1 hour',
        interval '1 hour'
    ) AS hour_start
) buckets
GROUP BY bucket_start, model, custom_llm_provider
"""

# Validated against the same catalog Postgres buckets with, so a name it accepts
# here is a name `AT TIME ZONE` will accept in the query above.
_TIMEZONE_EXISTS_QUERY = "SELECT EXISTS (SELECT 1 FROM pg_timezone_names WHERE name = $1) AS ok"


class _HourlySavingsRow(BaseModel):
    bucket_start: str
    model: str | None
    custom_llm_provider: str | None
    cache_read_input_tokens: int
    compression_saved_tokens: int


class _TimezoneExistsRow(BaseModel):
    ok: bool


_HOURLY_SAVINGS_ROWS = TypeAdapter(list[_HourlySavingsRow])
_TIMEZONE_EXISTS_ROWS = TypeAdapter(list[_TimezoneExistsRow])


def _parse_day(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {value}. Expected: 'YYYY-MM-DD'")


def build_hourly_buckets(rows: list[_HourlySavingsRow]) -> list[HourlySavingsBucket]:
    """
    Price each (hour, model, provider) group and fold the groups into one series
    per hour. Pricing cannot happen before this point: the rows are grouped by
    model precisely because `compute_savings_spend` needs the model's own rates.
    Every hour in the window is already present, including the empty ones, via
    the query's zero-row spine, so the hour labels come straight from the rows.
    """
    priced = sorted(
        (
            (
                row.bucket_start,
                compute_savings_spend(
                    model=row.model,
                    custom_llm_provider=row.custom_llm_provider,
                    compression_saved_tokens=row.compression_saved_tokens,
                    cache_read_input_tokens=row.cache_read_input_tokens,
                ),
            )
            for row in rows
        ),
        key=itemgetter(0),
    )
    grouped: list[tuple[str, tuple[SavingsSpend, ...]]] = [
        (label, tuple(spend for _, spend in group)) for label, group in groupby(priced, key=itemgetter(0))
    ]
    return [
        HourlySavingsBucket(
            bucket_start=label,
            compression_savings_spend=sum(spend.compression for spend in spends),
            prompt_caching_savings_spend=sum(spend.prompt_caching for spend in spends),
        )
        for label, spends in grouped
    ]


@router.get(
    "/v1/savings/hourly",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=HourlySavingsResponse,
)
async def get_hourly_savings(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str, Query(description="YYYY-MM-DD, read in the caller's timezone")],
    end_date: Annotated[str, Query(description="YYYY-MM-DD, inclusive, read in the caller's timezone")],
    timezone: Annotated[
        str,
        Query(
            description="IANA timezone name, so hours land on the caller's clock. Intl...resolvedOptions().timeZone",
        ),
    ] = "UTC",
):
    """
    Prompt-caching and compression savings bucketed by hour, for the Cost
    Optimization dashboard's short date ranges.

    Aggregated in Postgres over `LiteLLM_SpendLogs` and priced per model here,
    because token counts cannot be priced once they have been summed across
    models. Buckets are on the caller's clock via the IANA `timezone`, so
    historical days carry the offset that was in effect then rather than
    today's. Every hour in the window is returned, including empty ones. Capped
    at `MAX_HOURLY_SPAN_DAYS` days; longer ranges belong on the daily rollup,
    which needs no raw-log scan.
    """
    from litellm.proxy.proxy_server import disable_spend_logs, prisma_client

    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Only proxy admin roles can view hourly savings across the deployment",
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    local_start = _parse_day(start_date)
    local_end_exclusive = _parse_day(end_date) + timedelta(days=1)
    if local_end_exclusive <= local_start:
        raise HTTPException(status_code=400, detail=f"end_date {end_date} is before start_date {start_date}")
    if local_end_exclusive - local_start > timedelta(days=MAX_HOURLY_SPAN_DAYS):
        raise HTTPException(
            status_code=400,
            detail=f"Hourly savings are capped at {MAX_HOURLY_SPAN_DAYS} days; use the daily rollup for longer ranges",
        )

    if disable_spend_logs:
        return HourlySavingsResponse(
            buckets=[],
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
            spend_logs_disabled=True,
        )

    tz_rows = _TIMEZONE_EXISTS_ROWS.validate_python(await prisma_client.db.query_raw(_TIMEZONE_EXISTS_QUERY, timezone))
    if not tz_rows or not tz_rows[0].ok:
        raise HTTPException(status_code=400, detail=f"Unknown timezone: {timezone}")

    rows = await prisma_client.db.query_raw(
        _HOURLY_SAVINGS_QUERY,
        local_start.isoformat(),
        local_end_exclusive.isoformat(),
        timezone,
    )
    return HourlySavingsResponse(
        buckets=build_hourly_buckets(_HOURLY_SAVINGS_ROWS.validate_python(rows or [])),
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
        spend_logs_disabled=False,
    )
