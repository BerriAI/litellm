"""
COST SAVINGS READ ENDPOINTS

GET /v1/savings/hourly - prompt-caching and compression savings bucketed by hour

The daily rollup tables (`LiteLLM_DailyUserSpend` and friends) are keyed by a
`YYYY-MM-DD` string, so a one-day range renders as a single point. This endpoint
recomputes the same two savings drivers straight from `LiteLLM_SpendLogs` at
hour granularity, using the same extraction rules the daily writer uses, so the
hourly buckets for a day sum to that day's rollup row.
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

_HOURLY_SAVINGS_QUERY = f"""
SELECT
    to_char(
        date_trunc('hour', sl."startTime" + make_interval(mins => $3::int)),
        'YYYY-MM-DD"T"HH24:00'
    ) AS bucket_start,
    sl.model AS model,
    sl.custom_llm_provider AS custom_llm_provider,
    SUM({CACHE_READ_INPUT_TOKENS_SQL})::bigint AS cache_read_input_tokens,
    SUM({COMPRESSION_SAVED_TOKENS_SQL})::bigint AS compression_saved_tokens
FROM "LiteLLM_SpendLogs" sl
WHERE sl."startTime" >= $1::timestamp
  AND sl."startTime" < $2::timestamp
GROUP BY bucket_start, sl.model, sl.custom_llm_provider
"""


class _HourlySavingsRow(BaseModel):
    bucket_start: str
    model: str | None
    custom_llm_provider: str | None
    cache_read_input_tokens: int
    compression_saved_tokens: int


_HOURLY_SAVINGS_ROWS = TypeAdapter(list[_HourlySavingsRow])


def _parse_day(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {value}. Expected: 'YYYY-MM-DD'")


def _bucket_label(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:00")


def hour_bucket_labels(local_start: datetime, local_end_exclusive: datetime) -> tuple[str, ...]:
    """Every hour in the requested window, so quiet hours plot as $0 instead of a gap."""
    hours = int((local_end_exclusive - local_start).total_seconds() // 3600)
    return tuple(_bucket_label(local_start + timedelta(hours=index)) for index in range(hours))


def build_hourly_buckets(rows: list[_HourlySavingsRow], bucket_labels: tuple[str, ...]) -> list[HourlySavingsBucket]:
    """
    Price each (hour, model, provider) group and fold the groups into one series
    per hour. Pricing cannot happen before this point: the rows are grouped by
    model precisely because `compute_savings_spend` needs the model's own rates.
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
    by_bucket: dict[str, tuple[SavingsSpend, ...]] = {
        label: tuple(spend for _, spend in group) for label, group in groupby(priced, key=itemgetter(0))
    }
    return [
        HourlySavingsBucket(
            bucket_start=label,
            compression_savings_spend=sum(spend.compression for spend in by_bucket.get(label, ())),
            prompt_caching_savings_spend=sum(spend.prompt_caching for spend in by_bucket.get(label, ())),
        )
        for label in bucket_labels
    ]


@router.get(
    "/v1/savings/hourly",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=HourlySavingsResponse,
)
async def get_hourly_savings(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[str, Query(description="YYYY-MM-DD, read in the caller's local timezone")],
    end_date: Annotated[str, Query(description="YYYY-MM-DD, inclusive, read in the caller's local timezone")],
    utc_offset_minutes: Annotated[
        int,
        Query(
            ge=-1440,
            le=1440,
            description="Caller's offset from UTC, so hours land on the caller's clock. -new Date().getTimezoneOffset()",
        ),
    ] = 0,
):
    """
    Prompt-caching and compression savings bucketed by hour, for the Cost
    Optimization dashboard's short date ranges.

    Aggregated in Postgres over `LiteLLM_SpendLogs` and priced per model here,
    because token counts cannot be priced once they have been summed across
    models. Every hour in the window is returned, including the empty ones.
    Capped at `MAX_HOURLY_SPAN_DAYS` days; longer ranges belong on the daily
    rollup, which needs no raw-log scan.
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

    bucket_labels = hour_bucket_labels(local_start, local_end_exclusive)
    if disable_spend_logs:
        return HourlySavingsResponse(
            buckets=build_hourly_buckets([], bucket_labels),
            start_date=start_date,
            end_date=end_date,
            utc_offset_minutes=utc_offset_minutes,
            spend_logs_disabled=True,
        )

    offset = timedelta(minutes=utc_offset_minutes)
    rows = await prisma_client.db.query_raw(
        _HOURLY_SAVINGS_QUERY,
        (local_start - offset).isoformat(),
        (local_end_exclusive - offset).isoformat(),
        utc_offset_minutes,
    )
    return HourlySavingsResponse(
        buckets=build_hourly_buckets(_HOURLY_SAVINGS_ROWS.validate_python(rows or []), bucket_labels),
        start_date=start_date,
        end_date=end_date,
        utc_offset_minutes=utc_offset_minutes,
        spend_logs_disabled=False,
    )
