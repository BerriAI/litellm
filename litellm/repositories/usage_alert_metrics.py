"""Bounded reads of the daily rollups used by the Usage dashboard."""

from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import date, timedelta
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.repositories.daily_activity_buckets import api_key_bucket, model_bucket

Scope: TypeAlias = Literal["deployment", "key", "user", "team", "model"]


class DailyAlertMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    day: date
    spend: float
    requests: int
    errors: int
    response_time_ms: float
    timed_requests: int


class MetricsDatabase(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


_ROWS: Final = TypeAdapter(tuple[DailyAlertMetrics, ...])
_BATCH_SIZE: Final = 500


def _source(scope: Scope) -> tuple[str, str]:
    if scope == "team":
        return '"LiteLLM_DailyTeamSpend"', "team_id"
    if scope == "user":
        return '"LiteLLM_DailyUserSpend"', "user_id"
    if scope == "key":
        return '"LiteLLM_DailyUserSpend"', "api_key"
    if scope == "model":
        return '"LiteLLM_DailyUserSpend"', "COALESCE(NULLIF(model, ''), 'unknown-model')"
    return '"LiteLLM_DailyUserSpend"', "'deployment'::text"


def _query(scope: Scope) -> str:
    table, entity = _source(scope)
    # Identifiers come only from the closed scope mapping; request values are parameters.
    return f"""
WITH entities AS MATERIALIZED (
    SELECT DISTINCT {entity} AS entity_id FROM {table}
    WHERE date >= $1 AND date <= $2 AND {entity} >= COALESCE($3::text, '') AND ($3::text IS NULL OR {entity} <> $3)
    ORDER BY entity_id LIMIT $4
)
SELECT entities.entity_id, activity.date AS day,
       SUM(activity.spend) AS spend,
       SUM(activity.api_requests)::bigint AS requests,
       SUM(activity.failed_requests)::bigint AS errors,
       SUM(activity.total_response_time_ms) AS response_time_ms,
       SUM(activity.timed_requests)::bigint AS timed_requests
FROM {table} activity JOIN entities ON {entity} = entities.entity_id
WHERE activity.date >= $1 AND activity.date <= $2
GROUP BY entities.entity_id, activity.date
ORDER BY entities.entity_id, activity.date
"""


def _reporting_id(scope: Scope, row: DailyAlertMetrics) -> DailyAlertMetrics:
    if scope == "key":
        return row.model_copy(update={"entity_id": api_key_bucket(row.entity_id)})
    if scope == "model":
        return row.model_copy(update={"entity_id": model_bucket(row.entity_id)})
    return row


class UsageAlertMetricsRepository:
    def __init__(self, database: MetricsDatabase) -> None:
        self._database = database

    async def batches(
        self, scope: Scope, target: date, baseline_days: int
    ) -> AsyncIterator[tuple[DailyAlertMetrics, ...]]:
        start: Final = (target - timedelta(days=baseline_days)).isoformat()
        query: Final = _query(scope)
        cursor: str | None = None  # rebind-ok: keyset pagination retains only the current page
        while rows := _ROWS.validate_python(
            await self._database.query_raw(query, start, target.isoformat(), cursor, _BATCH_SIZE)
        ):
            yield tuple(_reporting_id(scope, row) for row in rows)
            cursor = rows[-1].entity_id
            if scope == "deployment":
                return
