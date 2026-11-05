"""Bounded reads of the daily rollups used by the Usage dashboard."""

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import TypeAdapter

from litellm.repositories.daily_activity_buckets import api_key_bucket, api_key_from_bucket, model_bucket
from litellm.types.proxy.model_deprecation import ModelDeprecationUsage

Scope: TypeAlias = Literal["deployment", "key", "user", "team", "model"]


@dataclass(frozen=True, slots=True)
class DailyAlertMetrics:
    entity_id: str
    day: date
    spend: float
    requests: int
    errors: int
    response_time_ms: float
    timed_requests: int


@dataclass(frozen=True, slots=True)
class IssueMetricWindow:
    reference_id: str
    entity_id: str
    start: date
    day: date


@dataclass(frozen=True, slots=True)
class IssueMetricDay:
    reference_id: str
    metrics: DailyAlertMetrics
    keys: int | None
    users: int | None


@dataclass(frozen=True, slots=True)
class IssueKeyAttribution:
    reference_id: str
    key_id: str
    equivalent: bool
    positive_change_share: float
    baseline_value: float
    observed_value: float


@dataclass(frozen=True, slots=True)
class IssueSourceProvider:
    reference_id: str
    provider: str


def _window_payload(scope: Scope, windows: tuple[IssueMetricWindow, ...]) -> str:
    if len(windows) > 500 or any(not 1 <= (window.day - window.start).days <= 30 for window in windows):
        raise ValueError("Read at most 500 issue windows with 1–30 reference days per batch")
    return json.dumps(
        tuple(
            {
                "reference_id": window.reference_id,
                "entity_id": api_key_from_bucket(window.entity_id) if scope == "key" else window.entity_id,
                "start_day": window.start.isoformat(),
                "target_day": window.day.isoformat(),
            }
            for window in windows
        )
    )


_WINDOW_INPUT: Final = """SELECT * FROM jsonb_to_recordset($1::jsonb)
    AS x(reference_id text,entity_id text,start_day text,target_day text)"""


def _window_rows(scope: Scope) -> str:
    table, entity = _source(scope)
    user: Final = "NULL::text" if scope == "team" else "a.user_id"
    return f"""SELECT x.reference_id,x.entity_id,x.start_day,x.target_day,
        COALESCE(a.api_key,'') AS api_key,COALESCE(a.model,'') AS model,
        {user} AS user_id,
        a.date AS day,a.spend,a.api_requests,a.failed_requests,a.total_response_time_ms,a.timed_requests
    FROM input x JOIN {table} a ON {entity}=x.entity_id
        AND a.date>=x.start_day AND a.date<=x.target_day"""


def _key_attribution_query(scope: Scope, metric: Literal["spend", "requests"]) -> str:
    measure: Final = "spend" if metric == "spend" else "api_requests"
    same_identity: Final = {
        "deployment": "true",
        "user": "k.users=1 AND k.user_id=p.entity_id",
        "model": "k.models=1 AND k.model=p.entity_id",
        "team": "k.team_days IS NOT DISTINCT FROM k.user_days",
        "key": "true",
    }[scope]
    team_days: Final = (
        """(SELECT jsonb_agg(to_jsonb(d) ORDER BY d.day) FROM (
            SELECT date AS day,SUM(spend) AS spend,SUM(api_requests)::bigint AS requests,
                SUM(failed_requests)::bigint AS errors,SUM(total_response_time_ms) AS ms,
                SUM(timed_requests)::bigint AS timed
            FROM "LiteLLM_DailyTeamSpend" WHERE api_key=w.api_key AND team_id=w.entity_id
                AND date>=w.start_day AND date<=w.target_day GROUP BY date
        ) d)"""
        if scope == "team"
        else "NULL::jsonb"
    )
    return f"""WITH input AS ({_WINDOW_INPUT}), source_rows AS MATERIALIZED (SELECT raw.*, {measure} AS score FROM ({_window_rows(scope)}) raw),
    population AS (
        SELECT reference_id,MIN(entity_id) AS entity_id,MIN(start_day) AS start_day,MIN(target_day) AS target_day,
            count(DISTINCT api_key) AS key_count,MIN(api_key) AS sole_key,
            count(DISTINCT day) FILTER(WHERE day<target_day) AS history_days,
            COALESCE(SUM(score) FILTER(WHERE day<target_day),0) /
                NULLIF(count(DISTINCT day) FILTER(WHERE day<target_day),0) AS baseline_value,
            COALESCE(SUM(score) FILTER(WHERE day=target_day),0) AS observed_value
        FROM source_rows GROUP BY reference_id
    ), key_growth AS (
        SELECT s.reference_id,s.api_key,
            count(DISTINCT s.day) FILTER(WHERE s.day<s.target_day)=p.history_days AND p.history_days>=3 AS comparable,
            COALESCE(SUM(s.score) FILTER(WHERE s.day=s.target_day),0) AS current_value,
            GREATEST(COALESCE(SUM(s.score) FILTER(WHERE s.day=s.target_day),0) -
                COALESCE(SUM(s.score) FILTER(WHERE s.day<s.target_day),0)/NULLIF(p.history_days,0),0) AS growth
        FROM source_rows s JOIN population p USING(reference_id)
        GROUP BY s.reference_id,s.api_key,p.history_days
    ), ranked AS (
        SELECT *,row_number() OVER(PARTITION BY reference_id ORDER BY
                CASE WHEN comparable THEN growth ELSE 0 END DESC,api_key) AS rank,
            SUM(CASE WHEN comparable THEN growth ELSE 0 END) OVER(PARTITION BY reference_id) AS positive_growth,
            SUM(CASE WHEN comparable THEN current_value ELSE 0 END) OVER(PARTITION BY reference_id) AS covered_value
        FROM key_growth
    ), candidate AS MATERIALIZED (
        SELECT p.*,CASE WHEN p.key_count=1 THEN p.sole_key ELSE r.api_key END AS api_key,
            CASE WHEN r.comparable AND ABS(r.covered_value-p.observed_value)<=1e-8*GREATEST(p.observed_value,1)
                THEN COALESCE(r.growth/NULLIF(r.positive_growth,0),0) ELSE 0 END AS positive_change_share
        FROM population p JOIN ranked r USING(reference_id) WHERE r.rank=1
    ), windows AS MATERIALIZED (
        SELECT DISTINCT api_key,entity_id,start_day,target_day FROM candidate WHERE api_key<>''
    ), key_population AS (
        SELECT w.*,
            (SELECT count(DISTINCT COALESCE(user_id,'')) FROM "LiteLLM_DailyUserSpend"
                WHERE api_key=w.api_key AND date>=w.start_day AND date<=w.target_day) AS users,
            (SELECT MIN(user_id) FROM "LiteLLM_DailyUserSpend"
                WHERE api_key=w.api_key AND date>=w.start_day AND date<=w.target_day) AS user_id,
            (SELECT count(DISTINCT COALESCE(NULLIF(model,''),'unknown-model')) FROM "LiteLLM_DailyUserSpend"
                WHERE api_key=w.api_key AND date>=w.start_day AND date<=w.target_day) AS models,
            (SELECT MIN(COALESCE(NULLIF(model,''),'unknown-model')) FROM "LiteLLM_DailyUserSpend"
                WHERE api_key=w.api_key AND date>=w.start_day AND date<=w.target_day) AS model,
            {team_days} AS team_days,
            (SELECT jsonb_agg(to_jsonb(d) ORDER BY d.day) FROM (
                SELECT date AS day,SUM(spend) AS spend,SUM(api_requests)::bigint AS requests,
                    SUM(failed_requests)::bigint AS errors,SUM(total_response_time_ms) AS ms,
                    SUM(timed_requests)::bigint AS timed
                FROM "LiteLLM_DailyUserSpend" WHERE api_key=w.api_key
                    AND date>=w.start_day AND date<=w.target_day GROUP BY date
            ) d) AS user_days
        FROM windows w
    ) SELECT p.reference_id,p.api_key AS key_id,
        p.key_count=1 AND ({same_identity}) AS equivalent,p.positive_change_share,
        COALESCE(p.baseline_value,0) AS baseline_value,p.observed_value
    FROM candidate p JOIN key_population k ON k.api_key=p.api_key AND k.entity_id=p.entity_id
        AND k.start_day=p.start_day AND k.target_day=p.target_day"""


class MetricsDatabase(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


_ROWS: Final = TypeAdapter(tuple[DailyAlertMetrics, ...])
_BATCH_SIZE: Final = 500
_DEPRECATION_USAGE: Final = TypeAdapter(tuple[ModelDeprecationUsage, ...])

_DEPRECATION_USAGE_QUERY: Final = """
WITH activity AS MATERIALIZED (
    SELECT model, date, api_key, user_id, api_requests FROM "LiteLLM_DailyUserSpend"
    WHERE date >= $1 AND date <= $2 AND model = ANY($3::text[]) AND api_requests > 0
), daily AS (
    SELECT model, date, SUM(api_requests)::bigint AS requests FROM activity GROUP BY model, date
), latest AS (
    SELECT DISTINCT ON (model) model, date AS last_used_date, requests AS last_day_requests
    FROM daily ORDER BY model, date DESC
), totals AS (
    SELECT model, SUM(api_requests)::bigint AS requests,
           COUNT(DISTINCT NULLIF(api_key, ''))::int AS keys,
           COUNT(DISTINCT NULLIF(NULLIF(user_id, ''), 'default_user_id'))::int AS users
    FROM activity GROUP BY model
), teams AS (
    SELECT model, COUNT(DISTINCT NULLIF(NULLIF(team_id, ''), 'litellm-dashboard'))::int AS teams
    FROM "LiteLLM_DailyTeamSpend"
    WHERE date >= $1 AND date <= $2 AND model = ANY($3::text[]) AND api_requests > 0
    GROUP BY model
)
SELECT totals.*, latest.last_used_date, latest.last_day_requests, COALESCE(teams.teams, 0) AS teams
FROM totals JOIN latest USING (model) LEFT JOIN teams USING (model) ORDER BY model
"""


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
        key: Final = api_key_bucket(row.entity_id)
        return row if key == row.entity_id else replace(row, entity_id=key)
    if scope == "model":
        model: Final = model_bucket(row.entity_id)
        return row if model == row.entity_id else replace(row, entity_id=model)
    return row


class UsageAlertMetricsRepository:
    def __init__(self, database: MetricsDatabase) -> None:
        self._database = database

    async def issue_windows(self, scope: Scope, windows: tuple[IssueMetricWindow, ...]) -> tuple[IssueMetricDay, ...]:
        if not windows:
            return ()
        users: Final = (
            "NULL::int"
            if scope == "team"
            else """CASE WHEN bool_or(day=target_day AND COALESCE(user_id,'') IN ('','default_user_id'))
                THEN NULL ELSE count(DISTINCT user_id) FILTER(WHERE day=target_day)::int END"""
        )
        return TypeAdapter(tuple[IssueMetricDay, ...]).validate_python(
            await self._database.query_raw(
                f"""WITH input AS ({_WINDOW_INPUT}), source_rows AS MATERIALIZED ({_window_rows(scope)}),
                impact AS (
                    SELECT reference_id,
                        CASE WHEN bool_or(day=target_day AND api_key='') THEN NULL
                            ELSE count(DISTINCT api_key) FILTER(WHERE day=target_day)::int END AS keys,
                        {users} AS users
                    FROM source_rows GROUP BY reference_id
                ), daily AS (
                    SELECT reference_id,jsonb_build_object(
                        'entity_id',MIN(entity_id),'day',day,'spend',SUM(spend),
                        'requests',SUM(api_requests),'errors',SUM(failed_requests),
                        'response_time_ms',SUM(total_response_time_ms),'timed_requests',SUM(timed_requests)
                    ) AS metrics
                    FROM source_rows GROUP BY reference_id,day
                ) SELECT daily.*,impact.keys,impact.users FROM daily JOIN impact USING(reference_id)
                ORDER BY reference_id,metrics->>'day'""",
                _window_payload(scope, windows),
            )
        )

    async def issue_key_attributions(
        self, scope: Scope, windows: tuple[IssueMetricWindow, ...], metric: Literal["spend", "requests"] = "spend"
    ) -> tuple[IssueKeyAttribution, ...]:
        if not windows:
            return ()
        rows: Final = TypeAdapter(tuple[IssueKeyAttribution, ...]).validate_python(
            await self._database.query_raw(_key_attribution_query(scope, metric), _window_payload(scope, windows))
        )
        return tuple(replace(row, key_id=api_key_bucket(row.key_id)) for row in rows)

    async def issue_source_providers(
        self, scope: Scope, windows: tuple[IssueMetricWindow, ...]
    ) -> tuple[IssueSourceProvider, ...]:
        if not windows:
            return ()
        table, entity = _source(scope)
        return TypeAdapter(tuple[IssueSourceProvider, ...]).validate_python(
            await self._database.query_raw(
                f"""WITH input AS ({_WINDOW_INPUT}), grouped AS (
                    SELECT x.reference_id,a.custom_llm_provider AS provider,SUM(a.timed_requests) AS timed
                    FROM input x JOIN {table} a ON {entity}=x.entity_id AND a.date=x.target_day
                    GROUP BY x.reference_id,a.custom_llm_provider
                ), ranked AS (
                    SELECT *,SUM(timed) OVER(PARTITION BY reference_id) AS total FROM grouped
                ) SELECT reference_id,provider FROM ranked
                WHERE provider<>'' AND timed>0 AND timed>=0.8*total""",
                _window_payload(scope, windows),
            )
        )

    async def deprecation_usage(
        self, models: tuple[str, ...], start: date, end: date
    ) -> tuple[ModelDeprecationUsage, ...]:
        if not models:
            return ()
        if len(models) > _BATCH_SIZE or not 0 <= (end - start).days < 30:
            raise ValueError("Deprecation usage accepts at most 500 models and 30 consecutive dates")
        return _DEPRECATION_USAGE.validate_python(
            await self._database.query_raw(_DEPRECATION_USAGE_QUERY, start.isoformat(), end.isoformat(), models)
        )

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
