"""Confirm shared latency changes using existing daily key/model counters."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Final, Literal, cast  # noqa: TID251  # generated Prisma structural interface

from pydantic import BaseModel, TypeAdapter

from litellm.constants import PTU_SENTINEL_API_KEY
from litellm.repositories.daily_activity_buckets import api_key_bucket, api_key_from_bucket
from litellm.repositories.usage_alert_metrics import MetricsDatabase
from litellm.types.proxy.issues import IssueCohort, IssueComparison, IssueImpact, IssueModelPair
from litellm.types.proxy.usage_alerts import UsageAlertRuleInput

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class ProviderLatencyEvidence:
    cohort: IssueCohort
    comparison: IssueComparison
    impact: IssueImpact
    comparable_timed_requests: int
    total_timed_requests: int


class _CohortRow(BaseModel, frozen=True):
    pairs: tuple[IssueModelPair, ...]
    observed: float
    expected: float
    sample_days: int | None
    comparable_timed: int
    total_timed: int
    keys: int
    users: int
    teams: int
    requests: int
    measured_keys: int


class _Member(BaseModel, frozen=True):
    entity_id: str


class _Recovery(BaseModel, frozen=True):
    recovered: bool


_PAIRS: Final = TypeAdapter(tuple[IssueModelPair, ...])
_COHORTS: Final = TypeAdapter(tuple[_CohortRow, ...])
_MEMBERS: Final = TypeAdapter(tuple[_Member, ...])
_CANDIDATES: Final = """
SELECT DISTINCT custom_llm_provider AS provider, model FROM "LiteLLM_DailyUserSpend"
WHERE date=$1 AND timed_requests>0 AND custom_llm_provider<>'' AND model<>''
  AND ($2::text IS NULL OR (custom_llm_provider,model)>($2,$3))
ORDER BY provider,model LIMIT $4
"""
_LATENCY: Final = """
WITH pairs AS (
    SELECT * FROM UNNEST($3::text[], $4::text[]) AS p(custom_llm_provider,model)
), daily AS MATERIALIZED (
    SELECT custom_llm_provider AS provider,model,api_key,date,
           SUM(timed_requests)::bigint AS timed,
           SUM(total_response_time_ms)::double precision AS response_ms,
           SUM(api_requests)::bigint AS requests
    FROM "LiteLLM_DailyUserSpend" JOIN pairs USING(custom_llm_provider,model)
    WHERE date >= $1 AND date <= $2
      AND ($10::text[] IS NULL OR api_key=ANY($10))
      AND ($11::text[] IS NULL OR model=ANY($11))
    GROUP BY custom_llm_provider,model,api_key,date
), history AS (
    SELECT provider,model,api_key,COUNT(*) FILTER(WHERE timed>0)::int AS sample_days,
           SUM(timed)::double precision AS timed,
           SUM(response_ms)/NULLIF(SUM(timed),0) AS baseline
    FROM daily WHERE date<$2 GROUP BY provider,model,api_key
), current AS (
    SELECT d.*,h.sample_days,h.baseline,
           COALESCE(h.sample_days >= $5 AND h.baseline>0
             AND h.timed/NULLIF(h.sample_days,0) >= $6 AND d.timed >= $6 AND d.timed>0
             AND d.api_key<>'' AND d.api_key<>$9, FALSE) AS comparable
    FROM daily d LEFT JOIN history h USING(provider,model,api_key) WHERE d.date=$2
), scored AS MATERIALIZED (
    SELECT *, comparable AND response_ms/NULLIF(timed,0) >= GREATEST(baseline*$7,baseline+$8) AS degraded
    FROM current
)
"""
_MODEL_SCORES: Final = """
, model_scores AS (
    SELECT provider,model,SUM(timed)::bigint AS total_timed,
           SUM(timed) FILTER(WHERE comparable)::bigint AS comparable_timed,
           SUM(response_ms) FILTER(WHERE comparable) AS observed_ms,
           SUM(timed*baseline) FILTER(WHERE comparable) AS expected_ms,
           MIN(sample_days) FILTER(WHERE comparable) AS min_days,
           MAX(sample_days) FILTER(WHERE comparable) AS max_days
    FROM scored GROUP BY provider,model
    HAVING COUNT(DISTINCT api_key) FILTER(WHERE degraded)>=2
       AND SUM(timed) FILTER(WHERE degraded) >= 0.8*SUM(timed)
)
"""
_SHARED_THRESHOLD: Final = """
HAVING COUNT(*)>=2 AND SUM(observed_ms)/SUM(comparable_timed) >=
    GREATEST(SUM(expected_ms)/SUM(comparable_timed)*$7,SUM(expected_ms)/SUM(comparable_timed)+$8)
"""
_COHORT: Final = (
    _LATENCY
    + _MODEL_SCORES
    + """
, selected AS MATERIALIZED (
    SELECT s.* FROM scored s JOIN model_scores USING(provider,model)
)
SELECT jsonb_agg(jsonb_build_object('provider',provider,'model',model) ORDER BY provider,model) AS pairs,
       SUM(observed_ms)/SUM(comparable_timed) AS observed,
       SUM(expected_ms)/SUM(comparable_timed) AS expected,
       CASE WHEN MIN(min_days)=MAX(max_days) THEN MIN(min_days) ELSE NULL END AS sample_days,
       SUM(comparable_timed)::bigint AS comparable_timed,SUM(total_timed)::bigint AS total_timed,
       (SELECT COUNT(DISTINCT api_key)::int FROM selected WHERE api_key<>'' AND api_key<>$9) AS keys,
       (SELECT COUNT(DISTINCT api_key)::int FROM selected WHERE degraded) AS measured_keys,
       (SELECT SUM(requests)::bigint FROM selected) AS requests,
       (SELECT COUNT(DISTINCT NULLIF(NULLIF(user_id,''),'default_user_id'))::int
        FROM "LiteLLM_DailyUserSpend" d JOIN model_scores s ON d.custom_llm_provider=s.provider AND d.model=s.model
        WHERE d.date=$2 AND api_requests>0) AS users,
       (SELECT COUNT(DISTINCT NULLIF(NULLIF(team_id,''),'litellm-dashboard'))::int
        FROM "LiteLLM_DailyTeamSpend" d JOIN model_scores s ON d.custom_llm_provider=s.provider AND d.model=s.model
        WHERE d.date=$2 AND api_requests>0) AS teams
FROM model_scores
"""
    + _SHARED_THRESHOLD
)
_RECOVERY_QUERY: Final = (
    _LATENCY
    + _MODEL_SCORES
    + """
, covered AS (
    SELECT provider,model FROM scored GROUP BY provider,model
    HAVING COUNT(DISTINCT api_key) FILTER(WHERE comparable)>=2
       AND SUM(timed) FILTER(WHERE comparable)>=0.8*SUM(timed)
)
SELECT (SELECT COUNT(*) FROM covered)=cardinality($4::text[]) AND NOT EXISTS (
    SELECT 1 FROM model_scores
"""
    + _SHARED_THRESHOLD
    + ") AS recovered"
)


def _arguments(
    pairs: tuple[IssueModelPair, ...],
    day: date,
    rule: UsageAlertRuleInput,
    keys: tuple[str, ...] | None = None,
    models: tuple[str, ...] | None = None,
) -> tuple[object, ...]:
    if not 1 <= len(pairs) <= 100 or len(frozenset(pairs)) != len(pairs):
        raise ValueError("Choose 1 to 100 unique provider/model pairs")
    if len({pair.provider for pair in pairs}) != 1:
        raise ValueError("A latency cohort belongs to one provider")
    if rule.metric != "latency" or rule.direction != "increase":
        raise ValueError("Provider latency evidence requires an increasing latency rule")
    return (
        (day - timedelta(days=rule.baseline_days)).isoformat(),
        day.isoformat(),
        tuple(pair.provider for pair in pairs),
        tuple(pair.model for pair in pairs),
        rule.min_history_days,
        rule.min_requests,
        rule.multiplier,
        rule.min_value,
        PTU_SENTINEL_API_KEY,
        keys,
        models,
    )


class ProviderLatencyRepository:
    def __init__(self, database: MetricsDatabase) -> None:
        self._database: Final = database

    @classmethod
    def from_prisma(cls, prisma: "PrismaClient") -> "ProviderLatencyRepository":
        return cls(cast(MetricsDatabase, prisma.writer_db))  # cast-ok: generated Prisma query_raw interface

    async def candidate_pairs(
        self, day: date, cursor: IssueModelPair | None = None, limit: int = 100
    ) -> tuple[IssueModelPair, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("Pair pages contain 1 to 100 candidates")
        return _PAIRS.validate_python(
            await self._database.query_raw(
                _CANDIDATES,
                day.isoformat(),
                cursor.provider if cursor else None,
                cursor.model if cursor else None,
                limit,
            )
        )

    async def latency_cohort(
        self, pairs: tuple[IssueModelPair, ...], day: date, rule: UsageAlertRuleInput
    ) -> ProviderLatencyEvidence | None:
        arguments: Final = _arguments(pairs, day, rule)
        if len(pairs) < 2:
            return None
        rows: Final = _COHORTS.validate_python(await self._database.query_raw(_COHORT, *arguments))
        if not rows:
            return None
        row: Final = rows[0]
        return ProviderLatencyEvidence(
            cohort=IssueCohort(scope_type="provider", scope_id=pairs[0].provider, model_pairs=row.pairs),
            comparison=IssueComparison(
                metric="latency",
                direction="increase",
                observed_value=row.observed,
                baseline_value=row.expected,
                threshold=max(row.expected * rule.multiplier, row.expected + rule.min_value),
                baseline_start=day - timedelta(days=rule.baseline_days),
                baseline_end=day - timedelta(days=1),
                baseline_sample_days=row.sample_days,
            ),
            impact=IssueImpact(
                keys=row.keys,
                users=row.users,
                teams=row.teams,
                models=len(row.pairs),
                requests=row.requests,
                measured_keys=row.measured_keys,
                measured_models=len(row.pairs),
            ),
            comparable_timed_requests=row.comparable_timed,
            total_timed_requests=row.total_timed,
        )

    async def source_members(
        self,
        pairs: tuple[IssueModelPair, ...],
        day: date,
        rule: UsageAlertRuleInput,
        scope_type: Literal["key", "model"],
        source_ids: tuple[str, ...],
    ) -> frozenset[str]:
        if not source_ids:
            return frozenset()
        if len(source_ids) > 500:
            raise ValueError("Source membership accepts at most 500 IDs")
        keys: Final = tuple(api_key_from_bucket(key) for key in source_ids) if scope_type == "key" else None
        arguments: Final = _arguments(pairs, day, rule, keys, source_ids if scope_type == "model" else None)
        column, parameter = ("api_key", 10) if scope_type == "key" else ("model", 11)
        query: Final = (
            _LATENCY
            + f"""
        , totals AS (
            SELECT {column},SUM(timed_requests)::bigint AS total
            FROM "LiteLLM_DailyUserSpend" WHERE date=$2 AND {column}=ANY(${parameter}) GROUP BY {column}
        )
        SELECT s.{column} AS entity_id FROM scored s JOIN totals USING({column}) GROUP BY s.{column},totals.total
        HAVING SUM(timed) FILTER(WHERE degraded) >= 0.8*totals.total AND totals.total>0
        """
        )
        rows: Final = _MEMBERS.validate_python(await self._database.query_raw(query, *arguments))
        return frozenset(api_key_bucket(row.entity_id) if scope_type == "key" else row.entity_id for row in rows)

    async def cohort_recovered(self, pairs: tuple[IssueModelPair, ...], day: date, rule: UsageAlertRuleInput) -> bool:
        arguments: Final = _arguments(pairs, day, rule)
        rows: Final = TypeAdapter(tuple[_Recovery, ...]).validate_python(
            await self._database.query_raw(_RECOVERY_QUERY, *arguments)
        )
        return rows[0].recovered
