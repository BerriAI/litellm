"""Read path for the auto-router benchmarks dashboard.

Sums pre-folded rows from ``LiteLLM_AutoRouterSession`` and nothing else. ``summarize``
produces both the per-router view and the totals, since averaging per-router percentages is
wrong. Every miss has one of four causes on one denominator: cold, prefix changed, aged out, or a turn whose cache state could not be established.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from functools import reduce
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

MAX_WINDOW_DAYS: Final = 30


@dataclass(frozen=True, slots=True)
class _Counters:
    sessions: int = 0
    turns: int = 0
    turns_with_usage: int = 0
    total_tokens: int = 0
    total_session_seconds: float = 0.0
    spend: float = 0.0
    baseline_spend: float = 0.0
    first_visit_turns: int = 0
    first_visit_hits: int = 0
    warm_turns: int = 0
    warm_hits: int = 0
    expired_turns: int = 0
    expired_hits: int = 0
    unordered_turns: int = 0
    unordered_hits: int = 0
    unknown_ttl_turns: int = 0
    unknown_ttl_hits: int = 0
    cache_5m_turns: int = 0
    cache_1h_turns: int = 0
    cache_ttl_unknown_turns: int = 0


class AutoRouterCacheBenchmark(BaseModel):
    turns: int
    hits: int
    misses: int
    hit_rate_pct: float
    coverage_pct: float
    first_visit_turns: int
    first_visit_hits: int
    first_visit_hit_rate_pct: float
    warm_turns: int
    warm_hits: int
    warm_hit_rate_pct: float
    expired_turns: int
    expired_hits: int
    expired_hit_rate_pct: float
    unordered_turns: int
    unordered_hits: int
    unknown_ttl_turns: int
    unknown_ttl_hits: int
    five_minute_cache_turns: int
    one_hour_cache_turns: int
    unknown_cache_ttl_turns: int
    cold_misses: int
    prefix_change_misses: int
    expired_misses: int
    unattributed_misses: int
    cold_miss_pct: float
    prefix_change_miss_pct: float
    expired_miss_pct: float
    unattributed_miss_pct: float


class AutoRouterBenchmark(BaseModel):
    sessions: int
    turns: int
    total_tokens: int
    spend: float
    baseline_spend: float
    savings: float
    savings_pct: float
    saved_per_session: float
    avg_turns_per_session: float
    avg_session_seconds: float
    avg_tokens_per_session: float
    cache: AutoRouterCacheBenchmark | None


class AutoRouterGroupBenchmark(BaseModel):
    model_group: str
    router_kind: str
    baseline_model: str | None
    benchmark: AutoRouterBenchmark


class AutoRouterBenchmarksResponse(BaseModel):
    start_date: date
    end_date: date
    routers_in_scope: int
    totals: AutoRouterBenchmark
    groups: tuple[AutoRouterGroupBenchmark, ...]


_DERIVED_COLUMNS: Final = ("sessions", "total_session_seconds")
_SUM_COLUMNS: Final = tuple(field.name for field in fields(_Counters) if field.name not in _DERIVED_COLUMNS)

_AGGREGATE_SQL: Final = f"""
SELECT
    model_group,
    MAX(router_kind) AS router_kind,
    MAX(baseline_model) AS baseline_model,
    COUNT(*) AS sessions,
    COALESCE(SUM(EXTRACT(EPOCH FROM (last_turn_at - first_turn_at))), 0) AS total_session_seconds,
    {", ".join(f"COALESCE(SUM({column}), 0) AS {column}" for column in _SUM_COLUMNS)}
FROM "LiteLLM_AutoRouterSession"
WHERE first_turn_at >= $1::timestamptz AT TIME ZONE \'UTC\'
  AND first_turn_at < $2::timestamptz AT TIME ZONE \'UTC\'
  AND ($3::text IS NULL OR model_group = $3::text)
GROUP BY model_group
ORDER BY SUM(spend) DESC
"""


def _pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _per(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _number(value: object) -> float:
    """Postgres aggregates surface through the driver as int, float, Decimal or a numeric
    string depending on the column type (SUM over BIGINT and EXTRACT both yield NUMERIC)."""
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _counters_from_row(row: Mapping[str, object]) -> _Counters:
    return _Counters(*(_number(row.get(field.name)) for field in fields(_Counters)))


def _combine(left: _Counters, right: _Counters) -> _Counters:
    return _Counters(*(a + b for a, b in zip(astuple(left), astuple(right))))


def _summarize_cache(counters: _Counters) -> AutoRouterCacheBenchmark | None:
    """The cache view, or ``None`` when no turn in scope reported cache behaviour."""
    if counters.turns_with_usage == 0:
        return None
    covered: Final = counters.turns_with_usage
    hits: Final = (
        counters.first_visit_hits
        + counters.warm_hits
        + counters.expired_hits
        + counters.unordered_hits
        + counters.unknown_ttl_hits
    )
    misses: Final = covered - hits
    cold: Final = counters.first_visit_turns - counters.first_visit_hits
    prefix_changed: Final = counters.warm_turns - counters.warm_hits
    savable: Final = counters.expired_turns - counters.expired_hits
    unattributed: Final = (
        counters.unordered_turns - counters.unordered_hits + counters.unknown_ttl_turns - counters.unknown_ttl_hits
    )
    return AutoRouterCacheBenchmark(
        turns=covered,
        hits=hits,
        misses=misses,
        hit_rate_pct=_pct(hits, covered),
        coverage_pct=_pct(counters.turns_with_usage, counters.turns),
        first_visit_turns=counters.first_visit_turns,
        first_visit_hits=counters.first_visit_hits,
        first_visit_hit_rate_pct=_pct(counters.first_visit_hits, counters.first_visit_turns),
        warm_turns=counters.warm_turns,
        warm_hits=counters.warm_hits,
        warm_hit_rate_pct=_pct(counters.warm_hits, counters.warm_turns),
        expired_turns=counters.expired_turns,
        expired_hits=counters.expired_hits,
        expired_hit_rate_pct=_pct(counters.expired_hits, counters.expired_turns),
        unordered_turns=counters.unordered_turns,
        unordered_hits=counters.unordered_hits,
        unknown_ttl_turns=counters.unknown_ttl_turns,
        unknown_ttl_hits=counters.unknown_ttl_hits,
        five_minute_cache_turns=counters.cache_5m_turns,
        one_hour_cache_turns=counters.cache_1h_turns,
        unknown_cache_ttl_turns=counters.cache_ttl_unknown_turns,
        cold_misses=cold,
        prefix_change_misses=prefix_changed,
        expired_misses=savable,
        unattributed_misses=unattributed,
        cold_miss_pct=_pct(cold, misses),
        prefix_change_miss_pct=_pct(prefix_changed, misses),
        expired_miss_pct=_pct(savable, misses),
        unattributed_miss_pct=_pct(unattributed, misses),
    )


def summarize(counters: _Counters) -> AutoRouterBenchmark:
    """One benchmark from raw counters, used for a single router and for the totals alike."""
    savings: Final = counters.baseline_spend - counters.spend
    return AutoRouterBenchmark(
        sessions=counters.sessions,
        turns=counters.turns,
        total_tokens=counters.total_tokens,
        spend=counters.spend,
        baseline_spend=counters.baseline_spend,
        savings=savings,
        savings_pct=_pct(savings, counters.baseline_spend),
        saved_per_session=_per(savings, counters.sessions),
        avg_turns_per_session=_per(counters.turns, counters.sessions),
        avg_session_seconds=_per(counters.total_session_seconds, counters.sessions),
        avg_tokens_per_session=_per(counters.total_tokens, counters.sessions),
        cache=_summarize_cache(counters),
    )


def clamp_window(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    """The half-open UTC interval to read, clamped to ``MAX_WINDOW_DAYS``; ``end_date`` is
    inclusive to the caller, so the upper bound is the start of the following day."""
    span_start: Final = max(start_date, end_date - timedelta(days=MAX_WINDOW_DAYS - 1))
    return (
        datetime.combine(span_start, time.min, tzinfo=timezone.utc),
        datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc),
    )


def build_response(
    rows: Sequence[Mapping[str, object]],
    start_date: date,
    end_date: date,
) -> AutoRouterBenchmarksResponse:
    per_group: Final = tuple((row, _counters_from_row(row)) for row in rows)
    totals: Final = reduce(_combine, (counters for _, counters in per_group), _Counters())
    return AutoRouterBenchmarksResponse(
        start_date=start_date,
        end_date=end_date,
        routers_in_scope=len(per_group),
        totals=summarize(totals),
        groups=tuple(
            AutoRouterGroupBenchmark(
                model_group=str(row.get("model_group") or ""),
                router_kind=str(row.get("router_kind") or ""),
                baseline_model=row.get("baseline_model") if isinstance(row.get("baseline_model"), str) else None,
                benchmark=summarize(counters),
            )
            for row, counters in per_group
        ),
    )


async def fetch_benchmarks(
    prisma_client: PrismaClient,
    start_date: date,
    end_date: date,
    model_group: str | None = None,
) -> AutoRouterBenchmarksResponse:
    """Benchmarks for the window actually read; sessions are attributed to the window they
    started in, and the response echoes the clamped dates rather than the requested ones."""
    window_start, window_end = clamp_window(start_date, end_date)
    rows: Final = await prisma_client.db.query_raw(
        _AGGREGATE_SQL,
        window_start.isoformat(),
        window_end.isoformat(),
        model_group,
    )
    return build_response(
        rows=rows,
        start_date=window_start.date(),
        end_date=window_end.date() - timedelta(days=1),
    )
