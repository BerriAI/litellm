"""Read side of the auto-router benchmarks dashboard.

Answers the customer question "what is the auto-router actually buying me": how
many turns a routed session runs, how long it lasts, how many tokens it burns,
how much cheaper the routed mix is than a single-model baseline, and how the
provider prompt cache behaves as the router moves a session between tiers.

Every one of those was folded when the turn happened (see
``auto_router_sessions``), so this module only sums pre-folded per-session rows.
It never reads ``LiteLLM_SpendLogs``: the sequential facts behind the cache
numbers cannot be recovered from per-request rows without window functions over
the whole window, which is what this replaced.

One aggregate query covers every auto-router, rather than four per router.
"""

from collections.abc import Mapping
from datetime import date, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, TypeAdapter

from litellm.proxy.spend_tracking.auto_router_sessions import PROMPT_CACHE_TTL_SECONDS

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

BENCHMARKS_MAX_WINDOW_DAYS = 30


class AutoRouterCacheBenchmark(BaseModel):
    """Provider prompt-cache behaviour for one auto-router.

    Sourced from the request's own ``cache_read_input_tokens``, which is the
    provider's prefix cache. LiteLLM's ``cache_hit`` column is a different
    mechanism entirely and reads false on turns the provider served from cache.

    The three turn buckets are mutually exclusive and exhaustive: every turn is
    either the router staying put, arriving somewhere new, or coming back to a
    tier this session already used, and they sum to ``turns``.
    """

    ttl_seconds: int
    usage_coverage_pct: float
    hit_rate_pct: float
    turns: int
    hits: int
    same_model_turns: int
    same_model_hits: int
    first_visit_turns: int
    first_visit_hits: int
    return_turns: int
    return_hits: int
    same_model_hit_rate_pct: float
    first_visit_hit_rate_pct: float
    return_hit_rate_pct: float
    stale_miss_share_pct: float
    warming_savable_miss_pct: float
    warming_break_even_pct: float
    stale_return_misses: int
    savable_return_misses: int
    warming_rescued_spend: float
    warming_replay_spend: float
    warming_net_spend: float


class AutoRouterGroupBenchmark(BaseModel):
    model_group: str
    router_kind: str
    baseline_model: str | None
    sessions: int
    turns: int
    avg_turns_per_session: float
    avg_session_length_seconds: float
    total_tokens: int
    avg_tokens_per_session: float
    actual_spend: float
    baseline_spend: float
    savings: float
    savings_pct: float
    cache: AutoRouterCacheBenchmark | None


class AutoRouterBenchmarksResponse(BaseModel):
    start_date: str
    end_date: str
    groups: tuple[AutoRouterGroupBenchmark, ...]


WARMING_BREAK_EVEN_PCT: Mapping[int, float] = MappingProxyType(
    {300: 9.0, 3600: 5.0}  # mutable-ok: a JSON object is a dict by definition
)  # mutable-ok: frozen by MappingProxyType on this line


class _GroupRow(BaseModel):
    """One folded auto-router, straight out of the aggregate."""

    model_group: str
    baseline_model: str | None
    router_kind: str
    sessions: int
    turns: int
    total_session_seconds: float
    total_tokens: int
    actual_spend: float
    baseline_spend: float
    turns_with_usage: int
    ephemeral_5m_tokens: int
    ephemeral_1h_tokens: int
    same_model_turns: int
    same_model_hits: int
    first_visit_turns: int
    first_visit_hits: int
    return_turns: int
    return_hits: int
    stale_return_misses: int
    savable_return_misses: int
    rescued_spend: float
    replay_spend: float


_GROUP_ROWS = TypeAdapter(tuple[_GroupRow, ...])


class _Window(NamedTuple):
    start: str
    end: str


_GROUP_SQL = """
SELECT
    model_group,
    MAX(baseline_model) AS baseline_model,
    MAX(router_kind) AS router_kind,
    COUNT(*)::bigint AS sessions,
    COALESCE(SUM(turns), 0)::bigint AS turns,
    COALESCE(SUM(EXTRACT(EPOCH FROM (last_turn_at - first_turn_at))), 0) AS total_session_seconds,
    COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
    COALESCE(SUM(spend), 0.0) AS actual_spend,
    COALESCE(SUM(baseline_spend), 0.0) AS baseline_spend,
    COALESCE(SUM(turns_with_usage), 0)::bigint AS turns_with_usage,
    COALESCE(SUM(ephemeral_5m_tokens), 0)::bigint AS ephemeral_5m_tokens,
    COALESCE(SUM(ephemeral_1h_tokens), 0)::bigint AS ephemeral_1h_tokens,
    COALESCE(SUM(same_model_turns), 0)::bigint AS same_model_turns,
    COALESCE(SUM(same_model_hits), 0)::bigint AS same_model_hits,
    COALESCE(SUM(first_visit_turns), 0)::bigint AS first_visit_turns,
    COALESCE(SUM(first_visit_hits), 0)::bigint AS first_visit_hits,
    COALESCE(SUM(return_turns), 0)::bigint AS return_turns,
    COALESCE(SUM(return_hits), 0)::bigint AS return_hits,
    COALESCE(SUM(stale_return_misses), 0)::bigint AS stale_return_misses,
    COALESCE(SUM(savable_return_misses), 0)::bigint AS savable_return_misses,
    COALESCE(SUM(rescued_spend), 0.0) AS rescued_spend,
    COALESCE(SUM(replay_spend), 0.0) AS replay_spend
FROM "LiteLLM_AutoRouterSession"
WHERE model_group = ANY($1::text[])
  AND last_turn_at >= ($2::timestamptz AT TIME ZONE 'UTC')
  AND first_turn_at < (($3::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
GROUP BY model_group
"""


def clamp_window(start_date: date, end_date: date) -> _Window:
    """Enforce ``start >= end - BENCHMARKS_MAX_WINDOW_DAYS``.

    The returned start reflects the window actually served, which the response
    echoes so the dashboard can label what it is showing rather than what it
    asked for. The dates arrive already parsed, because a malformed one is the
    route's contract to reject rather than this module's to discover.
    """
    floor = end_date - timedelta(days=BENCHMARKS_MAX_WINDOW_DAYS)
    return _Window(start=max(start_date, floor).isoformat(), end=end_date.isoformat())


def _rate_pct(part: int, whole: int) -> float:
    return (100.0 * part / whole) if whole else 0.0


def summarize_cache(row: _GroupRow) -> AutoRouterCacheBenchmark | None:
    """Fold one group's cache counters into the dashboard's cache view.

    ``stale_miss_share_pct`` narrows return-to-tier misses to those whose tier
    had gone idle past the TTL; the rest missed because the prefix changed, which
    keeping caches warm cannot fix.

    ``warming_savable_miss_pct`` narrows further and divides by every cache miss,
    so it reads as the share of all misses a refresher could actually have
    prevented. A miss qualifies only when the session returned to a tier it had
    already used, that tier had gone idle past the TTL, and it came back within
    two TTLs. The last bound is what one refresh fired just under the TTL can
    bridge; a tier idle longer needs a replay per elapsed TTL, and since every
    idle session pays those replays whether or not it returns, bridging further
    costs more than the write it avoids.
    """
    if row.turns_with_usage == 0:
        return None
    ttl_seconds = (
        PROMPT_CACHE_TTL_SECONDS["1h"]
        if row.ephemeral_1h_tokens > 0 and row.ephemeral_1h_tokens >= row.ephemeral_5m_tokens
        else PROMPT_CACHE_TTL_SECONDS["5m"]
    )
    hits = row.same_model_hits + row.first_visit_hits + row.return_hits
    bucketed_turns = row.same_model_turns + row.first_visit_turns + row.return_turns
    return_misses = row.return_turns - row.return_hits
    return AutoRouterCacheBenchmark(
        ttl_seconds=ttl_seconds,
        usage_coverage_pct=_rate_pct(row.turns_with_usage, row.turns),
        hit_rate_pct=_rate_pct(hits, bucketed_turns),
        turns=bucketed_turns,
        hits=hits,
        same_model_turns=row.same_model_turns,
        same_model_hits=row.same_model_hits,
        first_visit_turns=row.first_visit_turns,
        first_visit_hits=row.first_visit_hits,
        return_turns=row.return_turns,
        return_hits=row.return_hits,
        same_model_hit_rate_pct=_rate_pct(row.same_model_hits, row.same_model_turns),
        first_visit_hit_rate_pct=_rate_pct(row.first_visit_hits, row.first_visit_turns),
        return_hit_rate_pct=_rate_pct(row.return_hits, row.return_turns),
        stale_miss_share_pct=_rate_pct(row.stale_return_misses, return_misses),
        warming_savable_miss_pct=_rate_pct(row.savable_return_misses, bucketed_turns - hits),
        warming_break_even_pct=WARMING_BREAK_EVEN_PCT[ttl_seconds],
        stale_return_misses=row.stale_return_misses,
        savable_return_misses=row.savable_return_misses,
        warming_rescued_spend=row.rescued_spend,
        warming_replay_spend=row.replay_spend,
        warming_net_spend=row.rescued_spend - row.replay_spend,
    )


def summarize_group(row: _GroupRow) -> AutoRouterGroupBenchmark | None:
    """Fold one group's session rows into its benchmark.

    ``router_kind`` comes off the rows rather than from the configured groups,
    because the rows recorded which strategy actually served each turn while the
    config only says which are registered under the alias. Those differ when one
    alias owns several tagged strategies, and labelling the card from the config
    would name a router the numbers underneath it did not come from.

    ``savings`` keeps its sign. A router that thrashes the prompt cache can cost
    more than the baseline it is measured against, and an operator needs to be
    able to see that rather than have it floored to zero.
    """
    if row.sessions == 0:
        return None
    savings = row.baseline_spend - row.actual_spend
    return AutoRouterGroupBenchmark(
        model_group=row.model_group,
        router_kind=row.router_kind,
        baseline_model=row.baseline_model,
        sessions=row.sessions,
        turns=row.turns,
        avg_turns_per_session=row.turns / row.sessions,
        avg_session_length_seconds=row.total_session_seconds / row.sessions,
        total_tokens=row.total_tokens,
        avg_tokens_per_session=row.total_tokens / row.sessions,
        actual_spend=row.actual_spend,
        baseline_spend=row.baseline_spend,
        savings=savings,
        savings_pct=(100.0 * savings / row.baseline_spend) if row.baseline_spend > 0 else 0.0,
        cache=summarize_cache(row),
    )


async def compute_benchmarks(
    prisma_client: "PrismaClient",
    group_kinds: Mapping[str, str],
    start_date: date,
    end_date: date,
) -> AutoRouterBenchmarksResponse:
    """Aggregate the session rollup for every configured auto-router."""
    window = clamp_window(start_date, end_date)
    raw = await prisma_client.db.query_raw(
        _GROUP_SQL,
        list(group_kinds.keys()),  # mutable-ok: query_raw binds a list for the text[] parameter
        window.start,
        window.end,
    )
    summarized = (summarize_group(row) for row in _GROUP_ROWS.validate_python(raw))
    return AutoRouterBenchmarksResponse(
        start_date=window.start,
        end_date=window.end,
        groups=tuple(group for group in summarized if group is not None),
    )
