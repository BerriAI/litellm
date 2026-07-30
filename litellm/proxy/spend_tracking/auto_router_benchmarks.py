"""Session-level benchmarks for auto-router deployments.

Answers the customer question "what is the auto-router actually buying me": how
many turns a routed session runs, how long it lasts, how many tokens it burns,
and how much cheaper the routed model mix is than sending every request to a
single baseline model.

The four metrics are session-scoped, and ``session_id`` lives only on
``LiteLLM_SpendLogs`` (the per-request table), never on the daily rollups: a
session does not close on a day boundary, so it cannot be pre-aggregated the way
user/team spend is. This reads SpendLogs directly over a bounded window; the
durable answer is a per-session rollup table, tracked as a follow-up.

Only rows whose ``model_group`` is a configured auto-router alias are counted.
That filter is load-bearing: the auto-router's own LLM-classifier sub-calls land
in the same session but carry the judge model's group, not the alias, so
grouping by the alias yields one row per routed turn with no classifier noise
and no separate ``call_type`` filter.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Mapping, NamedTuple

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.router_utils.auto_router_model_naming import classify_strategy_router_model

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router


BENCHMARKS_MAX_WINDOW_DAYS = 30


class AutoRouterGroupBenchmark(BaseModel):
    model_group: str
    router_kind: str
    baseline_model: str
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


class AutoRouterBenchmarksResponse(BaseModel):
    start_date: str
    end_date: str
    groups: tuple[AutoRouterGroupBenchmark, ...]


class _BaselineRates(NamedTuple):
    model: str
    input_cost_per_token: float
    output_cost_per_token: float


class _SessionRow(NamedTuple):
    session_id: str
    turns: int
    session_length_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    actual_spend: float


def auto_router_groups(router: "Router") -> tuple[tuple[str, str, str | None], ...]:
    """Enumerate ``(public model_group, router kind, configured baseline)`` per auto-router.

    ``model_group`` is the public alias clients send and SpendLogs records; the
    ``litellm_params.model`` string is the ``auto_router/...`` discriminator. A
    per-router ``benchmark_baseline_model`` override, when set, pins the baseline
    to a fixed flagship; otherwise the caller derives it from the routed traffic.
    """
    return tuple(
        (
            str(entry["model_name"]),
            kind,
            _configured_baseline(entry.get("litellm_params")),
        )
        for entry in (router.model_list or [])
        if (model := _entry_model(entry)) is not None and (kind := classify_strategy_router_model(model)) is not None
    )


def _entry_model(entry: Mapping[str, object]) -> str | None:
    params = entry.get("litellm_params")
    if not isinstance(params, Mapping):
        return None
    model = params.get("model")
    return model if isinstance(model, str) else None


def _configured_baseline(params: object) -> str | None:
    if not isinstance(params, Mapping):
        return None
    baseline = params.get("benchmark_baseline_model")
    return baseline if isinstance(baseline, str) and baseline else None


def _baseline_rates(model: str) -> _BaselineRates:
    """Price a baseline model, trying provider-prefixed and bare name candidates.

    Daily/spend rows store models provider-prefixed (``anthropic/claude-opus-5``)
    while the cost map often keys them bare, so a single lookup silently prices
    zero. Falls open to zero rates for a fully unknown model, which surfaces as a
    zero baseline (and zero savings) rather than a raised error.
    """
    for candidate in _pricing_candidates(model):
        try:
            info = litellm.get_model_info(model=candidate)
        except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
            verbose_proxy_logger.debug("auto_router_benchmarks: no model info for %s (%s)", candidate, e)
            continue
        return _BaselineRates(
            model=model,
            input_cost_per_token=float(info.get("input_cost_per_token") or 0.0),
            output_cost_per_token=float(info.get("output_cost_per_token") or 0.0),
        )
    verbose_proxy_logger.warning(
        "auto_router_benchmarks: baseline model %s is not priced; savings will read zero", model
    )
    return _BaselineRates(model=model, input_cost_per_token=0.0, output_cost_per_token=0.0)


def _pricing_candidates(model: str) -> tuple[str, ...]:
    stripped = model.split("/", 1)[1] if "/" in model else model
    return tuple(dict.fromkeys((model, stripped)))


def _clamp_window(start_date: str, end_date: str) -> tuple[datetime, datetime, str, str]:
    """Parse the range and enforce ``start >= end - BENCHMARKS_MAX_WINDOW_DAYS``.

    The scan is over the unbounded per-request table, so the window is capped the
    way the tool-spend endpoint caps it; the returned start reflects the window
    actually served.
    """
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    floor = (end - timedelta(days=BENCHMARKS_MAX_WINDOW_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    clamped_start = max(start, floor)
    return clamped_start, end, clamped_start.date().isoformat(), end.date().isoformat()


def _derive_baseline_model(rows: tuple["_RoutedModelSpend", ...]) -> str | None:
    """Pick the priciest model actually routed to in the window as the baseline.

    Uniform across all four router types and always a model the router really
    used, so "vs baseline" is grounded in the deployment rather than a guess at
    which tier is the flagship. Priced by blended per-token rate so a model is
    not called the flagship purely because it emitted more tokens.
    """
    priced = tuple((row.model, rate) for row in rows if (rate := _blended_rate(row.model)) is not None)
    if not priced:
        return None
    return max(priced, key=lambda pair: pair[1])[0]


def _blended_rate(model: str) -> float | None:
    for candidate in _pricing_candidates(model):
        try:
            info = litellm.get_model_info(model=candidate)
        except Exception:  # noqa: BLE001  # unmapped model, try next candidate
            continue
        return float(info.get("input_cost_per_token") or 0.0) + float(info.get("output_cost_per_token") or 0.0)
    return None


class _RoutedModelSpend(NamedTuple):
    model: str
    prompt_tokens: int
    completion_tokens: int


def summarize_group(
    model_group: str,
    router_kind: str,
    session_rows: tuple[_SessionRow, ...],
    routed_models: tuple[_RoutedModelSpend, ...],
    configured_baseline: str | None,
) -> AutoRouterGroupBenchmark | None:
    """Fold per-session and per-model rows into one group benchmark.

    Returns None when the window holds no routed sessions for the group, so an
    idle auto-router is omitted rather than reported as a row of zeros.
    """
    if not session_rows:
        return None
    baseline_model = configured_baseline or _derive_baseline_model(routed_models)
    if baseline_model is None:
        return None
    rates = _baseline_rates(baseline_model)

    sessions = len(session_rows)
    turns = sum(row.turns for row in session_rows)
    total_tokens = sum(row.total_tokens for row in session_rows)
    actual_spend = sum(row.actual_spend for row in session_rows)
    baseline_spend = sum(
        row.prompt_tokens * rates.input_cost_per_token + row.completion_tokens * rates.output_cost_per_token
        for row in session_rows
    )
    savings = baseline_spend - actual_spend
    return AutoRouterGroupBenchmark(
        model_group=model_group,
        router_kind=router_kind,
        baseline_model=baseline_model,
        sessions=sessions,
        turns=turns,
        avg_turns_per_session=turns / sessions,
        avg_session_length_seconds=sum(row.session_length_seconds for row in session_rows) / sessions,
        total_tokens=total_tokens,
        avg_tokens_per_session=total_tokens / sessions,
        actual_spend=actual_spend,
        baseline_spend=baseline_spend,
        savings=savings,
        savings_pct=(100.0 * savings / baseline_spend) if baseline_spend > 0 else 0.0,
    )


_SESSION_SQL = """
SELECT
    session_id,
    COUNT(*)::bigint AS turns,
    EXTRACT(EPOCH FROM (MAX("endTime") - MIN("startTime"))) AS session_length_seconds,
    COALESCE(SUM(prompt_tokens), 0)::bigint AS prompt_tokens,
    COALESCE(SUM(completion_tokens), 0)::bigint AS completion_tokens,
    COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
    COALESCE(SUM(spend), 0.0) AS actual_spend
FROM "LiteLLM_SpendLogs"
WHERE model_group = $1
  AND session_id IS NOT NULL
  AND "startTime" >= ($2::timestamptz AT TIME ZONE 'UTC')
  AND "startTime" < (($3::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
GROUP BY session_id
"""

_ROUTED_MODEL_SQL = """
SELECT
    model,
    COALESCE(SUM(prompt_tokens), 0)::bigint AS prompt_tokens,
    COALESCE(SUM(completion_tokens), 0)::bigint AS completion_tokens
FROM "LiteLLM_SpendLogs"
WHERE model_group = $1
  AND session_id IS NOT NULL
  AND model IS NOT NULL
  AND "startTime" >= ($2::timestamptz AT TIME ZONE 'UTC')
  AND "startTime" < (($3::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
GROUP BY model
"""


class _RawSessionRow(BaseModel):
    session_id: str
    turns: int
    session_length_seconds: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    actual_spend: float


class _RawRoutedModelRow(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int


_SESSION_ROWS = TypeAdapter(tuple[_RawSessionRow, ...])
_ROUTED_MODEL_ROWS = TypeAdapter(tuple[_RawRoutedModelRow, ...])


async def compute_benchmarks(
    prisma_client: "PrismaClient",
    groups: tuple[tuple[str, str, str | None], ...],
    start_date: str,
    end_date: str,
) -> AutoRouterBenchmarksResponse:
    """Run the windowed session + routed-model queries per group and fold them.

    One pair of aggregate queries per auto-router group. Aggregation happens in
    Postgres; only per-session and per-model summaries cross the wire, never raw
    request rows.
    """
    _, _, served_start, served_end = _clamp_window(start_date, end_date)
    summarized = [
        summarize_group(
            model_group,
            router_kind,
            await _fetch_sessions(prisma_client, model_group, served_start, served_end),
            await _fetch_routed_models(prisma_client, model_group, served_start, served_end),
            configured_baseline,
        )
        for model_group, router_kind, configured_baseline in groups
    ]
    benchmarks = tuple(benchmark for benchmark in summarized if benchmark is not None)
    return AutoRouterBenchmarksResponse(start_date=served_start, end_date=served_end, groups=benchmarks)


async def _fetch_sessions(
    prisma_client: "PrismaClient", model_group: str, start: str, end: str
) -> tuple[_SessionRow, ...]:
    raw = await prisma_client.db.query_raw(_SESSION_SQL, model_group, start, end)
    return tuple(
        _SessionRow(
            session_id=row.session_id,
            turns=row.turns,
            session_length_seconds=row.session_length_seconds or 0.0,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            total_tokens=row.total_tokens,
            actual_spend=row.actual_spend,
        )
        for row in _SESSION_ROWS.validate_python(raw)
    )


async def _fetch_routed_models(
    prisma_client: "PrismaClient", model_group: str, start: str, end: str
) -> tuple[_RoutedModelSpend, ...]:
    raw = await prisma_client.db.query_raw(_ROUTED_MODEL_SQL, model_group, start, end)
    return tuple(
        _RoutedModelSpend(model=row.model, prompt_tokens=row.prompt_tokens, completion_tokens=row.completion_tokens)
        for row in _ROUTED_MODEL_ROWS.validate_python(raw)
    )
