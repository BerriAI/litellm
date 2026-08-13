"""
Per-session auto-router benchmarks rollup.

At request time the spend writer builds one AutoRouterTurnTransaction per successful
auto-routed request (a request whose metadata carries a routing_decision) and queues it
on the prisma client. The spend-log flush job drains the queue into
LiteLLM_AutoRouterSession with one conditional upsert per turn: the statement classifies
the turn (same model, first visit, return to a model the session already used, out of
order) against the row's own columns, so nothing is read before the write and concurrent
pods compose. The benchmarks endpoint aggregates these rows and never touches
LiteLLM_SpendLogs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from typing import TYPE_CHECKING, Final, NamedTuple

from litellm._logging import verbose_proxy_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.proxy._types import DB_RETRY_SAFE_ERROR_TYPES

if TYPE_CHECKING:
    from litellm.proxy._types import SpendLogsPayload
    from litellm.proxy.utils import PrismaClient

CACHE_TTL_5M_SECONDS: Final = 300
CACHE_TTL_1H_SECONDS: Final = 3600

AUTOROUTER_BENCHMARKS_SQL: Final = """
WITH windowed AS (
    SELECT * FROM "LiteLLM_AutoRouterSession"
    WHERE last_turn_at >= $1::timestamp AND first_turn_at < $2::timestamp
),
tier_maps AS (
    SELECT router_name, router_type, jsonb_object_agg(tier, tier_turns) AS tier_turns
    FROM (
        SELECT router_name, router_type, kv.key AS tier, SUM((kv.value)::int)::int AS tier_turns
        FROM windowed, LATERAL jsonb_each_text(tier_turns) AS kv
        GROUP BY router_name, router_type, kv.key
    ) per_tier
    GROUP BY router_name, router_type
)
SELECT
    agg.*,
    COALESCE(tier_maps.tier_turns, '{}'::jsonb) AS tier_turns
FROM (
SELECT
    router_name,
    router_type,
    COUNT(*)::int AS sessions,
    COALESCE(SUM(turns), 0)::int AS turns,
    COALESCE(SUM(unordered_turns), 0)::int AS unordered_turns,
    COALESCE(SUM(covered_turns), 0)::int AS covered_turns,
    COALESCE(SUM(cache_hits), 0)::int AS cache_hits,
    COALESCE(SUM(same_model_turns), 0)::int AS same_model_turns,
    COALESCE(SUM(same_model_hits), 0)::int AS same_model_hits,
    COALESCE(SUM(first_visit_turns), 0)::int AS first_visit_turns,
    COALESCE(SUM(first_visit_hits), 0)::int AS first_visit_hits,
    COALESCE(SUM(return_turns), 0)::int AS return_turns,
    COALESCE(SUM(return_hits), 0)::int AS return_hits,
    COALESCE(SUM(return_expired_misses), 0)::int AS return_expired_misses,
    COALESCE(SUM(return_within_ttl_misses), 0)::int AS return_within_ttl_misses,
    COALESCE(SUM(ttl_5m_turns), 0)::int AS ttl_5m_turns,
    COALESCE(SUM(ttl_1h_turns), 0)::int AS ttl_1h_turns,
    COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
    COALESCE(SUM(spend), 0)::float8 AS spend,
    COALESCE(SUM(saved_spend), 0)::float8 AS saved_spend,
    COALESCE(SUM(EXTRACT(EPOCH FROM (last_turn_at - first_turn_at))), 0)::float8 AS session_seconds
FROM windowed
GROUP BY router_name, router_type
) agg
LEFT JOIN tier_maps USING (router_name, router_type)
ORDER BY agg.spend DESC
"""


@dataclass(frozen=True, slots=True)
class AutoRouterTurnTransaction:
    api_key: str
    session_id: str
    router_name: str
    router_type: str
    model: str
    turn_at: datetime
    total_tokens: int
    spend: float
    saved_spend: float
    covered: bool
    cache_hit: bool
    cache_ttl_seconds: int | None
    cache_touched: bool
    tier: str | None = None


class TurnCacheFacts(NamedTuple):
    """One statement of a turn's cache interaction, derived from its usage record.

    ``touched`` is False only when telemetry positively shows the provider neither
    read from nor wrote to the cache; absent telemetry reads as touched, which is
    the conservative input for the per-model idle clock.
    """

    covered: bool
    read_tokens: int
    write_ttl_seconds: int | None
    touched: bool


def turn_cache_facts(usage_object: Mapping[str, object] | None) -> TurnCacheFacts:
    from litellm.proxy.spend_tracking.savings import extract_cache_read_tokens

    covered: Final = bool(usage_object)
    read_tokens: Final = extract_cache_read_tokens(usage_object)
    write_ttl_seconds: Final = _write_ttl_seconds(usage_object)
    return TurnCacheFacts(
        covered=covered,
        read_tokens=read_tokens,
        write_ttl_seconds=write_ttl_seconds,
        touched=not covered or read_tokens > 0 or write_ttl_seconds is not None,
    )


def _turn_time_utc(start_time_iso: str) -> datetime | None:
    try:
        parsed: Final = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _write_ttl_seconds(usage_object: Mapping[str, object] | None) -> int | None:
    """The TTL this turn's cache write used, or None when nothing was written.

    Providers that report a TTL split do so under prompt_tokens_details; a write with no
    split is the provider's default five-minute cache.
    """
    from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens

    if not usage_object:
        return None
    details: Final = usage_object.get("prompt_tokens_details")
    creation: Final = details.get("cache_creation_token_details") if isinstance(details, Mapping) else None
    if isinstance(creation, Mapping):
        if creation.get("ephemeral_1h_input_tokens"):
            return CACHE_TTL_1H_SECONDS
        if creation.get("ephemeral_5m_input_tokens"):
            return CACHE_TTL_5M_SECONDS
    if extract_cache_creation_tokens(usage_object) > 0:
        return CACHE_TTL_5M_SECONDS
    return None


SESSION_ID_MAX_CHARS: Final = 256


def _bounded_session_id(session_id: str) -> str:
    """The session id as stored, bounded so a caller-chosen identifier cannot exceed
    Postgres's B-tree index entry limit through the composite primary key. Oversized
    ids map to a stable digest, so their turns still aggregate into one session."""
    if len(session_id) <= SESSION_ID_MAX_CHARS:
        return session_id
    return "sha256:" + hashlib.sha256(session_id.encode("utf-8", errors="surrogatepass")).hexdigest()


def build_autorouter_turn_transaction(
    payload: SpendLogsPayload,
    metadata: Mapping[str, object],
    saved_spend: float,
) -> AutoRouterTurnTransaction | None:
    """One rollup transaction for a successful auto-routed turn, else None.

    The routing_decision record is what says a request was auto-routed at all, so a
    request without one (including the auto-router's own classifier sub-calls) never
    reaches the rollup. Internal sub-calls that DO carry one (a shadow eval's duplicate
    of a request through the router) are excluded by their internal_call_origin stamp:
    they are not traffic a user sent, so counting them would manufacture sessions and
    savings in the adoption metrics. Failed requests served nothing and are excluded.
    Cache facts are derived from the payload's own usage record through the savings
    owner, never handed in beside it.
    """
    if payload.get("status") != "success":
        return None
    if metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
        return None
    routing_decision: Final = metadata.get("routing_decision")
    if not isinstance(routing_decision, Mapping) or not routing_decision:
        return None
    router_name: Final = routing_decision.get("router_model_name") or payload.get("model_group")
    api_key: Final = payload.get("api_key")
    session_id: Final = payload.get("session_id")
    model: Final = payload.get("model")
    if not (isinstance(router_name, str) and router_name and api_key and session_id and model):
        return None
    turn_at: Final = _turn_time_utc(str(payload.get("startTime") or ""))
    if turn_at is None:
        return None
    usage_object_raw: Final = metadata.get("usage_object")
    cache: Final = turn_cache_facts(usage_object_raw if isinstance(usage_object_raw, Mapping) else None)
    tier_raw: Final = routing_decision.get("tier")
    return AutoRouterTurnTransaction(
        api_key=api_key,
        session_id=_bounded_session_id(session_id),
        router_name=router_name,
        router_type=str(routing_decision.get("router_type") or "unknown"),
        tier=tier_raw if isinstance(tier_raw, str) and tier_raw else None,
        model=model,
        turn_at=turn_at,
        total_tokens=int(payload.get("prompt_tokens") or 0) + int(payload.get("completion_tokens") or 0),
        spend=float(payload.get("spend") or 0.0),
        saved_spend=saved_spend,
        covered=cache.covered,
        cache_hit=cache.read_tokens > 0,
        cache_ttl_seconds=cache.write_ttl_seconds,
        cache_touched=cache.touched,
    )


_UPSERT_PARAM_FIELDS: Final = tuple(field.name for field in dataclasses.fields(AutoRouterTurnTransaction))


def _p(field_name: str) -> str:
    """Positional placeholder for a transaction field, numbered by the dataclass's own
    field order so the SQL and the argument tuple cannot disagree; typos fail at import."""
    return f"${_UPSERT_PARAM_FIELDS.index(field_name) + 1}"


_MODEL: Final = _p("model")
_TURN_AT: Final = _p("turn_at")
_COVERED: Final = _p("covered")
_CACHE_HIT: Final = _p("cache_hit")
_CACHE_TTL: Final = _p("cache_ttl_seconds")
_TOUCHED: Final = _p("cache_touched")
_TIER: Final = f"{_p('tier')}::text"
_TIER_DELTA: Final = f"(CASE WHEN {_TIER} IS NULL THEN '{{}}'::jsonb ELSE jsonb_build_object({_TIER}, 1) END)"

_IN_ORDER: Final = f"{_TURN_AT}::timestamp >= t.last_turn_at"
_SAME: Final = f"{_IN_ORDER} AND t.last_model = {_MODEL}"
_FIRST: Final = f"{_IN_ORDER} AND NOT t.models ? {_MODEL}"
_RETURN: Final = f"{_IN_ORDER} AND t.models ? {_MODEL} AND t.last_model <> {_MODEL}"
_RETURN_MISS: Final = (
    f"{_RETURN} AND {_COVERED}::int = 1 AND {_CACHE_HIT}::int = 0 AND (t.models -> {_MODEL} ->> 'ttl') IS NOT NULL"
)
_IDLE_SECONDS: Final = f"EXTRACT(EPOCH FROM {_TURN_AT}::timestamp) - (t.models -> {_MODEL} ->> 'at')::float8"
_CACHE_TOUCHED: Final = f"{_TOUCHED}::int = 1"

UPSERT_AUTOROUTER_SESSION_SQL: Final = f"""
INSERT INTO "LiteLLM_AutoRouterSession" AS t (
    api_key, session_id, router_name, router_type, first_turn_at, last_turn_at,
    last_model, models, turns, unordered_turns, covered_turns, cache_hits,
    same_model_turns, same_model_hits, first_visit_turns, first_visit_hits,
    return_turns, return_hits, return_expired_misses, return_within_ttl_misses,
    ttl_5m_turns, ttl_1h_turns, total_tokens, spend, saved_spend, tier_turns
)
VALUES (
    {_p("api_key")}, {_p("session_id")}, {_p("router_name")}, {_p("router_type")}, {_TURN_AT}::timestamp, {_TURN_AT}::timestamp,
    {_MODEL}, jsonb_build_object({_MODEL}, jsonb_build_object('at', EXTRACT(EPOCH FROM {_TURN_AT}::timestamp), 'ttl', {_CACHE_TTL}::int)),
    1, 0, {_COVERED}::int, {_CACHE_HIT}::int,
    0, 0, 1, {_CACHE_HIT}::int,
    0, 0, 0, 0,
    (CASE WHEN {_CACHE_TTL}::int = {CACHE_TTL_5M_SECONDS} THEN 1 ELSE 0 END),
    (CASE WHEN {_CACHE_TTL}::int = {CACHE_TTL_1H_SECONDS} THEN 1 ELSE 0 END),
    {_p("total_tokens")}::bigint, {_p("spend")}::float8, {_p("saved_spend")}::float8,
    {_TIER_DELTA}
)
ON CONFLICT (api_key, session_id, router_name) DO UPDATE SET
    turns = t.turns + 1,
    total_tokens = t.total_tokens + EXCLUDED.total_tokens,
    spend = t.spend + EXCLUDED.spend,
    saved_spend = t.saved_spend + EXCLUDED.saved_spend,
    covered_turns = t.covered_turns + EXCLUDED.covered_turns,
    cache_hits = t.cache_hits + EXCLUDED.cache_hits,
    ttl_5m_turns = t.ttl_5m_turns + EXCLUDED.ttl_5m_turns,
    ttl_1h_turns = t.ttl_1h_turns + EXCLUDED.ttl_1h_turns,
    unordered_turns = t.unordered_turns + (CASE WHEN NOT ({_IN_ORDER}) THEN 1 ELSE 0 END),
    same_model_turns = t.same_model_turns + (CASE WHEN {_SAME} THEN 1 ELSE 0 END),
    same_model_hits = t.same_model_hits + (CASE WHEN {_SAME} AND {_CACHE_HIT}::int = 1 THEN 1 ELSE 0 END),
    first_visit_turns = t.first_visit_turns + (CASE WHEN {_FIRST} THEN 1 ELSE 0 END),
    first_visit_hits = t.first_visit_hits + (CASE WHEN {_FIRST} AND {_CACHE_HIT}::int = 1 THEN 1 ELSE 0 END),
    return_turns = t.return_turns + (CASE WHEN {_RETURN} THEN 1 ELSE 0 END),
    return_hits = t.return_hits + (CASE WHEN {_RETURN} AND {_CACHE_HIT}::int = 1 THEN 1 ELSE 0 END),
    return_expired_misses = t.return_expired_misses
        + (CASE WHEN {_RETURN_MISS} AND {_IDLE_SECONDS} > (t.models -> {_MODEL} ->> 'ttl')::float8 THEN 1 ELSE 0 END),
    return_within_ttl_misses = t.return_within_ttl_misses
        + (CASE WHEN {_RETURN_MISS} AND {_IDLE_SECONDS} <= (t.models -> {_MODEL} ->> 'ttl')::float8 THEN 1 ELSE 0 END),
    models = t.models || jsonb_build_object({_MODEL}, jsonb_build_object(
        'at', (CASE WHEN {_CACHE_TOUCHED}
               THEN GREATEST(COALESCE((t.models -> {_MODEL} ->> 'at')::float8, 0), EXTRACT(EPOCH FROM {_TURN_AT}::timestamp))
               ELSE COALESCE((t.models -> {_MODEL} ->> 'at')::float8, EXTRACT(EPOCH FROM {_TURN_AT}::timestamp)) END),
        'ttl', (CASE WHEN {_IN_ORDER}
                THEN COALESCE({_CACHE_TTL}::int, (t.models -> {_MODEL} ->> 'ttl')::int)
                ELSE COALESCE((t.models -> {_MODEL} ->> 'ttl')::int, {_CACHE_TTL}::int) END)
    )),
    last_model = (CASE WHEN {_IN_ORDER} THEN {_MODEL} ELSE t.last_model END),
    tier_turns = (CASE WHEN {_TIER} IS NOT NULL AND t.router_type = {_p("router_type")}
        THEN t.tier_turns || jsonb_build_object({_TIER}, COALESCE((t.tier_turns ->> {_TIER})::int, 0) + 1)
        ELSE t.tier_turns END),
    first_turn_at = LEAST(t.first_turn_at, EXCLUDED.first_turn_at),
    last_turn_at = GREATEST(t.last_turn_at, EXCLUDED.last_turn_at)
"""


def _as_sql_param(value: str | float | bool | datetime | None) -> str | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _upsert_params(transaction: AutoRouterTurnTransaction) -> tuple[str | float | None, ...]:
    return tuple(_as_sql_param(getattr(transaction, name)) for name in _UPSERT_PARAM_FIELDS)


async def _upsert_turn_with_retry(
    prisma_client: PrismaClient,
    transaction: AutoRouterTurnTransaction,
    n_retry_times: int,
) -> None:
    for attempt in range(n_retry_times + 1):
        try:
            await prisma_client.db.execute_raw(UPSERT_AUTOROUTER_SESSION_SQL, *_upsert_params(transaction))
        except DB_RETRY_SAFE_ERROR_TYPES:
            if attempt >= n_retry_times:
                raise
            await asyncio.sleep(2**attempt + random.uniform(0, 1))
        else:
            return


async def flush_autorouter_turn_transactions(
    prisma_client: PrismaClient,
    transactions: Sequence[AutoRouterTurnTransaction],
    n_retry_times: int = 3,
) -> None:
    """Drain a queue batch into the rollup, one upsert per turn.

    Statements run sequentially in per-session event order: a turn's classification
    depends on the turns before it, and Postgres rejects one multi-row INSERT touching
    the same key twice. Only ConnectError is retried, per statement, because it proves
    that statement never reached the database. Any other failure drops the remaining
    turns of THAT session only, with an error log, and the flush continues with the
    next session: sessions are independent state machines, so one poisoned statement
    must not discard unrelated sessions, and a repeated increment is worse than an
    undercount. Callers must not add their own retry around this function.
    """
    if not transactions:
        return
    ordered: Final = sorted(
        transactions,
        key=lambda transaction: (
            transaction.api_key,
            transaction.session_id,
            transaction.router_name,
            transaction.turn_at,
        ),
    )
    for session_key, session_group in groupby(
        ordered,
        key=lambda transaction: (transaction.api_key, transaction.session_id, transaction.router_name),
    ):
        session_turns = tuple(session_group)
        for position, transaction in enumerate(session_turns):
            try:
                await _upsert_turn_with_retry(prisma_client, transaction, n_retry_times)
            except Exception as flush_err:  # noqa: BLE001  # a statement failure drops only its session's remainder by design
                verbose_proxy_logger.error(
                    "Spend tracking - auto-router session rollup flush failed for router %s; "
                    "%s of %s turn transactions dropped for one session: %s",
                    session_key[2],
                    len(session_turns) - position,
                    len(session_turns),
                    flush_err,
                )
                break
