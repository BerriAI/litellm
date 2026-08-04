"""Per-(api_key, session, auto-router) rollup behind the auto-router benchmarks dashboard.

A turn enters the cache view only when it touched the cache, so a model with caching off
contributes nothing to the buckets or the hit rate. ``tiers`` holds
``{model: [refreshed_at, ttl, prefix_tokens]}``, so a turn's bucket is
a question about one model's own record and the upsert answers it against the row it is
already writing: absent or nothing live is a first visit, an earlier start is unordered, and
otherwise warm or expired on the idle gap against the TTL the entry was written with.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.db_transaction_queue.base_update_queue import BaseUpdateQueue

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

CACHE_TTL_5M_SECONDS: Final = 300.0
CACHE_TTL_1H_SECONDS: Final = 3600.0


@dataclass(frozen=True, slots=True)
class TurnFacts:
    """One auto-routed turn, priced and reduced to what the rollup needs."""

    api_key: str
    session_id: str
    model_group: str
    router_kind: str
    baseline_model: str | None
    model: str
    started_at: float
    total_tokens: int
    spend: float
    baseline_spend: float
    cache_hit: bool
    cache_creation_tokens: int
    cached_prefix_tokens: int
    ttl_seconds: float


PARAM_NAMES: Final = (
    "api_key",
    "session_id",
    "model_group",
    "router_kind",
    "baseline_model",
    "model",
    "started_at",
    "total_tokens",
    "spend",
    "baseline_spend",
    "cache_hit",
    "written_tokens",
    "prefix_tokens",
    "one_hour",
    "ttl",
)

(
    _API_KEY,
    _SESSION_ID,
    _MODEL_GROUP,
    _ROUTER_KIND,
    _BASELINE_MODEL,
    _MODEL,
    _STARTED_AT,
    _TOTAL_TOKENS,
    _SPEND,
    _BASELINE_SPEND,
    _CACHE_HIT,
    _WRITTEN_TOKENS,
    _PREFIX_TOKENS,
    _ONE_HOUR,
    _TTL,
) = (f"${position}" for position in range(1, len(PARAM_NAMES) + 1))


def bind(turn: TurnFacts) -> tuple[object, ...]:
    """This turn's values, in ``PARAM_NAMES`` order."""
    return (
        turn.api_key,
        turn.session_id,
        turn.model_group,
        turn.router_kind,
        turn.baseline_model,
        turn.model,
        turn.started_at,
        turn.total_tokens,
        turn.spend,
        turn.baseline_spend,
        int(turn.cache_hit),
        turn.cache_creation_tokens,
        turn.cached_prefix_tokens,
        int(turn.ttl_seconds >= CACHE_TTL_1H_SECONDS),
        turn.ttl_seconds,
    )


_SEEN: Final = f"t.tiers ? {_MODEL}"
_CACHED_AT: Final = f"(t.tiers -> {_MODEL} ->> 0)::float8"
_CACHED_TTL: Final = f"GREATEST((t.tiers -> {_MODEL} ->> 1)::float8, 1)"
_CACHED_TOKENS: Final = f"(t.tiers -> {_MODEL} ->> 2)::float8"
_IDLE: Final = f"({_STARTED_AT}::float8 - {_CACHED_AT})"

_COVERED: Final = f"{_PREFIX_TOKENS}::float8 > 0"
_LIVE: Final = f"{_SEEN} AND {_CACHED_TOKENS} > 0"
_UNORDERED: Final = f"{_COVERED} AND {_LIVE} AND {_IDLE} < 0"
_WARM: Final = f"{_COVERED} AND {_LIVE} AND {_IDLE} >= 0 AND {_IDLE} <= {_CACHED_TTL}"
_EXPIRED: Final = f"{_COVERED} AND {_LIVE} AND {_IDLE} > {_CACHED_TTL}"
_REFRESHED: Final = f"NOT ({_SEEN}) OR ({_PREFIX_TOKENS}::float8 > 0 AND {_IDLE} >= 0)"
_REWROTE: Final = f"NOT ({_SEEN}) OR ({_WRITTEN_TOKENS}::float8 > 0 AND {_IDLE} >= 0)"

_UPSERT_SQL: Final = f"""
INSERT INTO "LiteLLM_AutoRouterSession" AS t (
    api_key, session_id, model_group, router_kind, baseline_model,
    turns, turns_with_usage, total_tokens, spend, baseline_spend,
    first_visit_turns, first_visit_hits, ephemeral_1h_turns,
    tiers, first_turn_at, last_turn_at, updated_at
)
VALUES (
    {_API_KEY}, {_SESSION_ID}, {_MODEL_GROUP}, {_ROUTER_KIND}, {_BASELINE_MODEL},
    1, CASE WHEN {_COVERED} THEN 1 ELSE 0 END, {_TOTAL_TOKENS}::bigint, {_SPEND}, {_BASELINE_SPEND},
    CASE WHEN {_COVERED} THEN 1 ELSE 0 END,
    CASE WHEN {_COVERED} THEN {_CACHE_HIT} ELSE 0 END,
    {_ONE_HOUR},
    jsonb_build_object({_MODEL}, jsonb_build_array(
        {_STARTED_AT}::float8, {_TTL}::float8, {_PREFIX_TOKENS}::float8
    )),
    to_timestamp({_STARTED_AT}::float8) AT TIME ZONE 'UTC',
    to_timestamp({_STARTED_AT}::float8) AT TIME ZONE 'UTC',
    NOW()
)
ON CONFLICT (api_key, session_id, model_group) DO UPDATE SET
    turns            = t.turns            + 1,
    turns_with_usage = t.turns_with_usage + CASE WHEN {_COVERED} THEN 1 ELSE 0 END,
    total_tokens     = t.total_tokens     + {_TOTAL_TOKENS}::bigint,
    spend            = t.spend            + {_SPEND},
    baseline_spend   = t.baseline_spend   + {_BASELINE_SPEND},

    first_visit_turns = t.first_visit_turns + CASE WHEN {_COVERED} AND NOT ({_LIVE}) THEN 1 ELSE 0 END,
    first_visit_hits  = t.first_visit_hits  + CASE WHEN {_COVERED} AND NOT ({_LIVE}) THEN {_CACHE_HIT} ELSE 0 END,
    unordered_turns   = t.unordered_turns   + CASE WHEN {_UNORDERED} THEN 1 ELSE 0 END,
    unordered_hits    = t.unordered_hits    + CASE WHEN {_UNORDERED} THEN {_CACHE_HIT} ELSE 0 END,
    warm_turns        = t.warm_turns        + CASE WHEN {_WARM} THEN 1 ELSE 0 END,
    warm_hits         = t.warm_hits         + CASE WHEN {_WARM} THEN {_CACHE_HIT} ELSE 0 END,
    expired_turns     = t.expired_turns     + CASE WHEN {_EXPIRED} THEN 1 ELSE 0 END,
    expired_hits      = t.expired_hits      + CASE WHEN {_EXPIRED} THEN {_CACHE_HIT} ELSE 0 END,


    ephemeral_1h_turns = t.ephemeral_1h_turns + {_ONE_HOUR},
    baseline_model = COALESCE(t.baseline_model, {_BASELINE_MODEL}),
    tiers = t.tiers || jsonb_build_object({_MODEL}, jsonb_build_array(
        CASE WHEN {_REFRESHED} THEN {_STARTED_AT}::float8 ELSE {_CACHED_AT} END,
        CASE WHEN {_REWROTE} THEN {_TTL}::float8 ELSE {_CACHED_TTL} END,
        CASE WHEN {_REFRESHED} THEN {_PREFIX_TOKENS}::float8 ELSE {_CACHED_TOKENS} END
    )),
    first_turn_at = LEAST(t.first_turn_at, to_timestamp({_STARTED_AT}::float8) AT TIME ZONE 'UTC'),
    last_turn_at  = GREATEST(t.last_turn_at, to_timestamp({_STARTED_AT}::float8) AT TIME ZONE 'UTC'),
    updated_at    = NOW()
"""


class AutoRouterSessionQueue(BaseUpdateQueue):
    """Stages turns in memory; writes an interval as one ordered batch on the spend flush.

    Batched because a busy interval drains up to ``MAX_IN_MEMORY_QUEUE_FLUSH_COUNT`` turns.
    Ordered because a session's classification depends on the turn before it. Never raises.
    """

    async def flush(self, prisma_client: PrismaClient) -> None:
        staged: Final[Sequence[TurnFacts]] = await self.flush_all_updates_from_in_memory_queue()
        if not staged:
            return
        try:
            async with prisma_client.db.batch_() as batcher:
                for turn in sorted(staged, key=lambda staged_turn: staged_turn.started_at):
                    batcher.execute_raw(_UPSERT_SQL, *bind(turn))
        except Exception as e:  # noqa: BLE001  # a dashboard rollup must never fail spend tracking
            verbose_proxy_logger.warning("auto_router_sessions: dropped %d turns (%s)", len(staged), e)


def ttl_seconds(usage_obj: Mapping[str, object] | None) -> float:
    """The cache tier this turn wrote against; no one-hour evidence means the 5m default."""
    if usage_obj is None:
        return CACHE_TTL_5M_SECONDS
    details: Final = usage_obj.get("cache_creation_token_details")
    if not isinstance(details, Mapping):
        return CACHE_TTL_5M_SECONDS
    if int(details.get("ephemeral_1h_input_tokens") or 0) > 0:
        return CACHE_TTL_1H_SECONDS
    return CACHE_TTL_5M_SECONDS


def _as_utc(moment: datetime) -> float:
    return (moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment).timestamp()


def _started_at(start_time: object) -> float | None:
    if isinstance(start_time, datetime):
        return _as_utc(start_time)
    if isinstance(start_time, str):
        try:
            return _as_utc(datetime.fromisoformat(start_time))
        except ValueError:
            return None
    return None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def build_turn_facts(
    payload: Mapping[str, object],
    metadata: Mapping[str, object],
    autorouter_savings: float,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> TurnFacts | None:
    """One spend-log payload as a rollup turn, or ``None`` if it was not auto-routed.

    A recorded ``routing_decision`` is what says the request was auto-routed, and names the kind.
    """
    decision: Final = metadata.get("routing_decision")
    if not isinstance(decision, Mapping):
        return None
    router_kind: Final = decision.get("router_type")
    api_key: Final = payload.get("api_key")
    session_id: Final = payload.get("session_id")
    model_group: Final = payload.get("model_group")
    model: Final = payload.get("model")
    if not (
        isinstance(router_kind, str)
        and isinstance(api_key, str)
        and isinstance(session_id, str)
        and isinstance(model_group, str)
        and isinstance(model, str)
        and api_key
        and session_id
        and model_group
        and model
    ):
        return None
    started_at: Final = _started_at(payload.get("startTime"))
    if started_at is None:
        return None
    usage_raw: Final = metadata.get("usage_object")
    usage_obj: Final = usage_raw if isinstance(usage_raw, Mapping) else None
    spend: Final = float(payload.get("spend") or 0.0)
    return TurnFacts(
        api_key=api_key,
        session_id=session_id,
        model_group=model_group,
        router_kind=router_kind,
        baseline_model=litellm.autorouter_savings_baseline_model,
        model=model,
        started_at=started_at,
        total_tokens=int(payload.get("prompt_tokens") or 0) + int(payload.get("completion_tokens") or 0),
        spend=spend,
        baseline_spend=spend + autorouter_savings,
        cache_hit=cache_read_tokens > 0,
        cache_creation_tokens=cache_creation_tokens,
        cached_prefix_tokens=max(cache_read_tokens, 0) + max(cache_creation_tokens, 0),
        ttl_seconds=ttl_seconds(usage_obj),
    )
