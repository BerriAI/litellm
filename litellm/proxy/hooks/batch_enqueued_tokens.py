"""
Enqueued-token accounting for batch submissions.

Opt-in via admin-set ``batch_enqueued_token_limit`` in key or team metadata: batch
submissions reserve their estimated token count against a long-lived
enqueued-token allowance instead of the per-minute rate-limit windows, and
the reservation is refunded when the batch reaches a terminal state
(completed, failed, expired, or cancelled).
"""

import asyncio
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import BATCH_ENQUEUED_TOKEN_LIMIT_METADATA_KEY, BATCH_ENQUEUED_TOKEN_TTL_SECONDS
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    Span = _Span
    InternalUsageCache = _InternalUsageCache

BATCH_ENQUEUED_REFUND_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "failed", "expired", "cancelled", "cancelling"}
)

ScopeKey: TypeAlias = Literal["api_key", "team"]

RESERVE_ENQUEUED_TOKENS_SCRIPT: Final = """
local amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current + amount > limit then
    return {0, current}
end
local updated = redis.call('INCRBY', KEYS[1], amount)
redis.call('EXPIRE', KEYS[1], ttl)
return {1, updated}
"""

REFUND_ENQUEUED_TOKENS_SCRIPT: Final = """
local updated = redis.call('DECRBY', KEYS[1], tonumber(ARGV[1]))
if updated <= 0 then
    redis.call('DEL', KEYS[1])
end
return 1
"""

SAVE_RESERVATION_SCRIPT: Final = """
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
return 1
"""

POP_RESERVATION_SCRIPT: Final = """
local value = redis.call('GET', KEYS[1])
if value and value ~= '' then
    redis.call('SET', KEYS[1], '', 'EX', tonumber(ARGV[1]))
end
return value
"""


@dataclass(frozen=True, slots=True)
class BatchEnqueuedTokenScope:
    key: ScopeKey
    value: str
    limit: int


ReservationBackend: TypeAlias = Literal["redis", "memory"]


@dataclass(frozen=True, slots=True)
class BatchEnqueuedTokenReservation:
    tokens: int
    scopes: tuple[BatchEnqueuedTokenScope, ...]
    backend: ReservationBackend = "redis"
    owner: str = ""
    reserved_at_monotonic: float = field(default_factory=time.monotonic, compare=False)


@dataclass(frozen=True, slots=True)
class BatchEnqueuedTokenOverLimit:
    scope: BatchEnqueuedTokenScope
    enqueued: int


BatchEnqueuedTokenOutcome: TypeAlias = BatchEnqueuedTokenReservation | BatchEnqueuedTokenOverLimit

_LIMIT_ADAPTER: Final = TypeAdapter(Annotated[int, Field(gt=0)])
_RESERVE_RESULT_ADAPTER: Final = TypeAdapter(tuple[int, int])
_POPPED_VALUE_ADAPTER: Final = TypeAdapter(str | bytes | None)
_STORED_COUNTER_ADAPTER: Final = TypeAdapter(int | None)
_RESERVATION_ADAPTER: Final = TypeAdapter(BatchEnqueuedTokenReservation)


class _ScriptRunner(Protocol):
    def __call__(self, keys: Sequence[str], args: Sequence[str | bytes | int | float]) -> Awaitable[object]: ...


def _read_metadata_limit(metadata: Mapping[str, object] | None) -> int | None:
    if not metadata:
        return None
    raw: Final = metadata.get(BATCH_ENQUEUED_TOKEN_LIMIT_METADATA_KEY)
    if raw is None:
        return None
    try:
        return _LIMIT_ADAPTER.validate_python(raw)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Ignoring invalid %s value %r; expected a positive integer",
            BATCH_ENQUEUED_TOKEN_LIMIT_METADATA_KEY,
            raw,
        )
        return None


def resolve_batch_enqueued_token_scopes(
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[BatchEnqueuedTokenScope, ...]:
    key_limit: Final = _read_metadata_limit(user_api_key_dict.metadata)
    team_limit: Final = _read_metadata_limit(user_api_key_dict.team_metadata)
    candidates: Final = (
        BatchEnqueuedTokenScope(key="api_key", value=user_api_key_dict.api_key, limit=key_limit)
        if key_limit is not None and user_api_key_dict.api_key
        else None,
        BatchEnqueuedTokenScope(key="team", value=user_api_key_dict.team_id, limit=team_limit)
        if team_limit is not None and user_api_key_dict.team_id
        else None,
    )
    return tuple(scope for scope in candidates if scope is not None)


def canonical_provider_batch_id(batch_id: str) -> str:
    from litellm.proxy.openai_files_endpoints.common_utils import (
        _is_base64_encoded_unified_file_id,  # pyright: ignore[reportPrivateUsage]  # canonical unified-id decoder has no public wrapper
        get_batch_id_from_unified_batch_id,
        get_original_file_id,
    )

    decoded: Final = _is_base64_encoded_unified_file_id(batch_id)
    if isinstance(decoded, str):
        if "llm_batch_id" in decoded or "generic_response_id" in decoded:
            return get_batch_id_from_unified_batch_id(decoded)
        return decoded
    return get_original_file_id(batch_id)


class _BatchResponseView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    status: str
    object: Literal["batch"]


def batch_response_view(response: object) -> _BatchResponseView | None:
    try:
        return _BatchResponseView.model_validate(response, from_attributes=True)
    except ValidationError:
        return None


class BatchEnqueuedTokenStore:
    """Tracks enqueued batch tokens per scope, plus per-batch reservation records for refunds.

    Counters and records live in Redis when Redis is configured, through
    single-key Lua scripts issued one scope at a time (Redis Cluster safe: no
    cross-slot commands), with an over-limit or failing scope rolling back the
    scopes reserved before it; otherwise a single-process in-memory fallback
    guarded by one asyncio lock is used. Reservations remember which backend
    granted them, and in-memory grants also remember the granting worker, so a
    refund never debits counters the grant did not charge. Everything expires after
    ``BATCH_ENQUEUED_TOKEN_TTL_SECONDS`` so a crash between submission and the
    terminal-state refund can never leak tokens forever, and reservation records
    expire no later than the counters they would refund, so a stale record can
    never debit an allowance re-granted after its counters expired.
    """

    def __init__(
        self,
        internal_usage_cache: "InternalUsageCache",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.internal_usage_cache = internal_usage_cache
        self._monotonic: Final = monotonic
        self._lock = asyncio.Lock()
        self._owner_token = uuid.uuid4().hex
        redis_cache = internal_usage_cache.dual_cache.redis_cache
        self._reserve_script: _ScriptRunner | None = (
            redis_cache.async_register_script(RESERVE_ENQUEUED_TOKENS_SCRIPT) if redis_cache is not None else None
        )
        self._refund_script: _ScriptRunner | None = (
            redis_cache.async_register_script(REFUND_ENQUEUED_TOKENS_SCRIPT) if redis_cache is not None else None
        )
        self._save_script: _ScriptRunner | None = (
            redis_cache.async_register_script(SAVE_RESERVATION_SCRIPT) if redis_cache is not None else None
        )
        self._pop_script: _ScriptRunner | None = (
            redis_cache.async_register_script(POP_RESERVATION_SCRIPT) if redis_cache is not None else None
        )

    @staticmethod
    def _counter_key(scope: BatchEnqueuedTokenScope) -> str:
        return f"batch_enqueued_tokens:{scope.key}:{scope.value}"

    @staticmethod
    def _record_key(batch_id: str) -> str:
        return f"batch_enqueued_token_reservation:{batch_id}"

    async def reserve(
        self,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
        litellm_parent_otel_span: "Span | None" = None,
    ) -> BatchEnqueuedTokenOutcome:
        if tokens <= 0 or not scopes:
            return BatchEnqueuedTokenReservation(tokens=max(tokens, 0), scopes=scopes)
        reserve_script: Final = self._reserve_script
        refund_script: Final = self._refund_script
        if reserve_script is not None and refund_script is not None:
            try:
                return await self._reserve_via_redis(reserve_script, refund_script, tokens=tokens, scopes=scopes)
            except Exception as e:  # noqa: BLE001  # any Redis failure must fall back to the in-memory counters
                verbose_proxy_logger.warning(
                    "Redis enqueued-token reserve failed, falling back to in-memory: %s", str(e)
                )
        return await self._reserve_in_memory(tokens=tokens, scopes=scopes, span=litellm_parent_otel_span)

    async def _reserve_via_redis(
        self,
        reserve_script: _ScriptRunner,
        refund_script: _ScriptRunner,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> BatchEnqueuedTokenOutcome:
        started: Final = self._monotonic()
        for index, scope in enumerate(scopes):
            result = await self._run_reserve_script(
                reserve_script,
                refund_script,
                tokens=tokens,
                scope=scope,
                already_reserved=scopes[:index],
            )
            if result[0] != 1:
                await self._rollback_partial_reserve(refund_script, tokens=tokens, scopes=scopes[:index])
                return BatchEnqueuedTokenOverLimit(scope=scope, enqueued=result[1])
        return BatchEnqueuedTokenReservation(
            tokens=tokens, scopes=scopes, backend="redis", reserved_at_monotonic=started
        )

    async def _run_reserve_script(
        self,
        reserve_script: _ScriptRunner,
        refund_script: _ScriptRunner,
        tokens: int,
        scope: BatchEnqueuedTokenScope,
        already_reserved: tuple[BatchEnqueuedTokenScope, ...],
    ) -> tuple[int, int]:
        try:
            raw_result: Final = await reserve_script(
                (self._counter_key(scope),),
                (tokens, BATCH_ENQUEUED_TOKEN_TTL_SECONDS, scope.limit),
            )
            return _RESERVE_RESULT_ADAPTER.validate_python(raw_result)
        except Exception:
            await self._rollback_partial_reserve(refund_script, tokens=tokens, scopes=already_reserved)
            raise

    async def _rollback_partial_reserve(
        self,
        refund_script: _ScriptRunner,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> None:
        try:
            await self._refund_via_redis(refund_script, tokens=tokens, scopes=scopes)
        except Exception as e:  # noqa: BLE001  # best-effort rollback: the leak is TTL-bounded and only tightens the allowance
            verbose_proxy_logger.warning(
                "Rollback of partially reserved enqueued tokens failed; leaked increments expire with the TTL: %s",
                str(e),
            )

    async def _refund_via_redis(
        self,
        refund_script: _ScriptRunner,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> None:
        for scope in scopes:
            await refund_script((self._counter_key(scope),), (tokens,))

    async def _reserve_in_memory(
        self,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
        span: "Span | None",
    ) -> BatchEnqueuedTokenOutcome:
        started: Final = self._monotonic()
        async with self._lock:
            currents: Final = tuple([await self._get_local_counter(scope, span) for scope in scopes])
            for scope, current in zip(scopes, currents):
                if current + tokens > scope.limit:
                    return BatchEnqueuedTokenOverLimit(scope=scope, enqueued=current)
            for scope, current in zip(scopes, currents):
                await self._set_local_counter(scope, current + tokens, span)
        return BatchEnqueuedTokenReservation(
            tokens=tokens, scopes=scopes, backend="memory", owner=self._owner_token, reserved_at_monotonic=started
        )

    async def refund(
        self,
        reservation: BatchEnqueuedTokenReservation,
        litellm_parent_otel_span: "Span | None" = None,
    ) -> None:
        if reservation.tokens <= 0 or not reservation.scopes:
            return
        if reservation.backend == "redis":
            await self._refund_redis_reservation(reservation)
            return
        if reservation.owner != self._owner_token:
            verbose_proxy_logger.warning(
                "Skipping enqueued-token refund granted in another worker's memory; its counters expire with the TTL"
            )
            return
        async with self._lock:
            for scope in reservation.scopes:
                current = await self._get_local_counter(scope, litellm_parent_otel_span)
                remaining = current - reservation.tokens
                if remaining <= 0:
                    self.internal_usage_cache.dual_cache.in_memory_cache.delete_cache(key=self._counter_key(scope))
                else:
                    await self._set_local_counter(scope, remaining, litellm_parent_otel_span)

    async def _refund_redis_reservation(self, reservation: BatchEnqueuedTokenReservation) -> None:
        refund_script: Final = self._refund_script
        if refund_script is None:
            verbose_proxy_logger.warning(
                "No Redis client for a Redis-granted enqueued-token refund; leaked increments expire with the TTL"
            )
            return
        try:
            await self._refund_via_redis(refund_script, tokens=reservation.tokens, scopes=reservation.scopes)
        except Exception as e:  # noqa: BLE001  # best-effort refund: the leak is TTL-bounded and only tightens the allowance
            verbose_proxy_logger.warning(
                "Redis enqueued-token refund failed; leaked increments expire with the TTL: %s", str(e)
            )

    async def save_reservation(
        self,
        batch_id: str,
        reservation: BatchEnqueuedTokenReservation,
        litellm_parent_otel_span: "Span | None" = None,
    ) -> None:
        serialized: Final = _RESERVATION_ADAPTER.dump_json(reservation).decode("utf-8")
        elapsed: Final = self._monotonic() - reservation.reserved_at_monotonic
        ttl: Final = max(1, BATCH_ENQUEUED_TOKEN_TTL_SECONDS - math.ceil(elapsed))
        if self._save_script is not None:
            try:
                await self._save_script(
                    (self._record_key(batch_id),),
                    (serialized, ttl),
                )
            except Exception as e:  # noqa: BLE001  # any Redis failure must fall back to the in-memory record
                verbose_proxy_logger.warning(
                    "Redis enqueued-token reservation save failed, falling back to in-memory: %s", str(e)
                )
            else:
                return
        await self.internal_usage_cache.async_set_cache(
            key=self._record_key(batch_id),
            value=serialized,
            ttl=ttl,
            litellm_parent_otel_span=litellm_parent_otel_span,
            local_only=True,
        )

    async def pop_reservation(
        self,
        batch_id: str,
        litellm_parent_otel_span: "Span | None" = None,
    ) -> BatchEnqueuedTokenReservation | None:
        redis_raw: Final = await self._pop_redis_record(batch_id)
        if redis_raw is not None and not redis_raw:
            # The Redis pop tombstones popped records in place, so a hit on the empty
            # tombstone means the batch was already refunded elsewhere; a local copy
            # left behind by a save that raised after landing must not refund again.
            await self._pop_local_record(batch_id, litellm_parent_otel_span)
            return None
        raw: Final = (
            redis_raw if redis_raw is not None else await self._pop_local_record(batch_id, litellm_parent_otel_span)
        )
        if raw is None:
            return None
        try:
            if isinstance(raw, (str, bytes)):
                return _RESERVATION_ADAPTER.validate_json(raw)
            return _RESERVATION_ADAPTER.validate_python(raw)
        except ValidationError:
            verbose_proxy_logger.warning("Discarding malformed enqueued-token reservation record for %s", batch_id)
            return None

    async def _pop_redis_record(self, batch_id: str) -> str | bytes | None:
        pop_script: Final = self._pop_script
        if pop_script is None:
            return None
        try:
            return _POPPED_VALUE_ADAPTER.validate_python(
                await pop_script((self._record_key(batch_id),), (BATCH_ENQUEUED_TOKEN_TTL_SECONDS,))
            )
        except Exception as e:  # noqa: BLE001  # any Redis failure must fall back to the in-memory record
            verbose_proxy_logger.warning(
                "Redis enqueued-token reservation pop failed, falling back to in-memory: %s", str(e)
            )
            return None

    async def _pop_local_record(self, batch_id: str, span: "Span | None") -> object:
        async with self._lock:
            stored = await self.internal_usage_cache.async_get_cache(
                key=self._record_key(batch_id),
                litellm_parent_otel_span=span,
                local_only=True,
            )
            if stored is None:
                return None
            self.internal_usage_cache.dual_cache.in_memory_cache.delete_cache(key=self._record_key(batch_id))
        return stored

    async def _get_local_counter(self, scope: BatchEnqueuedTokenScope, span: "Span | None") -> int:
        stored = await self.internal_usage_cache.async_get_cache(
            key=self._counter_key(scope),
            litellm_parent_otel_span=span,
            local_only=True,
        )
        return _STORED_COUNTER_ADAPTER.validate_python(stored) or 0

    async def _set_local_counter(self, scope: BatchEnqueuedTokenScope, value: int, span: "Span | None") -> None:
        await self.internal_usage_cache.async_set_cache(
            key=self._counter_key(scope),
            value=value,
            ttl=BATCH_ENQUEUED_TOKEN_TTL_SECONDS,
            litellm_parent_otel_span=span,
            local_only=True,
        )
