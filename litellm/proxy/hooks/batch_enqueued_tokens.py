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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.constants import BATCH_ENQUEUED_TOKEN_LIMIT_METADATA_KEY, BATCH_ENQUEUED_TOKEN_TTL_SECONDS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.redis.stores.enqueued_tokens import (
    AlreadyPopped,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenScope,
    EnqueuedTokensStore,
    OverLimit,
    Popped,
    Reserved,
    TransportEnqueuedTokensStore,
    counter_key,
    record_key,
)
from litellm.redis.transport.contract import RedisFailure
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    Span = _Span
    InternalUsageCache = _InternalUsageCache

BATCH_ENQUEUED_REFUND_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "failed", "expired", "cancelled", "cancelling"}
)


@dataclass(frozen=True, slots=True)
class BatchEnqueuedTokenOverLimit:
    scope: BatchEnqueuedTokenScope
    enqueued: int


BatchEnqueuedTokenOutcome: TypeAlias = BatchEnqueuedTokenReservation | BatchEnqueuedTokenOverLimit

_LIMIT_ADAPTER: Final = TypeAdapter(Annotated[int, Field(gt=0)])
_STORED_COUNTER_ADAPTER: Final = TypeAdapter(int | None)
_LOCAL_RECORD_ADAPTER: Final = TypeAdapter(object)


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
        get_batch_id_from_unified_batch_id,
        get_original_file_id,
        is_base64_encoded_unified_file_id,  # pyright: ignore[reportPrivateUsage]  # canonical unified-id decoder has no public wrapper
    )

    decoded: Final = is_base64_encoded_unified_file_id(batch_id)
    if isinstance(decoded, str):
        if "llm_batch_id" in decoded or "generic_response_id" in decoded:
            return get_batch_id_from_unified_batch_id(decoded)
        return decoded
    return get_original_file_id(batch_id)


class _BatchResponseView(LiteLLMBaseModel):
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
        redis_cache: Final = internal_usage_cache.dual_cache.redis_cache
        self._store: Final[EnqueuedTokensStore | None] = (
            TransportEnqueuedTokensStore(redis_cache.transport()) if redis_cache is not None else None
        )

    @staticmethod
    def _counter_key(scope: BatchEnqueuedTokenScope) -> str:
        return counter_key(scope.key, scope.value)

    @staticmethod
    def _record_key(batch_id: str) -> str:
        return record_key(batch_id)

    @with_service_target("rate_limits")
    async def reserve(
        self,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
        litellm_parent_otel_span: "Span | None" = None,
    ) -> BatchEnqueuedTokenOutcome:
        if tokens <= 0 or not scopes:
            return BatchEnqueuedTokenReservation(tokens=max(tokens, 0), scopes=scopes)
        if self._store is not None:
            outcome: Final = await self._reserve_via_redis(self._store, tokens=tokens, scopes=scopes)
            if not isinstance(outcome, RedisFailure):
                return outcome
        return await self._reserve_in_memory(tokens=tokens, scopes=scopes, span=litellm_parent_otel_span)

    async def _reserve_via_redis(
        self,
        store: EnqueuedTokensStore,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> BatchEnqueuedTokenOutcome | RedisFailure:
        started: Final = self._monotonic()
        ttl: Final = timedelta(seconds=BATCH_ENQUEUED_TOKEN_TTL_SECONDS)
        for index, scope in enumerate(scopes):
            reply = await store.reserve(scope.key, scope.value, tokens, limit=scope.limit, ttl=ttl)
            match reply:
                case Reserved():
                    continue
                case OverLimit(enqueued=enqueued):
                    await self._rollback_partial_reserve(store, tokens=tokens, scopes=scopes[:index])
                    return BatchEnqueuedTokenOverLimit(scope=scope, enqueued=enqueued)
                case RedisFailure():
                    await self._rollback_partial_reserve(store, tokens=tokens, scopes=scopes[:index])
                    return reply
        return BatchEnqueuedTokenReservation(
            tokens=tokens, scopes=scopes, backend="redis", reserved_at_monotonic=started
        )

    async def _rollback_partial_reserve(
        self,
        store: EnqueuedTokensStore,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> None:
        refunded: Final = await self._refund_via_redis(store, tokens=tokens, scopes=scopes)
        if isinstance(refunded, RedisFailure):
            verbose_proxy_logger.warning(
                "Rollback of partially reserved enqueued tokens failed; leaked increments expire with the TTL"
            )

    async def _refund_via_redis(
        self,
        store: EnqueuedTokensStore,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> None | RedisFailure:
        for scope in scopes:
            refunded = await store.refund(scope.key, scope.value, tokens)
            if refunded is not None:
                return refunded
        return None

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

    @with_service_target("rate_limits")
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
        if self._store is None:
            verbose_proxy_logger.warning(
                "No Redis client for a Redis-granted enqueued-token refund; leaked increments expire with the TTL"
            )
            return
        refunded: Final = await self._refund_via_redis(
            self._store, tokens=reservation.tokens, scopes=reservation.scopes
        )
        if isinstance(refunded, RedisFailure):
            verbose_proxy_logger.warning("Redis enqueued-token refund failed; leaked increments expire with the TTL")

    @with_service_target("rate_limits")
    async def save_reservation(
        self,
        batch_id: str,
        reservation: BatchEnqueuedTokenReservation,
        litellm_parent_otel_span: "Span | None" = None,
    ) -> None:
        elapsed: Final = self._monotonic() - reservation.reserved_at_monotonic
        ttl: Final = max(1, BATCH_ENQUEUED_TOKEN_TTL_SECONDS - math.ceil(elapsed))
        if self._store is not None:
            saved: Final = await self._store.save_record(batch_id, reservation, ttl=timedelta(seconds=ttl))
            if saved is None:
                return
        await self.internal_usage_cache.async_set_cache(
            key=self._record_key(batch_id),
            value=reservation,
            ttl=ttl,
            litellm_parent_otel_span=litellm_parent_otel_span,
            local_only=True,
        )

    @with_service_target("rate_limits")
    async def pop_reservation(
        self,
        batch_id: str,
        litellm_parent_otel_span: "Span | None" = None,
    ) -> BatchEnqueuedTokenReservation | None:
        popped: Final = await self._pop_redis_record(batch_id)
        if isinstance(popped, AlreadyPopped):
            # The Redis pop tombstones popped records in place, so a hit on the empty
            # tombstone means the batch was already refunded elsewhere; a local copy
            # left behind by a save that raised after landing must not refund again.
            await self._pop_local_record(batch_id, litellm_parent_otel_span)
            return None
        if isinstance(popped, Popped):
            return popped.reservation
        if isinstance(popped, RedisFailure):
            verbose_proxy_logger.warning("Discarding malformed enqueued-token reservation record for %s", batch_id)
            return None
        local: Final = await self._pop_local_record(batch_id, litellm_parent_otel_span)
        return local if isinstance(local, BatchEnqueuedTokenReservation) else None

    async def _pop_redis_record(self, batch_id: str) -> Popped | AlreadyPopped | RedisFailure | None:
        if self._store is None:
            return None
        popped: Final = await self._store.pop_record(
            batch_id, tombstone_ttl=timedelta(seconds=BATCH_ENQUEUED_TOKEN_TTL_SECONDS)
        )
        if isinstance(popped, RedisFailure) and popped.kind != "decode":
            return None
        return popped

    async def _pop_local_record(self, batch_id: str, span: "Span | None") -> object:
        async with self._lock:
            stored: Final = _LOCAL_RECORD_ADAPTER.validate_python(
                await self.internal_usage_cache.async_get_cache(
                    key=self._record_key(batch_id),
                    litellm_parent_otel_span=span,
                    local_only=True,
                )
            )
            if stored is None:
                return None
            self.internal_usage_cache.dual_cache.in_memory_cache.delete_cache(key=self._record_key(batch_id))
        return stored

    async def _get_local_counter(self, scope: BatchEnqueuedTokenScope, span: "Span | None") -> int:
        return (
            _STORED_COUNTER_ADAPTER.validate_python(
                await self.internal_usage_cache.async_get_cache(
                    key=self._counter_key(scope),
                    litellm_parent_otel_span=span,
                    local_only=True,
                )
            )
            or 0
        )

    async def _set_local_counter(self, scope: BatchEnqueuedTokenScope, value: int, span: "Span | None") -> None:
        await self.internal_usage_cache.async_set_cache(
            key=self._counter_key(scope),
            value=value,
            ttl=BATCH_ENQUEUED_TOKEN_TTL_SECONDS,
            litellm_parent_otel_span=span,
            local_only=True,
        )
