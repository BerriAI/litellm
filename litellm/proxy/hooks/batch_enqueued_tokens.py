"""
Enqueued-token accounting for batch submissions.

Opt-in via admin-set ``batch_enqueued_token_limit`` in key or team metadata: batch
submissions reserve their estimated token count against a long-lived
enqueued-token allowance instead of the per-minute rate-limit windows, and
the reservation is refunded when the batch reaches a terminal state
(completed, failed, expired, or cancelled).
"""

import asyncio
import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCache, log_redis_failure
from litellm.constants import BATCH_ENQUEUED_TOKEN_LIMIT_METADATA_KEY, BATCH_ENQUEUED_TOKEN_TTL_SECONDS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.caching import CacheFailure
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    Span = _Span
    InternalUsageCache = _InternalUsageCache

BATCH_ENQUEUED_REFUND_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "failed", "expired", "cancelled", "cancelling"}
)

ScopeKey: TypeAlias = Literal["api_key", "team"]


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
_STORED_COUNTER_ADAPTER: Final = TypeAdapter(int | None)
_RESERVATION_ADAPTER: Final = TypeAdapter(BatchEnqueuedTokenReservation)
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
        self._redis_cache = internal_usage_cache.dual_cache.redis_cache

    @staticmethod
    def _counter_key(scope: BatchEnqueuedTokenScope) -> str:
        return f"batch_enqueued_tokens:{scope.key}:{scope.value}"

    @staticmethod
    def _record_key(batch_id: str) -> str:
        return f"batch_enqueued_token_reservation:{batch_id}"

    @with_service_target("rate_limits")
    async def reserve(
        self,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
        litellm_parent_otel_span: "Span | None" = None,
    ) -> BatchEnqueuedTokenOutcome:
        if tokens <= 0 or not scopes:
            return BatchEnqueuedTokenReservation(tokens=max(tokens, 0), scopes=scopes)
        if self._redis_cache is not None:
            result: Final = await self._reserve_via_redis(self._redis_cache, tokens=tokens, scopes=scopes)
            if not isinstance(result, CacheFailure):
                return result
            log_redis_failure(
                verbose_proxy_logger,
                logging.WARNING,
                "Redis enqueued-token reserve failed, falling back to in-memory",
                result,
            )
        return await self._reserve_in_memory(tokens=tokens, scopes=scopes, span=litellm_parent_otel_span)

    async def _reserve_via_redis(
        self,
        redis_cache: RedisCache,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> BatchEnqueuedTokenOutcome | CacheFailure:
        started: Final = self._monotonic()
        for index, scope in enumerate(scopes):
            if (
                isinstance(
                    result := await redis_cache.async_reserve_counter(
                        key=self._counter_key(scope),
                        amount=tokens,
                        limit=scope.limit,
                        ttl_seconds=BATCH_ENQUEUED_TOKEN_TTL_SECONDS,
                    ),
                    CacheFailure,
                )
                or not result.accepted
            ):
                await self._rollback_partial_reserve(redis_cache, tokens=tokens, scopes=scopes[:index])
                return (
                    result
                    if isinstance(result, CacheFailure)
                    else BatchEnqueuedTokenOverLimit(scope=scope, enqueued=result.count)
                )
        return BatchEnqueuedTokenReservation(
            tokens=tokens, scopes=scopes, backend="redis", reserved_at_monotonic=started
        )

    async def _rollback_partial_reserve(
        self,
        redis_cache: RedisCache,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> None:
        result: Final = await self._refund_via_redis(redis_cache, tokens=tokens, scopes=scopes)
        if isinstance(result, CacheFailure):
            verbose_proxy_logger.warning(
                "Rollback of partially reserved enqueued tokens failed; leaked increments expire with the TTL: %s",
                result.message,
            )

    async def _refund_via_redis(
        self,
        redis_cache: RedisCache,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
    ) -> None | CacheFailure:
        for scope in scopes:
            if isinstance(
                result := await redis_cache.async_refund_counter(
                    key=self._counter_key(scope),
                    amount=tokens,
                ),
                CacheFailure,
            ):
                return result
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
        redis_cache: Final = self._redis_cache
        if redis_cache is None:
            verbose_proxy_logger.warning(
                "No Redis client for a Redis-granted enqueued-token refund; leaked increments expire with the TTL"
            )
            return
        result: Final = await self._refund_via_redis(redis_cache, tokens=reservation.tokens, scopes=reservation.scopes)
        if isinstance(result, CacheFailure):
            verbose_proxy_logger.warning(
                "Redis enqueued-token refund failed; leaked increments expire with the TTL: %s", result.message
            )

    @with_service_target("rate_limits")
    async def save_reservation(
        self,
        batch_id: str,
        reservation: BatchEnqueuedTokenReservation,
        litellm_parent_otel_span: "Span | None" = None,
    ) -> None:
        serialized: Final = _RESERVATION_ADAPTER.dump_json(reservation).decode("utf-8")
        elapsed: Final = self._monotonic() - reservation.reserved_at_monotonic
        ttl: Final = max(1, BATCH_ENQUEUED_TOKEN_TTL_SECONDS - math.ceil(elapsed))
        if self._redis_cache is not None:
            result: Final = await self._redis_cache.async_save_record(
                key=self._record_key(batch_id), value=serialized, ttl_seconds=ttl
            )
            if not isinstance(result, CacheFailure):
                return
            log_redis_failure(
                verbose_proxy_logger,
                logging.WARNING,
                "Redis enqueued-token reservation save failed, falling back to in-memory",
                result,
            )
        await self.internal_usage_cache.async_set_cache(
            key=self._record_key(batch_id),
            value=serialized,
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

    async def _pop_redis_record(self, batch_id: str) -> str | None:
        if self._redis_cache is None:
            return None
        result: Final = await self._redis_cache.async_pop_record(
            key=self._record_key(batch_id), tombstone_ttl_seconds=BATCH_ENQUEUED_TOKEN_TTL_SECONDS
        )
        if not isinstance(result, CacheFailure):
            return result
        log_redis_failure(
            verbose_proxy_logger,
            logging.WARNING,
            "Redis enqueued-token reservation pop failed, falling back to in-memory",
            result,
        )
        return None

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
