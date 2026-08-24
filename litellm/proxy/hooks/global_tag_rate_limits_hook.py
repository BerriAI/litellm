"""
Tag-scoped token, request, dollar, and concurrency rate limits declared once,
globally, in `litellm_settings.global_tag_rate_limits` -- enforced once per
request in `async_pre_call_hook`, before Router does any routing, so a limit
applies regardless of which model or fallback chain the request ends up
hitting.

This is the model-independent sibling of `model_based_tag_rate_limits_hook`,
which enforces the same `TagRateLimitEntry` shape but nested per-deployment
under `model_info.tag_rate_limits`, once per routing hop
(`async_filter_deployments`). A global entry has no deployment/routing-group
to reconcile -- there is exactly one config value, read once -- so this hook
reuses that sibling's free, already-hardened helper functions
(`_entry_applies`, the Lua atomic check-and-increment scripts, cache
partitioning, bucket-key hashing primitives) directly rather than duplicating
them, but implements its own, much smaller admission/accounting engine: no
`_LimitsIndex`, no routing-group or team-alias resolution, no per-deployment
dedup signatures.

Two independent entry-level knobs decide who a global entry applies to and
how its bucket is shared:

- `apply_to_key_alias`: unset means every request, any key, any model.  Set
  to a list of virtual-key aliases, only those keys' requests count.
- `scope_by_key_hash` (already exists on `TagRateLimitEntry`): whether the
  keys an entry applies to share one bucket, or each gets its own.

`async_pre_call_hook` runs before Router constructs `Logging`/`litellm_logging_obj`
for this request (see `common_request_processing.py`: `pre_call_hook` fires
well before `base_process_llm_request` builds the logging object), so unlike
`model_based_tag_rate_limits_hook` this hook cannot stash pending concurrency
reservations on `data["litellm_logging_obj"].model_call_details` -- that
object doesn't exist yet. Per-request state is instead kept on a
`ContextVar`-based stash, the same established pattern
`parallel_request_limiter_v3.py`'s v3 handler already uses for exactly this
problem: the ContextVar is inherited by every asyncio Task forked from this
request's own task (the SDK call, streaming generators, the logging worker),
so concurrent requests never see each other's stash regardless of a
caller-supplied `litellm_call_id` colliding, and `owner_litellm_call_id` only
exists to tell a nested LiteLLM call (e.g. a guardrail's own LLM judge call)
apart from the owning request.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.model_based_tag_rate_limits_hook import (
    _ATOMIC_UNITS,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _BACKGROUND_TASKS,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _CONCURRENCY_MIN_SAFETY_TTL_SECONDS,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _EMPTY_MAPPING,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _LIMIT_UNITS,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _UNIT_TO_GROUP_FIELD,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _UNIT_TO_RATE_LIMIT_TYPE,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    TAG_RL_CHECK_AND_INCR_SCRIPT,
    TAG_RL_DECR_FLOOR_ZERO_SCRIPT,
    _bucket_ttl_seconds,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _entry_applies,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _extract_identity,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _extract_key_alias,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _extract_key_hash,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _fixed_length_identity,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _LimitUnit,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _partition_key,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _PartitionKey,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _PartitionOperations,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _policy_fingerprint,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching model_based_tag_rate_limits_hook's identical import
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router_strategy.tag_based_routing import (
    _get_tags_from_request_kwargs,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching model_based_tag_rate_limits_hook's identical import
)
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.router import TagRateLimitEntry, TagRateLimits
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span: TypeAlias = _Span
else:
    Span: TypeAlias = object


def _hash_tag(entry: TagRateLimitEntry, unit: _LimitUnit, tag_value: str, key_hash: str | None) -> str:
    """
    Global-hook equivalent of `model_based_tag_rate_limits_hook._hash_tag`,
    without a `model_group`/deployment-scope/team-scope dimension -- a global
    entry has none of those. Namespaced under `tag_rl:global:` so it can never
    collide with that sibling hook's own `tag_rl:{model_group}:...` keys even
    if an operator names a deployment "global": every key also differs by
    `unit`/`name`/`tag_id`/`_policy_fingerprint`, and the two hooks' entries
    are never meant to share a bucket in the first place.
    """
    key_suffix: Final = f":key:{key_hash}" if key_hash is not None else ""
    policy_suffix: Final = f":policy:{_policy_fingerprint(entry)}"
    return f"tag_rl:global:{unit}:{entry.name}:{entry.tag_id}:{_fixed_length_identity(tag_value)}{key_suffix}{policy_suffix}"


def _bucket_key(
    entry: TagRateLimitEntry, unit: _LimitUnit, tag_value: str, bucket_id: int, key_hash: str | None
) -> str:
    return f"{{{_hash_tag(entry, unit, tag_value, key_hash)}}}:{bucket_id}"


def _inflight_key(entry: TagRateLimitEntry, unit: _LimitUnit, tag_value: str, key_hash: str | None) -> str:
    return f"{{{_hash_tag(entry, unit, tag_value, key_hash)}}}:inflight"


@dataclass(frozen=True, slots=True)
class _ClassifiedGlobalCheck:
    unit: _LimitUnit
    entry: TagRateLimitEntry
    tag_value: str
    key: str
    is_atomic: bool


@dataclass(frozen=True, slots=True)
class _CachePartition:
    internal_usage_cache: InternalUsageCache
    v3: _PROXY_MaxParallelRequestsHandler_v3


@dataclass(slots=True)
class _GlobalTagRateLimitStash:
    """Per-request bookkeeping `async_pre_call_hook` hands to the success/
    failure/disconnect callbacks -- see module docstring for why this is a
    `ContextVar`, not `model_call_details`."""

    owner_litellm_call_id: str | None = None
    admission_time: float | None = None
    pending_concurrency_keys: list[tuple[str, _PartitionKey]] = field(default_factory=list)  # mutable-ok: queue


_request_stash: Final[ContextVar[_GlobalTagRateLimitStash | None]] = ContextVar(
    "global_tag_rate_limits_request_stash", default=None
)


def _claim_stash_for_data(data: Mapping[str, object]) -> _GlobalTagRateLimitStash:
    stash = _request_stash.get()
    if stash is None:
        stash = _GlobalTagRateLimitStash()
        _request_stash.set(stash)
    owner_call_id: Final = data.get("litellm_call_id")
    if isinstance(owner_call_id, str):
        stash.owner_litellm_call_id = owner_call_id
    return stash


def _stash_for_call(litellm_call_id: str | None) -> _GlobalTagRateLimitStash | None:
    stash: Final = _request_stash.get()
    if stash is None:
        return None
    if stash.owner_litellm_call_id is None or litellm_call_id is None:
        return stash
    return stash if litellm_call_id == stash.owner_litellm_call_id else None


def _call_id_from_kwargs(kwargs: object) -> str | None:
    if not isinstance(kwargs, dict):
        return None
    call_id: Final = kwargs.get("litellm_call_id")
    return call_id if isinstance(call_id, str) else None


def _resolve_max_in_memory_cache_size() -> int | None:
    """Same shape as `model_based_tag_rate_limits_hook`'s own function, reading
    this hook's own `litellm_settings` knob instead."""
    configured: Final = litellm.global_tag_rate_limits_max_in_memory_cache_size
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return configured
    if configured is not None:
        verbose_proxy_logger.warning(
            "global_tag_rate_limits_hook: global_tag_rate_limits_max_in_memory_cache_size=%r is not a positive "
            "integer; falling back to the default in-memory cache size.",
            configured,
        )
    return None


class _PROXY_GlobalTagRateLimitsHook(  # pyright: ignore[reportUnusedClass]  # only referenced via the deferred import in litellm_logging.py's callback resolver; basedpyright doesn't trace that usage
    CustomLogger
):
    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._redis_cache: Final = internal_usage_cache.redis_cache
        self._time_provider = time_provider or datetime.now
        self._partitions: dict[_PartitionKey, _CachePartition] = {}  # mutable-ok: lazily memoized; see _partition_for
        self._partitions_lock = asyncio.Lock()
        default_partition: Final = self._build_partition(_resolve_max_in_memory_cache_size())
        self._partitions[None] = default_partition
        self.internal_usage_cache = default_partition.internal_usage_cache
        self._lock = asyncio.Lock()
        redis_cache: Final = self._redis_cache
        self._check_and_incr_script = (
            redis_cache.async_register_script(TAG_RL_CHECK_AND_INCR_SCRIPT) if redis_cache is not None else None
        )
        self._decr_floor_zero_script = (
            redis_cache.async_register_script(TAG_RL_DECR_FLOOR_ZERO_SCRIPT) if redis_cache is not None else None
        )
        self._config_cache_key: object | None = None
        self._config: TagRateLimits | None = None

    def _refresh_config(self) -> TagRateLimits | None:
        """Re-validates `litellm.global_tag_rate_limits` whenever the object
        identity changes (a config reload replaces it wholesale via
        `setattr(litellm, key, value)`), so a hot-reloaded config takes effect
        on the very next request with no staleness window and no TTL to tune."""
        raw: Final = getattr(litellm, "global_tag_rate_limits", None)
        if raw is not self._config_cache_key:
            self._config = TagRateLimits.model_validate(raw) if raw else None
            self._config_cache_key = raw
        return self._config

    def _build_partition(self, cache_size_override: int | None) -> _CachePartition:
        dual_cache: Final = DualCache(
            in_memory_cache=InMemoryCache(max_size_in_memory=cache_size_override),
            redis_cache=self._redis_cache,
        )
        cache: Final = InternalUsageCache(dual_cache=dual_cache)
        return _CachePartition(
            internal_usage_cache=cache,
            v3=_PROXY_MaxParallelRequestsHandler_v3(cache, time_provider=self._time_provider),
        )

    async def _partition_for(self, partition_key: _PartitionKey) -> _CachePartition:
        existing: Final = self._partitions.get(partition_key)
        if existing is not None:
            return existing
        async with self._partitions_lock:
            existing_after_lock: Final = self._partitions.get(partition_key)
            if existing_after_lock is not None:
                return existing_after_lock
            cache_size_override: Final = partition_key[-1] if partition_key is not None else None
            built: Final = self._build_partition(cache_size_override)
            self._partitions[partition_key] = (
                built  # mutable-ok: lazily memoized per distinct partition key, guarded by _partitions_lock above
            )
            return built

    async def _check_and_increment_one(
        self, cache: InternalUsageCache, key: str, limit: float, increment: float, ttl: int
    ) -> tuple[bool, float]:
        if self._check_and_incr_script is not None:
            raw: Final = await self._check_and_incr_script(keys=(key,), args=(limit, increment, ttl))
            return bool(raw[0]), float(raw[1])
        async with self._lock:
            current_value: Final = await cache.async_get_cache(key=key, litellm_parent_otel_span=None)
            current: Final = float(current_value) if current_value is not None else 0.0
            if current + increment > limit:
                return False, current
            new_value: Final = current + increment
            await cache.async_set_cache(key=key, value=new_value, ttl=ttl, litellm_parent_otel_span=None)
            return True, new_value

    async def _decrement_floor_zero(self, cache: InternalUsageCache, key: str, delta: float) -> None:
        if self._decr_floor_zero_script is not None:
            await self._decr_floor_zero_script(keys=(key,), args=(delta,))
            return
        async with self._lock:
            current_value: Final = await cache.async_get_cache(key=key, litellm_parent_otel_span=None)
            current: Final = float(current_value) if current_value is not None else 0.0
            await cache.async_set_cache(key=key, value=max(0.0, current + delta), litellm_parent_otel_span=None)

    async def _atomic_check_and_increment(
        self,
        checks: Sequence[tuple[InternalUsageCache, str, float, float, int]],
    ) -> tuple[int | None, tuple[float, ...]]:
        """All-or-nothing atomic admission across `checks` -- see
        `model_based_tag_rate_limits_hook._PROXY_ModelBasedTagRateLimitsHook._atomic_check_and_increment`'s
        own docstring for the full rationale (refund-on-rollback, why a
        raising key's own outcome is never refunded); identical logic,
        duplicated rather than shared since it lives as instance methods
        rather than free functions."""
        if not checks:
            return None, ()
        admitted_values: Final = []  # mutable-ok: sequential async accumulator, discardable on early rejection
        for index, (cache, key, limit, increment, ttl) in enumerate(checks):
            admitted = False
            try:
                admitted, value = await self._check_and_increment_one(cache, key, limit, increment, ttl)
            finally:
                if not admitted:
                    await self._refund_admitted(checks, up_to_index=index)
            if admitted:
                admitted_values.append(value)  # mutable-ok: see accumulator comment above
                continue
            return index, (value,)
        return None, tuple(admitted_values)

    async def _refund_admitted(
        self, checks: Sequence[tuple[InternalUsageCache, str, float, float, int]], up_to_index: int
    ) -> None:
        for refund_index in range(up_to_index):
            refund_cache, refund_key, _limit, refund_increment, _ttl = checks[refund_index]
            try:
                await self._decrement_floor_zero(refund_cache, refund_key, -refund_increment)
            except Exception as e:  # noqa: BLE001 - one failed refund must not block refunding the rest
                verbose_proxy_logger.warning(
                    "global_tag_rate_limits_hook: failed to refund %s on rollback: %s", refund_key, e
                )

    async def _release_keys(self, reservations: Sequence[tuple[str, _PartitionKey]]) -> None:
        for key, partition_key in reservations:
            try:
                partition = await self._partition_for(partition_key)  # not Final: rebound each loop iteration
                await self._decrement_floor_zero(partition.internal_usage_cache, key, -1.0)
            except Exception as e:  # noqa: BLE001 - releasing a slot must never raise into the caller's request path
                verbose_proxy_logger.warning(
                    "global_tag_rate_limits_hook: failed to release concurrency slot %s: %s", key, e
                )

    @staticmethod
    def _ttl_for(unit: _LimitUnit, entry: TagRateLimitEntry) -> int:
        if unit == "concurrency":
            requested_ttl: Final = entry.key_ttl_seconds if entry.key_ttl_seconds is not None else entry.period_seconds
            return max(requested_ttl, _CONCURRENCY_MIN_SAFETY_TTL_SECONDS)
        return _bucket_ttl_seconds(entry)

    def _classify(
        self, config: TagRateLimits, tags: Sequence[str], key_alias: str | None, key_hash: str | None, now: float
    ) -> tuple[_ClassifiedGlobalCheck, ...]:
        classified: Final = []  # mutable-ok: sequential accumulator, immediately frozen into a tuple below
        for unit in _LIMIT_UNITS:
            group = getattr(config, _UNIT_TO_GROUP_FIELD[unit])
            if group is None:
                continue
            for entry in group.limits:
                tag_value = _extract_identity(tags, entry.tag_id)
                if tag_value is None:
                    continue
                if not _entry_applies(entry, tag_value, tags, key_alias):
                    continue
                effective_key_hash = key_hash if entry.scope_by_key_hash else None
                if unit == "concurrency":
                    key = _inflight_key(entry, unit, tag_value, key_hash=effective_key_hash)
                    classified.append(
                        _ClassifiedGlobalCheck(unit, entry, tag_value, key, is_atomic=True)
                    )  # mutable-ok: see comment above
                    continue
                bucket_id = int(now) // entry.period_seconds
                key = _bucket_key(entry, unit, tag_value, bucket_id, key_hash=effective_key_hash)
                classified.append(  # mutable-ok: see comment above
                    _ClassifiedGlobalCheck(unit, entry, tag_value, key, is_atomic=unit in _ATOMIC_UNITS)
                )
        return tuple(classified)

    async def _read_only_values(
        self, read_only_checks: Sequence[_ClassifiedGlobalCheck], parent_otel_span: Span | None
    ) -> tuple[float | None, ...]:
        if not read_only_checks:
            return ()
        indices_by_partition: Final[dict[_PartitionKey, list[int]]] = {}  # mutable-ok: grouped, reassembled below
        for index, check in enumerate(read_only_checks):
            partition_key = _partition_key(check.entry)
            indices = indices_by_partition.setdefault(partition_key, [])  # mutable-ok: see above
            indices.append(index)  # mutable-ok: see comment above
        values_by_index: Final[dict[int, float | None]] = {}  # mutable-ok: see comment above
        for partition_key, indices in indices_by_partition.items():
            partition = await self._partition_for(partition_key)  # not Final: rebound each loop iteration
            keys = [read_only_checks[i].key for i in indices]  # mutable-ok: async_batch_get_cache needs a real list
            redis_cache = partition.internal_usage_cache.dual_cache.redis_cache
            if redis_cache is not None:
                redis_values: Mapping[str, object] = await redis_cache.async_batch_get_cache(
                    key_list=keys, parent_otel_span=parent_otel_span
                )
                resolved = [redis_values.get(key) for key in keys]  # mutable-ok: needs a real list
            else:
                current_values = await partition.internal_usage_cache.async_batch_get_cache(
                    keys=keys, parent_otel_span=parent_otel_span, local_only=True
                )
                missing = [None] * len(keys)  # mutable-ok: async_batch_get_cache requires a real list; see above
                resolved = current_values if current_values is not None else missing
            for i, value in zip(indices, resolved):
                values_by_index[i] = value  # mutable-ok: see comment above
        return tuple(values_by_index[i] for i in range(len(read_only_checks)))

    def _raise_if_over_limit(
        self,
        read_only_checks: Sequence[_ClassifiedGlobalCheck],
        current_values: Sequence[float | None],
        model: str | None,
    ) -> None:
        for check, current_value in zip(read_only_checks, current_values):
            current = float(current_value) if current_value is not None else 0.0
            if current < check.entry.limit:
                continue
            self._raise_over_limit(check.unit, check.entry, check.tag_value, model, current=current)

    def _raise_over_limit(
        self, unit: _LimitUnit, entry: TagRateLimitEntry, tag_value: str, model: str | None, current: float
    ) -> None:
        verbose_proxy_logger.debug(
            "global_tag_rate_limits_hook: OVER_LIMIT unit=%s name=%s tag_id=%s tag_value=%s current=%s limit=%s",
            unit,
            entry.name,
            entry.tag_id,
            tag_value,
            current,
            entry.limit,
        )
        raise ProxyRateLimitError(
            detail={  # mutable-ok: async_log_failure_event and generic proxy exception rendering branch on isinstance(exc.detail, dict)
                "error": "tag_rate_limit_exceeded",
                "type": unit,
                "tag_id": entry.tag_id,
                "tag_value": tag_value,
                "limit_name": entry.name,
                "limit": entry.limit,
                "period_seconds": entry.period_seconds,
            },
            headers={"retry-after": str(entry.period_seconds)},  # mutable-ok: same as detail
            rate_limit_type=_UNIT_TO_RATE_LIMIT_TYPE[unit],
            model=model,
            llm_provider="litellm_proxy",
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,  # mutable-ok: must match CustomLogger.async_pre_call_hook's own base signature exactly
        call_type: str,
    ) -> dict:  # mutable-ok: must match CustomLogger.async_pre_call_hook's own base signature exactly
        config: Final = self._refresh_config()
        if config is None:
            return data

        # Unlike model_based_tag_rate_limits_hook's async_filter_deployments
        # (called once per routing hop, so a still-queued reservation can
        # legitimately belong to an earlier, already-failed hop of the same
        # request), async_pre_call_hook fires exactly once per request --
        # there is no "prior hop" case here, so no stale-reservation release
        # is needed at the top of admission.
        stash: Final = _claim_stash_for_data(data)

        metadata_variable_name: Final = get_metadata_variable_name_from_kwargs(data)
        tags: Final = _get_tags_from_request_kwargs(data, metadata_variable_name=metadata_variable_name)
        key_alias: Final = user_api_key_dict.key_alias
        key_hash: Final = user_api_key_dict.api_key

        now: Final = self._time_provider().timestamp()
        stash.admission_time = now
        classified: Final = self._classify(config, tags, key_alias, key_hash, now)
        if not classified:
            return data

        read_only_checks: Final = tuple(c for c in classified if not c.is_atomic)
        atomic_checks: Final = tuple(c for c in classified if c.is_atomic)

        model: Final = data.get("model") if isinstance(data.get("model"), str) else None
        current_values: Final = await self._read_only_values(read_only_checks, parent_otel_span=None)
        self._raise_if_over_limit(read_only_checks, current_values, model)

        if atomic_checks:
            atomic_partitions_list: Final = []  # mutable-ok: sequential async lookups, one per atomic_checks entry
            for check in atomic_checks:
                atomic_partitions_list.append(
                    await self._partition_for(_partition_key(check.entry))
                )  # mutable-ok: see comment above
            atomic_partitions: Final = tuple(atomic_partitions_list)
            failing_index, values = await self._atomic_check_and_increment(
                tuple(
                    (
                        partition.internal_usage_cache,
                        check.key,
                        check.entry.limit,
                        1.0,
                        self._ttl_for(check.unit, check.entry),
                    )
                    for partition, check in zip(atomic_partitions, atomic_checks)
                )
            )
            if failing_index is not None:
                failing_check: Final = atomic_checks[failing_index]
                self._raise_over_limit(
                    failing_check.unit, failing_check.entry, failing_check.tag_value, model, current=values[0]
                )

            concurrency_reservations: Final = tuple(
                (check.key, _partition_key(check.entry)) for check in atomic_checks if check.unit == "concurrency"
            )
            if concurrency_reservations:
                stash.pending_concurrency_keys.extend(concurrency_reservations)  # mutable-ok: see field's own docstring

        return data

    async def async_release_disconnect_state_hook(self, request_data: Mapping[str, object]) -> None:
        stash: Final = _stash_for_call(_call_id_from_kwargs(dict(request_data)))
        if stash is None or not stash.pending_concurrency_keys:
            return
        release_keys: Final = tuple(stash.pending_concurrency_keys)
        stash.pending_concurrency_keys.clear()
        await self._release_keys(release_keys)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        if isinstance(kwargs.get("exception"), ProxyRateLimitError):
            detail: Final = (
                kwargs["exception"].detail if isinstance(kwargs["exception"].detail, dict) else _EMPTY_MAPPING
            )
            if detail.get("error") == "tag_rate_limit_exceeded":
                return

        stash: Final = _stash_for_call(_call_id_from_kwargs(kwargs))
        if stash is None or not stash.pending_concurrency_keys:
            return
        release_keys: Final = tuple(stash.pending_concurrency_keys)
        stash.pending_concurrency_keys.clear()
        await self._release_keys(release_keys)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        stash: Final = _stash_for_call(_call_id_from_kwargs(kwargs))
        if stash is not None and stash.pending_concurrency_keys:
            release_keys: Final = tuple(stash.pending_concurrency_keys)
            stash.pending_concurrency_keys.clear()
            release_task: Final = asyncio.create_task(self._release_keys(release_keys))
            _BACKGROUND_TASKS.add(release_task)  # mutable-ok: see _BACKGROUND_TASKS's own docstring
            release_task.add_done_callback(_BACKGROUND_TASKS.discard)

        config: Final = self._refresh_config()
        if config is None:
            return

        standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
        if standard_logging_object is None:
            return

        litellm_params_for_metadata: Final = kwargs.get("litellm_params") or kwargs
        metadata_variable_name: Final = get_metadata_variable_name_from_kwargs(litellm_params_for_metadata)
        key_hash: Final = _extract_key_hash(litellm_params_for_metadata, metadata_variable_name)
        key_alias: Final = _extract_key_alias(litellm_params_for_metadata, metadata_variable_name)

        tags: Final = _get_tags_from_request_kwargs(kwargs, metadata_variable_name=metadata_variable_name)
        if not tags:
            return

        now: Final = (
            stash.admission_time
            if stash is not None and stash.admission_time is not None
            else self._time_provider().timestamp()
        )
        increment_by_unit: Final[Mapping[_LimitUnit, float]] = MappingProxyType(
            {
                "tokens": float(standard_logging_object.get("total_tokens") or 0),
                "dollars": float(standard_logging_object.get("response_cost") or 0),
            }
        )

        operation_by_entry: Final = []  # mutable-ok: sequential accumulator over config groups, immediately used below
        for unit in ("tokens", "dollars"):
            group = getattr(config, _UNIT_TO_GROUP_FIELD[unit])
            if group is None:
                continue
            for entry in group.limits:
                tag_value = _extract_identity(tags, entry.tag_id)
                if tag_value is None:
                    continue
                if not _entry_applies(entry, tag_value, tags, key_alias):
                    continue
                increment_value = increment_by_unit[unit]
                if increment_value == 0:
                    continue
                bucket_id = int(now) // entry.period_seconds
                key_hash_for_entry = key_hash if entry.scope_by_key_hash else None
                key = _bucket_key(entry, unit, tag_value, bucket_id, key_hash=key_hash_for_entry)
                operation_by_entry.append(  # mutable-ok: see comment above
                    (
                        entry,
                        RedisPipelineIncrementOperation(
                            key=key, increment_value=increment_value, ttl=_bucket_ttl_seconds(entry)
                        ),
                    )
                )

        if not operation_by_entry:
            return

        operations_by_partition: Final[_PartitionOperations] = {}  # mutable-ok: grouped by cache partition below
        for entry, operation in operation_by_entry:
            partition_key = _partition_key(entry)
            operations = operations_by_partition.setdefault(partition_key, [])  # mutable-ok: see above
            operations.append(operation)  # mutable-ok: see comment above

        for partition_key, group_operations in operations_by_partition.items():
            partition = await self._partition_for(partition_key)  # not Final: rebound each loop iteration
            accounting_task = asyncio.create_task(  # not Final: rebound each loop iteration
                partition.v3.async_increment_tokens_with_ttl_preservation(
                    pipeline_operations=tuple(group_operations), parent_otel_span=None
                )
            )
            _BACKGROUND_TASKS.add(accounting_task)  # mutable-ok: see _BACKGROUND_TASKS's own docstring
            accounting_task.add_done_callback(_BACKGROUND_TASKS.discard)
