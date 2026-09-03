"""
Tag-scoped token, request, dollar, and concurrency rate limits declared once,
globally, in `litellm_settings.global_tag_rate_limits` and enforced in
`async_pre_call_hook`, before Router does any routing -- so a limit applies
regardless of which model or fallback chain the request ends up hitting.

Model-independent sibling of `model_based_tag_rate_limits_hook`, which
enforces the same `TagRateLimitEntry` shape per-deployment instead. Reuses
that sibling's shared helpers (`entry_applies`, the atomic Lua scripts, cache
partitioning, bucket-key hashing) from `tag_rate_limits_shared.py`, but has
its own smaller admission/accounting engine with no routing-group or
per-deployment concerns.

Three entry-level knobs: `apply_to_key_alias` scopes an entry to specific
virtual-key aliases; `apply_to_models` scopes it to specific caller-facing
model names, letting one entry cap a whole fallback chain as a unit (a
rejection then carries `detail["cross_model_scope"] = True` so
`_pre_call_with_fallbacks` re-raises instead of silently admitting the
request through an unlisted fallback model); `scope_by_key_hash` controls
whether matching keys share one bucket or each gets its own.

`data["litellm_logging_obj"]` does exist by the time `async_pre_call_hook`
runs, but `_pre_call_with_fallbacks` re-runs the whole pre-call pipeline
(building a fresh `Logging` object each time) once per fallback model on the
same `litellm_call_id`, so unlike `model_based_tag_rate_limits_hook`'s
per-Router-hop reservations, a stash keyed to one attempt's own
`model_call_details` wouldn't survive to a later attempt. Per-request state
instead lives on a `ContextVar`-based stash (the same pattern
`parallel_request_limiter_v3.py` uses for the identical admission-to-release
problem), keyed by `litellm_call_id` so a nested LiteLLM call sharing the
same inherited context (an LLM-judge guardrail, for example) gets its own
isolated entry instead of releasing the outer call's still-pending
reservation early. Confirmed live: a real streaming client disconnect
correctly releases its concurrency reservation through this mechanism.

`Router.abatch_completion`'s comma-separated `model` fans this one admission
out into several real, independent LLM calls below this hook entirely (it
never re-runs `async_pre_call_hook` per branch the way
`model_based_tag_rate_limits_hook`'s per-Router-hop admission does), each of
which reliably fires its own terminal success/failure event -- confirmed
live against the real dispatch. A single concurrency reservation for the
whole batch would be released by whichever branch finishes first, letting
the still-running siblings push real concurrent calls past the configured
cap, so admission instead reserves one unit per comma-separated model (see
`_non_racing_batch_width`) and each branch's own event releases just its own
share (see `_release_own_share`). The racing
`abatch_completion_fastest_response` variant is excluded from that: it
cancels every losing branch without ever firing a terminal event for it
(confirmed live), so reserving more than the single unit it already does
would leak every losing branch's share until the safety TTL.
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
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # shared private helper, reused by model_based_tag_rate_limits_hook too
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    ATOMIC_UNITS as _ATOMIC_UNITS,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    BACKGROUND_TASKS as _BACKGROUND_TASKS,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    CONCURRENCY_MIN_SAFETY_TTL_SECONDS as _CONCURRENCY_MIN_SAFETY_TTL_SECONDS,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    LIMIT_UNITS as _LIMIT_UNITS,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    TAG_RL_CHECK_AND_INCR_SCRIPT,
    TAG_RL_DECR_FLOOR_ZERO_SCRIPT,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    UNIT_TO_GROUP_FIELD as _UNIT_TO_GROUP_FIELD,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    UNIT_TO_RATE_LIMIT_TYPE as _UNIT_TO_RATE_LIMIT_TYPE,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    LimitUnit as _LimitUnit,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    PartitionKey as _PartitionKey,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    PartitionOperations as _PartitionOperations,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    bucket_ttl_seconds as _bucket_ttl_seconds,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    entry_applies as _entry_applies,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    extract_identity as _extract_identity,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    extract_key_alias as _extract_key_alias,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    extract_key_hash as _extract_key_hash,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    fixed_length_identity as _fixed_length_identity,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    order_tags_for_identity_resolution as _order_tags_for_identity_resolution,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    partition_key as _partition_key,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    policy_fingerprint as _policy_fingerprint,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    resolve_authoritative_metadata_variable_name as _resolve_authoritative_metadata_variable_name,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router_strategy.tag_based_routing import (
    _get_tags_from_request_kwargs,  # pyright: ignore[reportPrivateUsage]  # shared private helper, reused by model_based_tag_rate_limits_hook too
)
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.router import TagRateLimitEntry, TagRateLimits

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span: TypeAlias = _Span
else:
    Span: TypeAlias = object


def _entry_applies_any_candidate_model(
    entry: TagRateLimitEntry, tags: Sequence[str], key_alias: str | None, candidate_models: frozenset[str]
) -> bool:
    """Same as `_entry_applies`, except an `apply_to_models`-scoped entry
    counts as applying if ANY of `candidate_models` is in scope -- not just a
    single caller-visible name. Two call sites need this: at admission, a
    comma-separated `model` (a non-racing or racing batch dispatch) names
    several models at once, any of which should trip a chain-wide cap; at
    success-event accounting, a `_pre_call_with_fallbacks` retry re-admits
    with a different model for the same call_id, and an entry that matched
    an earlier attempt must still get its accounting."""
    if not candidate_models:
        return _entry_applies(entry, tags, key_alias, None)
    return any(_entry_applies(entry, tags, key_alias, model) for model in candidate_models)


def _own_concurrency_keys(
    config: TagRateLimits,
    tags: Sequence[str],
    key_alias: str | None,
    key_hash: str | None,
    candidate_models: frozenset[str],
) -> frozenset[str]:
    """The exact `_inflight_key`s this hop's own admission would have
    reserved, mirroring `_classify`'s own concurrency-unit matching. A
    request can match more than one concurrency-scoped `TagRateLimitEntry`
    (e.g. a global cap and a named per-team cap on the same tag), and each
    match is its own key -- a terminal event must release every one of
    them, not just one."""
    group: Final = getattr(config, _UNIT_TO_GROUP_FIELD["concurrency"])
    if group is None:
        return frozenset()
    return frozenset(
        _inflight_key(entry, "concurrency", tag_value, key_hash=key_hash if entry.scope_by_key_hash else None)
        for entry in group.limits
        if (tag_value := _extract_identity(tags, entry.tag_id)) is not None
        and _entry_applies_any_candidate_model(entry, tags, key_alias, candidate_models)
    )


def _individual_model_names(model: str | None, call_type: str) -> tuple[str, ...]:
    """`route_llm_request.py` splits a comma-separated `model` on this exact
    condition before fanning out through `Router.abatch_completion`/
    `abatch_completion_fastest_response` -- an `apply_to_models`-scoped entry
    must check each of those individual names, not the raw joined string
    (which is never a member of any caller-configured `apply_to_models`
    list), or a caller trivially bypasses a chain-wide cap by adding a
    second model to the comma list."""
    if model is None:
        return ()
    if call_type != "acompletion" or "," not in model:
        return (model,)
    return tuple(m.strip() for m in model.split(","))


def _non_racing_batch_width(data: Mapping[str, object], call_type: str) -> int:
    """`Router.abatch_completion` (not `abatch_completion_fastest_response`)
    fans this one admission out into one independent real LLM call per
    comma-separated model in `model`, and every one of those branches
    reliably fires its own terminal success/failure event -- confirmed
    empirically, no cancellation involved, unlike the racing
    `fastest_response` variant, which cancels every losing branch without
    ever firing a terminal event for it and must keep reserving a single
    unit released by whichever branch finishes first.

    A single concurrency reservation for the whole non-racing dispatch would
    get released by whichever branch finishes first, letting the
    still-running siblings push real concurrent calls past the configured
    cap. Reserving one unit per branch instead, released one at a time as
    each branch's own event fires, keeps the count accurate.

    Mirrors the exact condition `route_llm_request.py` uses to route into
    `abatch_completion` in the first place, so this only ever fires for a
    request that will actually take that path. `abatch_completion` itself
    also accepts a nested `messages: list[list[...]]` ("N requests to M
    models") and dispatches one branch per (message, model) pair, not one
    per model -- veria-ai's finding on this file.
    """
    if call_type != "acompletion" or data.get("fastest_response"):
        return 1
    model_field: Final = data.get("model")
    if not isinstance(model_field, str) or "," not in model_field:
        return 1
    model_count: Final = len(model_field.split(","))
    messages_field: Final = data.get("messages")
    message_list_count: Final = (
        len(messages_field)
        if isinstance(messages_field, list) and messages_field and all(isinstance(m, list) for m in messages_field)
        else 1
    )
    return model_count * message_list_count


def _hash_tag(entry: TagRateLimitEntry, unit: _LimitUnit, tag_value: str, key_hash: str | None) -> str:
    """Namespaced under `tag_rl:global:` so it never collides with
    `model_based_tag_rate_limits_hook`'s own `tag_rl:{model_group}:...` keys."""
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
    """Per-call bookkeeping `async_pre_call_hook` hands to that same call's
    success/failure/disconnect callbacks -- see module docstring for why this
    lives on a `ContextVar`, not `model_call_details`.

    Keyed by `litellm_call_id` rather than one shared mutable instance so a
    nested LiteLLM call (an LLM-judge guardrail) that mints its own call id
    but inherits the same context doesn't release the outer call's
    still-pending reservation early.
    """

    admission_time: float | None = None
    # Every model any admission attempt for this call_id has classified
    # entries against, accumulated rather than overwritten: a
    # _pre_call_with_fallbacks retry re-runs admission with a *different*
    # model for the same call_id, and an apply_to_models entry that matched
    # an earlier attempt must still get its accounting at success time even
    # though the request ultimately serves from a later attempt's model.
    admitted_models: frozenset[str] = field(default_factory=frozenset)
    # One entry per reserved unit, not one entry per distinct key: a
    # non-racing batch dispatch (see _non_racing_batch_width) reserves and
    # appends `batch_width` entries per matching policy, and each branch's
    # own terminal event pops exactly one entry per its own matching
    # policies -- see _release_own_share.
    pending_concurrency_keys: list[tuple[str, _PartitionKey]] = field(default_factory=list)  # mutable-ok: queue
    # Keys already charged for this call_id, so a fallback retry (same
    # litellm_call_id, different model) renews instead of double-charging.
    charged_request_keys: list[str] = field(default_factory=list)  # mutable-ok: see comment above
    # key_hash of whoever first claimed this stash. litellm_call_id is
    # caller-controlled (x-litellm-call-id), so only a later admission with
    # the same authenticated key_hash may renew this stash's charges.
    owner_key_hash: str | None = None


# Sentinel key for a call with no litellm_call_id at all (claim and lookup
# both fall back to this same key, so behavior for that degenerate case is
# unchanged: everything without a call id still shares one bucket).
_NO_CALL_ID: Final = "<no-call-id>"

_StashByCallId: TypeAlias = dict[
    str, _GlobalTagRateLimitStash
]  # mutable-ok: per-call-id entries added over a request's lifetime, see class docstring

_request_stash: Final[ContextVar[_StashByCallId | None]] = ContextVar(
    "global_tag_rate_limits_request_stash", default=None
)


def _claim_stash_for_data(data: Mapping[str, object]) -> _GlobalTagRateLimitStash:
    by_call_id: _StashByCallId | None = _request_stash.get()  # rebind-ok: lazily initialized below if never set
    if by_call_id is None:
        by_call_id = {}  # rebind-ok: see above  # mutable-ok: see _StashByCallId
        _request_stash.set(by_call_id)
    owner_call_id: Final = data.get("litellm_call_id")
    key: Final = owner_call_id if isinstance(owner_call_id, str) else _NO_CALL_ID
    stash = by_call_id.get(key)  # rebind-ok: reassigned just below when newly created
    if stash is None:
        stash = _GlobalTagRateLimitStash()  # rebind-ok: see above
        by_call_id[key] = stash  # mutable-ok: see class docstring
    return stash


def _stash_for_call(litellm_call_id: str | None) -> _GlobalTagRateLimitStash | None:
    by_call_id: Final = _request_stash.get()
    if by_call_id is None:
        return None
    key: Final = litellm_call_id if litellm_call_id is not None else _NO_CALL_ID
    return by_call_id.get(key)


def _call_id_from_kwargs(kwargs: Mapping[str, object]) -> str | None:
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
        self, cache: InternalUsageCache, key: str, limit: float, increment: float, ttl: int, refresh_ttl: bool
    ) -> tuple[bool, float]:
        if self._check_and_incr_script is not None:
            raw: Final = await self._check_and_incr_script(
                keys=(key,), args=(limit, increment, ttl, 1 if refresh_ttl else 0)
            )
            return bool(raw[0]), float(raw[1])
        async with self._lock:
            current_value: Final = await cache.async_get_cache(key=key, litellm_parent_otel_span=None)
            current: Final = float(current_value) if current_value is not None else 0.0
            if current + increment > limit:
                return False, current
            new_value: Final = current + increment
            await cache.async_set_cache(
                key=key, value=new_value, ttl=ttl, refresh_ttl=refresh_ttl, litellm_parent_otel_span=None
            )
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
        checks: Sequence[tuple[InternalUsageCache, str, float, float, int, bool]],
    ) -> tuple[int | None, tuple[float, ...]]:
        """All-or-nothing atomic admission across `checks`: on a rejection,
        refunds every check admitted earlier in this batch."""
        if not checks:
            return None, ()
        admitted_values: Final = []  # mutable-ok: sequential async accumulator, discardable on early rejection
        for index, (cache, key, limit, increment, ttl, refresh_ttl) in enumerate(checks):
            admitted = False
            try:
                admitted, value = await self._check_and_increment_one(cache, key, limit, increment, ttl, refresh_ttl)
            finally:
                if not admitted:
                    await self._refund_admitted(checks, up_to_index=index)
            if admitted:
                admitted_values.append(value)  # mutable-ok: see accumulator comment above
                continue
            return index, (value,)
        return None, tuple(admitted_values)

    async def _refund_admitted(
        self, checks: Sequence[tuple[InternalUsageCache, str, float, float, int, bool]], up_to_index: int
    ) -> None:
        for refund_index in range(up_to_index):
            refund_cache, refund_key, _limit, refund_increment, _ttl, _refresh_ttl = checks[refund_index]
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
    def _admitted_models_after(
        admitted_models: frozenset[str], candidate_models: frozenset[str], renewal_allowed: bool
    ) -> frozenset[str]:
        """Only called once this admission attempt has cleared every check
        without raising -- a rejected attempt's models must never join
        admitted_models, or a later successful attempt's accounting could
        wrongly credit an apply_to_models entry that never actually admitted
        this request under that model. `candidate_models` is every
        individual name a comma-separated `model` names (see
        `_individual_model_names`), not the raw joined string."""
        if renewal_allowed and candidate_models:
            return admitted_models | candidate_models
        return admitted_models

    @staticmethod
    def _ttl_for(unit: _LimitUnit, entry: TagRateLimitEntry) -> int:
        if unit == "concurrency":
            requested_ttl: Final = entry.key_ttl_seconds if entry.key_ttl_seconds is not None else entry.period_seconds
            return max(requested_ttl, _CONCURRENCY_MIN_SAFETY_TTL_SECONDS)
        return _bucket_ttl_seconds(entry)

    def _classify(
        self,
        config: TagRateLimits,
        tags: Sequence[str],
        key_alias: str | None,
        key_hash: str | None,
        now: float,
        candidate_models: frozenset[str],
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
                if not _entry_applies_any_candidate_model(entry, tags, key_alias, candidate_models):
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
                # tag_value deliberately excluded: it can resolve from
                # inherited_tags (server-assigned key/team/project metadata),
                # and echoing it back would disclose that identity to the
                # caller. verbose_proxy_logger.debug above still logs it
                # server-side for observability.
                "limit_name": entry.name,
                "limit": entry.limit,
                "period_seconds": entry.period_seconds,
                # ProxyBaseLLMRequestProcessing._pre_call_with_fallbacks reads this:
                # an apply_to_models entry caps an entire named chain as one unit, so
                # retrying against a fallback model outside that list would silently
                # defeat the very policy that just rejected this request.
                **({"cross_model_scope": True} if entry.apply_to_models is not None else {}),  # mutable-ok: see above
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

        # _pre_call_with_fallbacks can re-run this pipeline once per fallback
        # model on any ProxyRateLimitError, reusing the same litellm_call_id --
        # see charged_request_keys for how a repeat run renews instead of
        # re-charging.
        stash: Final = _claim_stash_for_data(data)

        # Not get_metadata_variable_name_from_kwargs (naive key-presence
        # check): a caller can forge an empty (or None) litellm_metadata on
        # an ordinary request to make that check pick it over the real,
        # populated metadata the proxy wrote identity/tags into, seeing no
        # tags at all and admitting past every configured limit.
        metadata_variable_name: Final = _resolve_authoritative_metadata_variable_name(data)
        tags: Final = _order_tags_for_identity_resolution(
            _get_tags_from_request_kwargs(data, metadata_variable_name=metadata_variable_name),
            data,
            metadata_variable_name,
        )
        key_alias: Final = user_api_key_dict.key_alias
        key_hash: Final = user_api_key_dict.api_key
        model: Final = data.get("model") if isinstance(data.get("model"), str) else None
        candidate_models: Final = frozenset(_individual_model_names(model, call_type))

        # First admission for this stash claims ownership; only a later one
        # with the same key_hash may renew its charges (see owner_key_hash).
        if stash.owner_key_hash is None:
            stash.owner_key_hash = key_hash
        renewal_allowed: Final = stash.owner_key_hash == key_hash

        now: Final = self._time_provider().timestamp()
        stash.admission_time = now
        classified: Final = self._classify(config, tags, key_alias, key_hash, now, candidate_models)
        if not classified:
            stash.admitted_models = self._admitted_models_after(
                stash.admitted_models, candidate_models, renewal_allowed
            )
            return data

        read_only_checks: Final = tuple(c for c in classified if not c.is_atomic)
        atomic_checks: Final = tuple(c for c in classified if c.is_atomic)

        current_values: Final = await self._read_only_values(read_only_checks, parent_otel_span=None)
        self._raise_if_over_limit(read_only_checks, current_values, model)

        if atomic_checks:
            atomic_partitions_list: Final = []  # mutable-ok: sequential async lookups, one per atomic_checks entry
            for check in atomic_checks:
                atomic_partitions_list.append(
                    await self._partition_for(_partition_key(check.entry))
                )  # mutable-ok: see comment above
            atomic_partitions: Final = tuple(atomic_partitions_list)
            already_reserved_concurrency_keys: Final = frozenset(
                key for key, _partition_key in stash.pending_concurrency_keys
            )
            non_racing_batch_width: Final = _non_racing_batch_width(data, call_type)
            failing_index, values = await self._atomic_check_and_increment(
                tuple(
                    (
                        partition.internal_usage_cache,
                        check.key,
                        check.entry.limit,
                        # A key already charged/reserved for this call_id (an
                        # earlier fallback attempt for the same request) renews
                        # at zero net cost instead of charging a second unit.
                        # Otherwise a concurrency check reserves one unit per
                        # non-racing batch branch (see _non_racing_batch_width),
                        # not just one for the whole dispatch.
                        0.0
                        if renewal_allowed
                        and (
                            (check.unit == "requests" and check.key in stash.charged_request_keys)
                            or (check.unit == "concurrency" and check.key in already_reserved_concurrency_keys)
                        )
                        else (float(non_racing_batch_width) if check.unit == "concurrency" else 1.0),
                        self._ttl_for(check.unit, check.entry),
                        check.unit == "concurrency",
                    )
                    for partition, check in zip(atomic_partitions, atomic_checks)
                )
            )
            if failing_index is not None:
                failing_check: Final = atomic_checks[failing_index]
                self._raise_over_limit(
                    failing_check.unit, failing_check.entry, failing_check.tag_value, model, current=values[0]
                )

            # Exclude already_reserved_concurrency_keys: that key renewed at
            # zero cost above, so re-adding it would make release decrement
            # twice for a counter only ever incremented once. One entry per
            # reserved unit (non_racing_batch_width of them, ordinarily 1) --
            # see pending_concurrency_keys's own docstring for why.
            concurrency_reservations: Final = tuple(
                (check.key, _partition_key(check.entry))
                for check in atomic_checks
                if check.unit == "concurrency" and check.key not in already_reserved_concurrency_keys
                for _ in range(non_racing_batch_width)
            )
            if concurrency_reservations:
                stash.pending_concurrency_keys.extend(concurrency_reservations)  # mutable-ok: see field's own docstring

            # Only recorded when renewal_allowed, so a call_id collision from
            # a different key_hash can't contaminate the rightful owner's
            # renewal tracking.
            request_keys: Final = (
                tuple(
                    check.key
                    for check in atomic_checks
                    if check.unit == "requests" and check.key not in stash.charged_request_keys
                )
                if renewal_allowed
                else ()
            )
            if request_keys:
                stash.charged_request_keys.extend(request_keys)  # mutable-ok: see field's own docstring

        stash.admitted_models = self._admitted_models_after(stash.admitted_models, candidate_models, renewal_allowed)
        return data

    async def _release_pending_for_call_id(self, request_kwargs: Mapping[str, object]) -> None:
        """Releases every reservation still pending for this call_id at once.
        Correct for a disconnect or a chain-exhausted failure: either aborts
        every not-yet-completed branch of a non-racing batch together (see
        _non_racing_batch_width), so none of them will ever fire its own
        terminal event to release its own share individually -- whatever's
        still pending here is exactly what those abandoned branches reserved,
        no more (any branch that already completed already popped its own
        entry via _release_one_pending_for_call_id) and no less."""
        stash: Final = _stash_for_call(_call_id_from_kwargs(request_kwargs))
        if stash is None or not stash.pending_concurrency_keys:
            return
        release_keys: Final = tuple(stash.pending_concurrency_keys)
        stash.pending_concurrency_keys.clear()
        await self._release_keys(release_keys)

    async def _release_one_pending_for_call_id(
        self, request_kwargs: Mapping[str, object]
    ) -> tuple[str, _PartitionKey] | None:
        """Releases exactly one reservation, not every reservation currently
        pending. Fallback only, for when this hop's own identity can't be
        resolved at release time (see _own_concurrency_keys_for_release):
        entries under one call_id are otherwise fungible (same key
        repeated), so releasing one arbitrary entry is safe even without
        knowing which policy it belongs to, but releasing every entry would
        risk sweeping up a still-running sibling branch's own share."""
        stash: Final = _stash_for_call(_call_id_from_kwargs(request_kwargs))
        if stash is None or not stash.pending_concurrency_keys:
            return None
        return stash.pending_concurrency_keys.pop()  # mutable-ok: see field's own docstring

    async def _pop_matching_keys_for_call_id(
        self, request_kwargs: Mapping[str, object], only_keys: frozenset[str]
    ) -> tuple[tuple[str, _PartitionKey], ...]:
        """Pops at most one reservation per key in only_keys, not every
        reservation currently pending: a request can match more than one
        concurrency-scoped entry (see _own_concurrency_keys), and each
        terminal event must release every one of its own matches, not just
        one -- while still leaving a still-running sibling branch's own
        share (a different key, or another fungible entry under a shared
        key) untouched. An empty only_keys means this hop's own identity
        resolved to zero matching policies, so it pops nothing."""
        stash: Final = _stash_for_call(_call_id_from_kwargs(request_kwargs))
        if stash is None or not stash.pending_concurrency_keys:
            return ()
        pending: Final = stash.pending_concurrency_keys
        matched_indices: Final = tuple(
            idx
            for key in only_keys
            if (idx := next((i for i, entry in enumerate(pending) if entry[0] == key), None)) is not None
        )
        released: Final = tuple(pending[idx] for idx in matched_indices)
        for entry in released:
            try:
                pending.remove(entry)  # mutable-ok: see field's own docstring
            except ValueError:
                pass
        return released

    def _own_concurrency_keys_for_release(self, kwargs: Mapping[str, object]) -> frozenset[str] | None:
        """The keys `_own_concurrency_keys` would compute for this hop, or
        `None` if identity/config can't be resolved at all -- distinct from
        resolving to zero matching policies, which is a real, releasable
        answer of "nothing to release here"."""
        config: Final = self._refresh_config()
        if config is None:
            return None
        litellm_params_raw: Final = kwargs.get("litellm_params")
        litellm_params_for_metadata: Final[Mapping[str, object]] = (
            litellm_params_raw if isinstance(litellm_params_raw, Mapping) else kwargs
        )
        metadata_variable_name: Final = _resolve_authoritative_metadata_variable_name(litellm_params_for_metadata)
        key_hash: Final = _extract_key_hash(litellm_params_for_metadata, metadata_variable_name)
        key_alias: Final = _extract_key_alias(litellm_params_for_metadata, metadata_variable_name)
        tags: Final = _order_tags_for_identity_resolution(
            _get_tags_from_request_kwargs(litellm_params_for_metadata, metadata_variable_name=metadata_variable_name),
            litellm_params_for_metadata,
            metadata_variable_name,
        )
        if not tags:
            return None
        model_raw: Final = kwargs.get("model")
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        model: Final = (
            model_raw
            if isinstance(model_raw, str)
            else standard_logging_object.get("model")
            if isinstance(standard_logging_object, dict)
            else None
        )
        candidate_models: Final = frozenset((model,)) if isinstance(model, str) else frozenset()
        return _own_concurrency_keys(config, tags, key_alias, key_hash, candidate_models)

    async def _release_own_share(self, kwargs: Mapping[str, object]) -> tuple[tuple[str, _PartitionKey], ...]:
        own_concurrency_keys: Final = self._own_concurrency_keys_for_release(kwargs)
        if own_concurrency_keys is not None:
            return await self._pop_matching_keys_for_call_id(kwargs, own_concurrency_keys)
        one_entry: Final = await self._release_one_pending_for_call_id(kwargs)
        return (one_entry,) if one_entry is not None else ()

    async def async_release_disconnect_state_hook(self, request_data: Mapping[str, object]) -> None:
        await self._release_pending_for_call_id(request_data)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,  # mutable-ok: must match CustomLogger.async_post_call_failure_hook's own base signature exactly
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        """
        A request that never reaches Router (every fallback model also
        rejected, or none configured) never runs the actual LLM call, so
        neither async_log_success_event nor async_log_failure_event -- both
        tied to that call's own wrapper -- ever fires for it. This is the
        only remaining release path for a reservation from an earlier,
        successful admission attempt in the same _pre_call_with_fallbacks
        chain. litellm_call_id survives proxy/utils.py's own stripping here
        (only litellm_logging_obj is popped), so the same ContextVar-based
        stash lookup as the other release hooks still works.
        """
        await self._release_pending_for_call_id(request_data)

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None:
        # Always release regardless of which hook raised: this hook's own
        # rejection never reserves a slot, so there is nothing to pop in
        # that case; a rejection from model_based_tag_rate_limits_hook (same
        # error marker) can still land after this hook already reserved its
        # own slot. Only this one branch's own share, not every reservation
        # still pending for a non-racing batch's other, still-running
        # branches -- see _release_own_share.
        released_entries: Final = await self._release_own_share(kwargs)
        if released_entries:
            await self._release_keys(released_entries)

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None:
        released_entries: Final = await self._release_own_share(kwargs)
        if released_entries:
            release_task: Final = asyncio.create_task(self._release_keys(released_entries))
            _BACKGROUND_TASKS.add(release_task)  # mutable-ok: see _BACKGROUND_TASKS's own docstring
            release_task.add_done_callback(_BACKGROUND_TASKS.discard)

        stash: Final = _stash_for_call(_call_id_from_kwargs(kwargs))
        config: Final = self._refresh_config()
        if config is None:
            return

        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if not isinstance(standard_logging_object, dict):
            return

        # kwargs here is Logging.model_call_details, not the router's flat
        # request kwargs admission sees: metadata/litellm_metadata are never
        # top-level here, only nested under kwargs["litellm_params"] (see
        # Logging.update_environment_variables).
        litellm_params_raw: Final = kwargs.get("litellm_params")
        litellm_params_for_metadata: Final[Mapping[str, object]] = (
            litellm_params_raw if isinstance(litellm_params_raw, Mapping) else kwargs
        )
        metadata_variable_name: Final = _resolve_authoritative_metadata_variable_name(litellm_params_for_metadata)
        key_hash: Final = _extract_key_hash(litellm_params_for_metadata, metadata_variable_name)
        key_alias: Final = _extract_key_alias(litellm_params_for_metadata, metadata_variable_name)

        tags: Final = _order_tags_for_identity_resolution(
            _get_tags_from_request_kwargs(litellm_params_for_metadata, metadata_variable_name=metadata_variable_name),
            litellm_params_for_metadata,
            metadata_variable_name,
        )
        if not tags:
            return

        now: Final = (
            stash.admission_time
            if stash is not None and stash.admission_time is not None
            else self._time_provider().timestamp()
        )
        admitted_models: Final = stash.admitted_models if stash is not None else frozenset()
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
                if not _entry_applies_any_candidate_model(entry, tags, key_alias, admitted_models):
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
                    pipeline_operations=group_operations, parent_otel_span=None
                )
            )
            _BACKGROUND_TASKS.add(accounting_task)  # mutable-ok: see _BACKGROUND_TASKS's own docstring
            accounting_task.add_done_callback(_BACKGROUND_TASKS.discard)
