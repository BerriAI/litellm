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

Three independent entry-level knobs decide who a global entry applies to and
how its bucket is shared:

- `apply_to_key_alias`: unset means every request, any key, any model.  Set
  to a list of virtual-key aliases, only those keys' requests count.
- `apply_to_models`: unset means every model. Set to a list of model names,
  only requests whose caller-facing `model` field is in that list count --
  letting one entry rate-limit a whole fallback chain as a single unit by
  naming every model in the chain. Each check is a fresh, independent
  evaluation of `_entry_applies` against whatever `model` is current at that
  moment, not a one-time decision that then sticks for the rest of the
  request. Two concrete consequences follow from that:
  (1) if the request's own model fails mid-flight and Router internally
  retries a different model for the *same* admitted call, that retry is
  never re-checked -- the original admission (against the originally
  requested model) already stands, so an operator who needs the limit to
  track whichever model actually ends up serving a request needs
  `model_info.tag_rate_limits` instead; but
  (2) if this hook's own admission *rejects* the request,
  `common_request_processing.py` would otherwise catch that rejection and
  retry the whole pre-call pipeline against
  `litellm_settings.fallbacks`/`router_settings.fallbacks`, with
  `data["model"]` mutated to the fallback target -- silently admitting the
  request via a model outside `apply_to_models`, defeating the cap. A
  rejection from an `apply_to_models`-scoped entry carries
  `detail["cross_model_scope"] = True` for exactly this reason:
  `_pre_call_with_fallbacks` checks that marker and re-raises immediately
  instead of trying any fallback, so this bypass is closed regardless of
  whether the fallback chain is also listed in `apply_to_models`.
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
problem, with one difference: the stash here is a dict keyed by
`litellm_call_id` rather than one shared mutable instance with an
overwritable "owner" field, so a nested LiteLLM call made inside the request
(e.g. a guardrail's own LLM judge call) -- which mints its own fresh call id
but inherits the same ContextVar-held ancestor context, not a separate one
-- gets its own isolated entry instead of overwriting the outer call's and
having its own success callback release the outer call's still-pending
reservations early.

`Router.abatch_completion`'s comma-separated `model` fans this one admission
out into several real, independent LLM calls below this hook entirely (it
never re-runs `async_pre_call_hook` per branch the way
`model_based_tag_rate_limits_hook`'s per-Router-hop admission does). Every
branch shares the one `litellm_logging_obj` this same request already
carries (attached in `common_request_processing.py` before Router ever
sees the request, then threaded unchanged through every branch's own
`**kwargs`), so `Logging.should_run_logging`'s per-event-type dedup fires
this hook's own success/failure callback at most once for a success and at
most once for a failure across the *whole* dispatch, never once per
branch -- confirmed live against the real dispatch, including with a real,
shared `litellm_logging_obj` attached the way production actually does it.
Reserving one unit per model and expecting each to release on its own
branch's event (an earlier version of this hook did exactly that) is
therefore not just occasionally racy: it guarantees every batch of width
more than 2 leaks the surplus for the safety TTL, every single call. This
hook instead reserves exactly one unit for the whole non-racing dispatch,
same as the racing `abatch_completion_fastest_response` variant below,
and releases it on whichever single event fires first -- a second event
firing (the mixed success-and-failure case) finds nothing left and is a
safe no-op. That undercounts the dispatch's real, brief concurrent
provider load, the same accepted tradeoff already made for
`abatch_completion_fastest_response`, whose cancelled losers never fire a
terminal event at all.
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
from litellm.proxy.hooks.model_based_tag_rate_limits_hook import (
    _ATOMIC_UNITS,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _BACKGROUND_TASKS,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _CONCURRENCY_MIN_SAFETY_TTL_SECONDS,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
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
    _order_tags_for_identity_resolution,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _partition_key,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _PartitionKey,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _PartitionOperations,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _policy_fingerprint,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
    _resolve_authoritative_metadata_variable_name,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, see module docstring
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


def _entry_applies_any_candidate_model(
    entry: TagRateLimitEntry, tags: Sequence[str], key_alias: str | None, candidate_models: frozenset[str]
) -> bool:
    """Same as `_entry_applies`, except an `apply_to_models`-scoped entry
    counts as applying if ANY of `candidate_models` is in scope -- not just a
    single caller-visible name. Two call sites need this: at admission, a
    comma-separated `model` (a batch dispatch) names several models at once,
    any of which should trip a chain-wide cap; at success-event accounting,
    a `_pre_call_with_fallbacks` retry re-admits with a different model for
    the same call_id, and an entry that matched an earlier attempt must
    still get its accounting."""
    if not candidate_models:
        return _entry_applies(entry, tags, key_alias, None)
    return any(_entry_applies(entry, tags, key_alias, model) for model in candidate_models)


def _individual_model_names(model: str | None, call_type: str) -> tuple[str, ...]:
    """`route_llm_request.py` splits a comma-separated `model` on this exact
    condition before fanning out through `Router.abatch_completion`/
    `abatch_completion_fastest_response` -- an `apply_to_models`-scoped entry
    must check each of those individual names, not the raw joined string,
    which is never a member of any caller-configured `apply_to_models` list."""
    if model is None:
        return ()
    if call_type != "acompletion" or "," not in model:
        return (model,)
    return tuple(m.strip() for m in model.split(","))


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
    """Per-call bookkeeping `async_pre_call_hook` hands to that same call's
    success/failure/disconnect callbacks -- see module docstring for why
    this lives on a `ContextVar`, not `model_call_details`.

    Keyed by `litellm_call_id` in the dict below rather than one shared
    mutable instance with an overwritable "owner" field: a nested LiteLLM
    call made inside the request (an LLM-judge guardrail, a silent
    experiment) that mints its own fresh call id runs inside the *same*
    inherited context, not a separate one, so a single shared instance's
    owner field would get reassigned to the nested call and its own
    success callback would then release the outer call's still-pending
    reservations early -- letting extra same-tag requests through while
    the outer request is still genuinely in flight. Keying by call id
    isolates each call's own reservations regardless of nesting.
    """

    # Claimed once by this stash's first admission attempt, like owner_key_hash
    # below -- a later _pre_call_with_fallbacks retry must not push this bucket
    # epoch forward, or a caller straddling a period rollover between the
    # in-scope attempt and an out-of-scope fallback gets its usage charged
    # against a bucket admission never actually checked.
    admission_time: float | None = None
    # Every model any admission attempt for this call_id has classified
    # entries against, accumulated rather than overwritten: a
    # _pre_call_with_fallbacks retry re-runs admission with a *different*
    # model for the same call_id, and an apply_to_models entry that matched
    # an earlier attempt must still get its accounting at success time even
    # though the request ultimately serves from a later attempt's model.
    admitted_models: frozenset[str] = field(default_factory=frozenset)
    pending_concurrency_keys: list[tuple[str, _PartitionKey]] = field(default_factory=list)  # mutable-ok: queue
    # "requests" keys already charged for this call_id -- veria-ai finding:
    # ProxyBaseLLMRequestProcessing._pre_call_with_fallbacks reruns the whole
    # pre-call pipeline (this hook included) once per fallback model on ANY
    # ProxyRateLimitError, not only one this hook itself raised, but reuses
    # the same litellm_call_id (self.data is mutated in place, only `model`
    # changes) across every attempt -- so this stash is the SAME object each
    # time. A "requests" check matching an already-charged key here renews
    # at zero net cost instead of charging a second unit for the same
    # logical request; see async_pre_call_hook's own comment for how.
    charged_request_keys: list[str] = field(default_factory=list)  # mutable-ok: see comment above
    # The server-authenticated key_hash (UserAPIKeyAuth.api_key) of whichever
    # call first claimed this stash. litellm_call_id is caller-controlled via
    # the x-litellm-call-id header (the exact forgery vector
    # model_based_tag_rate_limits_hook's own pending-reservations mirror was
    # hardened against earlier), so two unrelated requests sharing a
    # caller-chosen id must not be allowed to "renew" each other's charge --
    # only a later admission carrying this same, authenticated key_hash may.
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
        """All-or-nothing atomic admission across `checks` -- see
        `model_based_tag_rate_limits_hook._PROXY_ModelBasedTagRateLimitsHook._atomic_check_and_increment`'s
        own docstring for the full rationale (refund-on-rollback, why a
        raising key's own outcome is never refunded); identical logic,
        duplicated rather than shared since it lives as instance methods
        rather than free functions."""
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
    def _next_admitted_models(
        admitted_models: frozenset[str], candidate_models: frozenset[str], renewal_allowed: bool
    ) -> frozenset[str]:
        """Only meant to be applied once this admission attempt has cleared
        every check without raising -- a rejected attempt's models must
        never join admitted_models, or a later successful attempt's
        accounting could wrongly credit an apply_to_models entry that never
        actually admitted this request under that model. `candidate_models`
        is every individual name a comma-separated `model` names (see
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
                **(
                    MappingProxyType({"cross_model_scope": True})
                    if entry.apply_to_models is not None
                    else MappingProxyType({})
                ),
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

        # async_pre_call_hook fires once per request in the common case, but
        # ProxyBaseLLMRequestProcessing._pre_call_with_fallbacks can re-run
        # this same pipeline once per fallback model on any ProxyRateLimitError
        # (not only one this hook raised) -- see charged_request_keys' own
        # docstring for how a repeat run for the same call_id renews rather
        # than re-charges both "requests" and "concurrency" checks below.
        stash: Final = _claim_stash_for_data(data)

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

        # Only a repeat admission carrying the SAME authenticated key_hash as
        # whichever call first claimed this stash may renew its charges --
        # see owner_key_hash's own docstring for why a bare call_id match is
        # not enough. First admission for this stash claims ownership here.
        if stash.owner_key_hash is None:
            stash.owner_key_hash = key_hash
        renewal_allowed: Final = stash.owner_key_hash == key_hash

        now: Final = self._time_provider().timestamp()
        if stash.admission_time is None:
            stash.admission_time = now
        classified: Final = self._classify(config, tags, key_alias, key_hash, now, candidate_models)
        if not classified:
            stash.admitted_models = self._next_admitted_models(stash.admitted_models, candidate_models, renewal_allowed)
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
            failing_index, values = await self._atomic_check_and_increment(
                tuple(
                    (
                        partition.internal_usage_cache,
                        check.key,
                        check.entry.limit,
                        # A key already charged/reserved for this call_id (an
                        # earlier _pre_call_with_fallbacks attempt for the
                        # same logical request) renews at zero net cost
                        # instead of charging or reserving a second unit --
                        # folded into this same all-or-nothing batch so a
                        # rollback here (some other check in the batch
                        # rejecting) refunds that zero-cost renewal as a
                        # genuine no-op, same reasoning as
                        # model_based_tag_rate_limits_hook's identical fix
                        # for its own per-hop retries.
                        0.0
                        if renewal_allowed
                        and (
                            (check.unit == "requests" and check.key in stash.charged_request_keys)
                            or (check.unit == "concurrency" and check.key in already_reserved_concurrency_keys)
                        )
                        else 1.0,
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

            # Only genuinely new reservations, never a key already in
            # already_reserved_concurrency_keys: that key's own check just
            # renewed at zero net cost above, so re-adding it here would
            # make release (which decrements once per queued entry) decrement
            # twice for a counter that was only ever incremented once.
            concurrency_reservations: Final = tuple(
                (check.key, _partition_key(check.entry))
                for check in atomic_checks
                if check.unit == "concurrency" and check.key not in already_reserved_concurrency_keys
            )
            if concurrency_reservations:
                stash.pending_concurrency_keys.extend(concurrency_reservations)  # mutable-ok: see field's own docstring

            # Only recorded when renewal_allowed: an admission that didn't
            # own this stash (a call_id collision from a different key_hash)
            # must not contaminate the rightful owner's own renewal
            # tracking, or a later, genuine fallback retry from the owner
            # could wrongly treat the impostor's charge as its own and
            # renew for free.
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

        stash.admitted_models = self._next_admitted_models(stash.admitted_models, candidate_models, renewal_allowed)
        return data

    async def _release_pending_for_call_id(self, request_kwargs: Mapping[str, object]) -> None:
        """Releases every reservation still pending for this call_id at once.
        Every reservation is fungible under one call_id, and at most one
        success and one failure callback ever fires for the whole call_id
        (see the module docstring's explanation of `Logging.should_run_logging`'s
        per-event-type dedup across a batch dispatch's shared logging
        object) -- releasing everything still pending here is therefore
        always correct, never a still-genuinely-running branch's share,
        whether this fires from a disconnect, a chain-exhausted failure, or
        a terminal callback that already ran once for this call_id.
        """
        stash: Final = _stash_for_call(_call_id_from_kwargs(request_kwargs))
        if stash is None or not stash.pending_concurrency_keys:
            return
        release_keys: Final = tuple(stash.pending_concurrency_keys)
        stash.pending_concurrency_keys.clear()
        await self._release_keys(release_keys)

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
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        # Always release regardless of which hook raised: this hook's own
        # rejection never reserves a slot, so pending_concurrency_keys is
        # already empty in that case and the check below no-ops; a rejection
        # from model_based_tag_rate_limits_hook (same error marker) can still
        # land after this hook already reserved its own slot.
        await self._release_pending_for_call_id(kwargs)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        stash: Final = _stash_for_call(_call_id_from_kwargs(kwargs))
        if stash is not None and stash.pending_concurrency_keys:
            release_task: Final = asyncio.create_task(self._release_pending_for_call_id(kwargs))
            _BACKGROUND_TASKS.add(release_task)  # mutable-ok: see _BACKGROUND_TASKS's own docstring
            release_task.add_done_callback(_BACKGROUND_TASKS.discard)

        config: Final = self._refresh_config()
        if config is None:
            return

        standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get(  # pyright: ignore[reportAssignmentType]  # untyped callback kwargs, same as shadow_eval_logger.py's identical read
            "standard_logging_object"
        )
        if standard_logging_object is None:
            return

        # kwargs here is Logging.model_call_details, not the router's flat
        # request kwargs admission sees: metadata/litellm_metadata are never
        # top-level here, only nested under kwargs["litellm_params"] (see
        # Logging.update_environment_variables).
        litellm_params_raw: Final = kwargs.get("litellm_params")
        litellm_params_for_metadata: Final = litellm_params_raw if isinstance(litellm_params_raw, Mapping) else kwargs
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
