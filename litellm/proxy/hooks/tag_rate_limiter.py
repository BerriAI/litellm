"""
Tag-scoped token, request, dollar, and concurrency rate limits.

Each limit entry is keyed by an arbitrary caller-supplied tag value (not a
DB-provisioned entity, not composed with the calling API key) and enforced on
every routing attempt for a chain/model-group -- the primary hop and every
fallback hop, each checked against its own configuration.

Opt-in via `litellm_settings.callbacks: ["tag_rate_limiter"]` (not part of
`PROXY_HOOKS`), following the `dynamic_rate_limiter_v3` precedent: this hook
reuses `_PROXY_MaxParallelRequestsHandler_v3`'s Redis/TTL-preserving increment
machinery rather than duplicating it, and is never joined onto the default
limiter every proxy already runs.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import RateLimitType
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,
    get_metadata_variable_name_from_kwargs,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router import Router
from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import TagRateLimitEntry, TagRateLimits
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span | Any
else:
    Span = Any

_LimitUnit = Literal["tokens", "requests", "dollars", "concurrency"]
_LIMIT_UNITS: tuple[_LimitUnit, ...] = ("tokens", "requests", "dollars", "concurrency")
# Units whose admission must be atomic (check-and-increment in one Redis
# round trip) because the increment amount is known upfront (always 1).
# tokens/dollars can't be: real usage is only known after the response, so
# they stay a read-then-account-on-success check with a documented,
# unavoidable admit-vs-account race.
_ATOMIC_UNITS: frozenset[_LimitUnit] = frozenset({"requests", "concurrency"})

_UNIT_TO_GROUP_FIELD: dict[_LimitUnit, str] = {
    "tokens": "token_limits",
    "requests": "request_limits",
    "dollars": "dollar_limits",
    "concurrency": "concurrency_limits",
}
_UNIT_TO_RATE_LIMIT_TYPE: dict[_LimitUnit, RateLimitType] = {
    "tokens": RateLimitType.TOKENS,
    "requests": RateLimitType.REQUESTS,
    "dollars": RateLimitType.BUDGET,
    "concurrency": RateLimitType.CONCURRENT_REQUESTS,
}

# Single-key atomic check-and-increment. Deliberately one key per script call
# (never a batch of differently-hash-tagged keys in one call): every tag_rl
# key carries its own self-contained {..} hash tag so unrelated buckets never
# forcibly co-locate on the same Redis Cluster shard, which means a single Lua
# invocation can never span more than one key's slot without risking a
# cross-slot error. All-or-nothing across a hop's multiple atomic checks
# (e.g. requests + concurrency checked together) is achieved in Python by
# calling this once per key and refunding every earlier admission in the same
# batch if a later one is rejected -- the same refund-on-rollback shape as
# `atomic_check_and_increment_by_n` in parallel_request_limiter_v3.py, applied
# per-key instead of per-descriptor since each key already is one hash-tag
# group by construction.
TAG_RL_CHECK_AND_INCR_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local increment = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local current = tonumber(redis.call('GET', key) or 0)
if current + increment > limit then
    return { 0, current }
end
local new_value = redis.call('INCRBY', key, increment)
local current_ttl = redis.call('TTL', key)
if current_ttl == -1 and ttl > 0 then
    redis.call('EXPIRE', key, ttl)
end
return { 1, new_value }
"""

# Atomic decrement that never leaves a counter negative. Used both to refund
# an earlier admission when a later key in the same batch is rejected, and to
# release a concurrency reservation -- floors at 0 so a decrement that can't
# be attributed to the exact reservation that caused it (see
# `_release_keys`'s docstring) degrades to under-counting rather than a
# negative counter that would admit unlimited requests.
TAG_RL_DECR_FLOOR_ZERO_SCRIPT = """
local key = KEYS[1]
local delta = tonumber(ARGV[1])
local new_value = redis.call('INCRBY', key, delta)
if new_value < 0 then
    redis.call('SET', key, 0)
    new_value = 0
end
return new_value
"""


@dataclass(frozen=True)
class _ConfiguredLimit:
    unit: _LimitUnit
    entry: TagRateLimitEntry
    # None => chain-wide (every deployment in the model_group shares one
    # bucket). Otherwise the sorted deployment ids that declared this exact
    # value -- the bucket is shared among only those deployments.
    deployment_scope: Optional[tuple[str, ...]]


def _extract_identity(tags: list[str], tag_id: str) -> Optional[str]:
    """
    First tag matching `f"{tag_id}:"`, value after the colon. Tags starting
    with `!` are tag-routing negation markers, not identity tags, and are
    skipped so they can never be misread as an identity value.
    """
    prefix = f"{tag_id}:"
    for tag in tags:
        if tag.startswith("!"):
            continue
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def _deployment_id(deployment: dict) -> Optional[str]:
    return (deployment.get("model_info") or {}).get("id")


def _entries_for_unit(deployment: dict, unit: _LimitUnit) -> list[TagRateLimitEntry]:
    raw_tag_rate_limits = (deployment.get("model_info") or {}).get("tag_rate_limits")
    if not raw_tag_rate_limits:
        return []
    tag_rate_limits = TagRateLimits.model_validate(raw_tag_rate_limits)
    group = getattr(tag_rate_limits, _UNIT_TO_GROUP_FIELD[unit])
    return group.limits if group is not None else []


def _build_group_limits(deployments: list[dict], unit: _LimitUnit) -> list[_ConfiguredLimit]:
    """
    One `_ConfiguredLimit` per distinct (tag_id, name, limit, period_seconds)
    declared for `unit` across `deployments` (all sharing one `model_name`).

    A signature declared identically by every deployment in the group is
    chain-wide (one shared bucket, regardless of which deployment serves).
    A signature declared by only some deployments, or where deployments
    genuinely disagree on the value for the same (tag_id, name), becomes a
    per-deployment-scoped bucket shared by exactly the deployments that
    declared that value -- silently dropping a divergent deployment's config
    (as a naive dedupe-by-name index would) is the exact bug this guards
    against.

    `concurrency` is the one exception: a per-deployment-scoped reservation
    is never created for it. Admission for a hop reserves every scope whose
    deployments overlap `healthy_deployments`, but only one deployment ends
    up actually serving -- releasing the exact reservation(s) that were never
    used, without a per-request slot identity to track which reservation
    belongs to which hop, isn't solved correctly by this design (a
    since-fixed live bug: an admitted-then-failed call's per-deployment
    reservation was never released; a caller could also strand a sibling
    deployment's reservation just by never being routed to it). A divergent
    concurrency signature is dropped with a warning instead of silently
    creating a bucket that can leak; only chain-wide concurrency entries
    (identical across every deployment in the group) are supported.
    """
    declaring_ids_by_signature: dict[tuple[str, str, float, int], list[str]] = {}
    for deployment in deployments:
        dep_id = _deployment_id(deployment)
        if dep_id is None:
            continue
        for entry in _entries_for_unit(deployment, unit):
            signature = (entry.tag_id, entry.name, entry.limit, entry.period_seconds)
            declaring_ids_by_signature.setdefault(signature, []).append(dep_id)

    distinct_signatures_by_name: dict[tuple[str, str], int] = {}
    for tag_id, name, _limit, _period in declaring_ids_by_signature:
        key = (tag_id, name)
        distinct_signatures_by_name[key] = distinct_signatures_by_name.get(key, 0) + 1

    total_deployments = len(deployments)
    configured: list[_ConfiguredLimit] = []
    for signature, declaring_ids in declaring_ids_by_signature.items():
        tag_id, name, limit, period_seconds = signature
        is_chain_wide = distinct_signatures_by_name[(tag_id, name)] == 1 and len(declaring_ids) == total_deployments
        if unit == "concurrency" and not is_chain_wide:
            verbose_proxy_logger.warning(
                "tag_rate_limiter: concurrency_limits entry %r (tag_id=%s) is not declared identically by every "
                "deployment sharing this model_name; per-deployment-scoped concurrency limits are not supported "
                "and this entry is being skipped entirely.",
                name,
                tag_id,
            )
            continue
        configured.append(
            _ConfiguredLimit(
                unit=unit,
                entry=TagRateLimitEntry(name=name, tag_id=tag_id, limit=limit, period_seconds=period_seconds),
                deployment_scope=None if is_chain_wide else tuple(sorted(declaring_ids)),
            )
        )
    return configured


def _build_limits_index(model_list: list[dict]) -> dict[str, list[_ConfiguredLimit]]:
    """
    Keyed by every name a caller could pass as the `model` this hop is for.
    That's not always `deployment["model_name"]`: a team calling through its
    own `team_public_model_name` alias reaches `async_filter_deployments`
    with that alias as `model`, while the deployment dicts in
    `healthy_deployments` still carry their own real `model_name` --
    `Router` never rewrites it for this path (unlike `model_group_alias`,
    which is resolved to the real model_name before routing even starts).
    Without this, tag limits configured on a team-aliased chain would never
    be looked up at all. Grouping (which deployments share one bucket) is
    still by the deployment's own `model_name`; the alias is only an
    additional key pointing at that same computed group.
    """
    groups: dict[str, list[dict]] = {}
    for deployment in model_list:
        groups.setdefault(deployment["model_name"], []).append(deployment)

    index: dict[str, list[_ConfiguredLimit]] = {}
    for model_name, deployments in groups.items():
        configured = [limit for unit in _LIMIT_UNITS for limit in _build_group_limits(deployments, unit)]
        if not configured:
            continue
        index[model_name] = configured
        for deployment in deployments:
            team_public_model_name = (deployment.get("model_info") or {}).get("team_public_model_name")
            if team_public_model_name:
                index[team_public_model_name] = configured
    return index


# Upper bound on how stale the limits index may be after a length-preserving
# deployment update (e.g. editing an existing deployment's tag_rate_limits
# in place via the admin API, which never changes len(model_list)). Router
# exposes no generic "config changed" version counter to key off instead, so
# this bounds staleness by simply re-checking periodically.
_INDEX_TTL_SECONDS = 5.0

# Floor for a concurrency reservation's self-heal TTL, regardless of the
# configured period_seconds. A reservation that expires while its request is
# still genuinely in flight silently admits requests past the limit; this
# generous floor keeps that window far larger than any realistic request
# duration, at the cost of a leaked (crashed-worker) slot self-healing more
# slowly. period_seconds can still raise the TTL further, never lower it.
_CONCURRENCY_MIN_SAFETY_TTL_SECONDS = 3600


class _TagRateLimitIndex:
    """Rebuilds the limits index when `llm_router.model_list` changes, or at
    least every `_INDEX_TTL_SECONDS`, whichever comes first."""

    def __init__(self, time_provider: Callable[[], datetime]) -> None:
        self._time_provider = time_provider
        self._cache_key: Optional[tuple[int, int]] = None
        self._built_at: float = 0.0
        self._index: dict[str, list[_ConfiguredLimit]] = {}

    def get(self, llm_router: Router) -> dict[str, list[_ConfiguredLimit]]:
        model_list = llm_router.model_list or []
        cache_key = (id(llm_router), len(model_list))
        now = self._time_provider().timestamp()
        if cache_key != self._cache_key or (now - self._built_at) >= _INDEX_TTL_SECONDS:
            self._index = _build_limits_index(model_list)
            self._cache_key = cache_key
            self._built_at = now
        return self._index


def _scope_suffix(deployment_scope: Optional[tuple[str, ...]]) -> str:
    return "chain" if deployment_scope is None else "dep:" + "+".join(deployment_scope)


def _bucket_key(model_group: str, configured: _ConfiguredLimit, tag_value: str, bucket_id: int) -> str:
    scope = _scope_suffix(configured.deployment_scope)
    hash_tag = f"tag_rl:{model_group}:{configured.unit}:{configured.entry.name}:{scope}:{tag_value}"
    return f"{{{hash_tag}}}:{bucket_id}"


def _inflight_key(model_group: str, configured: _ConfiguredLimit, tag_value: str) -> str:
    """Concurrency counter key: not epoch-bucketed, since "how many are in
    flight right now" has no window to reset on -- it's released explicitly
    on completion, with a TTL fallback only for a leaked (crashed) reservation."""
    scope = _scope_suffix(configured.deployment_scope)
    hash_tag = f"tag_rl:{model_group}:{configured.unit}:{configured.entry.name}:{scope}:{tag_value}"
    return f"{{{hash_tag}}}:inflight"


class _PROXY_TagRateLimiter(CustomLogger):
    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Optional[Callable[[], datetime]] = None,
    ):
        self.internal_usage_cache = InternalUsageCache(dual_cache=internal_usage_cache)
        self._v3 = _PROXY_MaxParallelRequestsHandler_v3(self.internal_usage_cache, time_provider=time_provider)
        self._time_provider = time_provider or datetime.now
        self._index = _TagRateLimitIndex(time_provider=self._time_provider)
        self._lock = asyncio.Lock()
        self.llm_router: Optional[Router] = None
        redis_cache = self.internal_usage_cache.dual_cache.redis_cache
        self._check_and_incr_script = (
            redis_cache.async_register_script(TAG_RL_CHECK_AND_INCR_SCRIPT) if redis_cache is not None else None
        )
        self._decr_floor_zero_script = (
            redis_cache.async_register_script(TAG_RL_DECR_FLOOR_ZERO_SCRIPT) if redis_cache is not None else None
        )

    def update_variables(self, llm_router: Router) -> None:
        self.llm_router = llm_router

    async def _check_and_increment_one(self, key: str, limit: float, increment: float, ttl: int) -> tuple[bool, float]:
        """Single-key atomic check-and-increment. Always one key per Lua
        call -- see TAG_RL_CHECK_AND_INCR_SCRIPT's module docstring for why."""
        if self._check_and_incr_script is not None:
            raw = await self._check_and_incr_script(keys=[key], args=[limit, increment, ttl])
            return bool(raw[0]), float(raw[1])

        async with self._lock:
            current_value = await self.internal_usage_cache.async_get_cache(key=key, litellm_parent_otel_span=None)
            current = float(current_value) if current_value is not None else 0.0
            if current + increment > limit:
                return False, current
            new_value = current + increment
            await self.internal_usage_cache.async_set_cache(
                key=key, value=new_value, ttl=ttl, litellm_parent_otel_span=None
            )
            return True, new_value

    async def _decrement_floor_zero(self, key: str, delta: float) -> None:
        if self._decr_floor_zero_script is not None:
            await self._decr_floor_zero_script(keys=[key], args=[delta])
            return
        async with self._lock:
            current_value = await self.internal_usage_cache.async_get_cache(key=key, litellm_parent_otel_span=None)
            current = float(current_value) if current_value is not None else 0.0
            await self.internal_usage_cache.async_set_cache(
                key=key, value=max(0.0, current + delta), litellm_parent_otel_span=None
            )

    async def _atomic_check_and_increment(
        self,
        checks: list[tuple[str, float, float, int]],
    ) -> tuple[Optional[int], list[float]]:
        """
        All-or-nothing across every (key, limit, increment, ttl) in `checks`:
        if any would exceed its limit, none are incremented -- a single hop's
        requests-unit and concurrency-unit checks must commit together or not
        at all. Each key is checked/incremented in its own single-key Lua
        call (cluster-safe by construction); all-or-nothing across the batch
        is enforced here by refunding every earlier admission the moment a
        later key is rejected, not by a single multi-key script call.

        Refunds are best-effort: a refund that fails (e.g. a transient Redis
        error) is logged and skipped rather than raised, so one bad refund
        can't stop the rest of the batch from being refunded, and can't turn
        a clean rejection into an unhandled exception. A skipped refund
        self-heals via the key's TTL -- see `_ttl_for`.

        Returns (failing_index, values). On success, failing_index is None
        and values holds each key's new post-increment value, same order as
        `checks`. On rejection, failing_index is the 0-based index of the
        first key that would have exceeded its limit and values holds that
        one key's current (unmodified) value.
        """
        if not checks:
            return None, []

        admitted_values: list[float] = []
        for index, (key, limit, increment, ttl) in enumerate(checks):
            admitted, value = await self._check_and_increment_one(key, limit, increment, ttl)
            if admitted:
                admitted_values.append(value)
                continue
            for refund_index in range(index):
                refund_key, _limit, refund_increment, _ttl = checks[refund_index]
                try:
                    await self._decrement_floor_zero(refund_key, -refund_increment)
                except Exception as e:  # noqa: BLE001 - one failed refund must not block refunding the rest
                    verbose_proxy_logger.warning(f"tag_rate_limiter: failed to refund {refund_key} on rollback: {e}")
            return index, [value]

        return None, admitted_values

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict],
        messages: Optional[list[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> list[dict]:
        if not healthy_deployments or not isinstance(healthy_deployments, list) or self.llm_router is None:
            return healthy_deployments

        configured = self._index.get(self.llm_router).get(model)
        if not configured:
            return healthy_deployments

        request_kwargs = request_kwargs or {}
        metadata_variable_name = get_metadata_variable_name_from_kwargs(request_kwargs)
        tags = _get_tags_from_request_kwargs(request_kwargs, metadata_variable_name=metadata_variable_name)

        present_deployment_ids = {dep_id for d in healthy_deployments if (dep_id := _deployment_id(d)) is not None}

        now = self._time_provider().timestamp()
        read_only_checks: list[tuple[_ConfiguredLimit, str, str]] = []
        atomic_checks: list[tuple[_ConfiguredLimit, str, str]] = []
        for configured_limit in configured:
            if configured_limit.deployment_scope is not None and not (
                present_deployment_ids & set(configured_limit.deployment_scope)
            ):
                continue
            tag_value = _extract_identity(tags, configured_limit.entry.tag_id)
            if tag_value is None:
                continue
            if configured_limit.unit == "concurrency":
                key = _inflight_key(model, configured_limit, tag_value)
                atomic_checks.append((configured_limit, tag_value, key))
            elif configured_limit.unit in _ATOMIC_UNITS:
                bucket_id = int(now) // configured_limit.entry.period_seconds
                key = _bucket_key(model, configured_limit, tag_value, bucket_id)
                atomic_checks.append((configured_limit, tag_value, key))
            else:
                bucket_id = int(now) // configured_limit.entry.period_seconds
                key = _bucket_key(model, configured_limit, tag_value, bucket_id)
                read_only_checks.append((configured_limit, tag_value, key))

        current_values = await self._read_only_values(read_only_checks, parent_otel_span)
        self._raise_if_over_limit(read_only_checks, current_values, model)

        if atomic_checks:
            failing_index, values = await self._atomic_check_and_increment(
                [
                    (key, configured_limit.entry.limit, 1.0, self._ttl_for(configured_limit))
                    for configured_limit, _tag_value, key in atomic_checks
                ]
            )
            if failing_index is not None:
                configured_limit, tag_value, _key = atomic_checks[failing_index]
                self._raise_over_limit(configured_limit, tag_value, model, current=values[0])

        return healthy_deployments

    @staticmethod
    def _ttl_for(configured_limit: _ConfiguredLimit) -> int:
        if configured_limit.unit == "concurrency":
            # A reservation's TTL must comfortably outlast any real in-flight
            # request, or a slow request's reservation self-heals (expires)
            # while it is still genuinely running, silently admitting extra
            # requests past the configured limit. period_seconds is still
            # honored if the operator wants an even longer safety margin, but
            # never shortens the floor below it.
            return max(configured_limit.entry.period_seconds, _CONCURRENCY_MIN_SAFETY_TTL_SECONDS)
        return configured_limit.entry.period_seconds + 3600

    async def _read_only_values(
        self,
        read_only_checks: list[tuple[_ConfiguredLimit, str, str]],
        parent_otel_span: Optional[Span],
    ) -> list[Optional[float]]:
        if not read_only_checks:
            return []
        keys = [key for _cfg, _tag_value, key in read_only_checks]
        current_values = await self.internal_usage_cache.async_batch_get_cache(
            keys=keys,
            parent_otel_span=parent_otel_span,
            local_only=False,
        )
        return current_values if current_values is not None else [None] * len(keys)

    def _raise_if_over_limit(
        self,
        read_only_checks: list[tuple[_ConfiguredLimit, str, str]],
        current_values: list[Optional[float]],
        model: str,
    ) -> None:
        for (configured_limit, tag_value, _key), current_value in zip(read_only_checks, current_values):
            current = float(current_value) if current_value is not None else 0.0
            if current < configured_limit.entry.limit:
                continue
            self._raise_over_limit(configured_limit, tag_value, model, current=current)

    def _raise_over_limit(
        self,
        configured_limit: _ConfiguredLimit,
        tag_value: str,
        model: str,
        current: float,
    ) -> None:
        verbose_proxy_logger.debug(
            "tag_rate_limiter: OVER_LIMIT model=%s unit=%s name=%s tag_id=%s tag_value=%s current=%s limit=%s",
            model,
            configured_limit.unit,
            configured_limit.entry.name,
            configured_limit.entry.tag_id,
            tag_value,
            current,
            configured_limit.entry.limit,
        )
        raise ProxyRateLimitError(
            detail={
                "error": "tag_rate_limit_exceeded",
                "type": configured_limit.unit,
                "tag_id": configured_limit.entry.tag_id,
                "tag_value": tag_value,
                "limit_name": configured_limit.entry.name,
                "limit": configured_limit.entry.limit,
                "period_seconds": configured_limit.entry.period_seconds,
            },
            headers={"retry-after": str(configured_limit.entry.period_seconds)},
            rate_limit_type=_UNIT_TO_RATE_LIMIT_TYPE[configured_limit.unit],
            model=model,
            llm_provider="litellm_proxy",
        )

    async def _release_keys(self, keys: list[str]) -> None:
        """
        Release each key by one slot. This does not verify the completing
        request still owns a live reservation (no per-request slot identity
        is tracked -- see the concurrency design note above), so a request
        that outlives the safety TTL and gets its key reused by a fresh
        reservation could in principle decrement a reservation it never
        held. Flooring at 0 (TAG_RL_DECR_FLOOR_ZERO_SCRIPT) bounds the
        damage to under-counting (briefly under-enforcing the limit) rather
        than a negative counter, which would admit unlimited requests.
        """
        for key in keys:
            try:
                await self._decrement_floor_zero(key, -1.0)
            except Exception as e:  # noqa: BLE001 - releasing a slot must never raise into the caller's request path
                verbose_proxy_logger.warning("tag_rate_limiter: failed to release concurrency slot %s: %s", key, e)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ) -> None:
        """
        Release any concurrency slot reserved for this hop before the call
        failed. request_data carries no resolved model_group/deployment_id
        the way a successful call's standard_logging_object does, so this
        re-derives identity from request_data["model"] (the hop's own model
        group -- reliable, since a hop that reserved a slot did so under
        this exact name). Concurrency entries are always chain-wide (see
        `_build_group_limits`'s docstring), so no deployment_id is needed to
        resolve which bucket to release here.
        """
        if self.llm_router is None:
            return
        model_group = request_data.get("model")
        if not model_group:
            return
        configured = self._index.get(self.llm_router).get(model_group)
        if not configured:
            return
        metadata_variable_name = get_metadata_variable_name_from_kwargs(request_data)
        tags = _get_tags_from_request_kwargs(request_data, metadata_variable_name=metadata_variable_name)
        if not tags:
            return

        release_keys = [
            _inflight_key(model_group, configured_limit, tag_value)
            for configured_limit in configured
            if configured_limit.unit == "concurrency"
            and configured_limit.deployment_scope is None
            and (tag_value := _extract_identity(tags, configured_limit.entry.tag_id)) is not None
        ]
        if release_keys:
            await self._release_keys(release_keys)
        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        if self.llm_router is None:
            return

        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")
        if standard_logging_object is None:
            return

        model_group = standard_logging_object.get("model_group")
        if not model_group:
            return

        configured = self._index.get(self.llm_router).get(model_group)
        if not configured:
            return

        metadata_variable_name = get_metadata_variable_name_from_kwargs(kwargs)
        tags = _get_tags_from_request_kwargs(kwargs, metadata_variable_name=metadata_variable_name)
        if not tags:
            return

        deployment_id = standard_logging_object.get("model_id")
        now = self._time_provider().timestamp()
        increment_by_unit: dict[_LimitUnit, float] = {
            "tokens": float(standard_logging_object.get("total_tokens") or 0),
            "dollars": float(standard_logging_object.get("response_cost") or 0),
        }

        operations: list[RedisPipelineIncrementOperation] = []
        release_keys: list[str] = []
        for configured_limit in configured:
            if configured_limit.deployment_scope is not None and deployment_id not in configured_limit.deployment_scope:
                continue
            tag_value = _extract_identity(tags, configured_limit.entry.tag_id)
            if tag_value is None:
                continue
            if configured_limit.unit == "concurrency":
                release_keys.append(_inflight_key(model_group, configured_limit, tag_value))
                continue
            if configured_limit.unit not in increment_by_unit:
                continue  # "requests" is accounted atomically at admission, not here
            increment_value = increment_by_unit[configured_limit.unit]
            if increment_value == 0:
                continue
            bucket_id = int(now) // configured_limit.entry.period_seconds
            key = _bucket_key(model_group, configured_limit, tag_value, bucket_id)
            operations.append(
                RedisPipelineIncrementOperation(
                    key=key,
                    increment_value=increment_value,
                    ttl=configured_limit.entry.period_seconds + 3600,
                )
            )

        if release_keys:
            asyncio.create_task(self._release_keys(release_keys))

        if not operations:
            return

        asyncio.create_task(
            self._v3.async_increment_tokens_with_ttl_preservation(
                pipeline_operations=operations,
                parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
            )
        )
