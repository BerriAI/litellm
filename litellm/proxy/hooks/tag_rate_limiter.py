"""Tag-scoped token, request, dollar, and concurrency rate limits."""

import asyncio
import contextvars
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import RateLimitType
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching dynamic_rate_limiter_v3's identical import
    get_metadata_variable_name_from_kwargs,
)
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # this hook explicitly reuses its Redis/TTL-preserving increment machinery, see module docstring
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router import Router
from litellm.router_strategy.tag_based_routing import (
    _get_tags_from_request_kwargs,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching dynamic_rate_limiter_v3's identical import
)
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import TagRateLimitEntry, TagRateLimits
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span: TypeAlias = _Span
else:
    Span: TypeAlias = object

_LimitUnit: TypeAlias = Literal["tokens", "requests", "dollars", "concurrency"]
_LIMIT_UNITS: Final[tuple[_LimitUnit, ...]] = ("tokens", "requests", "dollars", "concurrency")
# Units whose admission must be atomic (check-and-increment in one Redis
# round trip) because the increment amount is known upfront (always 1).
# tokens/dollars can't be: real usage is only known after the response, so
# they stay a read-then-account-on-success check with a documented,
# unavoidable admit-vs-account race.
_ATOMIC_UNITS: Final[frozenset[_LimitUnit]] = frozenset({"requests", "concurrency"})

_UNIT_TO_GROUP_FIELD: Final[Mapping[_LimitUnit, str]] = MappingProxyType(
    {
        "tokens": "token_limits",
        "requests": "request_limits",
        "dollars": "dollar_limits",
        "concurrency": "concurrency_limits",
    }
)
_UNIT_TO_RATE_LIMIT_TYPE: Final[Mapping[_LimitUnit, RateLimitType]] = MappingProxyType(
    {
        "tokens": RateLimitType.TOKENS,
        "requests": RateLimitType.REQUESTS,
        "dollars": RateLimitType.BUDGET,
        "concurrency": RateLimitType.CONCURRENT_REQUESTS,
    }
)

# Shared read-only fallback for an absent/None mapping (request_kwargs,
# metadata, model_info, ...): avoids constructing a fresh mutable `{}` at
# every one of these call sites just to immediately call `.get()` on it.
_EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})

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
TAG_RL_CHECK_AND_INCR_SCRIPT: Final = """
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
TAG_RL_DECR_FLOOR_ZERO_SCRIPT: Final = """
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
    deployment_scope: tuple[str, ...] | None
    # The team_id this limit was resolved under via `by_team_alias`, or None
    # when resolved via `by_model_name`. team_public_model_name is only
    # unique per team, so two teams can publish the identical alias string;
    # without the team_id folded into the bucket key too, both teams'
    # identically-named, identically-configured limits would collide on the
    # same Redis counter despite the index itself correctly scoping the
    # lookup by (team_id, alias).
    team_scope: str | None = None
    # The real model_name this limit was found under when `resolve()`'s
    # direct lookup by the caller-visible model string missed and
    # `resolve_any()` fell back to resolving via a candidate deployment's
    # own model_name instead (routing groups, and any other indirection
    # where Router deliberately keeps the caller-visible name distinct from
    # every deployment's own model_name). None when resolved directly, in
    # which case the caller-visible name is already unambiguous and safe to
    # hash by. Set, this overrides the caller-visible name in the bucket key
    # so limits from two different underlying model_names sharing one
    # routing group never collide on one counter.
    resolved_group: str | None = None


def _extract_identity(tags: Sequence[str], tag_id: str) -> str | None:
    """
    First tag matching `f"{tag_id}:"`, value after the colon. Tags starting
    with `!` are tag-routing negation markers, not identity tags, and are
    skipped so they can never be misread as an identity value.
    """
    prefix: Final = f"{tag_id}:"
    for tag in tags:
        if tag.startswith("!"):
            continue
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def _deployment_id(deployment: Mapping[str, object]) -> str | None:
    return (deployment.get("model_info") or _EMPTY_MAPPING).get("id")


def _extract_team_id(request_kwargs: Mapping[str, object], metadata_variable_name: str) -> str | None:
    """Reads `user_api_key_team_id` from only the one field
    `get_metadata_variable_name_from_kwargs` names as authoritative for this
    request -- never falling back to the other field, since
    `litellm_pre_call_utils.py` writes the real, server-authenticated value
    into that one field alone and leaves the other exactly as the caller
    sent it. An OR-fallback across both would let a caller's own
    `metadata.user_api_key_team_id` (still present, unvalidated, on a route
    where `litellm_metadata` is the authoritative field) win over the real
    value."""
    active: Final = request_kwargs.get(metadata_variable_name) or _EMPTY_MAPPING
    team_id: Final = active.get("user_api_key_team_id")
    return team_id if isinstance(team_id, str) else None


def _extract_key_hash(request_kwargs: Mapping[str, object], metadata_variable_name: str) -> str | None:
    """Same single-authoritative-field lookup as `_extract_team_id`, but for
    the calling virtual key's hash: `LiteLLMProxyRequestSetup` sets
    `metadata["user_api_key"]` to `user_api_key_dict.api_key`, which despite
    the plain name is already the hashed token (see `litellm_pre_call_utils.py`).
    """
    active: Final = request_kwargs.get(metadata_variable_name) or _EMPTY_MAPPING
    key_hash: Final = active.get("user_api_key")
    return key_hash if isinstance(key_hash, str) else None


def _entries_for_unit(deployment: Mapping[str, object], unit: _LimitUnit) -> tuple[TagRateLimitEntry, ...]:
    raw_tag_rate_limits: Final = (deployment.get("model_info") or _EMPTY_MAPPING).get("tag_rate_limits")
    if not raw_tag_rate_limits:
        return ()
    tag_rate_limits: Final = TagRateLimits.model_validate(raw_tag_rate_limits)
    group: Final = getattr(tag_rate_limits, _UNIT_TO_GROUP_FIELD[unit])
    return tuple(group.limits) if group is not None else ()


def _configured_limit_for_signature(
    unit: _LimitUnit,
    signature: tuple[str, str, float, int, bool],
    declaring_ids: Sequence[str],
    is_chain_wide: bool,
) -> _ConfiguredLimit | None:
    tag_id, name, limit, period_seconds, scope_by_key_hash = signature
    if unit == "concurrency" and not is_chain_wide:
        verbose_proxy_logger.warning(
            "tag_rate_limiter: concurrency_limits entry %r (tag_id=%s) is not declared identically by every "
            "deployment sharing this model_name; per-deployment-scoped concurrency limits are not supported "
            "and this entry is being skipped entirely.",
            name,
            tag_id,
        )
        return None
    return _ConfiguredLimit(
        unit=unit,
        entry=TagRateLimitEntry(
            name=name,
            tag_id=tag_id,
            limit=limit,
            period_seconds=period_seconds,
            scope_by_key_hash=scope_by_key_hash,
        ),
        deployment_scope=None if is_chain_wide else tuple(sorted(declaring_ids)),
    )


def _build_group_limits(deployments: Sequence[Mapping[str, object]], unit: _LimitUnit) -> tuple[_ConfiguredLimit, ...]:
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
    # Insertion order here is load-bearing: it decides which limit's
    # ProxyRateLimitError surfaces first when several are breached by the
    # same hop (see async_filter_deployments). A presort-based
    # itertools.groupby would need to sort by signature to group it, which
    # would scramble that first-seen order, so this stays a plain
    # accumulator instead.
    declaring_ids_by_signature: Final = {}  # mutable-ok: first-seen order here decides which limit's error raises first (see comment above); sorting to use groupby would scramble it
    for deployment in deployments:
        dep_id = _deployment_id(deployment)
        if dep_id is None:
            continue
        for entry in _entries_for_unit(deployment, unit):
            signature = (entry.tag_id, entry.name, entry.limit, entry.period_seconds, entry.scope_by_key_hash)
            ids_for_signature = declaring_ids_by_signature.setdefault(signature, [])  # mutable-ok: see comment above
            ids_for_signature.append(dep_id)

    distinct_signature_count_by_name: Final[Mapping[tuple[str, str], int]] = MappingProxyType(
        {
            (tag_id, name): sum(
                1
                for other_tag_id, other_name, *_rest in declaring_ids_by_signature
                if (other_tag_id, other_name) == (tag_id, name)
            )
            for tag_id, name, *_rest in declaring_ids_by_signature
        }
    )

    total_deployments: Final = len(deployments)
    configured: Final = tuple(
        configured_limit
        for signature, declaring_ids in declaring_ids_by_signature.items()
        if (
            configured_limit := _configured_limit_for_signature(
                unit,
                signature,
                declaring_ids,
                is_chain_wide=(
                    distinct_signature_count_by_name[(signature[0], signature[1])] == 1
                    and len(declaring_ids) == total_deployments
                ),
            )
        )
        is not None
    )
    return configured


@dataclass(frozen=True, slots=True)
class _LimitsIndex:
    """
    Two lookup tables because a `team_public_model_name` alias is only
    unique per team, not globally: Router itself lets different teams
    publish the identical alias string for different deployments, and
    resolves each caller's own team's deployment by `(team_id, name)`, not
    by `name` alone (see `Router._update_team_model_index`). Keying alias
    limits by name alone here would let one team's config silently
    overwrite another's.
    """

    by_model_name: Mapping[str, tuple[_ConfiguredLimit, ...]]
    by_team_alias: Mapping[tuple[str, str], tuple[_ConfiguredLimit, ...]]

    def resolve(self, model: str, team_id: str | None) -> tuple[_ConfiguredLimit, ...]:
        if team_id is not None:
            scoped: Final = self.by_team_alias.get((team_id, model))
            if scoped is not None:
                return scoped
        return self.by_model_name.get(model, ())

    def resolve_any(
        self, model: str, team_id: str | None, candidate_model_names: Iterable[str]
    ) -> tuple[_ConfiguredLimit, ...]:
        """
        Like `resolve`, but falls back to each candidate deployment's own
        `model_name` when `model` itself matches neither table -- Router
        deliberately keeps `model` as a callable routing-group name distinct
        from every member deployment's own `model_name` (see
        `Router._get_routing_group_deployments`), so a group-addressed call
        would otherwise never match this index at all despite its member
        deployments carrying real `tag_rate_limits`. Each fallback result is
        stamped with the `model_name` it actually came from (`resolved_group`)
        so hashing stays namespaced per underlying group even when the
        candidates span more than one `model_name`.
        """
        direct: Final = self.resolve(model, team_id)
        if direct:
            return direct
        return tuple(
            replace(limit, resolved_group=name)
            for name in frozenset(candidate_model_names)
            for limit in self.by_model_name.get(name, ())
        )


def _team_alias_key(deployment: Mapping[str, object]) -> tuple[str, str] | None:
    model_info: Final = deployment.get("model_info") or _EMPTY_MAPPING
    team_id: Final = model_info.get("team_id")
    team_public_model_name: Final = model_info.get("team_public_model_name")
    if team_id and team_public_model_name:
        return (team_id, team_public_model_name)
    return None


def _build_limits_index(model_list: Sequence[Mapping[str, object]]) -> _LimitsIndex:
    """
    `by_model_name` is keyed by every deployment's own `model_name`, grouping
    deployments that share one.

    `by_team_alias` additionally covers `team_public_model_name`: a team
    calling through its own public alias reaches `async_filter_deployments`
    with that alias as `model`, while the deployment dicts in
    `healthy_deployments` still carry their own real `model_name` --
    `Router` never rewrites it for this path (unlike `model_group_alias`,
    which is resolved to the real model_name before routing even starts).
    Without this, tag limits configured on a team-aliased chain would never
    be looked up at all.

    This is a genuinely separate grouping from `by_model_name`, not a lookup
    into it: litellm auto-generates each team-added deployment's own
    `model_name` as `model_name_{team_id}_{uuid}` (see
    `model_listing_utils.py`), so multiple deployments sharing one
    `team_public_model_name` alias routinely have different, unique
    `model_name` values -- Router's own `team_model_to_deployment_indices`
    aggregates them by `(team_id, team_public_model_name)` regardless.
    Computing alias limits once per `model_name` group and keying the alias
    to whichever group happened to declare it would drop every other
    same-alias group's limits whenever more than one model_name shares an
    alias, since the last one processed would silently overwrite the rest.
    `_build_group_limits` has no `model_name`-specific logic (it only reads
    each deployment's own id and its own entries), so it's safe to reuse
    unchanged for a deployment set spanning multiple `model_name` values.

    Deployments are grouped via a stable sort + itertools.groupby rather than
    a setdefault-in-a-loop accumulator: `sorted` is stable, so deployments
    sharing a key keep the exact same relative order `_build_group_limits`
    would have seen them in without the sort, which is what keeps this safe
    (that relative order decides first-seen signature order downstream).
    """
    sorted_by_model_name: Final = sorted(model_list, key=lambda deployment: deployment["model_name"])
    by_model_name: Final[Mapping[str, tuple[_ConfiguredLimit, ...]]] = MappingProxyType(
        {
            model_name: configured
            for model_name, deployment_group in groupby(
                sorted_by_model_name, key=lambda deployment: deployment["model_name"]
            )
            for group in (tuple(deployment_group),)
            if (configured := tuple(limit for unit in _LIMIT_UNITS for limit in _build_group_limits(group, unit)))
        }
    )

    aliased: Final = tuple(
        (alias_key, deployment) for deployment in model_list if (alias_key := _team_alias_key(deployment)) is not None
    )
    sorted_by_alias: Final = sorted(aliased, key=lambda pair: pair[0])
    by_team_alias: Final[Mapping[tuple[str, str], tuple[_ConfiguredLimit, ...]]] = MappingProxyType(
        {
            alias_key: alias_configured
            for alias_key, alias_group in groupby(sorted_by_alias, key=lambda pair: pair[0])
            for aliased_group in (tuple(dep for _key, dep in alias_group),)
            if (
                alias_configured := tuple(
                    replace(limit, team_scope=alias_key[0])
                    for unit in _LIMIT_UNITS
                    for limit in _build_group_limits(aliased_group, unit)
                )
            )
        }
    )

    return _LimitsIndex(by_model_name=by_model_name, by_team_alias=by_team_alias)


# Upper bound on how stale the limits index may be after a length-preserving
# deployment update (e.g. editing an existing deployment's tag_rate_limits
# in place via the admin API, which never changes len(model_list)). Router
# exposes no generic "config changed" version counter to key off instead, so
# this bounds staleness by simply re-checking periodically.
_INDEX_TTL_SECONDS: Final = 5.0

# Floor for a concurrency reservation's self-heal TTL, regardless of the
# configured period_seconds. A reservation that expires while its request is
# still genuinely in flight silently admits requests past the limit; this
# generous floor keeps that window far larger than any realistic request
# duration, at the cost of a leaked (crashed-worker) slot self-healing more
# slowly. period_seconds can still raise the TTL further, never lower it.
_CONCURRENCY_MIN_SAFETY_TTL_SECONDS: Final = 3600


# Concurrency reservation keys accumulated for the current logical request,
# not yet released. Held via a ContextVar bound to a mutable holder object
# (not an immutable tuple rebound with `.set()`) because `asyncio.create_task`
# only copies which *object* a ContextVar is bound to, not a snapshot of that
# object's contents: a `.set()` performed inside a task forked off this
# context mutates only that task's own binding, invisible to the parent task
# that continues on to a fallback hop. Mutating a shared holder in place is
# visible from every task forked after the holder was first created,
# regardless of which task performs the mutation.
class _PendingConcurrencyKeys:
    __slots__ = ("keys",)

    def __init__(self) -> None:
        self.keys: list[str] = []  # mutable-ok: shared across asyncio.create_task forks by design; see class docstring


_pending_concurrency_keys: Final[contextvars.ContextVar[_PendingConcurrencyKeys | None]] = contextvars.ContextVar(
    "tag_rate_limiter_pending_concurrency_keys", default=None
)


def _pending_concurrency_holder() -> _PendingConcurrencyKeys:
    existing: Final = _pending_concurrency_keys.get()
    if existing is not None:
        return existing
    holder: Final = _PendingConcurrencyKeys()
    _pending_concurrency_keys.set(holder)
    return holder


class _TagRateLimitIndex:
    """Rebuilds the limits index when `llm_router.model_list` changes, or at
    least every `_INDEX_TTL_SECONDS`, whichever comes first."""

    def __init__(self, time_provider: Callable[[], datetime]) -> None:
        self._time_provider = time_provider
        self._cache_key: tuple[int, int] | None = None
        self._built_at: float = 0.0
        self._index: _LimitsIndex = _LimitsIndex(by_model_name=MappingProxyType({}), by_team_alias=MappingProxyType({}))

    def get(self, llm_router: Router) -> _LimitsIndex:
        model_list: Final = llm_router.model_list or ()
        cache_key: Final = (id(llm_router), len(model_list))
        now: Final = self._time_provider().timestamp()
        if cache_key != self._cache_key or (now - self._built_at) >= _INDEX_TTL_SECONDS:
            self._index = _build_limits_index(model_list)
            self._cache_key = cache_key
            self._built_at = now
        return self._index


def _scope_suffix(deployment_scope: tuple[str, ...] | None) -> str:
    return "chain" if deployment_scope is None else "dep:" + "+".join(deployment_scope)


def _hash_tag(model_group: str, configured: _ConfiguredLimit, tag_value: str, key_hash: str | None) -> str:
    # resolved_group overrides the caller-visible model_group when this
    # limit was found via resolve_any()'s per-deployment fallback (routing
    # groups): the caller-visible name is ambiguous there (shared by every
    # member model_name), so hashing by it would collide two different
    # underlying model_names' identically-named limits onto one counter.
    # See _ConfiguredLimit.resolved_group.
    effective_model_group: Final = configured.resolved_group if configured.resolved_group is not None else model_group
    scope: Final = _scope_suffix(configured.deployment_scope)
    # team_scope disambiguates two teams that publish the identical
    # team_public_model_name alias with identically-configured limits --
    # without it their buckets would collide despite the index correctly
    # scoping the lookup by (team_id, alias). See _ConfiguredLimit.team_scope.
    team_suffix: Final = f":team:{configured.team_scope}" if configured.team_scope is not None else ""
    key_suffix: Final = f":key:{key_hash}" if key_hash is not None else ""
    return (
        f"tag_rl:{effective_model_group}:{configured.unit}:{configured.entry.name}:{configured.entry.tag_id}:"
        f"{scope}{team_suffix}:{tag_value}{key_suffix}"
    )


def _bucket_key(
    model_group: str,
    configured: _ConfiguredLimit,
    tag_value: str,
    bucket_id: int,
    key_hash: str | None = None,
) -> str:
    return f"{{{_hash_tag(model_group, configured, tag_value, key_hash)}}}:{bucket_id}"


def _inflight_key(
    model_group: str,
    configured: _ConfiguredLimit,
    tag_value: str,
    key_hash: str | None = None,
) -> str:
    """Concurrency counter key: not epoch-bucketed, since "how many are in
    flight right now" has no window to reset on -- it's released explicitly
    on completion, with a TTL fallback only for a leaked (crashed) reservation."""
    return f"{{{_hash_tag(model_group, configured, tag_value, key_hash)}}}:inflight"


class _ClassifiedCheck(NamedTuple):
    configured_limit: _ConfiguredLimit
    tag_value: str
    key: str
    is_atomic: bool


def _classify_check(
    configured_limit: _ConfiguredLimit,
    model: str,
    tags: Sequence[str],
    present_deployment_ids: frozenset[str],
    request_kwargs: Mapping[str, object],
    metadata_variable_name: str,
    now: float,
) -> _ClassifiedCheck | None:
    if configured_limit.deployment_scope is not None and not (
        present_deployment_ids & frozenset(configured_limit.deployment_scope)
    ):
        return None
    tag_value: Final = _extract_identity(tags, configured_limit.entry.tag_id)
    if tag_value is None:
        return None
    key_hash: Final = (
        _extract_key_hash(request_kwargs, metadata_variable_name) if configured_limit.entry.scope_by_key_hash else None
    )
    if configured_limit.unit == "concurrency":
        inflight_key: Final = _inflight_key(model, configured_limit, tag_value, key_hash=key_hash)
        return _ClassifiedCheck(configured_limit, tag_value, inflight_key, is_atomic=True)
    bucket_id: Final = int(now) // configured_limit.entry.period_seconds
    bucket_key_value: Final = _bucket_key(model, configured_limit, tag_value, bucket_id, key_hash=key_hash)
    return _ClassifiedCheck(
        configured_limit, tag_value, bucket_key_value, is_atomic=configured_limit.unit in _ATOMIC_UNITS
    )


def _increment_operation_for_limit(
    configured_limit: _ConfiguredLimit,
    model_group: str,
    tags: Sequence[str],
    deployment_id: str | None,
    key_hash: str | None,
    increment_by_unit: Mapping[_LimitUnit, float],
    now: float,
) -> RedisPipelineIncrementOperation | None:
    if configured_limit.unit == "concurrency":
        return None  # released above, from _pending_concurrency_keys
    if configured_limit.deployment_scope is not None and deployment_id not in configured_limit.deployment_scope:
        return None
    tag_value: Final = _extract_identity(tags, configured_limit.entry.tag_id)
    if tag_value is None:
        return None
    if configured_limit.unit not in increment_by_unit:
        return None  # "requests" is accounted atomically at admission, not here
    increment_value: Final = increment_by_unit[configured_limit.unit]
    if increment_value == 0:
        return None
    bucket_id: Final = int(now) // configured_limit.entry.period_seconds
    key_hash_for_limit: Final = key_hash if configured_limit.entry.scope_by_key_hash else None
    key: Final = _bucket_key(model_group, configured_limit, tag_value, bucket_id, key_hash=key_hash_for_limit)
    return RedisPipelineIncrementOperation(
        key=key,
        increment_value=increment_value,
        ttl=configured_limit.entry.period_seconds + 3600,
    )


class _PROXY_TagRateLimiter(  # pyright: ignore[reportUnusedClass]  # only referenced via the deferred import in litellm_logging.py's callback resolver; basedpyright doesn't trace that usage
    CustomLogger
):
    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
    ) -> None:
        # A dedicated in-memory layer, not the proxy-wide `internal_usage_cache`
        # passed in: that instance is shared with the key/team parallel-request
        # limiter's own authentication-bound counters, and its default
        # InMemoryCache evicts at 200 items. Without this isolation, a caller
        # flooding this hook's own caller-controlled tag buckets past that
        # ceiling could evict an unrelated, authentication-bound counter and
        # exceed a limit nothing here configured. The real Redis connection
        # (if any) is still shared, so cross-instance correctness is unaffected.
        isolated_dual_cache: Final = DualCache(redis_cache=internal_usage_cache.redis_cache)
        self.internal_usage_cache = InternalUsageCache(dual_cache=isolated_dual_cache)
        self._v3 = _PROXY_MaxParallelRequestsHandler_v3(self.internal_usage_cache, time_provider=time_provider)
        self._time_provider = time_provider or datetime.now
        self._index = _TagRateLimitIndex(time_provider=self._time_provider)
        self._lock = asyncio.Lock()
        self.llm_router: Router | None = None
        redis_cache: Final = self.internal_usage_cache.dual_cache.redis_cache
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
            raw: Final = await self._check_and_incr_script(keys=(key,), args=(limit, increment, ttl))
            return bool(raw[0]), float(raw[1])

        async with self._lock:
            current_value: Final = await self.internal_usage_cache.async_get_cache(
                key=key, litellm_parent_otel_span=None
            )
            current: Final = float(current_value) if current_value is not None else 0.0
            if current + increment > limit:
                return False, current
            new_value: Final = current + increment
            await self.internal_usage_cache.async_set_cache(
                key=key, value=new_value, ttl=ttl, litellm_parent_otel_span=None
            )
            return True, new_value

    async def _decrement_floor_zero(self, key: str, delta: float) -> None:
        if self._decr_floor_zero_script is not None:
            await self._decr_floor_zero_script(keys=(key,), args=(delta,))
            return
        async with self._lock:
            current_value: Final = await self.internal_usage_cache.async_get_cache(
                key=key, litellm_parent_otel_span=None
            )
            current: Final = float(current_value) if current_value is not None else 0.0
            await self.internal_usage_cache.async_set_cache(
                key=key, value=max(0.0, current + delta), litellm_parent_otel_span=None
            )

    async def _atomic_check_and_increment(
        self,
        checks: Sequence[tuple[str, float, float, int]],
    ) -> tuple[int | None, tuple[float, ...]]:
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
            return None, ()

        # Sequential async admission: each element needs its own awaited
        # Redis round trip, and a rejection mid-loop discards everything
        # accumulated so far in favor of refunding and returning early, so
        # this can't be expressed as a one-shot comprehension.
        admitted_values: Final = []  # mutable-ok: sequential async accumulator, discardable on early rejection; see comment above
        for index, (key, limit, increment, ttl) in enumerate(checks):
            admitted, value = await self._check_and_increment_one(key, limit, increment, ttl)
            if admitted:
                admitted_values.append(value)  # mutable-ok: see accumulator comment above
                continue
            for refund_index in range(index):
                refund_key, _limit, refund_increment, _ttl = checks[refund_index]
                try:
                    await self._decrement_floor_zero(refund_key, -refund_increment)
                except Exception as e:  # noqa: BLE001 - one failed refund must not block refunding the rest
                    verbose_proxy_logger.warning("tag_rate_limiter: failed to refund %s on rollback: %s", refund_key, e)
            return index, (value,)

        return None, tuple(admitted_values)

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,  # mutable-ok: must match CustomLogger's base signature exactly, or basedpyright flags reportIncompatibleMethodOverride
        messages: list[AllMessageValues] | None,  # mutable-ok: see reason above
        request_kwargs: dict | None = None,  # mutable-ok: see reason above
        parent_otel_span: Span | None = None,
    ) -> list[dict]:  # mutable-ok: see reason above
        if (
            not healthy_deployments
            or not isinstance(healthy_deployments, list)  # pyright: ignore[reportUnnecessaryIsInstance]  # defensive at runtime despite the static list annotation Router's own callers aren't guaranteed to honor
            or self.llm_router is None
        ):
            return healthy_deployments

        resolved_request_kwargs: Final = request_kwargs or _EMPTY_MAPPING
        metadata_variable_name: Final = get_metadata_variable_name_from_kwargs(resolved_request_kwargs)
        team_id: Final = _extract_team_id(resolved_request_kwargs, metadata_variable_name)
        candidate_model_names: Final = tuple(
            name for d in healthy_deployments if isinstance(name := d.get("model_name"), str)
        )
        configured: Final = self._index.get(self.llm_router).resolve_any(model, team_id, candidate_model_names)
        if not configured:
            return healthy_deployments

        tags: Final = _get_tags_from_request_kwargs(
            resolved_request_kwargs, metadata_variable_name=metadata_variable_name
        )

        present_deployment_ids: Final[frozenset[str]] = frozenset(
            dep_id for d in healthy_deployments if (dep_id := _deployment_id(d)) is not None
        )

        now: Final = self._time_provider().timestamp()
        classified: Final = tuple(
            check
            for configured_limit in configured
            if (
                check := _classify_check(
                    configured_limit,
                    model,
                    tags,
                    present_deployment_ids,
                    resolved_request_kwargs,
                    metadata_variable_name,
                    now,
                )
            )
            is not None
        )
        read_only_checks: Final = tuple((c.configured_limit, c.tag_value, c.key) for c in classified if not c.is_atomic)
        atomic_checks: Final = tuple((c.configured_limit, c.tag_value, c.key) for c in classified if c.is_atomic)

        current_values: Final = await self._read_only_values(read_only_checks, parent_otel_span)
        self._raise_if_over_limit(read_only_checks, current_values, model)

        if atomic_checks:
            failing_index, values = await self._atomic_check_and_increment(
                tuple(
                    (key, configured_limit.entry.limit, 1.0, self._ttl_for(configured_limit))
                    for configured_limit, _tag_value, key in atomic_checks
                )
            )
            if failing_index is not None:
                configured_limit, tag_value, _key = atomic_checks[failing_index]
                self._raise_over_limit(configured_limit, tag_value, model, current=values[0])

            concurrency_keys: Final = tuple(
                key for configured_limit, _tag_value, key in atomic_checks if configured_limit.unit == "concurrency"
            )
            if concurrency_keys:
                _pending_concurrency_holder().keys.extend(concurrency_keys)

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
        read_only_checks: Sequence[tuple[_ConfiguredLimit, str, str]],
        parent_otel_span: Span | None,
    ) -> tuple[float | None, ...]:
        if not read_only_checks:
            return ()
        keys: Final = tuple(key for _cfg, _tag_value, key in read_only_checks)
        current_values: Final = await self.internal_usage_cache.async_batch_get_cache(
            keys=list(keys),  # mutable-ok: async_batch_get_cache requires a real list; converted only at this boundary
            parent_otel_span=parent_otel_span,
            local_only=False,
        )
        return tuple(current_values) if current_values is not None else tuple(None for _ in keys)

    def _raise_if_over_limit(
        self,
        read_only_checks: Sequence[tuple[_ConfiguredLimit, str, str]],
        current_values: Sequence[float | None],
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
            detail={  # mutable-ok: must stay a real dict -- async_log_failure_event below (and generic proxy exception rendering, e.g. proxy/utils.py, guardrail hooks) branch on isinstance(exc.detail, dict); a MappingProxyType silently falls through those checks
                "error": "tag_rate_limit_exceeded",
                "type": configured_limit.unit,
                "tag_id": configured_limit.entry.tag_id,
                "tag_value": tag_value,
                "limit_name": configured_limit.entry.name,
                "limit": configured_limit.entry.limit,
                "period_seconds": configured_limit.entry.period_seconds,
            },
            headers={"retry-after": str(configured_limit.entry.period_seconds)},  # mutable-ok: same as detail
            rate_limit_type=_UNIT_TO_RATE_LIMIT_TYPE[configured_limit.unit],
            model=model,
            llm_provider="litellm_proxy",
        )

    async def _release_keys(self, keys: Sequence[str]) -> None:
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

    @staticmethod
    def _pop_pending_concurrency_keys() -> tuple[str, ...]:
        # Snapshot then remove only those exact keys, never a blanket clear:
        # a sibling hop can still be live and appending to the same shared
        # holder concurrently (see the holder's own comment above), so
        # wiping the whole list here would silently strand that hop's
        # reservation instead of releasing it later.
        holder: Final = _pending_concurrency_keys.get()
        if holder is None or not holder.keys:
            return ()
        keys: Final = tuple(holder.keys)
        for key in keys:
            try:
                holder.keys.remove(key)
            except ValueError:
                pass
        return keys

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        if isinstance(kwargs.get("exception"), ProxyRateLimitError):
            detail: Final = (
                kwargs["exception"].detail if isinstance(kwargs["exception"].detail, dict) else _EMPTY_MAPPING
            )
            if detail.get("error") == "tag_rate_limit_exceeded":
                return

        release_keys: Final = self._pop_pending_concurrency_keys()
        if release_keys:
            await self._release_keys(release_keys)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        release_keys: Final = self._pop_pending_concurrency_keys()
        if release_keys:
            asyncio.create_task(self._release_keys(release_keys))

        if self.llm_router is None:
            return

        standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
        if standard_logging_object is None:
            return

        model_group: Final = standard_logging_object.get("model_group")
        if not model_group:
            return

        standard_logging_metadata: Final = standard_logging_object.get("metadata") or _EMPTY_MAPPING
        team_id: Final = standard_logging_metadata.get("user_api_key_team_id")
        key_hash: Final = standard_logging_metadata.get("user_api_key_hash")
        # model_group is the caller-visible name, which Router deliberately
        # keeps distinct from the serving deployment's own model_name for a
        # routing-group call (see resolve_any's docstring); fall back to the
        # one deployment that actually served this hop.
        deployment_id: Final = standard_logging_object.get("model_id")
        serving_deployment: Final = (
            self.llm_router.get_deployment(deployment_id) if isinstance(deployment_id, str) else None
        )
        candidate_model_names: Final = (serving_deployment.model_name,) if serving_deployment is not None else ()
        configured: Final = self._index.get(self.llm_router).resolve_any(model_group, team_id, candidate_model_names)
        if not configured:
            return

        # kwargs here is Logging.model_call_details, not the router's flat
        # request kwargs admission sees: metadata/litellm_metadata are never
        # top-level here, only nested under kwargs["litellm_params"] (see
        # Logging.update_environment_variables). Resolving the field name
        # against kwargs itself always picks the "metadata" default, so on
        # LITELLM_METADATA_ROUTES (/v1/messages, /responses, ...) this read
        # the caller's native, tag-less metadata instead of the real,
        # server-computed litellm_metadata.tags admission already used.
        litellm_params_for_metadata: Final = kwargs.get("litellm_params") or kwargs
        metadata_variable_name: Final = get_metadata_variable_name_from_kwargs(litellm_params_for_metadata)
        tags: Final = _get_tags_from_request_kwargs(kwargs, metadata_variable_name=metadata_variable_name)
        if not tags:
            return

        now: Final = self._time_provider().timestamp()
        increment_by_unit: Final[Mapping[_LimitUnit, float]] = MappingProxyType(
            {
                "tokens": float(standard_logging_object.get("total_tokens") or 0),
                "dollars": float(standard_logging_object.get("response_cost") or 0),
            }
        )

        operations: Final = tuple(
            operation
            for configured_limit in configured
            if (
                operation := _increment_operation_for_limit(
                    configured_limit, model_group, tags, deployment_id, key_hash, increment_by_unit, now
                )
            )
            is not None
        )

        if not operations:
            return

        asyncio.create_task(
            self._v3.async_increment_tokens_with_ttl_preservation(
                pipeline_operations=operations,
                parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
            )
        )
