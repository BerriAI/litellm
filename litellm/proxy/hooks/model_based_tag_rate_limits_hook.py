"""Tag-scoped token, request, dollar, and concurrency rate limits."""

import asyncio
import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
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
from litellm.types.router import TagRateLimitEntry, TagRateLimits, TagRateLimitScope
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span: TypeAlias = _Span
else:
    Span: TypeAlias = object

_LimitUnit: TypeAlias = Literal["tokens", "requests", "dollars", "concurrency"]
_LIMIT_UNITS: Final[tuple[_LimitUnit, ...]] = ("tokens", "requests", "dollars", "concurrency")
# A (tag_id, values) pair mirroring TagRateLimitScope's own fields, used only
# to fold `enabled_for`/`disabled_for` into `_DedupSignature` below without
# depending on TagRateLimitScope's own hashability.
_ScopeSignature: TypeAlias = tuple[str, tuple[str, ...]] | None
# (tag_id, name, limit, period_seconds, scope_by_key_hash, enabled_for,
# disabled_for, apply_to_key_alias) -- the fields that decide whether two
# deployments' entries are the same rate limit for dedup purposes; see
# _build_group_limits. Two deployments that agree on the first five but
# disagree on any scoping field are declaring genuinely different policies
# (e.g. one excludes a user the other doesn't) and must not be merged into
# one shared bucket -- the same class of bug this signature already guards
# against for a plain divergent `limit`.
_DedupSignature: TypeAlias = tuple[
    str,
    str,
    float,
    int,
    bool,
    _ScopeSignature,
    _ScopeSignature,
    tuple[str, ...] | None,
]
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

# `asyncio.create_task`'s own docs: "Save a reference to the result of this
# function, to avoid a task disappearing mid-execution. The event loop only
# keeps weak references to tasks. A task that isn't referenced elsewhere may
# get garbage collected at any time, even before it's done." The success path
# deliberately fires-and-forgets its concurrency release and its token/dollar
# accounting increment (unlike the failure/disconnect paths, which await
# concurrency release directly) to keep the hot success-response path from
# waiting on a Redis round trip; by the time either background task would
# run, the state it needs (popped pending keys, or the request's own usage
# figures) is only available in that task's own closure, so a collected
# task's work is unrecoverable, not just delayed. Holding a strong reference
# here until each task's own completion callback discards it is the standard
# fix, shared by every fire-and-forget task this hook creates.
_BACKGROUND_TASKS: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: see comment above

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
# negative counter that would admit unlimited requests. Floors via DEL, not
# `SET key 0`: releasing a reservation whose key already expired makes
# INCRBY recreate it with no TTL, and a plain SET would leave that recreated
# key permanently in Redis (SET clears any TTL); DEL removes it outright,
# which reads back identically to 0 everywhere this key is read (`GET key or 0`).
TAG_RL_DECR_FLOOR_ZERO_SCRIPT: Final = """
local key = KEYS[1]
local delta = tonumber(ARGV[1])
local new_value = redis.call('INCRBY', key, delta)
if new_value < 0 then
    redis.call('DEL', key)
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


def _scope_signature(scope: TagRateLimitScope | None) -> _ScopeSignature:
    """Normalizes a `TagRateLimitScope` into a plain, hashable tuple for use
    in `_DedupSignature` -- see that alias's own comment for why two
    deployments disagreeing on `enabled_for`/`disabled_for` must be treated
    as genuinely different policies rather than merged into one bucket."""
    return None if scope is None else (scope.tag_id, scope.values)


def _entry_applies(entry: TagRateLimitEntry, tags: Sequence[str], key_alias: str | None) -> bool:
    """
    Applies `entry`'s own scoping fields (`enabled_for`/`disabled_for`/
    `apply_to_key_alias`), evaluated in this order -- deny overrides allow,
    checked before either allowlist:

      1. `disabled_for`: the gate tag (often a SECOND, independent tag, but
         `disabled_for.tag_id` can equally be set to this entry's own
         `tag_id` to gate on a subset of its own resolved identity) is
         present and its value is in `disabled_for.values` -> doesn't apply.
         Absent gate tag never triggers this -- nothing to match against a
         denylist.
      2. `enabled_for`: the gate tag is absent, or present but its value is
         NOT in `enabled_for.values` -> doesn't apply. Unlike `disabled_for`,
         absence DOES fail this check -- an allowlist gate requires an
         explicit match, so "not tagged at all" means "not in scope".
      3. `apply_to_key_alias`: the calling key's own alias is absent, or
         present but not in the list -> doesn't apply. Same allowlist
         semantics as `enabled_for` -- a key with no alias set never
         satisfies this gate.

    An entry with none of these fields set always applies -- this is the
    unscoped behavior every existing entry has today, unchanged.
    """
    if entry.disabled_for is not None:
        disabled_gate_value: Final = _extract_identity(tags, entry.disabled_for.tag_id)
        if disabled_gate_value is not None and disabled_gate_value in entry.disabled_for.values:
            return False
    if entry.enabled_for is not None:
        enabled_gate_value: Final = _extract_identity(tags, entry.enabled_for.tag_id)
        if enabled_gate_value is None or enabled_gate_value not in entry.enabled_for.values:
            return False
    if entry.apply_to_key_alias is None:
        return True
    return key_alias in entry.apply_to_key_alias


def _deployment_id(deployment: Mapping[str, object]) -> str | None:
    return (deployment.get("model_info") or _EMPTY_MAPPING).get("id")


def _resolve_success_event_metadata_variable_name(
    litellm_params_for_metadata: Mapping[str, object],
) -> Literal["metadata", "litellm_metadata"]:
    """`get_metadata_variable_name_from_kwargs` only checks key presence, which
    misresolves at `async_log_success_event` time: `kwargs["litellm_params"]`
    always carries a `litellm_metadata` key (typically `None`) alongside the
    real, populated `metadata` dict for a standard (non
    LITELLM_METADATA_ROUTES) request, so the key-presence check always picks
    `litellm_metadata` there and silently reads no tags/identity at all.
    Requiring the value to actually be a populated dict, matching
    `_get_request_tags`'s own truthiness check in litellm_logging.py, only
    ever prefers `litellm_metadata` when it is genuinely the field the proxy
    wrote identity/tags into (LITELLM_METADATA_ROUTES pre-seed it before
    admission runs, so it is always a populated dict by success time there)."""
    litellm_metadata: Final = litellm_params_for_metadata.get("litellm_metadata")
    if isinstance(litellm_metadata, Mapping) and litellm_metadata:
        return "litellm_metadata"
    return "metadata"


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


def _extract_key_alias(request_kwargs: Mapping[str, object], metadata_variable_name: str) -> str | None:
    """Same single-authoritative-field lookup as `_extract_team_id`, but for
    the calling virtual key's own `key_alias`: `LiteLLMProxyRequestSetup` sets
    `metadata["user_api_key_alias"]` to `user_api_key_dict.key_alias`
    (see `litellm_pre_call_utils.py`)."""
    active: Final = request_kwargs.get(metadata_variable_name) or _EMPTY_MAPPING
    key_alias: Final = active.get("user_api_key_alias")
    return key_alias if isinstance(key_alias, str) else None


def _entries_for_unit(deployment: Mapping[str, object], unit: _LimitUnit) -> tuple[TagRateLimitEntry, ...]:
    raw_tag_rate_limits: Final = (deployment.get("model_info") or _EMPTY_MAPPING).get("tag_rate_limits")
    if not raw_tag_rate_limits:
        return ()
    tag_rate_limits: Final = TagRateLimits.model_validate(raw_tag_rate_limits)
    group: Final = getattr(tag_rate_limits, _UNIT_TO_GROUP_FIELD[unit])
    return tuple(group.limits) if group is not None else ()


def _configured_limit_for_signature(
    unit: _LimitUnit,
    entry: TagRateLimitEntry,
    declaring_ids: Sequence[str],
    is_chain_wide: bool,
) -> _ConfiguredLimit | None:
    if unit == "concurrency" and not is_chain_wide:
        verbose_proxy_logger.warning(
            "model_based_tag_rate_limits_hook: concurrency_limits entry %r (tag_id=%s) is not declared identically by every "
            "deployment sharing this model_name; per-deployment-scoped concurrency limits are not supported "
            "and this entry is being skipped entirely.",
            entry.name,
            entry.tag_id,
        )
        return None
    return _ConfiguredLimit(
        unit=unit,
        entry=entry,
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
    # The dedup signature is deliberately narrower than the full entry: two
    # deployments agreeing on (tag_id, name, limit, period_seconds,
    # scope_by_key_hash) share one bucket even if they set key_ttl_seconds or
    # max_in_memory_cache_size differently. Whichever deployment's entry is
    # seen first for a given signature supplies those fields for the whole
    # group -- an arbitrary but deterministic tie-break, consistent with the
    # first-seen-order precedent already established above.
    representative_entry_by_signature: Final[dict[_DedupSignature, TagRateLimitEntry]] = {}  # mutable-ok: see above
    for deployment in deployments:
        dep_id = _deployment_id(deployment)
        if dep_id is None:
            continue
        for entry in _entries_for_unit(deployment, unit):
            signature = (
                entry.tag_id,
                entry.name,
                entry.limit,
                entry.period_seconds,
                entry.scope_by_key_hash,
                _scope_signature(entry.enabled_for),
                _scope_signature(entry.disabled_for),
                entry.apply_to_key_alias,
            )
            ids_for_signature = declaring_ids_by_signature.setdefault(signature, [])  # mutable-ok: see comment above
            # One deployment declaring the identical entry twice (a config
            # duplicate) must count once, or len(declaring_ids) inflates past
            # total_deployments below, making is_chain_wide false for an
            # entry every deployment actually agrees on -- for concurrency
            # that silently drops the entry entirely (see the docstring
            # above), disabling enforcement rather than degrading it.
            if dep_id not in ids_for_signature:
                ids_for_signature.append(dep_id)  # mutable-ok: see comment above
            representative_entry_by_signature.setdefault(signature, entry)  # mutable-ok: see comment above

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
                representative_entry_by_signature[signature],
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

        Members declaring the identical signature and scope are deduped to
        one shared entry: only one deployment in the group ends up actually
        serving a given hop, but every member's own `model_name` is resolved
        independently above, so an undeduped union would check and charge
        every member's bucket for that one hop -- request/concurrency
        capacity a caller never actually used, and a false 429 for a sibling
        member that was never over its own limit. Divergent configs across
        model_names (different limit/period/scope for the same tag_id+name)
        are left as separate entries, same as before this dedup: resolving
        that ambiguity needs knowing which deployment will be picked, which
        isn't known yet at this admission-time hook.

        Candidates are deduped in sorted order, not raw `frozenset` iteration
        order: `frozenset` order depends on the process's hash seed, so two
        workers resolving the identical candidate set could otherwise pick
        different members as `resolved_group` and end up checking/accounting
        against different Redis keys for what's meant to be one shared bucket.
        """
        direct: Final = self.resolve(model, team_id)
        if direct:
            return direct
        deduped: Final[dict[tuple[object, ...], _ConfiguredLimit]] = {}  # mutable-ok: see docstring above
        for name in sorted(frozenset(candidate_model_names)):
            for limit in self.by_model_name.get(name, ()):
                key = (
                    limit.unit,
                    limit.entry.tag_id,
                    limit.entry.name,
                    limit.entry.limit,
                    limit.entry.period_seconds,
                    limit.entry.scope_by_key_hash,
                    _scope_signature(limit.entry.enabled_for),
                    _scope_signature(limit.entry.disabled_for),
                    limit.entry.apply_to_key_alias,
                    limit.deployment_scope,
                    limit.team_scope,
                )
                deduped.setdefault(key, replace(limit, resolved_group=name))  # mutable-ok: see docstring above
        return tuple(deduped.values())


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
# not yet released, paired with the cache-size override (from
# TagRateLimitEntry.max_in_memory_cache_size) each reservation was
# incremented under: releasing a reservation must decrement the exact same
# cache partition it was incremented on, or the release silently no-ops on
# the wrong (default) partition and the reservation leaks forever.
#
# Stashed directly on `Logging.model_call_details` under this field, not a
# `contextvars.ContextVar`: the real proxy request pipeline forks the
# streaming response through several distinct asyncio Tasks (the disconnect
# race in `create_response`, the streaming generator's own task, ...), and a
# ContextVar only propagates forward into tasks forked *after* a value was
# `.set()` -- a task that isn't a descendant of admission's task never sees
# it, so release silently finds nothing and every reservation leaks until
# `_CONCURRENCY_MIN_SAFETY_TTL_SECONDS`, disconnect or not (confirmed live:
# even a fully-completed, non-disconnected streaming request never released
# its slot). `model_call_details` is a single dict, explicitly passed by
# object reference through both admission's `request_kwargs` (as
# `request_kwargs["litellm_logging_obj"].model_call_details`) and release's
# `kwargs` (`async_log_success_event`/`async_log_failure_event`'s `kwargs`
# argument *is* `model_call_details` -- see their own callers), so it
# survives task boundaries by construction, not by ambient context.
#
# Deliberately not keyed by `litellm_call_id` instead: that field is
# caller-controlled via the `x-litellm-call-id` request header, so two
# unrelated concurrent requests sharing a caller-chosen id would merge their
# reservations under a shared identifier -- letting one request's release
# free a different request's still-live slot. `model_call_details` is a
# plain Python object with no caller-visible identifier, created fresh
# server-side per logical request (and shared across that request's own
# fallback hops, matching the original chain-wide release semantics), so it
# can't be forged or guessed.
_PENDING_CONCURRENCY_KEYS_FIELD: Final[str] = "_model_based_tag_rate_limits_pending_concurrency_keys"

# The admission-time timestamp a hop's token/dollar checks classified their
# bucket against, stashed on the same model_call_details object so success
# accounting recomputes the identical bucket_id (int(now) // period_seconds)
# instead of a fresh one. A completion can take long enough for a fresh
# timestamp at success time to land in the *next* window than the one
# admission actually checked, letting a burst of calls admitted against one
# (still-under-limit) window get charged entirely into the next window's
# fresh, unrelated counter -- silently bypassing the limit right around each
# rollover. Overwritten by each hop's own admission (last-write-wins), which
# is correct: success only ever fires for whichever hop actually served the
# request, so its own most recent admission timestamp is the right one.
_ADMISSION_TIME_FIELD: Final[str] = "_model_based_tag_rate_limits_admission_time"


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


def _fixed_length_identity(tag_value: str) -> str:
    """
    `tag_value` is caller-controlled (whatever follows the tag_id prefix in
    a caller-supplied tag) with no length or content bound. Embedding it
    directly would let a caller inflate this hook's own in-memory dict keys
    past what `max_in_memory_cache_size` bounds (that caps item *count*, not
    key bytes) and grow unbounded Redis keys with no cap at all. Hashing to
    a fixed-length digest bounds this hook's own contribution to key size
    regardless of the caller's input, while still preserving distinctness
    (two different tag values still resolve to two different buckets).
    """
    return hashlib.sha256(tag_value.encode()).hexdigest()


def _policy_fingerprint(entry: TagRateLimitEntry) -> str:
    """
    Two entries can share a `name` and `tag_id` while genuinely disagreeing
    on `limit`, `period_seconds`, `scope_by_key_hash`, or any of the scoping
    fields -- `_DedupSignature`/`resolve_any` already treat that as two
    distinct policies (see `distinct_signature_count_by_name` in
    `_build_group_limits`), so the Redis/in-memory bucket key must too, or two
    differently-configured entries that happen to share a name check and
    charge the identical counter. `scope_by_key_hash` specifically needs its
    own slot here rather than relying on `_hash_tag`'s `key_hash`-derived
    suffix to carry it: that suffix is empty whenever `key_hash` resolves to
    `None` (no virtual key on the call), which would otherwise collide an
    unscoped entry with a key-hash-scoped one that agrees on every other
    field. Hashed to a fixed-length digest for the same reason
    `_fixed_length_identity` hashes `tag_value`: an operator's own
    `enabled_for`/`disabled_for`/`apply_to_key_alias` list has no length bound.
    """
    fingerprint_source: Final = (
        entry.limit,
        entry.period_seconds,
        entry.scope_by_key_hash,
        _scope_signature(entry.enabled_for),
        _scope_signature(entry.disabled_for),
        entry.apply_to_key_alias,
    )
    return hashlib.sha256(repr(fingerprint_source).encode()).hexdigest()[:16]


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
    # Two entries can share `name`/`tag_id` while disagreeing on limit,
    # period_seconds, or scoping (see _policy_fingerprint) -- included so
    # they never collide onto the same counter despite the shared name.
    policy_suffix: Final = f":policy:{_policy_fingerprint(configured.entry)}"
    return (
        f"tag_rl:{effective_model_group}:{configured.unit}:{configured.entry.name}:{configured.entry.tag_id}:"
        f"{scope}{team_suffix}:{_fixed_length_identity(tag_value)}{key_suffix}{policy_suffix}"
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
    key_alias: str | None,
) -> _ClassifiedCheck | None:
    if configured_limit.deployment_scope is not None and not (
        present_deployment_ids & frozenset(configured_limit.deployment_scope)
    ):
        return None
    tag_value: Final = _extract_identity(tags, configured_limit.entry.tag_id)
    if tag_value is None:
        return None
    if not _entry_applies(configured_limit.entry, tags, key_alias):
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


def _bucket_ttl_seconds(entry: TagRateLimitEntry) -> int:
    """Redis (and in-memory fallback) TTL for a non-concurrency bucket key.
    `entry.key_ttl_seconds` overrides the default of period_seconds + 3600
    when set -- see TagRateLimitEntry.key_ttl_seconds."""
    return entry.key_ttl_seconds if entry.key_ttl_seconds is not None else entry.period_seconds + 3600


def _increment_operation_for_limit(
    configured_limit: _ConfiguredLimit,
    model_group: str,
    tags: Sequence[str],
    deployment_id: str | None,
    key_hash: str | None,
    key_alias: str | None,
    increment_by_unit: Mapping[_LimitUnit, float],
    now: float,
) -> RedisPipelineIncrementOperation | None:
    if configured_limit.unit == "concurrency":
        return None  # released above, via _pop_pending_concurrency_keys
    if configured_limit.deployment_scope is not None and deployment_id not in configured_limit.deployment_scope:
        return None
    tag_value: Final = _extract_identity(tags, configured_limit.entry.tag_id)
    if tag_value is None:
        return None
    if not _entry_applies(configured_limit.entry, tags, key_alias):
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
        ttl=_bucket_ttl_seconds(configured_limit.entry),
    )


def _resolve_max_in_memory_cache_size() -> int | None:
    """
    `litellm_settings` values reach `litellm.model_based_tag_rate_limits_max_in_memory_cache_size`
    via a plain, unvalidated `setattr`, so a config typo (a negative number, or a
    string like "500" from an unresolved os.environ/ substitution) can reach here.
    InMemoryCache raises when comparing its size against a non-positive-int
    max_size_in_memory, and DualCache.async_set_cache swallows that exception, so
    an invalid value would otherwise silently disable every counter write for this
    hook rather than fail loudly -- rejected here in favor of the safe default instead.
    """
    configured: Final = litellm.model_based_tag_rate_limits_max_in_memory_cache_size
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return configured
    if configured is not None:
        verbose_proxy_logger.warning(
            "model_based_tag_rate_limits_hook: model_based_tag_rate_limits_max_in_memory_cache_size=%r is not a positive integer; "
            "falling back to the default in-memory cache size.",
            configured,
        )
    return None


# None => this entry shares the hook's single default cache partition
# (matching every entry's behavior before this override existed). Otherwise
# a value-stable signature -- not the override int alone -- so two different
# entries that happen to choose the identical max_in_memory_cache_size don't
# get merged into one shared partition; the same entry (same config content)
# always resolves to the same signature across index rebuilds, which is what
# keeps _PROXY_ModelBasedTagRateLimitsHook._partitions from leaking a fresh partition
# every time _TagRateLimitIndex rebuilds and reconstructs `_ConfiguredLimit`s.
_PartitionKey: TypeAlias = tuple[str, str, float, int, bool, int] | None
# Grouping type for async_log_success_event's per-partition tokens/dollars
# pipeline dispatch -- named only so the declaration fits on one line; see
# that method for why the grouping is needed.
_PartitionOperations: TypeAlias = dict[_PartitionKey, list[RedisPipelineIncrementOperation]]


def _partition_key(entry: TagRateLimitEntry) -> _PartitionKey:
    if entry.max_in_memory_cache_size is None:
        return None
    return (
        entry.tag_id,
        entry.name,
        entry.limit,
        entry.period_seconds,
        entry.scope_by_key_hash,
        entry.max_in_memory_cache_size,
    )


def _queue_pending_concurrency_reservations(
    request_kwargs: Mapping[str, object], reservations: Sequence[tuple[str, _PartitionKey]]
) -> None:
    """Stash reservations on the request's own `model_call_details` -- see
    `_PENDING_CONCURRENCY_KEYS_FIELD`'s docstring for why this, not a
    ContextVar or `litellm_call_id`. Silently a no-op without a real logging
    object (defensive only; every real request has one): the reservation
    still self-heals via `_CONCURRENCY_MIN_SAFETY_TTL_SECONDS`, just later.
    """
    logging_obj: Final = request_kwargs.get("litellm_logging_obj")
    model_call_details: Final = getattr(logging_obj, "model_call_details", None)
    if not isinstance(model_call_details, dict):
        return
    pending = model_call_details.get(_PENDING_CONCURRENCY_KEYS_FIELD)  # rebind-ok: lazily initialized below when absent
    if pending is None:
        pending = []  # mutable-ok: shared, request-scoped accumulator; see field's own docstring  # rebind-ok: lazily initialized only when absent
        model_call_details[_PENDING_CONCURRENCY_KEYS_FIELD] = pending
    pending.extend(reservations)  # mutable-ok: see comment above


def _record_admission_time(request_kwargs: Mapping[str, object], now: float) -> None:
    """Stash this hop's admission timestamp -- see `_ADMISSION_TIME_FIELD`'s
    docstring for why. Silently a no-op without a real logging object
    (defensive only; every real request has one): success accounting falls
    back to its own fresh timestamp, same as before this fix existed."""
    logging_obj: Final = request_kwargs.get("litellm_logging_obj")
    model_call_details: Final = getattr(logging_obj, "model_call_details", None)
    if isinstance(model_call_details, dict):
        model_call_details[_ADMISSION_TIME_FIELD] = now


def _admission_time_or(kwargs: Mapping[str, object], fallback: float) -> float:
    recorded: Final = kwargs.get(_ADMISSION_TIME_FIELD)
    return recorded if isinstance(recorded, float) else fallback


@dataclass(frozen=True, slots=True)
class _CachePartition:
    internal_usage_cache: InternalUsageCache
    v3: _PROXY_MaxParallelRequestsHandler_v3


class _PROXY_ModelBasedTagRateLimitsHook(  # pyright: ignore[reportUnusedClass]  # only referenced via the deferred import in litellm_logging.py's callback resolver; basedpyright doesn't trace that usage
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
        # (if any) is still shared across every partition (see _build_partition),
        # so cross-instance correctness is unaffected regardless of partitioning.
        self._redis_cache: Final = internal_usage_cache.redis_cache
        self._time_provider = time_provider or datetime.now
        # Every distinct _PartitionKey gets its own dedicated partition
        # (in-memory cache + its own v3 handler), lazily built and memoized
        # here -- see _partition_for. None (the key every entry uses unless
        # it sets its own max_in_memory_cache_size) is this hook's single
        # default partition, sized by
        # litellm.model_based_tag_rate_limits_max_in_memory_cache_size (200 if that's
        # also unset), matching today's behavior for every entry that doesn't
        # opt into its own partition.
        self._partitions: dict[_PartitionKey, _CachePartition] = {}  # mutable-ok: lazily memoized; see _partition_for
        self._partitions_lock = asyncio.Lock()
        default_partition: Final = self._build_partition(_resolve_max_in_memory_cache_size())
        self._partitions[None] = default_partition
        self.internal_usage_cache = default_partition.internal_usage_cache
        self._v3 = default_partition.v3
        self._index = _TagRateLimitIndex(time_provider=self._time_provider)
        self._lock = asyncio.Lock()
        self.llm_router: Router | None = None
        redis_cache: Final = self._redis_cache
        self._check_and_incr_script = (
            redis_cache.async_register_script(TAG_RL_CHECK_AND_INCR_SCRIPT) if redis_cache is not None else None
        )
        self._decr_floor_zero_script = (
            redis_cache.async_register_script(TAG_RL_DECR_FLOOR_ZERO_SCRIPT) if redis_cache is not None else None
        )

    def update_variables(self, llm_router: Router) -> None:
        self.llm_router = llm_router

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
        """Single-key atomic check-and-increment. Always one key per Lua
        call -- see TAG_RL_CHECK_AND_INCR_SCRIPT's module docstring for why."""
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
        """
        All-or-nothing across every (cache, key, limit, increment, ttl) in
        `checks`: if any would exceed its limit, none are incremented -- a
        single hop's requests-unit and concurrency-unit checks must commit
        together or not at all, even when they span more than one cache
        partition. Each key is checked/incremented in its own single-key Lua
        call (cluster-safe by construction); all-or-nothing across the batch
        is enforced here by refunding every earlier admission the moment a
        later key is rejected, not by a single multi-key script call.

        Refunds are best-effort: a refund that fails (e.g. a transient Redis
        error) is logged and skipped rather than raised, so one bad refund
        can't stop the rest of the batch from being refunded, and can't turn
        a clean rejection into an unhandled exception. A skipped refund
        self-heals via the key's TTL -- see `_ttl_for`.

        A later key's own admission raising (a transient Redis error, or
        this coroutine being cancelled mid-call, e.g. the caller
        disconnecting) is treated the same as a normal rejection for refund
        purposes: only the earlier admissions in this batch are refunded,
        never the raising key's own key. This is deliberate, not an
        oversight: a raise gives no guarantee that key's own increment
        didn't already commit server-side (Redis can run the INCRBY and
        still have the call raise if the response back to us is lost), but
        these are shared, chain-wide buckets with no per-request ownership
        tracking -- decrementing on that guess is just as likely to erase a
        *different*, legitimately-admitted concurrent request's charge on
        the same key as it is to undo our own. That failure mode (an
        attacker repeatedly cancelling requests to erase other callers'
        charges and exceed the configured limit) is worse than the
        alternative this accepts instead: a key that did commit but never
        gets refunded self-heals via its own TTL -- see `_ttl_for`. The
        earlier admissions refunded here are never ambiguous like this: they
        are this same request's own confirmed-successful increments, so
        undoing them is always safe.

        Refunds are best-effort: a refund that fails (e.g. a transient Redis
        error) is logged and skipped rather than raised, so one bad refund
        can't stop the rest of the batch from being refunded, and can't turn
        a clean rejection into an unhandled exception.

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
        for index, (cache, key, limit, increment, ttl) in enumerate(checks):
            admitted = False
            try:
                admitted, value = await self._check_and_increment_one(cache, key, limit, increment, ttl)
            finally:
                # Runs on a normal rejection (admitted stays False) and on
                # any exception/cancellation from the awaited call above
                # (admitted never gets assigned, so it's still the False set
                # just before the try) -- either way, only the earlier,
                # known-safe admissions are refunded; see the docstring
                # above for why this key's own ambiguous outcome is not.
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
                    "model_based_tag_rate_limits_hook: failed to refund %s on rollback: %s", refund_key, e
                )

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
        await self._release_stale_hop_reservations(resolved_request_kwargs)
        metadata_variable_name: Final = get_metadata_variable_name_from_kwargs(resolved_request_kwargs)
        team_id: Final = _extract_team_id(resolved_request_kwargs, metadata_variable_name)
        # Built from the full routing-group membership, not `healthy_deployments`
        # (Router's own cooldown-filtered list for this hop): a member that's
        # merely cooled down right now is still a real member of the group for
        # the purpose of deciding resolved_group, and success accounting has no
        # way to know which members were healthy at admission time -- it can
        # only reconstruct the full, static membership (see its own comment
        # below). Deriving both sides from the same full-membership source is
        # the only way they're guaranteed to dedup to the identical bucket
        # regardless of cooldown state at either point in time.
        routing_group_deployments: Final = self.llm_router._get_routing_group_deployments(  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching resolve_any's own reliance on this method
            model=model, team_id=team_id
        )
        candidate_model_names: Final = (
            tuple(dep["model_name"] for dep in routing_group_deployments)
            if routing_group_deployments is not None
            else tuple(name for d in healthy_deployments if isinstance(name := d.get("model_name"), str))
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

        key_alias: Final = _extract_key_alias(resolved_request_kwargs, metadata_variable_name)
        now: Final = self._time_provider().timestamp()
        _record_admission_time(resolved_request_kwargs, now)
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
                    key_alias,
                )
            )
            is not None
        )
        read_only_checks: Final = tuple((c.configured_limit, c.tag_value, c.key) for c in classified if not c.is_atomic)
        atomic_checks: Final = tuple((c.configured_limit, c.tag_value, c.key) for c in classified if c.is_atomic)

        current_values: Final = await self._read_only_values(read_only_checks, parent_otel_span)
        self._raise_if_over_limit(read_only_checks, current_values, model)

        if atomic_checks:
            atomic_partitions_list: Final = []  # mutable-ok: sequential async lookups, one per atomic_checks entry (a genexpr can't `await` here); zipped with atomic_checks immediately below
            for configured_limit, _tag_value, _key in atomic_checks:
                atomic_partitions_list.append(
                    await self._partition_for(_partition_key(configured_limit.entry))
                )  # mutable-ok: see comment above
            atomic_partitions: Final = tuple(atomic_partitions_list)
            failing_index, values = await self._atomic_check_and_increment(
                tuple(
                    (
                        partition.internal_usage_cache,
                        key,
                        configured_limit.entry.limit,
                        1.0,
                        self._ttl_for(configured_limit),
                    )
                    for partition, (configured_limit, _tag_value, key) in zip(atomic_partitions, atomic_checks)
                )
            )
            if failing_index is not None:
                failing_limit, failing_tag_value, _ = atomic_checks[failing_index]
                self._raise_over_limit(failing_limit, failing_tag_value, model, current=values[0])

            concurrency_reservations: Final = tuple(
                (key, _partition_key(configured_limit.entry))
                for configured_limit, _tag_value, key in atomic_checks
                if configured_limit.unit == "concurrency"
            )
            if concurrency_reservations:
                _queue_pending_concurrency_reservations(resolved_request_kwargs, concurrency_reservations)

        return healthy_deployments

    @staticmethod
    def _ttl_for(configured_limit: _ConfiguredLimit) -> int:
        if configured_limit.unit == "concurrency":
            # A reservation's TTL must comfortably outlast any real in-flight
            # request, or a slow request's reservation self-heals (expires)
            # while it is still genuinely running, silently admitting extra
            # requests past the configured limit. period_seconds (or an
            # explicit key_ttl_seconds override) is still honored if the
            # operator wants an even longer safety margin, but this floor is
            # never lowered below it, even by an explicit override.
            entry: Final = configured_limit.entry
            requested_ttl: Final = entry.key_ttl_seconds if entry.key_ttl_seconds is not None else entry.period_seconds
            return max(requested_ttl, _CONCURRENCY_MIN_SAFETY_TTL_SECONDS)
        return _bucket_ttl_seconds(configured_limit.entry)

    async def _read_only_values(
        self,
        read_only_checks: Sequence[tuple[_ConfiguredLimit, str, str]],
        parent_otel_span: Span | None,
    ) -> tuple[float | None, ...]:
        if not read_only_checks:
            return ()

        # Grouped by cache partition (one batched read per partition), then
        # reassembled back into read_only_checks's original order: a hop can
        # mix entries from more than one partition (e.g. a default-cache
        # dollar_limits entry alongside a dedicated-partition request_limits
        # entry), and _raise_if_over_limit below zips this result positionally
        # against read_only_checks, so order must be preserved exactly.
        indices_by_partition: Final[dict[_PartitionKey, list[int]]] = {}  # mutable-ok: grouped, reassembled below
        for index, (configured_limit, _tag_value, _key) in enumerate(read_only_checks):
            partition_key = _partition_key(configured_limit.entry)
            indices = indices_by_partition.setdefault(partition_key, [])  # mutable-ok: see above
            indices.append(index)  # mutable-ok: see comment above

        values_by_index: Final[dict[int, float | None]] = {}  # mutable-ok: see comment above
        for partition_key, indices in indices_by_partition.items():
            # not `Final`: rebound each loop iteration, which basedpyright's
            # LIT010/Final-in-loop check forbids
            partition = await self._partition_for(partition_key)
            keys = [read_only_checks[i][2] for i in indices]  # mutable-ok: async_batch_get_cache needs a real list
            redis_cache = partition.internal_usage_cache.dual_cache.redis_cache
            if redis_cache is not None:
                # async_log_success_event increments these buckets straight
                # through a Lua script on this same redis_cache, bypassing
                # DualCache/InternalUsageCache entirely -- so its in-memory
                # layer never learns about that write. DualCache's own
                # async_batch_get_cache treats any non-None in-memory hit as
                # authoritative and never re-checks Redis for that key (see
                # _reserve_redis_batch_keys), so once a key is backfilled
                # in-memory it silently freezes for up to the in-memory TTL
                # (10 minutes by default) while the real Redis counter keeps
                # moving underneath it -- reading straight from Redis here,
                # bypassing that in-memory layer, is the only way this
                # read-then-later-increment split stays coherent.
                # not `Final`: rebound each loop iteration, which basedpyright's
                # LIT010/Final-in-loop check forbids; explicitly typed (as the
                # read-only supertype, since this is never mutated) since
                # RedisCache.async_batch_get_cache's own signature returns a
                # bare, unparameterized dict
                redis_values: Mapping[str, object] = await redis_cache.async_batch_get_cache(
                    key_list=keys, parent_otel_span=parent_otel_span
                )
                resolved = [redis_values.get(key) for key in keys]  # mutable-ok: needs a real list
            else:
                current_values = await partition.internal_usage_cache.async_batch_get_cache(
                    keys=keys,
                    parent_otel_span=parent_otel_span,
                    local_only=True,
                )
                missing = [None] * len(keys)  # mutable-ok: async_batch_get_cache requires a real list; see above
                resolved = current_values if current_values is not None else missing
            for i, value in zip(indices, resolved):
                values_by_index[i] = value  # mutable-ok: see comment above

        return tuple(values_by_index[i] for i in range(len(read_only_checks)))

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
            "model_based_tag_rate_limits_hook: OVER_LIMIT model=%s unit=%s name=%s tag_id=%s tag_value=%s current=%s limit=%s",
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

    async def _release_keys(self, reservations: Sequence[tuple[str, _PartitionKey]]) -> None:
        """
        Release each key by one slot. This does not verify the completing
        request still owns a live reservation (no per-request slot identity
        is tracked -- see the concurrency design note above), so a request
        that outlives the safety TTL and gets its key reused by a fresh
        reservation could in principle decrement a reservation it never
        held. Flooring at 0 (TAG_RL_DECR_FLOOR_ZERO_SCRIPT) bounds the
        damage to under-counting (briefly under-enforcing the limit) rather
        than a negative counter, which would admit unlimited requests.

        Each reservation is released against the exact cache partition
        (`_partition_for(partition_key)`) its increment used -- see
        `_PENDING_CONCURRENCY_KEYS_FIELD`'s docstring for why this must match.
        """
        for key, partition_key in reservations:
            try:
                partition = await self._partition_for(partition_key)  # not Final: rebound each loop iteration
                await self._decrement_floor_zero(partition.internal_usage_cache, key, -1.0)
            except Exception as e:  # noqa: BLE001 - releasing a slot must never raise into the caller's request path
                verbose_proxy_logger.warning(
                    "model_based_tag_rate_limits_hook: failed to release concurrency slot %s: %s", key, e
                )

    async def _release_stale_hop_reservations(self, request_kwargs: Mapping[str, object]) -> None:
        """
        A concurrency reservation still queued when a *new* hop's admission
        runs can only belong to an earlier hop of this same request that
        already concluded and failed: Router awaits one hop's entire attempt
        (call plus its own failure handling) before starting the next, and a
        hop that instead succeeded ends the request there via
        async_log_success_event, which already pops everything -- so
        admission is never re-entered while an earlier hop's reservation is
        still legitimately in flight.

        LiteLLM only invokes a request's CustomLogger.async_log_failure_event
        once per request, for whichever hop fails first (its internal
        has_logged_async_failure dedup silently skips every later hop's own
        failure), so every hop after that one would otherwise never release
        its predecessor's key until _CONCURRENCY_MIN_SAFETY_TTL_SECONDS.
        Releasing here, at the one point guaranteed to re-run before every
        subsequent hop, closes that gap for every hop except a final one
        whose own failure exhausts the retry chain -- that residual case
        still self-heals via the same TTL floor.
        """
        logging_obj: Final = request_kwargs.get("litellm_logging_obj")
        model_call_details: Final = getattr(logging_obj, "model_call_details", None)
        if not isinstance(model_call_details, dict):
            return
        release_keys: Final = self._pop_pending_concurrency_keys(model_call_details)
        if release_keys:
            await self._release_keys(release_keys)

    @staticmethod
    def _pop_pending_concurrency_keys(kwargs: Mapping[str, object]) -> tuple[tuple[str, _PartitionKey], ...]:
        # Snapshot then remove only those exact keys, never a blanket clear:
        # a sibling hop sharing this same request's model_call_details can
        # still be live and appending concurrently (see the field's own
        # docstring), so wiping the whole list here would silently strand
        # that hop's reservation instead of releasing it later.
        pending: Final = kwargs.get(_PENDING_CONCURRENCY_KEYS_FIELD)
        if not isinstance(pending, list) or not pending:
            return ()
        keys: Final = tuple(pending)
        for key in keys:
            try:
                pending.remove(key)
            except ValueError:
                pass
        return keys

    async def async_release_disconnect_state_hook(self, request_data: Mapping[str, object]) -> None:
        """
        A client disconnecting before the first streamed chunk raises
        CancelledError/GeneratorExit, which bypasses both async_log_success_event
        and async_log_failure_event below -- the only two places a concurrency
        reservation queued during admission is normally popped and released.
        Without this, the reservation would sit held until _CONCURRENCY_MIN_SAFETY_TTL_SECONDS
        expires, letting a caller who repeatedly opens and immediately drops
        streaming requests exhaust their own tag's concurrency limit for free.
        """
        logging_obj: Final = request_data.get("litellm_logging_obj")
        model_call_details: Final = getattr(logging_obj, "model_call_details", None)
        if not isinstance(model_call_details, dict):
            return
        release_keys: Final = self._pop_pending_concurrency_keys(model_call_details)
        if release_keys:
            await self._release_keys(release_keys)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        # No special-case skip for this hook's own tag_rate_limit_exceeded
        # rejection: a hop whose own admission rejects never reaches the
        # point where a concurrency reservation is queued (see
        # async_filter_deployments), so _pop_pending_concurrency_keys already
        # returns nothing to release in that case. Skipping release based on
        # the exception's error marker alone would be wrong here, since
        # global_tag_rate_limits_hook raises the identical marker -- that
        # rejection can land after this hook already reserved a slot for the
        # same request, and that slot must still be released.
        release_keys: Final = self._pop_pending_concurrency_keys(kwargs)
        if release_keys:
            await self._release_keys(release_keys)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        release_keys: Final = self._pop_pending_concurrency_keys(kwargs)
        if release_keys:
            release_task: Final = asyncio.create_task(self._release_keys(release_keys))
            _BACKGROUND_TASKS.add(release_task)  # mutable-ok: see _BACKGROUND_TASKS's own docstring
            release_task.add_done_callback(_BACKGROUND_TASKS.discard)

        if self.llm_router is None:
            return

        standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
        if standard_logging_object is None:
            return

        model_group: Final = standard_logging_object.get("model_group")
        if not model_group:
            return

        # kwargs here is Logging.model_call_details, not the router's flat
        # request kwargs admission sees: metadata/litellm_metadata are never
        # top-level here, only nested under kwargs["litellm_params"] (see
        # Logging.update_environment_variables).
        litellm_params_for_metadata: Final = kwargs.get("litellm_params") or kwargs
        metadata_variable_name: Final = _resolve_success_event_metadata_variable_name(litellm_params_for_metadata)
        team_id: Final = _extract_team_id(litellm_params_for_metadata, metadata_variable_name)
        key_hash: Final = _extract_key_hash(litellm_params_for_metadata, metadata_variable_name)
        key_alias: Final = _extract_key_alias(litellm_params_for_metadata, metadata_variable_name)
        # model_group is the caller-visible name, which Router deliberately
        # keeps distinct from the serving deployment's own model_name for a
        # routing-group call (see resolve_any's docstring). Passing only the
        # one deployment that actually served this hop as the sole candidate
        # would make resolve_any's dedup independently re-derive a
        # *different* resolved_group than admission did whenever the group
        # has more than one member: admission sees every member and picks
        # whichever one frozenset(candidate_model_names) yields first for a
        # shared signature, so success accounting must reconstruct that same
        # full candidate set to land on the identical bucket, not just
        # whichever deployment happened to serve -- otherwise a token/dollar
        # limit is checked against one bucket at admission and accounted
        # against a different one on success, letting usage silently bypass
        # the configured limit. Falls back to the serving deployment alone
        # only when `model_group` isn't a routing group at all (a plain
        # single-model_name chain, where resolve() already matches directly
        # and this candidate set is never actually consulted).
        deployment_id: Final = standard_logging_object.get("model_id")
        serving_deployment: Final = (
            self.llm_router.get_deployment(deployment_id) if isinstance(deployment_id, str) else None
        )
        routing_group_deployments: Final = self.llm_router._get_routing_group_deployments(  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching resolve_any's own reliance on this method
            model=model_group, team_id=team_id
        )
        candidate_model_names: Final = (
            tuple(dep["model_name"] for dep in routing_group_deployments)
            if routing_group_deployments is not None
            else ((serving_deployment.model_name,) if serving_deployment is not None else ())
        )
        configured: Final = self._index.get(self.llm_router).resolve_any(model_group, team_id, candidate_model_names)
        if not configured:
            return

        tags: Final = _get_tags_from_request_kwargs(kwargs, metadata_variable_name=metadata_variable_name)
        if not tags:
            return

        now: Final = _admission_time_or(kwargs, fallback=self._time_provider().timestamp())
        increment_by_unit: Final[Mapping[_LimitUnit, float]] = MappingProxyType(
            {
                "tokens": float(standard_logging_object.get("total_tokens") or 0),
                "dollars": float(standard_logging_object.get("response_cost") or 0),
            }
        )

        operation_by_limit: Final = tuple(
            (configured_limit, operation)
            for configured_limit in configured
            if (
                operation := _increment_operation_for_limit(
                    configured_limit, model_group, tags, deployment_id, key_hash, key_alias, increment_by_unit, now
                )
            )
            is not None
        )

        if not operation_by_limit:
            return

        # Grouped by cache partition: a hop's tokens/dollars entries can span
        # more than one partition, and each partition owns its own v3
        # handler (see _build_partition), so each group's operations are
        # pipelined through that partition's own handler.
        operations_by_partition: Final[_PartitionOperations] = {}  # mutable-ok: see comment above
        for configured_limit, operation in operation_by_limit:
            partition_key = _partition_key(configured_limit.entry)
            operations = operations_by_partition.setdefault(partition_key, [])  # mutable-ok: see above
            operations.append(operation)  # mutable-ok: see comment above

        parent_otel_span: Final = _get_parent_otel_span_from_kwargs(kwargs)
        for partition_key, group_operations in operations_by_partition.items():
            partition = await self._partition_for(partition_key)  # not Final: rebound each loop iteration
            accounting_task = asyncio.create_task(  # not Final: rebound each loop iteration
                partition.v3.async_increment_tokens_with_ttl_preservation(
                    pipeline_operations=tuple(group_operations),
                    parent_otel_span=parent_otel_span,
                )
            )
            _BACKGROUND_TASKS.add(accounting_task)  # mutable-ok: see _BACKGROUND_TASKS's own docstring
            accounting_task.add_done_callback(_BACKGROUND_TASKS.discard)
