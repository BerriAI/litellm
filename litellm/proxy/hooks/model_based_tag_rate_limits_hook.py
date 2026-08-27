"""
Tag-scoped token, request, dollar, and concurrency rate limits, admitted
once per routing hop via `async_filter_deployments`, for limits declared
per-deployment under `model_info.tag_rate_limits`.

Shares its identity/scope-extraction, policy-fingerprinting, bucket-key
hashing, and cache-partitioning primitives with the model-independent
`global_tag_rate_limits_hook.py` via `tag_rate_limits_shared.py`; this
module owns everything specific to per-deployment admission instead:
routing-group resolution, the (team-alias-aware) limits index, and
per-deployment dedup.
"""

import asyncio
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching dynamic_rate_limiter_v3's identical import
    get_metadata_variable_name_from_kwargs,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # this hook explicitly reuses its Redis/TTL-preserving increment machinery, see module docstring
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
    EMPTY_MAPPING as _EMPTY_MAPPING,
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
    ScopeSignature as _ScopeSignature,
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
    resolve_success_event_metadata_variable_name as _resolve_success_event_metadata_variable_name,
)
from litellm.proxy.hooks.tag_rate_limits_shared import (
    scope_signature as _scope_signature,
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

# (tag_id, name, limit, period_seconds, scope_by_key_hash, enabled_for,
# disabled_for, apply_to_key_alias, apply_to_models) -- the fields that
# decide whether two deployments' entries are the same rate limit for dedup
# purposes; see _build_group_limits. Two deployments that agree on the first
# five but disagree on any scoping field are declaring genuinely different
# policies (e.g. one excludes a user the other doesn't) and must not be
# merged into one shared bucket -- the same class of bug this signature
# already guards against for a plain divergent `limit`.
_DedupSignature: TypeAlias = tuple[
    str,
    str,
    float,
    int,
    bool,
    _ScopeSignature,
    _ScopeSignature,
    tuple[str, ...] | None,
    tuple[str, ...] | None,
]


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


def _deployment_id(deployment: Mapping[str, object]) -> str | None:
    return (deployment.get("model_info") or _EMPTY_MAPPING).get("id")


def _model_name_of(deployment: Mapping[str, object]) -> str:
    """`model_name` is a required field on every deployment dict; `str(...)`
    (rather than a bare index) gives `sorted`/`groupby`'s key functions a
    provably orderable, hashable return type without an unsafe cast -- every
    real deployment's value here is already a string, so this is a no-op
    coercion in practice."""
    return str(deployment["model_name"])


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
                entry.apply_to_models,
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
                    limit.entry.apply_to_models,
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
    sorted_by_model_name: Final = sorted(model_list, key=_model_name_of)
    by_model_name: Final[Mapping[str, tuple[_ConfiguredLimit, ...]]] = MappingProxyType(
        {
            model_name: (
                # `Router.should_include_deployment` lets a same-team caller
                # reach a team-owned deployment by its own internal
                # model_name, not only its team_public_model_name alias
                # (litellm auto-generates a name unique per (team_id, uuid),
                # so every deployment in this group shares one team_id when
                # any does) -- stamping the identical team_scope here as the
                # alias entry below gets keeps both paths resolving to the
                # same bucket, so a caller can't split its usage across two
                # independent counters just by alternating which name it calls.
                tuple(replace(limit, team_scope=team_scope) for limit in configured)
                if (team_scope := next((key[0] for dep in group if (key := _team_alias_key(dep))), None)) is not None
                else configured
            )
            for model_name, deployment_group in groupby(sorted_by_model_name, key=_model_name_of)
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

# Same `model_call_details`-stashing rationale as the field above, for a
# different unit: "requests" is atomic and admitted once per hop (see
# _ATOMIC_UNITS), same as concurrency, but a "requests" limit is meant to cap
# logical client requests, not internal routing attempts -- a chain that
# fails once before succeeding must still consume exactly one unit overall,
# not one per hop. _release_stale_hop_reservations refunds a stale entry
# here the same way it releases a stale concurrency reservation, since its
# own invariant (a queued entry still present when a new hop's admission
# runs can only belong to an earlier hop of this same request that already
# failed) holds identically for either unit. Unlike concurrency, a
# successful (or chain-final-failing) hop's own entry here is deliberately
# never refunded -- exactly one unit must survive per logical request -- so
# async_log_success_event/async_log_failure_event must leave this field
# completely untouched: litellm's has_logged_async_failure dedup lets the
# *first* failing hop's own failure event through (not only a chain's final
# failure), so popping this field there -- even just to discard it -- would
# strand the very entry the *next* hop's admission is relying on being able
# to refund. There is no final-hop/cache-mirror problem to solve for this
# field either: the one hop that never gets superseded is exactly the one
# whose charge should stick, with nothing left to clean up.
_PENDING_REQUEST_INCREMENTS_FIELD: Final[str] = "_model_based_tag_rate_limits_pending_request_increments"

# Mirrors the latest hop's own queued reservation in the same external cache
# the reservations themselves live in, keyed by (litellm_call_id, key_hash),
# for the one release path that cannot reach model_call_details at all:
# proxy/utils.py's post_call_failure_hook deliberately pops litellm_logging_obj
# off request_data before invoking any callback's async_post_call_failure_hook
# ("Remove before callbacks iterate — not serialisable"), so a fallback
# chain's own final, chain-exhausting failure -- which only this hook fires
# for, since litellm's has_logged_async_failure dedup blocks
# async_log_failure_event for every hop after the first -- has no
# model_call_details to pop a reservation from.
#
# Neither a ContextVar nor the flat request_kwargs dict works here (both
# confirmed live, not just reasoned about): a ContextVar's value only
# propagates into descendant tasks, and Router's own per-hop/per-attempt
# execution does not keep the task that later calls post_call_failure_hook
# a descendant of the task that ran the final hop's own admission, so a
# value set there is invisible by the time this fires. request_kwargs is a
# distinct object every hop (confirmed via id()), is a third, unrelated
# object again by the time post_call_failure_hook runs, and mutating it
# directly leaks the mutated key into the actual provider call as an
# `extra_body` param, since litellm forwards unrecognized kwargs verbatim.
# litellm_call_id is the one identifier that is stable across every one of
# those objects, so an external cache keyed by it -- the same Redis/
# in-memory store the reservations themselves already live in -- is the
# only channel that survives all three failure modes at once.
#
# litellm_call_id alone is not enough to key this cache: it comes from the
# caller-controlled x-litellm-call-id header, so two unrelated requests that
# choose the identical id would overwrite each other's mirror entry, letting
# one caller's terminal failure release a completely different caller's
# still-live reservation. Folding in key_hash -- the calling virtual key's
# hash, resolved server-side (UserAPIKeyAuth.api_key in
# async_post_call_failure_hook, metadata["user_api_key"] everywhere else,
# both authenticated before this hook ever runs) -- confines a collision to
# one caller reusing its own call_id across two of its own concurrent
# requests. This is not confined to "weakens only that caller's own cap":
# for a chain-wide (non scope_by_key_hash) tag, the released reservation is
# on a bucket that tag value's other callers share too, so the forging
# caller's own terminal failure can release a slot on a bucket a different
# caller is also drawing from. Closing this fully needs a per-admission
# identifier that is both server-generated (unlike litellm_call_id) and
# survives proxy/utils.py's post_call_failure_hook stripping
# litellm_logging_obj (unlike everything this comment already ruled out
# above) -- tracked as a known follow-up rather than attempted here.
_PENDING_RESERVATIONS_CACHE_KEY_PREFIX: Final = "model_based_tag_rate_limits:pending_reservations:"


def _pending_reservations_cache_key(call_id: str, key_hash: str | None) -> str:
    # call_id is caller-controlled (the x-litellm-call-id header) with no
    # length bound -- same unbounded-cache-key concern _fixed_length_identity
    # documents for tag values, reused here rather than duplicated.
    return f"{_PENDING_RESERVATIONS_CACHE_KEY_PREFIX}{_fixed_length_identity(call_id)}:{key_hash or ''}"


def _encode_reservations(reservations: Sequence[tuple[str, "_PartitionKey"]]) -> str:
    return json.dumps(
        tuple(
            (key, partition_key if partition_key is None else tuple(partition_key))
            for key, partition_key in reservations
        )
    )


def _as_decoded_list(raw: object) -> Sequence[object] | None:
    # InMemoryCache.get_cache always attempts json.loads on read regardless
    # of what was stored (see its own implementation), so a value written as
    # our own already-JSON-encoded string comes back pre-decoded into a list
    # when served from the in-memory layer; only a real Redis round trip
    # hands back the raw string that still needs decoding here.
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        decoded: Final = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, list) else None


def _decode_reservations(raw: object) -> tuple[tuple[str, "_PartitionKey"], ...]:
    decoded: Final = _as_decoded_list(raw)
    if decoded is None:
        return ()
    entries: Final = []  # mutable-ok: accumulator over an externally-decoded, untrusted-shape list; immediately frozen below
    for item in decoded:
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[0], str)):
            continue
        partition_key_raw = item[1]
        partition_key: _PartitionKey = tuple(partition_key_raw) if isinstance(partition_key_raw, list) else None  # pyright: ignore[reportGeneralTypeIssues]  # decoded from our own _encode_reservations output; shape validated above
        entries.append((item[0], partition_key))  # mutable-ok: see comment above
    return tuple(entries)


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

# The routing-group membership (candidate_model_names) admission actually
# resolved against, stashed the same way _ADMISSION_TIME_FIELD is. resolve_any
# dedupes divergent per-deployment entries by picking the alphabetically first
# member model_name sharing a signature (resolved_group) -- a pure function of
# this exact candidate set. Router's live routing-group membership can change
# between admission and success (a deployment added or removed mid-request via
# /model/new or a config hot-reload), and success independently re-deriving
# candidate_model_names from *live* membership at that later point can pick a
# different resolved_group than admission did, hashing to a different Redis
# key -- so success accounting silently misses the bucket admission actually
# checked, letting real usage escape the enforced cap. Reusing admission's own
# snapshot keeps resolve_any's output identical at both points regardless of
# what changed in between.
_ROUTING_GROUP_CANDIDATES_FIELD: Final[str] = "_model_based_tag_rate_limits_routing_group_candidates"


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
    # A breach of a deployment-scoped check (this function's own
    # `deployment_scope is not None` branch, and every one of its callers'
    # `_raise_over_limit`) deliberately rejects the whole routing attempt for
    # this hop, not just the deployment(s) that own it -- it does not filter
    # them out of `healthy_deployments` and let a sibling in the same group
    # serve instead. An earlier design considered filter-and-retry-sibling
    # semantics (matching how native tag routing filters candidates rather
    # than rejecting the hop) and deliberately did not adopt it: rejecting
    # the whole hop is simpler, and avoids a caller silently succeeding
    # against a deployment whose limit configuration they didn't intend to
    # satisfy.
    if configured_limit.deployment_scope is not None and not (
        present_deployment_ids & frozenset(configured_limit.deployment_scope)
    ):
        return None
    tag_value: Final = _extract_identity(tags, configured_limit.entry.tag_id)
    if tag_value is None:
        return None
    if not _entry_applies(configured_limit.entry, tags, key_alias, model):
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
    if not _entry_applies(configured_limit.entry, tags, key_alias, model_group):
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


def _queue_pending_reservations(
    request_kwargs: Mapping[str, object], field: str, reservations: Sequence[tuple[str, _PartitionKey]]
) -> None:
    """Stash reservations on the request's own `model_call_details`, under
    `field` -- see `_PENDING_CONCURRENCY_KEYS_FIELD`'s docstring for why this,
    not a ContextVar or `litellm_call_id`. Silently a no-op without a real
    logging object (defensive only; every real request has one): a queued
    concurrency reservation still self-heals via
    `_CONCURRENCY_MIN_SAFETY_TTL_SECONDS`, just later.
    """
    logging_obj: Final = request_kwargs.get("litellm_logging_obj")
    model_call_details: Final = getattr(logging_obj, "model_call_details", None)
    if not isinstance(model_call_details, dict):
        return
    pending = model_call_details.get(field)  # rebind-ok: lazily initialized below when absent
    if pending is None:
        pending = []  # mutable-ok: shared, request-scoped accumulator; see field's own docstring  # rebind-ok: lazily initialized only when absent
        model_call_details[field] = pending
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


def _record_routing_group_candidates(
    request_kwargs: Mapping[str, object], candidate_model_names: tuple[str, ...]
) -> None:
    """Stash the routing-group membership admission resolved against -- see
    `_ROUTING_GROUP_CANDIDATES_FIELD`'s docstring for why. Silently a no-op
    without a real logging object (defensive only; every real request has
    one): success accounting falls back to its own live reconstruction, same
    as before this fix existed."""
    logging_obj: Final = request_kwargs.get("litellm_logging_obj")
    model_call_details: Final = getattr(logging_obj, "model_call_details", None)
    if isinstance(model_call_details, dict):
        model_call_details[_ROUTING_GROUP_CANDIDATES_FIELD] = candidate_model_names


def _routing_group_candidates_or(kwargs: Mapping[str, object], fallback: tuple[str, ...]) -> tuple[str, ...]:
    recorded: Final = kwargs.get(_ROUTING_GROUP_CANDIDATES_FIELD)
    return recorded if isinstance(recorded, tuple) else fallback


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
        self, cache: InternalUsageCache, key: str, limit: float, increment: float, ttl: int, refresh_ttl: bool
    ) -> tuple[bool, float]:
        """Single-key atomic check-and-increment. Always one key per Lua
        call -- see TAG_RL_CHECK_AND_INCR_SCRIPT's module docstring for why,
        and for why `refresh_ttl` must be True for a concurrency key and
        False for a requests key."""
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
        for index, (cache, key, limit, increment, ttl, refresh_ttl) in enumerate(checks):
            admitted = False
            try:
                admitted, value = await self._check_and_increment_one(cache, key, limit, increment, ttl, refresh_ttl)
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
        self, checks: Sequence[tuple[InternalUsageCache, str, float, float, int, bool]], up_to_index: int
    ) -> None:
        for refund_index in range(up_to_index):
            refund_cache, refund_key, _limit, refund_increment, _ttl, _refresh_ttl = checks[refund_index]
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
        stale_request_keys: Final = await self._release_stale_hop_reservations(resolved_request_kwargs)
        metadata_variable_name: Final = get_metadata_variable_name_from_kwargs(resolved_request_kwargs)
        team_id: Final = _extract_team_id(resolved_request_kwargs, metadata_variable_name)
        # Built from the full routing-group membership, not `healthy_deployments`
        # (Router's own cooldown-filtered list for this hop): a member that's
        # merely cooled down right now is still a real member of the group for
        # the purpose of deciding resolved_group, and success accounting has no
        # way to know which members were healthy at admission time -- it can
        # only reconstruct the full, static membership (see its own comment
        # below). Deriving both sides from the same full-membership source
        # handles cooldown-state drift between the two points in time; actual
        # membership drift (a deployment added or removed mid-request) still
        # needs admission's own snapshot stashed and reused -- see
        # _ROUTING_GROUP_CANDIDATES_FIELD's docstring.
        routing_group_deployments: Final = self.llm_router._get_routing_group_deployments(  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching resolve_any's own reliance on this method
            model=model, team_id=team_id
        )
        candidate_model_names: Final = (
            tuple(dep["model_name"] for dep in routing_group_deployments)
            if routing_group_deployments is not None
            else tuple(name for d in healthy_deployments if isinstance(name := d.get("model_name"), str))
        )
        _record_routing_group_candidates(resolved_request_kwargs, candidate_model_names)
        configured: Final = self._index.get(self.llm_router).resolve_any(model, team_id, candidate_model_names)
        if not configured:
            return healthy_deployments

        tags: Final = _order_tags_for_identity_resolution(
            _get_tags_from_request_kwargs(resolved_request_kwargs, metadata_variable_name=metadata_variable_name),
            resolved_request_kwargs,
            metadata_variable_name,
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
                        # A "requests" key matching one already charged by a
                        # superseded earlier hop of this same request (see
                        # _release_stale_hop_reservations) renews that same
                        # charge at zero net cost instead of adding a second
                        # unit on top of it -- folded into this same
                        # all-or-nothing batch so a hop that goes on to fail
                        # a *different* check here never commits a refund
                        # with nothing to replace it.
                        0.0 if configured_limit.unit == "requests" and key in stale_request_keys else 1.0,
                        self._ttl_for(configured_limit),
                        configured_limit.unit == "concurrency",
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
                _queue_pending_reservations(
                    resolved_request_kwargs, _PENDING_CONCURRENCY_KEYS_FIELD, concurrency_reservations
                )
                await self._mirror_pending_reservations(
                    resolved_request_kwargs.get("litellm_call_id"),
                    _extract_key_hash(resolved_request_kwargs, metadata_variable_name),
                    concurrency_reservations,
                )

            # Only genuinely new keys, never one already in stale_request_keys:
            # that key's own check just renewed at zero net cost above and is
            # still sitting in the field (see _release_stale_hop_reservations'
            # own comment on why this is a peek, not a pop) -- appending it
            # again here would grow the list with a duplicate entry on every
            # hop of a long retry chain without changing what it means.
            request_increments: Final = tuple(
                (key, _partition_key(configured_limit.entry))
                for configured_limit, _tag_value, key in atomic_checks
                if configured_limit.unit == "requests" and key not in stale_request_keys
            )
            if request_increments:
                _queue_pending_reservations(
                    resolved_request_kwargs, _PENDING_REQUEST_INCREMENTS_FIELD, request_increments
                )

        return healthy_deployments

    async def _mirror_pending_reservations(
        self, call_id: object, key_hash: str | None, reservations: Sequence[tuple[str, "_PartitionKey"]]
    ) -> None:
        if not isinstance(call_id, str):
            return
        try:
            await self.internal_usage_cache.async_set_cache(
                key=_pending_reservations_cache_key(call_id, key_hash),
                value=_encode_reservations(reservations),
                ttl=_CONCURRENCY_MIN_SAFETY_TTL_SECONDS,
                litellm_parent_otel_span=None,
            )
        except Exception as e:  # noqa: BLE001 - a failed mirror write must never block admission; the reservation still self-heals via its own TTL
            verbose_proxy_logger.warning(
                "model_based_tag_rate_limits_hook: failed to mirror pending reservations for call_id=%s: %s", call_id, e
            )

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

    async def _release_stale_hop_reservations(self, request_kwargs: Mapping[str, object]) -> frozenset[str]:
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
        whose own failure exhausts the retry chain -- async_post_call_failure_hook
        closes that residual case instead, via the cache mirror
        `_PENDING_RESERVATIONS_CACHE_KEY_PREFIX` documents.

        A "requests" atomic increment queued under
        `_PENDING_REQUEST_INCREMENTS_FIELD` is never refunded here, even
        though the identical staleness invariant holds for it too: an
        unconditional refund followed by this hop's own admission is not one
        atomic operation, so a hop that goes on to fail a *different* check
        (a read-only limit, or another entry in the same atomic batch) would
        leave the refund committed with nothing to replace it, undercounting
        a logical request that genuinely made an earlier, real attempt. The
        returned keys let `async_filter_deployments` fold the swap into its
        own atomic batch instead -- see its own comment for how.

        Deliberately a peek, not a pop, for that same field: an earlier
        version popped it here and only re-queued on a fully successful
        atomic batch, so a hop that failed *before* reaching that point (a
        read-only check, or a different entry in its own batch) silently
        dropped the bookkeeping -- the real counter was untouched (0.0
        renewals roll back to a no-op), but the *next* hop's own peek would
        come back empty, no longer recognize the key as already charged, and
        charge a fresh unit on top of the one still sitting in the real
        counter. Peeking leaves the field exactly as it was for whichever
        hop reads it next, regardless of how many hops in between fail
        before ever reaching their own successful queuing step.
        """
        logging_obj: Final = request_kwargs.get("litellm_logging_obj")
        model_call_details: Final = getattr(logging_obj, "model_call_details", None)
        if not isinstance(model_call_details, dict):
            return frozenset()
        release_keys: Final = await self._pop_pending_concurrency_keys(model_call_details)
        if release_keys:
            await self._release_keys(release_keys)
        pending_request_increments: Final = model_call_details.get(_PENDING_REQUEST_INCREMENTS_FIELD)
        if not isinstance(pending_request_increments, list):
            return frozenset()
        return frozenset(key for key, _partition_key in pending_request_increments)

    async def _pop_pending_concurrency_keys(
        self, kwargs: Mapping[str, object]
    ) -> tuple[tuple[str, _PartitionKey], ...]:
        # Every caller of this method is itself a normal release path, so
        # also clear the async_post_call_failure_hook cache mirror for the
        # same call_id right here: whatever this pop is about to release
        # must never be found there later and double-released.
        call_id: Final = kwargs.get("litellm_call_id")
        if isinstance(call_id, str):
            # Not `get_metadata_variable_name_from_kwargs` (naive key-presence
            # check): at this point `kwargs` is `model_call_details`, which
            # carries `litellm_metadata` present-but-`None` alongside the
            # real, populated `metadata` for a standard request -- see
            # `_resolve_success_event_metadata_variable_name`'s own docstring.
            litellm_params_raw: Final = kwargs.get("litellm_params")
            litellm_params_for_metadata: Final = (
                litellm_params_raw if isinstance(litellm_params_raw, Mapping) else kwargs
            )
            metadata_variable_name: Final = _resolve_success_event_metadata_variable_name(litellm_params_for_metadata)
            key_hash: Final = _extract_key_hash(litellm_params_for_metadata, metadata_variable_name)
            try:
                await self.internal_usage_cache.dual_cache.async_delete_cache(
                    _pending_reservations_cache_key(call_id, key_hash)
                )
            except Exception as e:  # noqa: BLE001 - a failed mirror clear must never block the real release below
                verbose_proxy_logger.warning(
                    "model_based_tag_rate_limits_hook: failed to clear mirrored reservations for call_id=%s: %s",
                    call_id,
                    e,
                )
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
        release_keys: Final = await self._pop_pending_concurrency_keys(model_call_details)
        if release_keys:
            await self._release_keys(release_keys)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,  # mutable-ok: must match CustomLogger.async_post_call_failure_hook's own base signature exactly
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        """
        litellm's Logging object sets has_logged_async_failure=True after
        the first hop of a fallback chain fails, which blocks
        async_log_failure_event for every later hop (see
        fallback_event_handlers.py's own docstring) -- so a chain's own
        final, chain-exhausting failure never reaches that callback at all,
        and _release_stale_hop_reservations only cleans up a stale
        reservation when a *next* hop's admission runs, which never happens
        after the last one. This hook fires exactly once per proxy request,
        at the point the proxy gives up and returns an error to the caller,
        regardless of how many hops ran or whether the completion-level
        callback was suppressed for this one.

        Reads the cache mirror written by `_mirror_pending_reservations`, not
        `model_call_details`: proxy/utils.py's post_call_failure_hook pops
        `litellm_logging_obj` off `request_data` before invoking any callback
        here ("Remove before callbacks iterate — not serialisable"), and
        neither a ContextVar nor `request_data` itself survives to this
        point either (see `_PENDING_RESERVATIONS_CACHE_KEY_PREFIX`'s own
        docstring for why, confirmed live for each).

        Keyed by `user_api_key_dict.api_key`, not a value read out of
        `request_data`: the proxy's own auth middleware establishes
        `user_api_key_dict` before any hook runs, so it can't be forged the
        way `request_data["litellm_call_id"]` (the `x-litellm-call-id`
        header) can -- see `_PENDING_RESERVATIONS_CACHE_KEY_PREFIX`'s
        docstring for what a caller-forgeable-only key would let a caller do.
        """
        call_id: Final = request_data.get("litellm_call_id")
        if not isinstance(call_id, str):
            return
        cache_key: Final = _pending_reservations_cache_key(call_id, user_api_key_dict.api_key)
        try:
            raw: Final = await self.internal_usage_cache.async_get_cache(key=cache_key, litellm_parent_otel_span=None)
        except Exception as e:  # noqa: BLE001 - a failed mirror read must never raise into the caller's request path
            verbose_proxy_logger.warning(
                "model_based_tag_rate_limits_hook: failed to read mirrored reservations for call_id=%s: %s", call_id, e
            )
            return
        release_keys: Final = _decode_reservations(raw)
        if not release_keys:
            return
        try:
            await self.internal_usage_cache.dual_cache.async_delete_cache(cache_key)
        except Exception as e:  # noqa: BLE001 - a failed mirror clear must never block the real release below
            verbose_proxy_logger.warning(
                "model_based_tag_rate_limits_hook: failed to clear mirrored reservations for call_id=%s: %s", call_id, e
            )
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
        release_keys: Final = await self._pop_pending_concurrency_keys(kwargs)
        if release_keys:
            await self._release_keys(release_keys)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        release_keys: Final = await self._pop_pending_concurrency_keys(kwargs)
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
        #
        # Reconstructing live membership here is itself only a fallback: it
        # can still disagree with admission's own candidate set if the
        # routing group's actual membership changed between the two points
        # in time (not just cooldown/health state) -- _routing_group_candidates_or
        # below prefers admission's own stashed snapshot whenever one exists.
        # See _ROUTING_GROUP_CANDIDATES_FIELD's docstring.
        deployment_id: Final = standard_logging_object.get("model_id")
        serving_deployment: Final = (
            self.llm_router.get_deployment(deployment_id) if isinstance(deployment_id, str) else None
        )
        routing_group_deployments: Final = self.llm_router._get_routing_group_deployments(  # pyright: ignore[reportPrivateUsage]  # reused across module boundaries, matching resolve_any's own reliance on this method
            model=model_group, team_id=team_id
        )
        live_candidate_model_names: Final = (
            tuple(dep["model_name"] for dep in routing_group_deployments)
            if routing_group_deployments is not None
            else ((serving_deployment.model_name,) if serving_deployment is not None else ())
        )
        candidate_model_names: Final = _routing_group_candidates_or(kwargs, fallback=live_candidate_model_names)
        configured: Final = self._index.get(self.llm_router).resolve_any(model_group, team_id, candidate_model_names)
        if not configured:
            return

        tags: Final = _order_tags_for_identity_resolution(
            _get_tags_from_request_kwargs(kwargs, metadata_variable_name=metadata_variable_name),
            kwargs,
            metadata_variable_name,
        )
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
                    pipeline_operations=group_operations,
                    parent_otel_span=parent_otel_span,
                )
            )
            _BACKGROUND_TASKS.add(accounting_task)  # mutable-ok: see _BACKGROUND_TASKS's own docstring
            accounting_task.add_done_callback(_BACKGROUND_TASKS.discard)
