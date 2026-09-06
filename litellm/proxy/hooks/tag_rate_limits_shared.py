"""Primitives shared by both tag-scoped rate-limit hooks (model_based_tag_rate_limits_hook.py,
global_tag_rate_limits_hook.py): identity/scope extraction, policy fingerprinting, bucket-key
hashing, and cache-partitioning. Every name here is a genuine public export (no leading
underscore); each hook imports what it needs aliased back to its own historical private name."""

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from litellm.exceptions import RateLimitType
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.router import TagRateLimitEntry, TagRateLimitScope

LimitUnit: TypeAlias = Literal["tokens", "requests", "dollars", "concurrency"]
LIMIT_UNITS: Final[tuple[LimitUnit, ...]] = ("tokens", "requests", "dollars", "concurrency")

# requests/concurrency admit via atomic check-and-increment; tokens/dollars are only known
# after the response, so they stay a read-then-account-on-success check with an unavoidable race.
ATOMIC_UNITS: Final[frozenset[LimitUnit]] = frozenset({"requests", "concurrency"})

UNIT_TO_GROUP_FIELD: Final[Mapping[LimitUnit, str]] = MappingProxyType(
    {
        "tokens": "token_limits",
        "requests": "request_limits",
        "dollars": "dollar_limits",
        "concurrency": "concurrency_limits",
    }
)
UNIT_TO_RATE_LIMIT_TYPE: Final[Mapping[LimitUnit, RateLimitType]] = MappingProxyType(
    {
        "tokens": RateLimitType.TOKENS,
        "requests": RateLimitType.REQUESTS,
        "dollars": RateLimitType.BUDGET,
        "concurrency": RateLimitType.CONCURRENT_REQUESTS,
    }
)

# shared read-only fallback for an absent/None mapping, so call sites don't build a fresh {}
EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})

# holds a strong reference to every fire-and-forget background task (concurrency release,
# token/dollar accounting) so the event loop's weak-ref-only tracking can't garbage-collect
# one mid-execution; each task's own completion callback discards its entry
BACKGROUND_TASKS: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: see comment above

# floor for a concurrency reservation's self-heal ttl, regardless of period_seconds, so an
# expiring-while-in-flight reservation can't silently admit past the limit
CONCURRENCY_MIN_SAFETY_TTL_SECONDS: Final = 3600

# single-key atomic check-and-increment (one key per call: every tag_rl key carries its own
# {..} hash tag, so a multi-key call could span shards and cross-slot error). refresh_ttl
# (ARGV[4]) distinguishes callers: "requests" is a fixed window whose ttl is set once and never
# extended; "concurrency" isn't windowed, so its crash-safety ttl must refresh on every admission.
TAG_RL_CHECK_AND_INCR_SCRIPT: Final = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local increment = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local refresh_ttl = tonumber(ARGV[4])
local current = tonumber(redis.call('GET', key) or 0)
if current + increment > limit then
    return { 0, current }
end
local new_value = redis.call('INCRBY', key, increment)
if ttl > 0 then
    if refresh_ttl == 1 then
        redis.call('EXPIRE', key, ttl)
    else
        local current_ttl = redis.call('TTL', key)
        if current_ttl == -1 then
            redis.call('EXPIRE', key, ttl)
        end
    end
end
return { 1, new_value }
"""

# atomic decrement floored at 0 (refund or concurrency release); floors via DEL rather than
# `SET key 0`, since a release on an already-expired key would otherwise recreate it with no ttl
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

# a (tag_id, values) pair mirroring TagRateLimitScope, used to fold enabled_for/disabled_for
# into a hashable form without depending on TagRateLimitScope's own hashability
ScopeSignature: TypeAlias = tuple[str, tuple[str, ...]] | None


def scope_signature(scope: TagRateLimitScope | None) -> ScopeSignature:
    """Normalizes a TagRateLimitScope into a hashable tuple for policy fingerprinting/dedup."""
    return None if scope is None else (scope.tag_id, scope.values)


def extract_identity(tags: Sequence[str], tag_id: str) -> str | None:
    """First tag matching `f"{tag_id}:"`, value after the colon. `!`-prefixed tag-routing
    negation markers are skipped so they're never misread as an identity value."""
    prefix: Final = f"{tag_id}:"
    for tag in tags:
        if tag.startswith("!"):
            continue
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def entry_applies(entry: TagRateLimitEntry, tags: Sequence[str], key_alias: str | None, model: str | None) -> bool:
    """Applies entry's own scoping fields in order (deny before allow): disabled_for excludes a
    matching value; enabled_for requires a matching value (absence fails, unlike disabled_for);
    apply_to_models and apply_to_key_alias are allowlists an absent model/alias never satisfies.
    An entry with none of these set always applies."""
    if entry.disabled_for is not None:
        disabled_gate_value: Final = extract_identity(tags, entry.disabled_for.tag_id)
        if disabled_gate_value is not None and disabled_gate_value in entry.disabled_for.values:
            return False
    if entry.enabled_for is not None:
        enabled_gate_value: Final = extract_identity(tags, entry.enabled_for.tag_id)
        if enabled_gate_value is None or enabled_gate_value not in entry.enabled_for.values:
            return False
    if entry.apply_to_models is not None and model not in entry.apply_to_models:
        return False
    if entry.apply_to_key_alias is None:
        return True
    return key_alias in entry.apply_to_key_alias


def resolve_authoritative_metadata_variable_name(
    metadata_source: Mapping[str, object],
) -> Literal["metadata", "litellm_metadata"]:
    """Picks metadata vs litellm_metadata by the unforgeable `user_api_key_auth` marker
    `add_litellm_data_to_request` stamps into whichever bucket is actually authoritative --
    key presence or truthiness alone can be forged by a caller onto the wrong bucket."""
    litellm_metadata: Final = metadata_source.get("litellm_metadata")
    if isinstance(litellm_metadata, Mapping) and "user_api_key_auth" in litellm_metadata:
        return "litellm_metadata"
    return "metadata"


def _active_metadata_bucket(request_kwargs: Mapping[str, object], metadata_variable_name: str) -> Mapping[str, object]:
    """request_kwargs carries metadata at its own top level at admission time, but only nested
    under litellm_params by async_log_success_event/async_log_failure_event time; checks both."""
    top_level: Final = request_kwargs.get(metadata_variable_name)
    if isinstance(top_level, Mapping):
        return top_level
    litellm_params: Final = request_kwargs.get("litellm_params")
    if isinstance(litellm_params, Mapping):
        nested: Final = litellm_params.get(metadata_variable_name)
        if isinstance(nested, Mapping):
            return nested
    return EMPTY_MAPPING


def extract_key_hash(request_kwargs: Mapping[str, object], metadata_variable_name: str) -> str | None:
    """Reads the calling virtual key's hash from the authoritative metadata bucket (nested under
    litellm_params by log time, same as _active_metadata_bucket's other callers); `metadata["user_api_key"]`
    is already the hashed token despite the plain name."""
    active: Final = _active_metadata_bucket(request_kwargs, metadata_variable_name)
    key_hash: Final = active.get("user_api_key")
    return key_hash if isinstance(key_hash, str) else None


def extract_key_alias(request_kwargs: Mapping[str, object], metadata_variable_name: str) -> str | None:
    """Reads the calling virtual key's own key_alias from the authoritative metadata bucket."""
    active: Final = _active_metadata_bucket(request_kwargs, metadata_variable_name)
    key_alias: Final = active.get("user_api_key_alias")
    return key_alias if isinstance(key_alias, str) else None


def order_tags_for_identity_resolution(
    tags: Sequence[str], request_kwargs: Mapping[str, object], metadata_variable_name: str
) -> tuple[str, ...]:
    """Puts server-computed `metadata.inherited_tags` ahead of caller-supplied tags, so a caller
    can't submit e.g. `company_id:attacker-chosen` and shadow the calling key's real, same-prefix
    tag; extract_identity/entry_applies both resolve tag_id via first-match-by-prefix."""
    active: Final = _active_metadata_bucket(request_kwargs, metadata_variable_name)
    inherited_tags: Final = active.get("inherited_tags")
    if not isinstance(inherited_tags, (list, tuple)) or not inherited_tags:
        return tuple(tags)
    return tuple(dict.fromkeys((*inherited_tags, *tags)))


def fixed_length_identity(tag_value: str) -> str:
    """Hashes a caller-controlled, unbounded-length tag value to a fixed-length digest, bounding
    a hook's own contribution to an in-memory cache key or Redis key regardless of input size."""
    return hashlib.sha256(tag_value.encode()).hexdigest()


def policy_fingerprint(entry: TagRateLimitEntry) -> str:
    """Fixed-length digest folding every policy-distinguishing field of an entry (limit,
    period_seconds, scope_by_key_hash, enabled_for/disabled_for, apply_to_key_alias/models) into
    one bucket-key signature, so two differently-configured entries sharing a name never share
    a counter."""
    fingerprint_source: Final = (
        entry.limit,
        entry.period_seconds,
        entry.scope_by_key_hash,
        scope_signature(entry.enabled_for),
        scope_signature(entry.disabled_for),
        entry.apply_to_key_alias,
        entry.apply_to_models,
    )
    return hashlib.sha256(repr(fingerprint_source).encode()).hexdigest()[:16]


def bucket_ttl_seconds(entry: TagRateLimitEntry) -> int:
    """Redis (and in-memory fallback) ttl for a non-concurrency bucket key: entry.key_ttl_seconds
    overrides the default of period_seconds + 3600 when set."""
    return entry.key_ttl_seconds if entry.key_ttl_seconds is not None else entry.period_seconds + 3600


# None => this entry shares the hook's single default cache partition. Otherwise a value-stable
# signature (policy_fingerprint(entry), not the override int alone) so two entries sharing a
# max_in_memory_cache_size don't merge into one partition; max_in_memory_cache_size stays the
# trailing element since `partition_key[-1]` reads it directly to size the partition's cache.
PartitionKey: TypeAlias = tuple[str, str, str, int] | None
PartitionOperations: TypeAlias = dict[PartitionKey, list[RedisPipelineIncrementOperation]]


def partition_key(entry: TagRateLimitEntry) -> PartitionKey:
    if entry.max_in_memory_cache_size is None:
        return None
    return (
        entry.tag_id,
        entry.name,
        policy_fingerprint(entry),
        entry.max_in_memory_cache_size,
    )
