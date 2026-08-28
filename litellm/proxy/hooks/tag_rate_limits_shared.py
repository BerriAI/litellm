"""
Primitives shared by both tag-scoped rate-limit hooks:
`model_based_tag_rate_limits_hook.py` (per-deployment limits nested under
`model_info.tag_rate_limits`, admitted once per routing hop) and
`global_tag_rate_limits_hook.py` (model-independent limits declared once
under `litellm_settings.global_tag_rate_limits`, admitted once per request).

Both hooks enforce the identical `TagRateLimitEntry` shape and need the same
identity/scope extraction, policy fingerprinting, bucket-key hashing, and
cache-partitioning primitives, so those live here rather than in either
hook's own module -- keeping neither hook reaching into the other's private
internals to reuse them. Every name here is this module's own public
interface (no leading underscore): each hook imports what it needs aliased
back to its own historical, underscore-prefixed local name (e.g.
`entry_applies as _entry_applies`), so this is a genuine export, not a
private symbol either hook reaches across a module boundary to grab.
"""

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

# Units whose admission must be atomic (check-and-increment in one Redis
# round trip) because the increment amount is known upfront (always 1).
# tokens/dollars can't be: real usage is only known after the response, so
# they stay a read-then-account-on-success check with a documented,
# unavoidable admit-vs-account race.
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

# Shared read-only fallback for an absent/None mapping (request_kwargs,
# metadata, model_info, ...): avoids constructing a fresh mutable `{}` at
# every one of these call sites just to immediately call `.get()` on it.
EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})

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
# fix, shared by every fire-and-forget task either hook creates.
BACKGROUND_TASKS: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: see comment above

# Floor for a concurrency reservation's self-heal TTL, regardless of the
# configured period_seconds. A reservation that expires while its request is
# still genuinely in flight silently admits requests past the limit; this
# generous floor keeps that window far larger than any realistic request
# duration, at the cost of a leaked (crashed-worker) slot self-healing more
# slowly. period_seconds can still raise the TTL further, never lower it.
CONCURRENCY_MIN_SAFETY_TTL_SECONDS: Final = 3600

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

# A (tag_id, values) pair mirroring TagRateLimitScope's own fields, used to
# fold `enabled_for`/`disabled_for` into a hashable form (a policy
# fingerprint, or a per-deployment dedup signature) without depending on
# TagRateLimitScope's own hashability.
ScopeSignature: TypeAlias = tuple[str, tuple[str, ...]] | None


def scope_signature(scope: TagRateLimitScope | None) -> ScopeSignature:
    """Normalizes a `TagRateLimitScope` into a plain, hashable tuple, so
    `enabled_for`/`disabled_for` can be folded into a policy fingerprint (or
    a per-deployment dedup signature) -- two entries disagreeing on either
    field must be treated as genuinely different policies rather than
    merged into one bucket."""
    return None if scope is None else (scope.tag_id, scope.values)


def extract_identity(tags: Sequence[str], tag_id: str) -> str | None:
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


def entry_applies(entry: TagRateLimitEntry, tags: Sequence[str], key_alias: str | None, model: str | None) -> bool:
    """
    Applies `entry`'s own scoping fields (`enabled_for`/`disabled_for`/
    `apply_to_key_alias`/`apply_to_models`), evaluated in this order -- deny
    overrides allow, checked before any allowlist:

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
      3. `apply_to_models`: `model` is absent, or present but not in the
         list -> doesn't apply. Same allowlist semantics as `enabled_for` --
         a request with no `model` never satisfies this gate.
      4. `apply_to_key_alias`: the calling key's own alias is absent, or
         present but not in the list -> doesn't apply. Same allowlist
         semantics as `enabled_for` -- a key with no alias set never
         satisfies this gate.

    An entry with none of these fields set always applies -- this is the
    unscoped behavior every existing entry has today, unchanged.
    """
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


def resolve_success_event_metadata_variable_name(
    litellm_params_for_metadata: Mapping[str, object],
) -> Literal["metadata", "litellm_metadata"]:
    """`get_metadata_variable_name_from_kwargs` only checks key presence, which
    misresolves at `async_log_success_event` time: `kwargs["litellm_params"]`
    always carries a `litellm_metadata` key (typically `None`) alongside the
    real, populated `metadata` dict for a standard (non
    LITELLM_METADATA_ROUTES) request, so the key-presence check always picks
    `litellm_metadata` there and silently reads no tags/identity at all.

    A plain truthiness check on `litellm_metadata` isn't enough either: a
    caller can populate it with unrelated, non-empty content on a route where
    `metadata` is the field the proxy actually wrote identity into, and
    truthiness alone would still misresolve to the caller-controlled bucket.
    `add_user_api_key_auth_to_request_metadata` (litellm_pre_call_utils.py)
    unconditionally stamps a `user_api_key_auth` key into whichever bucket it
    resolved as authoritative, overwriting anything a caller pre-populated
    there -- so requiring that marker's presence, not mere truthiness, only
    ever prefers `litellm_metadata` when it is genuinely the field the proxy
    wrote identity/tags into."""
    litellm_metadata: Final = litellm_params_for_metadata.get("litellm_metadata")
    if isinstance(litellm_metadata, Mapping) and "user_api_key_auth" in litellm_metadata:
        return "litellm_metadata"
    return "metadata"


def extract_key_hash(request_kwargs: Mapping[str, object], metadata_variable_name: str) -> str | None:
    """Same single-authoritative-field lookup as
    `model_based_tag_rate_limits_hook._extract_team_id`, but for the calling
    virtual key's hash: `LiteLLMProxyRequestSetup` sets
    `metadata["user_api_key"]` to `user_api_key_dict.api_key`, which despite
    the plain name is already the hashed token (see `litellm_pre_call_utils.py`).
    """
    active: Final = request_kwargs.get(metadata_variable_name) or EMPTY_MAPPING
    key_hash: Final = active.get("user_api_key")
    return key_hash if isinstance(key_hash, str) else None


def extract_key_alias(request_kwargs: Mapping[str, object], metadata_variable_name: str) -> str | None:
    """Same single-authoritative-field lookup as
    `model_based_tag_rate_limits_hook._extract_team_id`, but for the calling
    virtual key's own `key_alias`: `LiteLLMProxyRequestSetup` sets
    `metadata["user_api_key_alias"]` to `user_api_key_dict.key_alias`
    (see `litellm_pre_call_utils.py`)."""
    active: Final = request_kwargs.get(metadata_variable_name) or EMPTY_MAPPING
    key_alias: Final = active.get("user_api_key_alias")
    return key_alias if isinstance(key_alias, str) else None


def order_tags_for_identity_resolution(
    tags: Sequence[str], request_kwargs: Mapping[str, object], metadata_variable_name: str
) -> tuple[str, ...]:
    """`extract_identity`/`entry_applies` both resolve a `tag_id` via
    first-match-by-prefix. `_merge_tags` (litellm_pre_call_utils.py) appends
    key/team/project tags only if not already present, keeping caller-supplied
    tags first in the merged `tags` list -- so an authenticated caller could
    submit e.g. `company_id:attacker-chosen` ahead of the calling key's real
    `company_id:real-company` tag and have every entry scoped to `company_id`
    resolve to the caller's own value instead of the key's. `metadata.inherited_tags`
    is a separate, server-computed snapshot of only the tags the calling
    key/team/project's own config contributed (see that field's docstring in
    litellm_pre_call_utils.py), so putting it first makes a policy-backed tag
    win over a same-prefix caller-supplied one.
    """
    active: Final = request_kwargs.get(metadata_variable_name) or EMPTY_MAPPING
    inherited_tags: Final = active.get("inherited_tags") if isinstance(active, Mapping) else None
    if not isinstance(inherited_tags, (list, tuple)) or not inherited_tags:
        return tuple(tags)
    return tuple(dict.fromkeys((*inherited_tags, *tags)))


def fixed_length_identity(tag_value: str) -> str:
    """
    `tag_value` is caller-controlled (whatever follows the tag_id prefix in
    a caller-supplied tag) with no length or content bound. Embedding it
    directly would let a caller inflate a hook's own in-memory dict keys
    past what `max_in_memory_cache_size` bounds (that caps item *count*, not
    key bytes) and grow unbounded Redis keys with no cap at all. Hashing to
    a fixed-length digest bounds a hook's own contribution to key size
    regardless of the caller's input, while still preserving distinctness
    (two different tag values still resolve to two different buckets).
    """
    return hashlib.sha256(tag_value.encode()).hexdigest()


def policy_fingerprint(entry: TagRateLimitEntry) -> str:
    """
    Two entries can share a `name` and `tag_id` while genuinely disagreeing
    on `limit`, `period_seconds`, `scope_by_key_hash`, or any of the scoping
    fields -- both hooks already treat that as two distinct policies
    elsewhere (model_based_tag_rate_limits_hook.py's own per-deployment
    dedup signature is one example), so the Redis/in-memory bucket key must
    too, or two differently-configured entries that happen to share a name
    check and charge the identical counter. `scope_by_key_hash` specifically
    needs its own slot here rather than relying on a hook's own `_hash_tag`
    `key_hash`-derived suffix to carry it: that suffix is empty whenever
    `key_hash` resolves to `None` (no virtual key on the call), which would
    otherwise collide an unscoped entry with a key-hash-scoped one that
    agrees on every other field. Hashed to a fixed-length digest for the
    same reason `fixed_length_identity` hashes `tag_value`: an operator's
    own `enabled_for`/`disabled_for`/`apply_to_key_alias`/`apply_to_models`
    list has no length bound.
    """
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
    """Redis (and in-memory fallback) TTL for a non-concurrency bucket key.
    `entry.key_ttl_seconds` overrides the default of period_seconds + 3600
    when set -- see TagRateLimitEntry.key_ttl_seconds."""
    return entry.key_ttl_seconds if entry.key_ttl_seconds is not None else entry.period_seconds + 3600


# None => this entry shares the hook's single default cache partition
# (matching every entry's behavior before this override existed). Otherwise
# a value-stable signature -- not the override int alone -- so two different
# entries that happen to choose the identical max_in_memory_cache_size don't
# get merged into one shared partition; the same entry (same config content)
# always resolves to the same signature across config/index rebuilds, which
# is what keeps each hook's own `_partitions` cache from leaking a fresh
# partition every time its configuration is re-resolved.
# The str before the trailing int is `policy_fingerprint(entry)`: two
# entries can share tag_id/name while genuinely disagreeing on
# limit/period_seconds/scope_by_key_hash/enabled_for/disabled_for/
# apply_to_key_alias/apply_to_models -- both hooks already treat that as two
# distinct policies for bucket-key purposes, so a shared
# max_in_memory_cache_size must not route them onto the same partition
# either, or one entry's high-cardinality traffic can evict the other's
# active counters from a cache neither entry asked to share.
# max_in_memory_cache_size stays the trailing element: `partition_key[-1]`
# reads it directly to size the partition's cache.
PartitionKey: TypeAlias = tuple[str, str, str, int] | None
# Grouping type for async_log_success_event's per-partition tokens/dollars
# pipeline dispatch -- named only so the declaration fits on one line; see
# that method for why the grouping is needed.
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
