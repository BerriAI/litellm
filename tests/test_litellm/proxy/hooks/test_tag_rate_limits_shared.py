"""
Unit tests for the primitives shared by both tag-scoped rate-limit hooks
(`model_based_tag_rate_limits_hook.py` and `global_tag_rate_limits_hook.py`).

These test the hook-independent logic in isolation: identity extraction,
`entry_applies` scoping, and the partition/bucket-TTL key helpers. Each
hook's own test file covers everything specific to how it wires these
primitives into its own admission/accounting engine.
"""

from litellm.proxy.hooks.tag_rate_limits_shared import (
    bucket_ttl_seconds,
    entry_applies,
    extract_identity,
    extract_key_alias,
    extract_key_hash,
    fixed_length_identity,
    order_tags_for_identity_resolution,
    partition_key,
    resolve_success_event_metadata_variable_name,
)
from litellm.types.router import TagRateLimitEntry, TagRateLimitScope

# ---------------------------------------------------------------------------
# extract_identity
# ---------------------------------------------------------------------------


def test_extract_identity_matches_prefixed_tag():
    assert extract_identity(["team_id:t1", "end_user_id:u1"], "end_user_id") == "u1"


def test_extract_identity_returns_none_when_absent():
    assert extract_identity(["team_id:t1"], "end_user_id") is None


def test_extract_identity_skips_negation_tags():
    """A `!end_user_id:u1` routing-negation marker must never be read as identity."""
    assert extract_identity(["!end_user_id:u1"], "end_user_id") is None


# ---------------------------------------------------------------------------
# order_tags_for_identity_resolution -- veria-ai finding on PR #38292: a
# caller-supplied tag must not shadow a policy-backed (key/team/project)
# tag sharing the same tag_id prefix
# ---------------------------------------------------------------------------


def test_order_tags_for_identity_resolution_prefers_inherited_tag_over_caller_supplied():
    request_kwargs = {"metadata": {"inherited_tags": ["company_id:real-company"]}}
    tags = ["company_id:attacker-chosen", "end_user_id:u1"]
    ordered = order_tags_for_identity_resolution(tags, request_kwargs, "metadata")
    assert extract_identity(ordered, "company_id") == "real-company"


def test_order_tags_for_identity_resolution_falls_back_to_caller_tags_when_nothing_inherited():
    request_kwargs = {"metadata": {}}
    tags = ["end_user_id:u1"]
    ordered = order_tags_for_identity_resolution(tags, request_kwargs, "metadata")
    assert extract_identity(ordered, "end_user_id") == "u1"


def test_order_tags_for_identity_resolution_keeps_caller_only_tags_not_shadowed_by_a_different_tag_id():
    request_kwargs = {"metadata": {"inherited_tags": ["company_id:real-company"]}}
    tags = ["end_user_id:u1"]
    ordered = order_tags_for_identity_resolution(tags, request_kwargs, "metadata")
    assert extract_identity(ordered, "end_user_id") == "u1"
    assert extract_identity(ordered, "company_id") == "real-company"


def test_order_tags_for_identity_resolution_deduplicates_identical_tag_present_in_both_sources():
    request_kwargs = {"metadata": {"inherited_tags": ["company_id:real-company"]}}
    tags = ["company_id:real-company"]
    ordered = order_tags_for_identity_resolution(tags, request_kwargs, "metadata")
    assert ordered.count("company_id:real-company") == 1


# ---------------------------------------------------------------------------
# fixed_length_identity -- tag_value is caller-controlled with no length
# bound; a hook's own contribution to a cache key must not grow with it
# ---------------------------------------------------------------------------


def test_fixed_length_identity_bounds_key_contribution_regardless_of_input_size():
    """
    A caller can submit an arbitrarily long tag value (no length or content
    bound is enforced upstream of either hook). Without hashing, that value
    would go straight into an in-memory dict key (bypassing
    max_in_memory_cache_size, which caps item *count* not key bytes) and an
    unbounded-length Redis key (Redis has no key-count or key-size cap at
    all here). A fixed-length digest bounds a hook's own contribution to
    the key regardless of input size.
    """
    huge_value = "x" * 5_000_000
    digest = fixed_length_identity(huge_value)
    assert len(digest) == 64  # sha256 hex digest length, independent of input size


def test_fixed_length_identity_preserves_distinctness():
    """Hashing must not collapse two different tag values onto one bucket."""
    assert fixed_length_identity("user-a") != fixed_length_identity("user-b")
    assert fixed_length_identity("user-a") == fixed_length_identity("user-a")


# ---------------------------------------------------------------------------
# extract_key_hash -- must read only the one field the server actually
# authenticates into, never fall back to the other
# ---------------------------------------------------------------------------


def test_extract_key_hash_ignores_a_forged_value_in_the_non_authoritative_field():
    """
    On a route where litellm_metadata is authoritative, the server writes
    the real hash there and never touches metadata -- so a caller-supplied
    metadata.user_api_key must not be read at all, let alone win.
    """
    request_kwargs = {
        "metadata": {"user_api_key": "forged-by-caller"},
        "litellm_metadata": {"user_api_key": "real-authenticated-hash"},
    }
    assert extract_key_hash(request_kwargs, "litellm_metadata") == "real-authenticated-hash"


def test_extract_key_hash_reads_metadata_when_it_is_the_authoritative_field():
    request_kwargs = {"metadata": {"user_api_key": "real-hash"}}
    assert extract_key_hash(request_kwargs, "metadata") == "real-hash"


def test_extract_key_hash_ignores_a_non_mapping_authoritative_field():
    """Bugbot finding: metadata can arrive as an unparsed JSON string (see
    apply_client_tag_policy_pre_auth's own docstring on multipart/extra_body
    routes); a truthy non-Mapping must not reach .get() and crash."""
    request_kwargs = {"metadata": '{"user_api_key": "forged"}'}
    assert extract_key_hash(request_kwargs, "metadata") is None


def test_extract_key_alias_ignores_a_non_mapping_authoritative_field():
    request_kwargs = {"metadata": '{"user_api_key_alias": "forged"}'}
    assert extract_key_alias(request_kwargs, "metadata") is None


# ---------------------------------------------------------------------------
# resolve_success_event_metadata_variable_name -- veria-ai finding surfaced on
# PR #38347: a caller-supplied, non-empty litellm_metadata must not be
# selected over the metadata the proxy actually wrote authenticated tags into
# ---------------------------------------------------------------------------


def test_resolve_success_event_metadata_variable_name_ignores_caller_forged_non_empty_litellm_metadata():
    """A caller-supplied litellm_metadata with unrelated content (no
    user_api_key_auth marker) must not be picked over metadata, even though
    it is a non-empty dict."""
    litellm_params_for_metadata = {"litellm_metadata": {"marker": True}}
    assert resolve_success_event_metadata_variable_name(litellm_params_for_metadata) == "metadata"


def test_resolve_success_event_metadata_variable_name_selects_litellm_metadata_when_server_written():
    """A genuinely server-populated litellm_metadata (LITELLM_METADATA_ROUTES)
    always carries the user_api_key_auth marker stamped by
    add_user_api_key_auth_to_request_metadata."""
    litellm_params_for_metadata = {
        "litellm_metadata": {"user_api_key_auth": object(), "tags": ["team_id:t1"]}
    }
    assert resolve_success_event_metadata_variable_name(litellm_params_for_metadata) == "litellm_metadata"


def test_resolve_success_event_metadata_variable_name_defaults_to_metadata_when_litellm_metadata_absent():
    assert resolve_success_event_metadata_variable_name({}) == "metadata"


def test_resolve_success_event_metadata_variable_name_defaults_to_metadata_when_litellm_metadata_none():
    assert resolve_success_event_metadata_variable_name({"litellm_metadata": None}) == "metadata"


def test_resolve_success_event_metadata_variable_name_defaults_to_metadata_when_litellm_metadata_empty():
    assert resolve_success_event_metadata_variable_name({"litellm_metadata": {}}) == "metadata"


# ---------------------------------------------------------------------------
# entry_applies -- enabled_for / disabled_for / apply_to_key_alias / apply_to_models
# ---------------------------------------------------------------------------


def test_entry_applies_with_none_of_the_scoping_fields_set():
    entry = TagRateLimitEntry(name="daily", tag_id="end_user_id", limit=500, period_seconds=86400)
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is True


def test_entry_applies_disabled_for_on_its_own_tag_id_excludes_a_listed_value():
    """disabled_for's `tag_id` can be set to the entry's own tag_id, gating on
    a subset of its own resolved identity rather than a second tag."""
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        disabled_for=TagRateLimitScope(tag_id="end_user_id", values=("u1",)),
    )
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is False
    assert entry_applies(entry, ["end_user_id:u2"], None, None) is True


def test_entry_applies_enabled_for_on_its_own_tag_id_restricts_to_a_listed_value():
    """enabled_for's `tag_id` can likewise be set to the entry's own tag_id,
    admitting only a hand-picked subset of its own resolved identity."""
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        enabled_for=TagRateLimitScope(tag_id="end_user_id", values=("u2", "u3")),
    )
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is False
    assert entry_applies(entry, ["end_user_id:u2"], None, None) is True


def test_entry_applies_matches_an_enabled_for_gate():
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        enabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",)),
    )
    assert entry_applies(entry, ["end_user_id:u1", "company_id:1032"], None, None) is True


def test_entry_applies_skips_when_enabled_for_gate_tag_is_absent():
    """
    enabled_for is an allowlist gate: absence of the gate tag must not
    satisfy it, unlike disabled_for below.
    """
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        enabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",)),
    )
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is False


def test_entry_applies_skips_when_disabled_for_gate_matches():
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        disabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",)),
    )
    assert entry_applies(entry, ["end_user_id:u1", "company_id:1032"], None, None) is False


def test_entry_applies_when_disabled_for_gate_tag_is_absent():
    """disabled_for is a denylist gate: absence of the gate tag has nothing
    to match against, so the entry still applies."""
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        disabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",)),
    )
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is True


def test_entry_applies_disabled_for_overrides_a_matching_enabled_for_gate():
    """Deny (disabled_for) takes effect independently of whether the
    enabled_for gate itself matched, even when both target the same tag."""
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        enabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",)),
        disabled_for=TagRateLimitScope(tag_id="end_user_id", values=("u1",)),
    )
    assert entry_applies(entry, ["end_user_id:u1", "company_id:1032"], None, None) is False


def test_entry_applies_with_apply_to_key_alias_unset_applies_to_every_key():
    entry = TagRateLimitEntry(name="daily", tag_id="end_user_id", limit=500, period_seconds=86400)
    assert entry_applies(entry, ["end_user_id:u1"], "any-key-alias", None) is True
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is True


def test_entry_applies_admits_a_key_alias_on_the_allowlist():
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_key_alias=("team-a-key",)
    )
    assert entry_applies(entry, ["end_user_id:u1"], "team-a-key", None) is True


def test_entry_applies_rejects_a_key_alias_missing_from_the_allowlist():
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_key_alias=("team-a-key",)
    )
    assert entry_applies(entry, ["end_user_id:u1"], "team-b-key", None) is False


def test_entry_applies_rejects_when_key_has_no_alias_but_allowlist_is_set():
    """apply_to_key_alias is an allowlist gate: a key with no alias at all
    never satisfies it, same as enabled_for's absent-gate-tag semantics."""
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_key_alias=("team-a-key",)
    )
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is False


def test_entry_applies_with_apply_to_models_unset_applies_to_every_model():
    entry = TagRateLimitEntry(name="daily", tag_id="end_user_id", limit=500, period_seconds=86400)
    assert entry_applies(entry, ["end_user_id:u1"], None, "opus-chain") is True
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is True


def test_entry_applies_admits_a_model_on_the_apply_to_models_allowlist():
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_models=("opus-chain",)
    )
    assert entry_applies(entry, ["end_user_id:u1"], None, "opus-chain") is True


def test_entry_applies_rejects_a_model_missing_from_the_apply_to_models_allowlist():
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_models=("opus-chain",)
    )
    assert entry_applies(entry, ["end_user_id:u1"], None, "sonnet-chain") is False


def test_entry_applies_rejects_when_model_is_absent_but_apply_to_models_is_set():
    """apply_to_models is an allowlist gate: a request with no model at all
    never satisfies it, same as apply_to_key_alias's absent-key semantics."""
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_models=("opus-chain",)
    )
    assert entry_applies(entry, ["end_user_id:u1"], None, None) is False


def test_entry_applies_apply_to_models_composes_with_apply_to_key_alias():
    """Both gates must pass: a request against the listed model but a
    non-listed key alias must not apply, even though apply_to_models alone
    would have admitted it."""
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        apply_to_models=("opus-chain",),
        apply_to_key_alias=("premium-key",),
    )
    assert entry_applies(entry, ["end_user_id:u1"], "premium-key", "opus-chain") is True
    assert entry_applies(entry, ["end_user_id:u1"], "other-key", "opus-chain") is False
    assert entry_applies(entry, ["end_user_id:u1"], "premium-key", "sonnet-chain") is False


# ---------------------------------------------------------------------------
# partition_key -- entries that share max_in_memory_cache_size but disagree
# on any policy-fingerprinted field must never share a cache partition
# ---------------------------------------------------------------------------


def test_partition_key_distinguishes_entries_that_differ_only_by_scope_by_key_hash():
    """
    scope_by_key_hash is part of the partition-key signature: two entries
    identical in every other field but differing only on this flag are
    different rate limits (different bucket keys per each hook's own
    `_hash_tag`) and must never be routed to the same cache partition.
    """
    unscoped = TagRateLimitEntry(
        name="per_minute", tag_id="end_user_id", limit=5, period_seconds=60, max_in_memory_cache_size=100
    )
    scoped = TagRateLimitEntry(
        name="per_minute",
        tag_id="end_user_id",
        limit=5,
        period_seconds=60,
        scope_by_key_hash=True,
        max_in_memory_cache_size=100,
    )
    assert partition_key(unscoped) != partition_key(scoped)


def test_partition_key_distinguishes_entries_that_differ_only_by_scoping_fields():
    """
    A plain, unscoped entry and a scoped override can legitimately share
    name/tag_id/limit/period_seconds/scope_by_key_hash while disagreeing on
    enabled_for/disabled_for/apply_to_key_alias/apply_to_models --
    policy_fingerprint already treats that as two distinct policies, so a
    shared max_in_memory_cache_size must not route them onto the same
    in-memory partition either, or one entry's high-cardinality traffic can
    evict the other's active counters from a cache neither entry asked to
    share.
    """
    base_kwargs = {
        "name": "daily",
        "tag_id": "end_user_id",
        "limit": 100,
        "period_seconds": 86400,
        "max_in_memory_cache_size": 50,
    }
    unscoped = TagRateLimitEntry(**base_kwargs)
    enabled_for_scoped = TagRateLimitEntry(
        **base_kwargs, enabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",))
    )
    disabled_for_scoped = TagRateLimitEntry(
        **base_kwargs, disabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",))
    )
    alias_scoped = TagRateLimitEntry(**base_kwargs, apply_to_key_alias=("premium-key",))
    models_scoped = TagRateLimitEntry(**base_kwargs, apply_to_models=("opus-chain",))

    keys = {
        partition_key(unscoped),
        partition_key(enabled_for_scoped),
        partition_key(disabled_for_scoped),
        partition_key(alias_scoped),
        partition_key(models_scoped),
    }
    assert len(keys) == 5


# ---------------------------------------------------------------------------
# bucket_ttl_seconds -- per-tag Redis/bucket key TTL override
# ---------------------------------------------------------------------------


def test_bucket_ttl_seconds_defaults_to_period_plus_one_hour_when_unset():
    entry = TagRateLimitEntry(name="per_minute", tag_id="end_user_id", limit=1, period_seconds=60)
    assert bucket_ttl_seconds(entry) == 60 + 3600


def test_bucket_ttl_seconds_honors_key_ttl_seconds_override():
    entry = TagRateLimitEntry(
        name="per_minute", tag_id="end_user_id", limit=1, period_seconds=60, key_ttl_seconds=120
    )
    assert bucket_ttl_seconds(entry) == 120
