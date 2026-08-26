"""
Unit tests for tag-scoped token/request/dollar rate limiting.
"""

import asyncio
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.model_based_tag_rate_limits_hook import (
    _BACKGROUND_TASKS,
    _CONCURRENCY_MIN_SAFETY_TTL_SECONDS,
    _PENDING_CONCURRENCY_KEYS_FIELD,
    _bucket_key,
    _bucket_ttl_seconds,
    _build_group_limits,
    _build_limits_index,
    _ConfiguredLimit,
    _entry_applies,
    _extract_identity,
    _extract_key_hash,
    _extract_team_id,
    _fixed_length_identity,
    _inflight_key,
    _partition_key,
    _pending_reservations_cache_key,
    _PROXY_ModelBasedTagRateLimitsHook,
)
from litellm.types.router import RoutingGroup, TagRateLimitEntry, TagRateLimitScope


class TimeController:
    def __init__(self):
        self._current = datetime(2026, 1, 1, 0, 0, 0)

    def now(self) -> datetime:
        return self._current

    def advance(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)


@pytest.fixture
def time_controller():
    return TimeController()


def _make_limiter(time_controller: TimeController) -> _PROXY_ModelBasedTagRateLimitsHook:
    return _PROXY_ModelBasedTagRateLimitsHook(
        internal_usage_cache=DualCache(),
        time_provider=time_controller.now,
    )


def _call_context(tags: list[str]) -> tuple[dict, dict]:
    """
    A (request_kwargs, kwargs) pair sharing one `model_call_details` dict,
    mirroring production: admission reads `request_kwargs["litellm_logging_obj"]
    .model_call_details`, and the `kwargs` passed to async_log_success_event /
    async_log_failure_event / async_release_disconnect_state_hook's
    request_data *is* that same model_call_details dict (or carries the same
    logging_obj) -- see _PENDING_CONCURRENCY_KEYS_FIELD's docstring. A plain
    SimpleNamespace stands in for the real Logging object; only its
    model_call_details attribute is used.
    """
    model_call_details: dict = {}
    logging_obj = SimpleNamespace(model_call_details=model_call_details)
    request_kwargs = {"metadata": {"tags": tags}, "litellm_logging_obj": logging_obj}
    # kwargs must be the *same* dict object model_call_details is, so that
    # admission's writes onto model_call_details are visible when this kwargs
    # is later passed to a release hook -- see the docstring above.
    model_call_details["litellm_logging_obj"] = logging_obj
    model_call_details["metadata"] = {"tags": tags}
    return request_kwargs, model_call_details


def _deployment(model_name: str, deployment_id: str, tag_rate_limits: dict) -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": "gpt-4o", "mock_response": "ok"},
        "model_info": {"id": deployment_id, "tag_rate_limits": tag_rate_limits},
    }


def _expected_bucket_key(
    model_group: str,
    unit: str,
    name: str,
    tag_id: str,
    tag_value: str,
    period_seconds: int,
    now: float,
    deployment_scope: tuple | None = None,
    team_scope: str | None = None,
    resolved_group: str | None = None,
    key_hash: str | None = None,
    limit: float = 1,
    enabled_for: dict | None = None,
    disabled_for: dict | None = None,
    scope_by_key_hash: bool = False,
) -> str:
    """
    Builds the exact key the real code would compute (via _hash_tag's
    fixed-length hashing of tag_value), instead of hand-writing the raw
    tag value into a literal string -- the internal key format (hashed or
    not) is an implementation detail these tests shouldn't hardcode.

    `limit` and the scoping fields default to values that produce a
    stable fingerprint for tests that don't care about it, but must be
    passed matching the real entry's own configuration whenever a test's
    router declares a `limit` other than 1 (or any scoping) for the entry
    whose key this reproduces -- see _policy_fingerprint, which folds them
    into the key precisely so two differently-configured entries sharing a
    name never collide onto the same counter.
    """
    configured = _ConfiguredLimit(
        unit=unit,
        entry=TagRateLimitEntry(
            name=name,
            tag_id=tag_id,
            limit=limit,
            period_seconds=period_seconds,
            enabled_for=enabled_for,
            disabled_for=disabled_for,
            scope_by_key_hash=scope_by_key_hash,
        ),
        deployment_scope=deployment_scope,
        team_scope=team_scope,
        resolved_group=resolved_group,
    )
    bucket_id = int(now) // period_seconds
    return _bucket_key(model_group, configured, tag_value, bucket_id, key_hash=key_hash)


# ---------------------------------------------------------------------------
# _extract_identity
# ---------------------------------------------------------------------------


def test_extract_identity_matches_prefixed_tag():
    assert _extract_identity(["team_id:t1", "end_user_id:u1"], "end_user_id") == "u1"


def test_extract_identity_returns_none_when_absent():
    assert _extract_identity(["team_id:t1"], "end_user_id") is None


def test_extract_identity_skips_negation_tags():
    """A `!end_user_id:u1` routing-negation marker must never be read as identity."""
    assert _extract_identity(["!end_user_id:u1"], "end_user_id") is None


# ---------------------------------------------------------------------------
# _fixed_length_identity -- tag_value is caller-controlled with no length
# bound; this hook's own contribution to a cache key must not grow with it
# ---------------------------------------------------------------------------


def test_fixed_length_identity_bounds_key_contribution_regardless_of_input_size():
    """
    A caller can submit an arbitrarily long tag value (no length or content
    bound is enforced upstream of this hook). Without hashing, that value
    would go straight into an in-memory dict key (bypassing
    max_in_memory_cache_size, which caps item *count* not key bytes) and an
    unbounded-length Redis key (Redis has no key-count or key-size cap at
    all here). A fixed-length digest bounds this hook's own contribution to
    the key regardless of input size.
    """
    huge_value = "x" * 5_000_000
    digest = _fixed_length_identity(huge_value)
    assert len(digest) == 64  # sha256 hex digest length, independent of input size


def test_fixed_length_identity_preserves_distinctness():
    """Hashing must not collapse two different tag values onto one bucket."""
    assert _fixed_length_identity("user-a") != _fixed_length_identity("user-b")
    assert _fixed_length_identity("user-a") == _fixed_length_identity("user-a")


def test_pending_reservations_cache_key_bounds_call_id_regardless_of_input_size():
    """
    veria-ai finding on PR #36541: litellm_call_id comes straight from the
    caller-controlled x-litellm-call-id header with no length bound, and was
    embedded directly in the pending-reservations mirror key -- a caller
    submitting long ids across many in-flight tagged requests could inflate
    Redis/in-memory key size disproportionately. Hashed via
    _fixed_length_identity, same as every other caller-controlled value this
    hook puts in a cache key.
    """
    huge_call_id = "x" * 5_000_000
    key = _pending_reservations_cache_key(huge_call_id, "some-key-hash")
    assert len(key) < 200


def test_pending_reservations_cache_key_preserves_distinctness():
    assert _pending_reservations_cache_key("call-a", "kh") != _pending_reservations_cache_key("call-b", "kh")
    assert _pending_reservations_cache_key("call-a", "kh") == _pending_reservations_cache_key("call-a", "kh")


@pytest.mark.asyncio
async def test_an_oversized_tag_value_does_not_inflate_the_bucket_key(time_controller):
    """
    End-to-end: a request tagged with a multi-megabyte end_user_id value
    must still resolve to a short, fixed-length bucket key, not one whose
    size scales with the caller's input.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 1000, "period_seconds": 60}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    huge_tag_value = "y" * 2_000_000

    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": [f"end_user_id:{huge_tag_value}"]}},
    )

    (only_key,) = limiter.internal_usage_cache.dual_cache.in_memory_cache.cache_dict.keys()
    assert len(only_key) < 200


# ---------------------------------------------------------------------------
# _extract_key_hash / _extract_team_id -- must read only the one field the
# server actually authenticates into, never fall back to the other
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
    assert _extract_key_hash(request_kwargs, "litellm_metadata") == "real-authenticated-hash"


def test_extract_key_hash_reads_metadata_when_it_is_the_authoritative_field():
    request_kwargs = {"metadata": {"user_api_key": "real-hash"}}
    assert _extract_key_hash(request_kwargs, "metadata") == "real-hash"


def test_extract_team_id_ignores_a_forged_value_in_the_non_authoritative_field():
    request_kwargs = {
        "metadata": {"user_api_key_team_id": "forged-team"},
        "litellm_metadata": {"user_api_key_team_id": "real-team"},
    }
    assert _extract_team_id(request_kwargs, "litellm_metadata") == "real-team"


# ---------------------------------------------------------------------------
# TagRateLimitEntry -- limit validation
# ---------------------------------------------------------------------------


def test_tag_rate_limit_entry_rejects_nan_limit():
    """
    NaN compares False against every ordering operator, so a NaN limit makes
    the atomic requests/concurrency check-and-increment (rejects when the new
    value exceeds the limit) admit indefinitely, while the read-only
    tokens/dollars check (admits when the current value is under the limit)
    rejects every tagged request -- either way silently defeating the entry.
    """
    with pytest.raises(ValidationError, match="limit must not be NaN"):
        TagRateLimitEntry(name="n", limit=float("nan"), period_seconds=60)


def test_tag_rate_limit_entry_rejects_infinite_limit():
    """
    Positive infinity makes the atomic requests/concurrency
    current + increment > limit check always false, so admission never
    rejects; negative infinity makes it always true, rejecting every tagged
    request. Same silent-misconfiguration class as NaN, just via a different
    non-finite float rather than a non-ordering one.
    """
    with pytest.raises(ValidationError, match="limit must be finite"):
        TagRateLimitEntry(name="n", limit=float("inf"), period_seconds=60)
    with pytest.raises(ValidationError, match="limit must be finite"):
        TagRateLimitEntry(name="n", limit=float("-inf"), period_seconds=60)


def test_tag_rate_limit_entry_rejects_zero_or_negative_limit():
    """
    A limit of 0 (or negative) makes the atomic requests/concurrency check
    (current + increment > limit) reject every admission and the read-only
    tokens/dollars check (current < limit) never admit, same silent
    always-reject-everything failure mode as a negative-infinity limit --
    almost certainly a config typo, not an intentional "block everything"
    policy, so reject it at config load time instead.
    """
    with pytest.raises(ValidationError, match="limit must be a positive number"):
        TagRateLimitEntry(name="n", limit=0, period_seconds=60)
    with pytest.raises(ValidationError, match="limit must be a positive number"):
        TagRateLimitEntry(name="n", limit=-1, period_seconds=60)


# ---------------------------------------------------------------------------
# TagRateLimitEntry -- period_seconds validation
# ---------------------------------------------------------------------------


def test_tag_rate_limit_entry_rejects_zero_period_seconds():
    with pytest.raises(ValidationError, match="period_seconds must be a positive integer"):
        TagRateLimitEntry(name="n", limit=1, period_seconds=0)


def test_tag_rate_limit_entry_rejects_negative_period_seconds():
    with pytest.raises(ValidationError, match="period_seconds must be a positive integer"):
        TagRateLimitEntry(name="n", limit=1, period_seconds=-1)


def test_tag_rate_limit_entry_accepts_positive_period_seconds():
    entry = TagRateLimitEntry(name="n", limit=1, period_seconds=60)
    assert entry.period_seconds == 60


# ---------------------------------------------------------------------------
# _build_group_limits -- chain-wide vs per-deployment scoping
# ---------------------------------------------------------------------------


def test_build_group_limits_chain_wide_when_all_deployments_agree():
    deployments = [
        _deployment(
            "grp", "dep-1", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}}
        ),
        _deployment(
            "grp", "dep-2", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}}
        ),
    ]
    configured = _build_group_limits(deployments, "tokens")
    assert len(configured) == 1
    assert configured[0].deployment_scope is None


def test_build_group_limits_per_deployment_when_values_diverge():
    """
    Regression test: a naive index that dedupes by (model_name, limit name) and
    keeps whichever deployment it encounters first silently drops the second
    deployment's config. Divergent values must produce two independent
    per-deployment-scoped entries instead.
    """
    deployments = [
        _deployment(
            "grp", "dep-1", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}}
        ),
        _deployment(
            "grp", "dep-2", {"token_limits": {"limits": [{"name": "daily", "limit": 999, "period_seconds": 86400}]}}
        ),
    ]
    configured = _build_group_limits(deployments, "tokens")
    assert len(configured) == 2
    scopes = {c.deployment_scope for c in configured}
    assert scopes == {("dep-1",), ("dep-2",)}
    limits = {c.deployment_scope: c.entry.limit for c in configured}
    assert limits[("dep-1",)] == 500
    assert limits[("dep-2",)] == 999


def test_build_group_limits_per_deployment_when_only_some_declare_it():
    deployments = [
        _deployment(
            "grp", "dep-1", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}}
        ),
        _deployment("grp", "dep-2", {}),
    ]
    configured = _build_group_limits(deployments, "tokens")
    assert len(configured) == 1
    assert configured[0].deployment_scope == ("dep-1",)


def test_build_group_limits_empty_when_no_deployment_configures_unit():
    deployments = [_deployment("grp", "dep-1", {}), _deployment("grp", "dep-2", {})]
    assert _build_group_limits(deployments, "tokens") == ()


# ---------------------------------------------------------------------------
# _entry_applies -- enabled_for / disabled_for / apply_to_key_alias
# ---------------------------------------------------------------------------


def test_entry_applies_with_none_of_the_scoping_fields_set():
    entry = TagRateLimitEntry(name="daily", tag_id="end_user_id", limit=500, period_seconds=86400)
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is True


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
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is False
    assert _entry_applies(entry, ["end_user_id:u2"], None, None) is True


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
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is False
    assert _entry_applies(entry, ["end_user_id:u2"], None, None) is True


def test_entry_applies_matches_an_enabled_for_gate():
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        enabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",)),
    )
    assert _entry_applies(entry, ["end_user_id:u1", "company_id:1032"], None, None) is True


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
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is False


def test_entry_applies_skips_when_disabled_for_gate_matches():
    entry = TagRateLimitEntry(
        name="daily",
        tag_id="end_user_id",
        limit=500,
        period_seconds=86400,
        disabled_for=TagRateLimitScope(tag_id="company_id", values=("1032",)),
    )
    assert _entry_applies(entry, ["end_user_id:u1", "company_id:1032"], None, None) is False


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
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is True


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
    assert _entry_applies(entry, ["end_user_id:u1", "company_id:1032"], None, None) is False


def test_entry_applies_with_apply_to_key_alias_unset_applies_to_every_key():
    entry = TagRateLimitEntry(name="daily", tag_id="end_user_id", limit=500, period_seconds=86400)
    assert _entry_applies(entry, ["end_user_id:u1"], "any-key-alias", None) is True
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is True


def test_entry_applies_admits_a_key_alias_on_the_allowlist():
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_key_alias=("team-a-key",)
    )
    assert _entry_applies(entry, ["end_user_id:u1"], "team-a-key", None) is True


def test_entry_applies_rejects_a_key_alias_missing_from_the_allowlist():
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_key_alias=("team-a-key",)
    )
    assert _entry_applies(entry, ["end_user_id:u1"], "team-b-key", None) is False


def test_entry_applies_rejects_when_key_has_no_alias_but_allowlist_is_set():
    """apply_to_key_alias is an allowlist gate: a key with no alias at all
    never satisfies it, same as enabled_for's absent-gate-tag semantics."""
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_key_alias=("team-a-key",)
    )
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is False


def test_entry_applies_with_apply_to_models_unset_applies_to_every_model():
    entry = TagRateLimitEntry(name="daily", tag_id="end_user_id", limit=500, period_seconds=86400)
    assert _entry_applies(entry, ["end_user_id:u1"], None, "opus-chain") is True
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is True


def test_entry_applies_admits_a_model_on_the_apply_to_models_allowlist():
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_models=("opus-chain",)
    )
    assert _entry_applies(entry, ["end_user_id:u1"], None, "opus-chain") is True


def test_entry_applies_rejects_a_model_missing_from_the_apply_to_models_allowlist():
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_models=("opus-chain",)
    )
    assert _entry_applies(entry, ["end_user_id:u1"], None, "sonnet-chain") is False


def test_entry_applies_rejects_when_model_is_absent_but_apply_to_models_is_set():
    """apply_to_models is an allowlist gate: a request with no model at all
    never satisfies it, same as apply_to_key_alias's absent-key semantics."""
    entry = TagRateLimitEntry(
        name="daily", tag_id="end_user_id", limit=500, period_seconds=86400, apply_to_models=("opus-chain",)
    )
    assert _entry_applies(entry, ["end_user_id:u1"], None, None) is False


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
    assert _entry_applies(entry, ["end_user_id:u1"], "premium-key", "opus-chain") is True
    assert _entry_applies(entry, ["end_user_id:u1"], "other-key", "opus-chain") is False
    assert _entry_applies(entry, ["end_user_id:u1"], "premium-key", "sonnet-chain") is False


# ---------------------------------------------------------------------------
# TagRateLimitEntry / TagRateLimitScope -- scoping field validation
# ---------------------------------------------------------------------------


def test_tag_rate_limit_scope_rejects_empty_values():
    with pytest.raises(ValidationError, match="values must be a non-empty list"):
        TagRateLimitScope(tag_id="company_id", values=())


def test_tag_rate_limit_entry_rejects_enabled_for_missing_values():
    with pytest.raises(ValidationError):
        TagRateLimitEntry(name="daily", limit=1, period_seconds=60, enabled_for={"tag_id": "company_id"})


def test_tag_rate_limit_scope_normalizes_values_order_and_duplicates():
    scope = TagRateLimitScope(tag_id="company_id", values=("1032", "1001", "1001"))
    assert scope.values == ("1001", "1032")


def test_tag_rate_limit_entry_rejects_empty_apply_to_key_alias():
    with pytest.raises(ValidationError, match="apply_to_key_alias must be a non-empty list"):
        TagRateLimitEntry(name="daily", limit=1, period_seconds=60, apply_to_key_alias=())


def test_tag_rate_limit_entry_normalizes_apply_to_key_alias_order_and_duplicates():
    entry = TagRateLimitEntry(
        name="daily", limit=1, period_seconds=60, apply_to_key_alias=("team-b-key", "team-a-key", "team-a-key")
    )
    assert entry.apply_to_key_alias == ("team-a-key", "team-b-key")


def test_tag_rate_limit_entry_rejects_empty_apply_to_models():
    with pytest.raises(ValidationError, match="apply_to_models must be a non-empty list"):
        TagRateLimitEntry(name="daily", limit=1, period_seconds=60, apply_to_models=())


def test_tag_rate_limit_entry_normalizes_apply_to_models_order_and_duplicates():
    entry = TagRateLimitEntry(
        name="daily", limit=1, period_seconds=60, apply_to_models=("sonnet-chain", "opus-chain", "opus-chain")
    )
    assert entry.apply_to_models == ("opus-chain", "sonnet-chain")


# ---------------------------------------------------------------------------
# _hash_tag / _bucket_key -- policy identity folds into the Redis key itself
# ---------------------------------------------------------------------------


def test_bucket_key_differs_for_same_named_entries_with_different_limits():
    """
    A plain, unscoped entry and a stricter, scoped override can legitimately
    share a `name` (the worked example in the docs uses distinct names, but
    nothing in validation requires that) -- resolve_any/_build_group_limits
    already treat differing limit/scoping as genuinely distinct policies for
    dedup purposes, so the actual counter key must too, or two
    differently-configured entries that happen to share a name check and
    charge the identical Redis/in-memory bucket.
    """
    now = 0.0
    default_key = _expected_bucket_key("grp", "requests", "daily", "end_user_id", "u1", 86400, now, limit=2500)
    override_key = _expected_bucket_key(
        "grp",
        "requests",
        "daily",
        "end_user_id",
        "u1",
        86400,
        now,
        limit=1,
        enabled_for={"tag_id": "company_id", "values": ["1032"]},
    )
    assert default_key != override_key


def test_bucket_key_differs_for_same_named_entries_with_different_scoping_only():
    now = 0.0
    excluding_u1 = _expected_bucket_key(
        "grp",
        "requests",
        "daily",
        "end_user_id",
        "u2",
        86400,
        now,
        limit=100,
        disabled_for={"tag_id": "end_user_id", "values": ["u1"]},
    )
    excluding_u2 = _expected_bucket_key(
        "grp",
        "requests",
        "daily",
        "end_user_id",
        "u2",
        86400,
        now,
        limit=100,
        disabled_for={"tag_id": "end_user_id", "values": ["u2"]},
    )
    assert excluding_u1 != excluding_u2


def test_bucket_key_differs_for_same_named_entries_diverging_only_on_scope_by_key_hash():
    """
    _DedupSignature already folds scope_by_key_hash into dedup (two
    deployments declaring the same name/tag_id but different
    scope_by_key_hash become two distinct _ConfiguredLimit entries, not one
    merged one), but _policy_fingerprint didn't fold it into the bucket-key
    hash. When a request's key_hash resolves to None -- e.g. no virtual key
    on the call -- both entries' key_hash-derived suffix is empty too, so an
    unscoped entry and a key-hash-scoped entry that otherwise share every
    other field collided onto the identical counter, letting one entry's
    admission or accounting silently corrupt the other's.
    """
    now = 0.0
    unscoped = _expected_bucket_key(
        "grp", "requests", "daily", "end_user_id", "u1", 86400, now, limit=100, scope_by_key_hash=False
    )
    key_hash_scoped_but_no_key_present = _expected_bucket_key(
        "grp", "requests", "daily", "end_user_id", "u1", 86400, now, limit=100, scope_by_key_hash=True, key_hash=None
    )
    assert unscoped != key_hash_scoped_but_no_key_present


# ---------------------------------------------------------------------------
# _build_group_limits -- scoping fields fold into the dedup signature
# ---------------------------------------------------------------------------


def test_build_group_limits_per_deployment_when_disabled_for_diverges():
    """
    Regression test: two deployments agreeing on tag_id/limit/period_seconds
    but declaring different disabled_for scopes are genuinely different
    policies and must not be silently merged into one shared bucket -- the
    same class of bug test_build_group_limits_per_deployment_when_values_diverge
    already guards against for a plain divergent limit value.
    """
    deployments = [
        _deployment(
            "grp",
            "dep-1",
            {
                "token_limits": {
                    "limits": [
                        {
                            "name": "daily",
                            "limit": 500,
                            "period_seconds": 86400,
                            "disabled_for": {"tag_id": "end_user_id", "values": ["u1"]},
                        }
                    ]
                }
            },
        ),
        _deployment(
            "grp",
            "dep-2",
            {
                "token_limits": {
                    "limits": [
                        {
                            "name": "daily",
                            "limit": 500,
                            "period_seconds": 86400,
                            "disabled_for": {"tag_id": "end_user_id", "values": ["u2"]},
                        }
                    ]
                }
            },
        ),
    ]
    configured = _build_group_limits(deployments, "tokens")
    assert len(configured) == 2
    scopes = {c.deployment_scope for c in configured}
    assert scopes == {("dep-1",), ("dep-2",)}


def test_build_group_limits_chain_wide_when_disabled_for_agrees():
    deployments = [
        _deployment(
            "grp",
            "dep-1",
            {
                "token_limits": {
                    "limits": [
                        {
                            "name": "daily",
                            "limit": 500,
                            "period_seconds": 86400,
                            "disabled_for": {"tag_id": "end_user_id", "values": ["u1"]},
                        }
                    ]
                }
            },
        ),
        _deployment(
            "grp",
            "dep-2",
            {
                "token_limits": {
                    "limits": [
                        {
                            "name": "daily",
                            "limit": 500,
                            "period_seconds": 86400,
                            "disabled_for": {"tag_id": "end_user_id", "values": ["u1"]},
                        }
                    ]
                }
            },
        ),
    ]
    configured = _build_group_limits(deployments, "tokens")
    assert len(configured) == 1
    assert configured[0].deployment_scope is None


def test_build_group_limits_chain_wide_when_disabled_for_agrees_in_different_order():
    """
    Two deployments declaring the identical disabled_for values set, just in
    a different config order, must dedup to one chain-wide entry -- config
    order is not a policy difference. Relies on TagRateLimitScope's own
    normalization (sorting) of values at construction time, not on this
    dedup path re-sorting them itself.
    """
    deployments = [
        _deployment(
            "grp",
            "dep-1",
            {
                "token_limits": {
                    "limits": [
                        {
                            "name": "daily",
                            "limit": 500,
                            "period_seconds": 86400,
                            "disabled_for": {"tag_id": "end_user_id", "values": ["u1", "u2"]},
                        }
                    ]
                }
            },
        ),
        _deployment(
            "grp",
            "dep-2",
            {
                "token_limits": {
                    "limits": [
                        {
                            "name": "daily",
                            "limit": 500,
                            "period_seconds": 86400,
                            "disabled_for": {"tag_id": "end_user_id", "values": ["u2", "u1"]},
                        }
                    ]
                }
            },
        ),
    ]
    configured = _build_group_limits(deployments, "tokens")
    assert len(configured) == 1
    assert configured[0].deployment_scope is None


# ---------------------------------------------------------------------------
# async_filter_deployments -- enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_deployments_noop_without_config(time_controller):
    limiter = _make_limiter(time_controller)
    router = litellm.Router(model_list=[_deployment("grp", "dep-1", {})])
    limiter.update_variables(llm_router=router)

    healthy = router.model_list
    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_filter_deployments_allows_under_limit_and_rejects_at_limit(time_controller):
    """
    "requests" admission is atomic check-and-increment at the filter step
    itself (not a separate read-then-account-later pass), so two concurrent
    requests can never both read "1 under limit" and both get admitted past
    a limit of 2 -- each call's own increment is immediately visible to the
    next. Calling the filter 3 times with limit=2 must admit exactly 2 and
    reject the 3rd.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 2, "period_seconds": 60}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    for _ in range(2):
        result = await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
        assert result == healthy

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["tag_value"] == "u1"
    assert exc_info.value.detail["limit_name"] == "per_minute"


@pytest.mark.asyncio
async def test_filter_deployments_falls_back_to_deployment_model_name_for_routing_group_calls(time_controller):
    """
    Router keeps a callable routing-group name distinct from every member
    deployment's own model_name (see Router._get_routing_group_deployments),
    so async_filter_deployments can be called with model="my-group" while
    healthy_deployments carries the group's real member deployments. The
    limiter must still resolve and enforce each member's own configured
    limits rather than silently no-opping because "my-group" itself never
    appears in the index.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "backend-a",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 2, "period_seconds": 60}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    for _ in range(2):
        result = await limiter.async_filter_deployments(
            model="my-group",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
        assert result == healthy

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="my-group",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )


@pytest.mark.asyncio
async def test_filter_deployments_routing_group_does_not_collide_across_different_model_names(time_controller):
    """
    A routing group can span deployments from different model_names that
    happen to declare an identically-named, identically-configured limit.
    Each must get its own bucket (keyed by its own model_name via
    resolved_group), not share one just because the caller addressed both
    through the same group name.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "backend-a",
                "dep-a",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
                    }
                },
            ),
            _deployment(
                "backend-b",
                "dep-b",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
                    }
                },
            ),
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # Exhaust backend-a's limit (limit=1) via the group-addressed call.
    await limiter.async_filter_deployments(
        model="my-group",
        healthy_deployments=[healthy[0]],
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="my-group",
            healthy_deployments=[healthy[0]],
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )

    # backend-b's own bucket must be untouched -- same group, same tag, same
    # limit name, but a different underlying model_name.
    result = await limiter.async_filter_deployments(
        model="my-group",
        healthy_deployments=[healthy[1]],
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == [healthy[1]]


def test_resolve_any_dedups_identical_signature_across_member_model_names():
    """
    A real routing-group hop presents every member simultaneously (Router
    resolves the group name to its full member list for one filtering pass,
    then picks exactly one afterwards), and resolve_any is called once for
    that one hop with every member's model_name as a candidate. Two members
    declaring the identical concurrency signature must resolve to one shared
    entry for that hop, not two: `async_filter_deployments` checks and
    atomically increments every entry `resolve_any` returns as belonging to
    this one hop, so two entries here means the hop reserves capacity twice
    (once per member) even though only one deployment will actually serve --
    over-charging the caller's own usage and risking a false 429 against a
    sibling member that was never over its own limit.

    Admission-level round-trip tests can't distinguish this from "two
    separate entries with identical limits, always incremented together":
    every hop that presents the same member set moves both buckets in
    lockstep regardless of whether they're actually one shared entry or two,
    so the dedup can only be verified directly at this level.
    """
    concurrency_limits = {
        "concurrency_limits": {
            "limits": [{"name": "inflight", "tag_id": "end_user_id", "limit": 1, "period_seconds": 300}]
        }
    }
    index = _build_limits_index(
        [
            _deployment("backend-a", "dep-a", concurrency_limits),
            _deployment("backend-b", "dep-b", concurrency_limits),
        ]
    )
    resolved = index.resolve_any("my-group", team_id=None, candidate_model_names=("backend-a", "backend-b"))
    assert len(resolved) == 1
    assert resolved[0].unit == "concurrency"
    assert resolved[0].entry.limit == 1


def test_resolve_any_keeps_divergent_signatures_across_member_model_names_separate():
    """
    Companion to the dedup test above: members that genuinely disagree on
    the limit for the same tag_id+name must not be silently collapsed --
    which of two different limits would even apply isn't knowable at this
    admission-time hook, before a specific deployment is picked, so both
    stay as their own entries (today's pre-existing behavior for a
    divergent config, left unchanged by the identical-signature dedup).
    """
    index = _build_limits_index(
        [
            _deployment(
                "backend-a",
                "dep-a",
                {
                    "concurrency_limits": {
                        "limits": [{"name": "inflight", "tag_id": "end_user_id", "limit": 1, "period_seconds": 300}]
                    }
                },
            ),
            _deployment(
                "backend-b",
                "dep-b",
                {
                    "concurrency_limits": {
                        "limits": [{"name": "inflight", "tag_id": "end_user_id", "limit": 2, "period_seconds": 300}]
                    }
                },
            ),
        ]
    )
    resolved = index.resolve_any("my-group", team_id=None, candidate_model_names=("backend-a", "backend-b"))
    assert len(resolved) == 2
    assert {c.entry.limit for c in resolved} == {1, 2}


def test_resolve_any_keeps_divergent_disabled_for_across_member_model_names_separate():
    """
    resolve_any's own dedup key omitted enabled_for/disabled_for/
    apply_to_key_alias, so two routing-group members agreeing on
    tag_id/limit/period_seconds but declaring different disabled_for scopes
    collapsed to whichever model_name sorted first -- silently applying the
    wrong member's policy (and, for the discarded one, no enforcement or
    accounting at all for callers only that policy covers). This is the same
    class of bug test_build_group_limits_per_deployment_when_disabled_for_diverges
    already guards against for the sibling load-balanced-group dedup path.
    """
    concurrency_limits_excluding_u1 = {
        "concurrency_limits": {
            "limits": [
                {
                    "name": "inflight",
                    "tag_id": "end_user_id",
                    "limit": 1,
                    "period_seconds": 300,
                    "disabled_for": {"tag_id": "end_user_id", "values": ["u1"]},
                }
            ]
        }
    }
    concurrency_limits_excluding_u2 = {
        "concurrency_limits": {
            "limits": [
                {
                    "name": "inflight",
                    "tag_id": "end_user_id",
                    "limit": 1,
                    "period_seconds": 300,
                    "disabled_for": {"tag_id": "end_user_id", "values": ["u2"]},
                }
            ]
        }
    }
    index = _build_limits_index(
        [
            _deployment("backend-a", "dep-a", concurrency_limits_excluding_u1),
            _deployment("backend-b", "dep-b", concurrency_limits_excluding_u2),
        ]
    )
    resolved = index.resolve_any("my-group", team_id=None, candidate_model_names=("backend-a", "backend-b"))
    assert len(resolved) == 2
    assert {c.entry.disabled_for.values for c in resolved} == {("u1",), ("u2",)}


def test_resolve_any_keeps_divergent_apply_to_models_across_member_model_names_separate():
    """Same class of bug as the disabled_for test above, for apply_to_models:
    two routing-group members agreeing on tag_id/limit/period_seconds but
    scoped to different apply_to_models lists must not collapse to one."""
    concurrency_limits_for_opus = {
        "concurrency_limits": {
            "limits": [
                {
                    "name": "inflight",
                    "tag_id": "end_user_id",
                    "limit": 1,
                    "period_seconds": 300,
                    "apply_to_models": ["opus-chain"],
                }
            ]
        }
    }
    concurrency_limits_for_sonnet = {
        "concurrency_limits": {
            "limits": [
                {
                    "name": "inflight",
                    "tag_id": "end_user_id",
                    "limit": 1,
                    "period_seconds": 300,
                    "apply_to_models": ["sonnet-chain"],
                }
            ]
        }
    }
    index = _build_limits_index(
        [
            _deployment("backend-a", "dep-a", concurrency_limits_for_opus),
            _deployment("backend-b", "dep-b", concurrency_limits_for_sonnet),
        ]
    )
    resolved = index.resolve_any("my-group", team_id=None, candidate_model_names=("backend-a", "backend-b"))
    assert len(resolved) == 2
    assert {c.entry.apply_to_models for c in resolved} == {("opus-chain",), ("sonnet-chain",)}


def test_resolve_any_picks_the_same_resolved_group_regardless_of_hash_seed():
    """
    Two members with an identical signature dedup to whichever one
    `frozenset(candidate_model_names)` iterates first. Plain `frozenset`
    iteration order for strings is seeded from `PYTHONHASHSEED`, which is
    randomized per process by default, so two proxy worker processes (or the
    same process across a restart) resolving the identical member set could
    pick different members as `resolved_group` -- fragmenting what's meant to
    be one shared Redis bucket into two. This can't be observed from within
    one interpreter (a single process has one fixed seed for its lifetime),
    so this spawns two real subprocesses pinned to seeds empirically known to
    order these three names differently under a plain, unsorted frozenset --
    see the bug report this regression-tests for the exact reproduction.
    """
    script = (
        "from litellm.proxy.hooks.model_based_tag_rate_limits_hook import _build_limits_index\n"
        "def _deployment(model_name, deployment_id, tag_rate_limits):\n"
        "    return {'model_name': model_name, 'litellm_params': {'model': 'gpt-4o'},"
        " 'model_info': {'id': deployment_id, 'tag_rate_limits': tag_rate_limits}}\n"
        "limits = {'concurrency_limits': {'limits': [{'name': 'inflight', 'tag_id': 'end_user_id',"
        " 'limit': 1, 'period_seconds': 300}]}}\n"
        "index = _build_limits_index(["
        "_deployment('backend-a', 'dep-a', limits),"
        "_deployment('backend-b', 'dep-b', limits),"
        "_deployment('backend-c', 'dep-c', limits)])\n"
        "resolved = index.resolve_any('my-group', team_id=None,"
        " candidate_model_names=('backend-a', 'backend-b', 'backend-c'))\n"
        "print(resolved[0].resolved_group)\n"
    )
    # seed=1 and seed=3 are empirically confirmed to order these three
    # literal strings differently under plain (unsorted) frozenset iteration.
    results = {
        seed: subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "3")
    }
    assert results["1"] == results["3"] == "backend-a"


@pytest.mark.asyncio
async def test_filter_deployments_per_entry_fail_open_when_tag_absent(time_controller):
    """
    Two entries on the same chain, different tag_ids. Only the tag that's
    actually present in the request gets checked; the other has zero effect.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400},
                            {"name": "monthly", "tag_id": "team_id", "limit": 1, "period_seconds": 2592000},
                        ]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # Only end_user_id is present -- team_id-keyed entry must not raise or touch Redis.
    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy

    now = time_controller.now().timestamp()
    team_bucket_id = int(now) // 2592000
    team_key = f"{{tag_rl:grp:requests:monthly:team_id:chain:whatever}}:{team_bucket_id}"
    assert await limiter.internal_usage_cache.async_get_cache(key=team_key, litellm_parent_otel_span=None) is None


def _company_tiered_cap_router(default_limit: int, override_limit: int) -> "litellm.Router":
    return litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {
                                "name": "default_daily",
                                "tag_id": "end_user_id",
                                "limit": default_limit,
                                "period_seconds": 86400,
                            },
                            {
                                "name": "company_1032_daily",
                                "tag_id": "end_user_id",
                                "limit": override_limit,
                                "period_seconds": 86400,
                                "enabled_for": {"tag_id": "company_id", "values": ["1032"]},
                                "disabled_for": {"tag_id": "end_user_id", "values": ["u1"]},
                            },
                        ]
                    }
                },
            )
        ]
    )


@pytest.mark.asyncio
async def test_filter_deployments_scoped_override_skips_for_an_excluded_identity(time_controller):
    """
    Company-tiered-cap example from the plan: a stricter override entry
    gated to one company via enabled_for, with a handful of named users
    excluded from it via disabled_for on the entry's own tag_id. An excluded
    user must fall through to the unscoped default entry entirely -- the
    override never enforces or accounts for them.
    """
    limiter = _make_limiter(time_controller)
    router = _company_tiered_cap_router(default_limit=3, override_limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    for _ in range(3):
        result = await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1", "company_id:1032"]}},
        )
        assert result == healthy

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1", "company_id:1032"]}},
        )
    assert exc_info.value.detail["limit_name"] == "default_daily"


@pytest.mark.asyncio
async def test_filter_deployments_scoped_override_enforces_for_a_non_excluded_identity_in_scope(time_controller):
    """
    The same override applies, and enforces its own stricter limit, for a
    company-1032 user who is not disabled_for's excluded identity, proving
    the two entries are independently enforced rather than one silently
    replacing the other.
    """
    limiter = _make_limiter(time_controller)
    router = _company_tiered_cap_router(default_limit=3, override_limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u2", "company_id:1032"]}},
    )
    assert result == healthy

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u2", "company_id:1032"]}},
        )
    assert exc_info.value.detail["limit_name"] == "company_1032_daily"


@pytest.mark.asyncio
async def test_filter_deployments_scoped_override_does_not_apply_outside_its_enabled_for_gate(time_controller):
    """A user not tagged with the gate company at all only ever hits the
    unscoped default entry, even though the override's own limit is looser
    and would otherwise still have room."""
    limiter = _make_limiter(time_controller)
    router = _company_tiered_cap_router(default_limit=1, override_limit=5)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u3"]}},
    )
    assert result == healthy

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u3"]}},
        )
    assert exc_info.value.detail["limit_name"] == "default_daily"


@pytest.mark.asyncio
async def test_different_tag_ids_with_same_name_do_not_share_a_counter(time_controller):
    """
    Regression test: the key format used to omit `tag_id`, so two
    independently configured entries sharing the same unit/name (here both
    named "daily") but keyed on different tag_ids would collide whenever a
    caller's value for one tag_id happened to equal another caller's value
    for the other tag_id. With equal limits of 1, a colliding shared counter
    would make team_id "u1"'s very first request get wrongly rejected right
    after end_user_id "u1"'s own first (and separately limited) request --
    the failure mode a higher team_id limit would have masked.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400},
                            {"name": "daily", "tag_id": "team_id", "limit": 1, "period_seconds": 86400},
                        ]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # end_user_id "u1" makes its one allowed request.
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )

    # team_id "u1" -- identical value, different tag_id, its own untouched
    # limit of 1 -- must still admit its first request.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["team_id:u1"]}}
    )
    assert result == healthy

    # Both identities are now genuinely at their own limit of 1.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["team_id:u1"]}},
        )


# ---------------------------------------------------------------------------
# Load-balanced group: chain-wide vs per-deployment enforcement end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_balanced_group_per_deployment_breach_rejects_whole_hop(time_controller):
    """
    Two deployments share one model_name with divergent per-deployment
    limits. Breaching one deployment's bucket rejects the hop even though the
    other deployment (still present in healthy_deployments) is comfortably
    under its own limit -- this does NOT filter the breaching deployment out
    and let the router retry the sibling.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
                    }
                },
            ),
            _deployment(
                "grp",
                "dep-2",
                {
                    "request_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 999, "period_seconds": 86400}]
                    }
                },
            ),
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    request_kwargs = {"metadata": {"tags": ["end_user_id:u1"]}}

    now = time_controller.now().timestamp()
    dep1_key = _expected_bucket_key(
        "grp", "requests", "daily", "end_user_id", "u1", 86400, now, deployment_scope=("dep-1",)
    )
    await limiter.internal_usage_cache.async_set_cache(key=dep1_key, value=1, ttl=86400, litellm_parent_otel_span=None)

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
        )


# ---------------------------------------------------------------------------
# async_log_success_event -- accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_success_event_increments_configured_units(time_controller):
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "token_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_seconds": 86400}]
                    },
                    "request_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 100, "period_seconds": 86400}]
                    },
                    "dollar_limits": {
                        "limits": [
                            {"name": "monthly", "tag_id": "end_user_id", "limit": 50.0, "period_seconds": 2592000}
                        ]
                    },
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)

    kwargs = {
        "metadata": {"tags": ["end_user_id:u1"]},
        "standard_logging_object": {
            "model_group": "grp",
            "model_id": "dep-1",
            "total_tokens": 42,
            "response_cost": 0.01,
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    # accounting is fired via asyncio.create_task; let it run.
    await asyncio.sleep(0)

    now = time_controller.now().timestamp()
    token_key = _expected_bucket_key("grp", "tokens", "daily", "end_user_id", "u1", 86400, now, limit=500000)
    dollar_key = _expected_bucket_key("grp", "dollars", "monthly", "end_user_id", "u1", 2592000, now, limit=50.0)

    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    )
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=dollar_key, litellm_parent_otel_span=None)) == 0.01
    )

    # "requests" is accounted atomically at admission (async_filter_deployments),
    # not here -- async_log_success_event must not touch its bucket at all.
    request_key = _expected_bucket_key("grp", "requests", "daily", "end_user_id", "u1", 86400, now, limit=100)
    assert await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None) is None


@pytest.mark.asyncio
async def test_log_success_event_accounts_when_litellm_params_carries_a_null_litellm_metadata_key(time_controller):
    """
    kwargs at async_log_success_event time is Logging.model_call_details, not
    the flat dict admission sees -- for a plain (non LITELLM_METADATA_ROUTES)
    chat completion, kwargs["litellm_params"] carries a "litellm_metadata" key
    that is always present but set to None, alongside the real, populated
    "metadata" dict. get_metadata_variable_name_from_kwargs only checks key
    presence, so it always resolved to "litellm_metadata" here and read no
    tags/identity at all, silently dropping every token/dollar accounting for
    this route shape.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "token_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_seconds": 86400}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)

    kwargs = {
        "litellm_params": {
            "litellm_metadata": None,
            "metadata": {"tags": ["end_user_id:u1"]},
        },
        "standard_logging_object": {
            "model_group": "grp",
            "model_id": "dep-1",
            "total_tokens": 42,
            "response_cost": 0.01,
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    now = time_controller.now().timestamp()
    token_key = _expected_bucket_key("grp", "tokens", "daily", "end_user_id", "u1", 86400, now, limit=500000)
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    )


@pytest.mark.asyncio
async def test_log_success_event_reads_nested_litellm_metadata_when_that_is_authoritative(time_controller):
    """
    kwargs here is Logging.model_call_details: on LITELLM_METADATA_ROUTES
    (/v1/messages, /responses, ...) metadata/litellm_metadata are never
    top-level keys, only nested under kwargs["litellm_params"] -- and the
    caller's own native "metadata" can be present there with no tags at all,
    while the real, server-computed tags live in "litellm_metadata".
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "token_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_seconds": 86400}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)

    kwargs = {
        "litellm_params": {
            "metadata": {"tags": []},
            "litellm_metadata": {"tags": ["end_user_id:u1"]},
        },
        "standard_logging_object": {
            "model_group": "grp",
            "model_id": "dep-1",
            "total_tokens": 42,
            "response_cost": 0.01,
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    now = time_controller.now().timestamp()
    token_key = _expected_bucket_key("grp", "tokens", "daily", "end_user_id", "u1", 86400, now, limit=500000)
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    )


@pytest.mark.asyncio
async def test_log_success_event_falls_back_to_serving_deployment_model_name_for_routing_group_calls(
    time_controller,
):
    """
    standard_logging_object["model_group"] is the caller-visible name from
    Router._update_kwargs_before_fallbacks -- for a routing-group call this
    is the group name too, which never appears in the index. Success
    accounting must fall back to the model_name of the deployment that
    actually served this hop (standard_logging_object["model_id"]).
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "backend-a",
                "dep-1",
                {
                    "token_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_seconds": 86400}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)

    kwargs = {
        "metadata": {"tags": ["end_user_id:u1"]},
        "standard_logging_object": {
            "model_group": "my-group",
            "model_id": "dep-1",
            "total_tokens": 42,
            "response_cost": 0.01,
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    now = time_controller.now().timestamp()
    token_key = _expected_bucket_key("backend-a", "tokens", "daily", "end_user_id", "u1", 86400, now, limit=500000)
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    )


@pytest.mark.asyncio
async def test_log_success_event_accounts_against_the_same_bucket_admission_checked(time_controller):
    """
    resolve_any dedups an identical signature across a routing group's
    members into one shared entry, stamped with resolved_group from
    whichever member frozenset(candidate_model_names) yields first (see
    resolve_any's own docstring). Success accounting for tokens/dollars only
    learns the one deployment that actually served this hop; passing just
    that single name as resolve_any's sole candidate would make its dedup
    trivially resolve to that deployment's own name -- which can differ from
    whichever member admission's full-group view picked, silently
    accounting usage against a bucket admission never checked and letting a
    token/dollar limit be bypassed. Success accounting must reconstruct the
    full routing-group candidate set so it lands on the identical bucket
    regardless of which member actually served.
    """
    token_limits = {
        "token_limits": {
            "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_seconds": 86400}]
        }
    }
    router = litellm.Router(
        model_list=[
            _deployment("backend-a", "dep-a", token_limits),
            _deployment("backend-b", "dep-b", token_limits),
        ],
        routing_groups=[
            RoutingGroup(group_name="my-group", models=["backend-a", "backend-b"], routing_strategy="simple-shuffle")
        ],
    )
    limiter = _make_limiter(time_controller)
    limiter.update_variables(llm_router=router)

    # What admission would check: it sees every member, and resolve_any's
    # dedup picks whichever one frozenset yields first for the shared entry.
    admitted = limiter._index.get(router).resolve_any(
        "my-group", team_id=None, candidate_model_names=("backend-a", "backend-b")
    )
    assert len(admitted) == 1
    admission_bucket_group = admitted[0].resolved_group

    # Force the deployment that actually serves to be the *other* member --
    # deterministic regardless of which one frozenset happened to pick above,
    # so this test always exercises the mismatch the fix guards against.
    serving_model_name = "backend-b" if admission_bucket_group == "backend-a" else "backend-a"
    serving_deployment_id = "dep-b" if serving_model_name == "backend-b" else "dep-a"

    kwargs = {
        "metadata": {"tags": ["end_user_id:u1"]},
        "standard_logging_object": {
            "model_group": "my-group",
            "model_id": serving_deployment_id,
            "total_tokens": 42,
            "response_cost": 0.01,
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    now = time_controller.now().timestamp()
    token_key = _expected_bucket_key(
        "my-group", "tokens", "daily", "end_user_id", "u1", 86400, now, resolved_group=admission_bucket_group, limit=500000
    )
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    )


@pytest.mark.asyncio
async def test_admission_dedups_against_the_full_group_not_just_currently_healthy_members(time_controller):
    """
    `healthy_deployments` is Router's cooldown-filtered list for this one
    hop -- a member merely cooled down right now is excluded from it, but
    it's still a real member of the routing group. Deriving resolve_any's
    candidate set from `healthy_deployments` instead of the full group would
    make admission's resolved_group choice depend on which members happen to
    be healthy at that exact moment, while success accounting (which has no
    way to know what was healthy at admission time) always reconstructs the
    full, static membership -- landing the two sides on different buckets
    whenever a member is cooled down. Admission must dedup against the same
    full membership success does, regardless of which members are currently
    healthy.
    """
    token_limits = {
        "token_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 10, "period_seconds": 86400}]}
    }
    router = litellm.Router(
        model_list=[
            _deployment("backend-a", "dep-a", token_limits),
            _deployment("backend-b", "dep-b", token_limits),
        ],
        routing_groups=[
            RoutingGroup(group_name="my-group", models=["backend-a", "backend-b"], routing_strategy="simple-shuffle")
        ],
    )
    limiter = _make_limiter(time_controller)
    limiter.update_variables(llm_router=router)

    # The shared entry always dedups to "backend-a" (alphabetically first).
    # Pre-load *that* bucket over the limit; the "backend-b" bucket (what a
    # healthy_deployments-derived candidate set would wrongly resolve to,
    # since backend-a is the only one excluded below) stays empty.
    now = time_controller.now().timestamp()
    over_limit_key = _expected_bucket_key(
        "my-group", "tokens", "daily", "end_user_id", "u1", 86400, now, resolved_group="backend-a", limit=10
    )
    await limiter.internal_usage_cache.async_set_cache(key=over_limit_key, value=20.0, litellm_parent_otel_span=None)

    # Simulate backend-a being cooled down: Router would exclude it from the
    # healthy_deployments list passed to this hop's admission.
    healthy_excluding_backend_a = [d for d in router.model_list if d["model_name"] == "backend-b"]
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="my-group",
            healthy_deployments=healthy_excluding_backend_a,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )


@pytest.mark.asyncio
async def test_log_success_event_accounts_against_the_key_hash_admission_checked(time_controller):
    """
    Admission's `_extract_key_hash` reads `metadata.user_api_key` unconditionally
    whenever scope_by_key_hash is set -- that field is already the hashed
    token by the time it reaches this hook (see the function's own
    docstring), regardless of its shape. `standard_logging_object.metadata`
    only ever carries the derived `user_api_key_hash` field, and only when the
    raw value happens to look like a SHA-256 hex digest (see
    litellm_logging.py's get_standard_logging_metadata) -- a virtual key
    represented any other way makes that field silently absent, so reading it
    on the success side would account against key_hash=None while admission
    scoped the check against the real value, letting usage silently bypass a
    per-key limit whenever the key's own representation isn't SHA-256-shaped.
    """
    token_limits = {
        "token_limits": {
            "limits": [
                {"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_seconds": 86400, "scope_by_key_hash": True}
            ]
        }
    }
    router = litellm.Router(model_list=[_deployment("grp", "dep-1", token_limits)])
    limiter = _make_limiter(time_controller)
    limiter.update_variables(llm_router=router)

    # "keyA" deliberately isn't SHA-256-shaped, so standard_logging_object's
    # own redaction/derivation step would never populate user_api_key_hash
    # for it -- it's simply absent, matching production for a key hash that
    # doesn't pass that shape check.
    kwargs = {
        "metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"},
        "standard_logging_object": {
            "model_group": "grp",
            "model_id": "dep-1",
            "total_tokens": 42,
            "response_cost": 0.01,
            "metadata": {},
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    now = time_controller.now().timestamp()
    keyed_bucket = _expected_bucket_key(
        "grp", "tokens", "daily", "end_user_id", "u1", 86400, now, key_hash="keyA", limit=500000, scope_by_key_hash=True
    )
    unkeyed_bucket = _expected_bucket_key(
        "grp", "tokens", "daily", "end_user_id", "u1", 86400, now, key_hash=None, limit=500000, scope_by_key_hash=True
    )
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=keyed_bucket, litellm_parent_otel_span=None))
        == 42.0
    )
    assert await limiter.internal_usage_cache.async_get_cache(key=unkeyed_bucket, litellm_parent_otel_span=None) is None


@pytest.mark.asyncio
async def test_log_success_event_charges_the_window_admission_checked_not_a_later_one(time_controller):
    """
    Admission classifies its bucket as int(now) // period_seconds at filter
    time; success accounting used to recompute a fresh now of its own, so a
    call slow enough to cross a period_seconds boundary between admission and
    completion got admitted against one window's (still-open) counter but
    charged into the next window's fresh, empty one -- silently bypassing the
    limit for calls straddling each rollover. Success must charge the exact
    window admission classified against, not whatever window happens to be
    current when the response finishes.
    """
    token_limits = {
        "token_limits": {"limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 500, "period_seconds": 60}]}
    }
    router = litellm.Router(model_list=[_deployment("grp", "dep-1", token_limits)])
    limiter = _make_limiter(time_controller)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    request_kwargs, kwargs = _call_context(["end_user_id:u1"])

    admission_time = time_controller.now().timestamp()
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy

    # The response takes long enough to cross into the next 60s window before
    # completing.
    time_controller.advance(61)
    kwargs["standard_logging_object"] = {
        "model_group": "grp",
        "model_id": "dep-1",
        "total_tokens": 42,
        "response_cost": 0.01,
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    admitted_window_bucket = _expected_bucket_key(
        "grp", "tokens", "per_minute", "end_user_id", "u1", 60, admission_time, limit=500
    )
    later_window_bucket = _expected_bucket_key(
        "grp", "tokens", "per_minute", "end_user_id", "u1", 60, time_controller.now().timestamp(), limit=500
    )
    assert (
        float(
            await limiter.internal_usage_cache.async_get_cache(key=admitted_window_bucket, litellm_parent_otel_span=None)
        )
        == 42.0
    )
    assert (
        await limiter.internal_usage_cache.async_get_cache(key=later_window_bucket, litellm_parent_otel_span=None)
        is None
    )


@pytest.mark.asyncio
async def test_log_success_event_accounts_against_the_team_id_admission_checked(time_controller):
    """
    Admission resolves team_id via `_extract_team_id`, the single
    metadata_variable_name-authoritative field lookup -- success used to read
    `standard_logging_object.metadata.user_api_key_team_id` instead, a
    separately-constructed field that isn't guaranteed to come from the same
    field admission used (e.g. on LITELLM_METADATA_ROUTES, where
    `litellm_metadata` is authoritative but `standard_logging_object` may
    still reflect a different resolution). A mismatched team_id changes
    team_scope, which is hashed into the bucket key, so success would charge
    a different bucket than the one admission's team-aliased lookup checked.
    """
    deployment = _deployment(
        "real-model-name",
        "dep-1",
        {"token_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 500, "period_seconds": 86400}]}},
    )
    deployment["model_info"]["team_id"] = "team-1"
    deployment["model_info"]["team_public_model_name"] = "team-alias-name"
    router = litellm.Router(model_list=[deployment])
    limiter = _make_limiter(time_controller)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # LITELLM_METADATA_ROUTES shape: litellm_metadata is the authoritative
    # field, and team-alias resolution requires the real team_id from it.
    request_kwargs = {"litellm_metadata": {"tags": ["end_user_id:u1"], "user_api_key_team_id": "team-1"}}
    result = await limiter.async_filter_deployments(
        model="team-alias-name", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy

    # standard_logging_object's own team_id deliberately disagrees with the
    # real one in litellm_params.litellm_metadata, simulating litellm_logging.py
    # resolving a different field than the one admission used.
    kwargs = {
        "litellm_params": {"litellm_metadata": {"tags": ["end_user_id:u1"], "user_api_key_team_id": "team-1"}},
        "standard_logging_object": {
            "model_group": "team-alias-name",
            "model_id": "dep-1",
            "total_tokens": 42,
            "response_cost": 0.01,
            "metadata": {"user_api_key_team_id": None},
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    now = time_controller.now().timestamp()
    correct_bucket = _expected_bucket_key(
        "team-alias-name", "tokens", "daily", "end_user_id", "u1", 86400, now, team_scope="team-1", limit=500
    )
    wrong_bucket = _expected_bucket_key(
        "team-alias-name", "tokens", "daily", "end_user_id", "u1", 86400, now, limit=500
    )
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=correct_bucket, litellm_parent_otel_span=None))
        == 42.0
    )
    assert await limiter.internal_usage_cache.async_get_cache(key=wrong_bucket, litellm_parent_otel_span=None) is None


# ---------------------------------------------------------------------------
# concurrency limits -- reserve at admission, release on success/failure
# ---------------------------------------------------------------------------


def _concurrency_router(limit: int) -> "litellm.Router":
    return litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "concurrency_limits": {
                        "limits": [{"name": "inflight", "tag_id": "end_user_id", "limit": limit, "period_seconds": 300}]
                    }
                },
            )
        ]
    )


@pytest.mark.asyncio
async def test_cross_unit_rejection_does_not_leave_a_phantom_increment(time_controller):
    """
    Regression test: a chain with BOTH a requests limit and a concurrency
    limit on the same tag checks both atomically in one
    async_filter_deployments call. If the concurrency check rejects the hop,
    the requests-unit check (evaluated in the same call, and which would have
    been admitted on its own) must NOT have incremented its counter --
    otherwise a rejected hop silently burns through the caller's requests
    budget for a call that never actually went through.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 10, "period_seconds": 60}]
                    },
                    "concurrency_limits": {
                        "limits": [{"name": "inflight", "tag_id": "end_user_id", "limit": 1, "period_seconds": 300}]
                    },
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # Occupy the one concurrency slot.
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )

    # A second attempt: requests-unit alone would admit (well under 10), but
    # concurrency is exhausted, so the whole hop must reject -- and the
    # requests counter must remain untouched by this rejected attempt.
    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
    assert exc_info.value.detail["type"] == "concurrency"

    now = time_controller.now().timestamp()
    request_key = _expected_bucket_key("grp", "requests", "per_minute", "end_user_id", "u1", 60, now, limit=10)
    requests_value = await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None)
    assert (float(requests_value) if requests_value is not None else 0.0) == 1.0


@pytest.mark.asyncio
async def test_concurrency_limit_rejects_third_concurrent_reservation(time_controller):
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=2)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    kwargs_1 = {"metadata": {"tags": ["end_user_id:u1"]}}
    kwargs_2 = {"metadata": {"tags": ["end_user_id:u1"]}}
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs_1
    )
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs_2
    )

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
    assert exc_info.value.detail["type"] == "concurrency"


@pytest.mark.asyncio
async def test_requests_admission_is_race_free_under_genuine_concurrency(time_controller):
    """
    The concrete race a read-then-account-later design would allow: N
    coroutines all read "under limit" before any of them increments, and all
    N get admitted even past the limit. Firing many concurrent filter calls
    at once (asyncio.gather, not sequential awaits) must admit exactly
    `limit`, never more -- the atomic check-and-increment closes the window
    a plain GET-then-SET could not.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 5, "period_seconds": 60}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    async def attempt():
        try:
            await limiter.async_filter_deployments(
                model="grp",
                healthy_deployments=healthy,
                messages=None,
                request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
            )
            return True
        except ProxyRateLimitError:
            return False

    results = await asyncio.gather(*(attempt() for _ in range(20)))
    assert sum(results) == 5


@pytest.mark.asyncio
async def test_index_refreshes_after_ttl_for_length_preserving_update(time_controller):
    """
    Editing an existing deployment's tag_rate_limits in place (same
    len(model_list), so the (id(router), len) staleness check alone can't
    detect it) must eventually be picked up -- bounded by the index TTL,
    not indefinitely stale.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )

    # Same length, deployment mutated in place -- raise the limit to 100.
    router.model_list[0]["model_info"]["tag_rate_limits"] = {
        "request_limits": {
            "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 100, "period_seconds": 86400}]
        }
    }

    time_controller.advance(6)  # past _INDEX_TTL_SECONDS

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=router.model_list,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == router.model_list


@pytest.mark.asyncio
async def test_concurrency_slot_released_on_success_frees_capacity(time_controller):
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    request_kwargs, kwargs = _call_context(["end_user_id:u1"])
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )

    # At capacity: a second concurrent request is rejected.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )

    # The first request completes -- its slot is released -- freeing capacity again.
    kwargs["standard_logging_object"] = {
        "model_group": "grp",
        "model_id": "dep-1",
        "total_tokens": 0,
        "response_cost": 0,
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_background_release_tasks_registry_holds_a_reference_until_done():
    """
    async_log_success_event fires its release via a bare asyncio.create_task
    (unlike failure/disconnect, which await it directly) to keep the hot
    success path from waiting on a Redis round trip. asyncio.create_task's
    own docs warn the event loop only holds a *weak* reference to a task, so
    one with no other referrer can be garbage collected before it runs --
    and by the time it would run here, its keys are already popped out of
    model_call_details, so a collected task's release is unrecoverable, not
    merely delayed. _BACKGROUND_TASKS exists to hold a strong
    reference for exactly as long as the task is pending, then release it via
    the task's own done-callback -- exercised directly here (an Event gate
    gives a deterministic pending window; going through the real
    async_log_success_event doesn't, since its own further awaits let a fast
    in-memory release resolve before a test could ever observe it pending).
    """
    assert len(_BACKGROUND_TASKS) == 0
    gate = asyncio.Event()

    async def _pending_release():
        await gate.wait()

    task = asyncio.create_task(_pending_release())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    assert task in _BACKGROUND_TASKS

    gate.set()
    await task

    # The done-callback removes it -- the registry doesn't grow unbounded
    # across requests.
    assert task not in _BACKGROUND_TASKS
    assert len(_BACKGROUND_TASKS) == 0


@pytest.mark.asyncio
async def test_success_event_release_is_wired_through_the_background_registry(time_controller):
    """
    End-to-end check that async_log_success_event's fire-and-forget release
    is genuinely wired through _BACKGROUND_TASKS, not a bare
    unreferenced asyncio.create_task -- the registry must be empty again once
    the (fast, in-memory) release has had a chance to run, and the release
    itself must have actually happened.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    request_kwargs, kwargs = _call_context(["end_user_id:u1"])
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )

    kwargs["standard_logging_object"] = {
        "model_group": "grp",
        "model_id": "dep-1",
        "total_tokens": 0,
        "response_cost": 0,
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)

    # The registry was actually populated: proves the release ran through
    # _BACKGROUND_TASKS, not a bare unreferenced asyncio.create_task
    # (which would never touch this set at all, and an "empty at the end"
    # check alone can't tell the two apart -- an empty registry throughout
    # would satisfy that just as well as one that filled and drained).
    assert len(_BACKGROUND_TASKS) == 1

    # Two ticks: one for the release task itself to finish (it may already be
    # done by the time async_log_success_event returns, given that method's
    # own further awaits), and one for its done-callback -- scheduled via
    # call_soon when the task completes -- to actually run and discard it.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(_BACKGROUND_TASKS) == 0

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_success_event_token_accounting_is_wired_through_the_background_registry(time_controller):
    """
    Same gap as the concurrency release above, in a second fire-and-forget
    task on the same success path: token/dollar accounting is also fired
    via a bare asyncio.create_task per cache partition, with no strong
    reference of its own. A collected task here drops a usage increment
    that can never be recovered (the figures it needed only exist in that
    task's own closure), silently under-counting a caller's token/dollar
    usage against its configured limit. Must be tracked the same way.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "token_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_seconds": 86400}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)

    kwargs = {
        "metadata": {"tags": ["end_user_id:u1"]},
        "standard_logging_object": {
            "model_group": "grp",
            "model_id": "dep-1",
            "total_tokens": 42,
            "response_cost": 0.01,
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)

    # The registry was actually populated: proves accounting ran through
    # _BACKGROUND_TASKS, not a bare unreferenced asyncio.create_task.
    assert len(_BACKGROUND_TASKS) == 1

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(_BACKGROUND_TASKS) == 0

    now = time_controller.now().timestamp()
    token_key = _expected_bucket_key("grp", "tokens", "daily", "end_user_id", "u1", 86400, now, limit=500000)
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    )


@pytest.mark.asyncio
async def test_concurrency_slot_released_on_disconnect_frees_capacity(time_controller):
    """
    A client disconnecting before the first streamed chunk raises
    CancelledError/GeneratorExit, which bypasses both async_log_success_event
    and async_log_failure_event entirely -- neither fires, so the reservation
    would otherwise sit held until the safety-net TTL. The proxy's disconnect
    cleanup calls async_release_disconnect_state_hook instead in that case.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    request_kwargs, kwargs = _call_context(["end_user_id:u1"])
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )

    # At capacity: a second concurrent request is rejected.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )

    # The first request's client disconnects -- neither logging callback fires --
    # but the disconnect hook still releases its slot, freeing capacity again.
    await limiter.async_release_disconnect_state_hook(request_kwargs)

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_concurrency_slot_released_on_failure_frees_capacity(time_controller):
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    request_kwargs, kwargs = _call_context(["end_user_id:u1"])
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs=request_kwargs,
    )

    kwargs["standard_logging_object"] = {"model_group": "grp"}
    await limiter.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=0,
        end_time=0,
    )

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_concurrency_slot_released_by_post_call_failure_hook_on_the_final_fallback_hop(time_controller):
    """
    litellm's Logging object sets has_logged_async_failure=True after the
    first hop's failure and blocks async_log_failure_event for every later
    hop (see fallback_event_handlers.py), so a fallback chain's own final,
    chain-exhausting failure never reaches async_log_failure_event at all --
    _release_stale_hop_reservations only cleans up a stale reservation when
    a *next* hop's admission runs, and there is no next hop after the last
    one. async_post_call_failure_hook fires exactly once, at the point the
    proxy gives up and returns an error to the caller, regardless of how
    many hops ran or whether the completion-level callback was suppressed --
    it must release whatever reservation is still pending at that point.

    request_data here is a distinct dict object from admission's own
    request_kwargs, with no litellm_logging_obj at all: proxy/utils.py's
    post_call_failure_hook pops that key off request_data before invoking
    any callback ("Remove before callbacks iterate — not serialisable"),
    and confirmed live, request_data is a third, unrelated object from
    every hop's own request_kwargs by the time this fires. litellm_call_id
    is the only identifier stable across all of them.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # This hop's admission reserves the slot; its own failure is the chain's
    # final one, so async_log_failure_event never fires for it (simulating
    # litellm's has_logged_async_failure dedup blocking the callback here).
    request_kwargs = {
        "metadata": {"tags": ["end_user_id:u1"], "user_api_key": "hash"},
        "litellm_call_id": "call-final",
    }
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )

    await limiter.async_post_call_failure_hook(
        request_data={"litellm_call_id": "call-final"},
        original_exception=Exception("all deployments failed"),
        user_api_key_dict=UserAPIKeyAuth(api_key="hash"),
    )

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_post_call_failure_hook_cannot_release_a_different_keys_reservation(time_controller):
    """
    Security regression: litellm_call_id comes from the caller-controlled
    x-litellm-call-id header, so two different callers choosing the identical
    id must not be able to release each other's reservation through the
    pending-reservations cache mirror. Request A (key-a, tag victim_user) and
    request B (key-b, tag attacker_user) share one call_id; A's own terminal
    failure must only ever be able to find and release A's own mirror entry,
    keyed by A's server-authenticated key hash, never B's.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    victim_request_kwargs = {
        "metadata": {"tags": ["end_user_id:victim_user"], "user_api_key": "key-a-hash"},
        "litellm_call_id": "shared-call-id",
    }
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=victim_request_kwargs
    )

    attacker_request_kwargs = {
        "metadata": {"tags": ["end_user_id:attacker_user"], "user_api_key": "key-b-hash"},
        "litellm_call_id": "shared-call-id",
    }
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=attacker_request_kwargs
    )

    # Simulates request A's own fallback chain exhausting -- its terminal
    # failure hook must not touch request B's still-live reservation just
    # because both requests share a caller-chosen call_id.
    await limiter.async_post_call_failure_hook(
        request_data={"litellm_call_id": "shared-call-id"},
        original_exception=Exception("all deployments failed"),
        user_api_key_dict=UserAPIKeyAuth(api_key="key-a-hash"),
    )

    # attacker_user's own reservation must still be held: key-a's failure
    # hook releasing it would let key-b bypass its own concurrency cap.
    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:attacker_user"], "user_api_key": "key-b-hash"}},
        )
    assert exc_info.value.detail["type"] == "concurrency"

    # victim_user's own slot was correctly released by its own key's
    # failure hook -- the legitimate single-key path still works.
    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:victim_user"], "user_api_key": "key-a-hash"}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_concurrency_slot_released_when_a_different_hook_rejects_the_request(time_controller):
    """
    global_tag_rate_limits_hook raises the identical ProxyRateLimitError
    shape (detail["error"] == "tag_rate_limit_exceeded") this hook's own
    admission does. async_log_failure_event fires on every registered
    CustomLogger regardless of which one raised, so this hook must still
    release its own successfully reserved concurrency slot when the *other*
    hook is what rejected the request -- skipping release based on the
    shared marker alone would leak this hook's own slot until the safety
    TTL, even though nothing about this hook's own admission failed.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    request_kwargs, kwargs = _call_context(["end_user_id:u1"])
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs=request_kwargs,
    )

    other_hooks_rejection = ProxyRateLimitError(
        detail={"error": "tag_rate_limit_exceeded", "type": "requests", "tag_id": "end_user_id"},
        headers={"retry-after": "60"},
        rate_limit_type=None,
        model="grp",
        llm_provider="litellm_proxy",
    )
    kwargs["exception"] = other_hooks_rejection
    await limiter.async_log_failure_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_concurrency_slot_released_on_fallback_recovered_hop_failure(time_controller):
    """
    A hop that fails but gets recovered by a later fallback never reaches
    a terminal, request-level hook -- there is none, deliberately, since
    there's no reliable, caller-uncontrolled way to correlate multiple hops
    of one logical request from inside a CustomLogger hook (see
    async_log_failure_event's docstring for why litellm_call_id, the one
    candidate, can't be trusted for this). async_log_failure_event fires
    per hop, on every failure, recomputing this hop's own key independently,
    so this specific case -- exactly one prior failure, then a fallback that
    succeeds -- is still handled correctly.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    request_kwargs, kwargs = _call_context(["end_user_id:u1"])
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs=request_kwargs,
    )
    kwargs["standard_logging_object"] = {"model_group": "grp", "model_id": "dep-1"}
    await limiter.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=0,
        end_time=0,
    )

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_pending_concurrency_reservations_do_not_leak_across_unrelated_requests(time_controller):
    """
    Security regression test, current design: pending concurrency keys are
    stashed on the admitting request's own `model_call_details` dict (see
    `_PENDING_CONCURRENCY_KEYS_FIELD`'s docstring), never in a registry keyed
    by anything caller-visible or by ambient asyncio context. Two unrelated
    concurrent requests each get their own `model_call_details` in
    production, so one request's release can never see or drain a different
    request's still-pending reservation, regardless of which asyncio task
    each happens to run in and even when both share the identical tag value
    (an earlier design keyed reservations by `litellm_call_id` -- settable by
    the caller via the `x-litellm-call-id` header -- which let two unrelated
    requests merge reservations simply by choosing the same id).
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    request_a, kwargs_a = _call_context(["end_user_id:shared"])
    request_b, kwargs_b = _call_context(["end_user_id:shared"])

    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_a
    )
    # B shares A's tag value but is a genuinely separate request/object: at
    # capacity (limit=1), B is rejected and never reserves anything.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_b
        )

    # B's own failure event releases via its own (empty) model_call_details --
    # this must not accidentally drain A's still-live reservation.
    kwargs_b["standard_logging_object"] = {"model_group": "grp", "model_id": "dep-1"}
    await limiter.async_log_failure_event(kwargs=kwargs_b, response_obj=None, start_time=0, end_time=0)

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:shared"]}},
        )

    # A's own success event correctly releases its own reservation.
    kwargs_a["standard_logging_object"] = {
        "model_group": "grp",
        "model_id": "dep-1",
        "total_tokens": 0,
        "response_cost": 0,
    }
    await limiter.async_log_success_event(kwargs=kwargs_a, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:shared"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_concurrency_released_for_every_hop_across_a_real_task_boundary(time_controller):
    """
    litellm dedupes `async_log_failure_event` to fire once per logical
    request: only the first failed hop's failure reaches it (see
    `Logging.has_run_logging`'s `has_logged_async_failure` guard); a later
    failed hop (a retry or a further fallback) never gets its own failure
    event at all. Reservations still accumulate at admission for every hop
    regardless (onto the request's own `model_call_details`, shared across
    every hop of one logical request -- see `_PENDING_CONCURRENCY_KEYS_FIELD`'s
    docstring), so whichever event fires next must release everything
    accumulated since the last release, not just its own hop's key.

    Hop 3's eventual success is fired as a child task of the same admission
    chain -- exactly like litellm's real dispatch, where `wrapper_async`
    create_task's the success path and `LoggingWorker.enqueue` explicitly
    propagates the calling context -- to prove the fix survives the actual
    task boundary a real success event crosses in production, not just a
    same-coroutine call that would pass regardless of whether the pending
    keys lived on a real shared object or an ordinary per-task variable.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=2)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    async def _one_logical_request():
        # All three hops of this one logical request share the same
        # model_call_details, exactly as real fallback hops share one
        # Logging object -- only litellm_call_id differs per hop.
        request_kwargs, kwargs = _call_context(["end_user_id:u1"])

        # Hop 1 admits and fails; its failure event is the one that fires
        # (dedup allows exactly the first failure through), releasing its
        # own key immediately.
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs=request_kwargs,
        )
        kwargs["standard_logging_object"] = {"model_group": "grp"}
        await limiter.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=0,
            end_time=0,
        )

        # Hop 2 (a retry or fallback) admits and also fails, but -- per
        # litellm's dedup -- no async_log_failure_event call follows it.
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs=request_kwargs,
        )

        # Hop 3 admits and succeeds. Its success event, dispatched as a
        # child task (mirroring the real worker hop), must release both
        # hop 2's still-pending reservation and its own.
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs=request_kwargs,
        )

        async def _hop_3_success_event():
            kwargs["standard_logging_object"] = {
                "model_group": "grp",
                "model_id": "dep-1",
                "total_tokens": 0,
                "response_cost": 0,
            }
            await limiter.async_log_success_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=0,
                end_time=0,
            )

        await asyncio.create_task(_hop_3_success_event())

    await asyncio.create_task(_one_logical_request())
    await asyncio.sleep(0)

    # Full capacity (2) is free again -- both hop 2's leaked reservation and
    # hop 3's own were released. If the earlier hop's leaked reservation
    # hadn't been released too, only one of these two admissions would succeed.
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_next_hops_admission_releases_a_prior_hops_leaked_reservation(time_controller):
    """
    Regression test for a leak that a success/failure-event-only release
    strategy can never close: litellm's has_logged_async_failure dedup lets
    exactly one hop's async_log_failure_event fire per logical request (see
    test_concurrency_released_for_every_hop_across_a_real_task_boundary), so
    a hop that fails *after* that one event has already fired gets no
    failure event of its own at all -- not "delayed until the next event",
    genuinely never. Only the next hop's own admission call is guaranteed to
    run afterward, so release must happen there, not wait for some later
    success/failure event that this specific hop will never get.

    Concurrency limit of 1 makes this observable directly: if hop 2's
    admission doesn't release hop 1's leaked reservation before checking its
    own, it raises ProxyRateLimitError against a bucket that's actually free.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    request_kwargs, _kwargs = _call_context(["end_user_id:u1"])

    # Hop 1 admits (the only slot) and then fails with no failure event ever
    # following it -- simulating every hop after litellm's one dedup-allowed
    # failure event has already fired for an earlier hop of this request.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy

    # Hop 2's own admission call must release hop 1's stale reservation
    # before checking its own -- if it didn't, this raises ProxyRateLimitError.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy


def _request_limit_router(limit: int) -> "litellm.Router":
    return litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_period", "tag_id": "end_user_id", "limit": limit, "period_seconds": 300}]
                    }
                },
            )
        ]
    )


@pytest.mark.asyncio
async def test_next_hops_admission_refunds_a_prior_failed_hops_request_increment(time_controller):
    """
    Regression test for Cursor Bugbot's "fallback hops burn request budget"
    finding on PR #36541, live-confirmed against a real proxy: a "requests"
    limit is meant to cap logical client requests, not internal routing
    attempts, but without a refund a chain that fails once before succeeding
    burned 2 units of a 1-request-per-period budget for one logical call --
    live reproduction showed the retry's own admission rejected with
    current=1.0 limit=1.0 even though the client only made one call.

    Concurrency's next-hop-releases-the-prior-hop's-stale-reservation pattern
    (see test_next_hops_admission_releases_a_prior_hops_leaked_reservation)
    generalizes cleanly here: since Router only re-enters admission for a hop
    that already failed, the prior hop's own "requests" increment must be
    refunded there too, before this hop's own check runs.
    """
    limiter = _make_limiter(time_controller)
    router = _request_limit_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    request_kwargs, _kwargs = _call_context(["end_user_id:u1"])

    # Hop 1 admits (the only unit) then fails -- no failure event follows,
    # mirroring the "already consumed litellm's one dedup-allowed failure
    # event" scenario the sibling concurrency test documents.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy

    # Hop 2's own admission must refund hop 1's now-stale "requests"
    # increment before checking its own -- if it didn't, this raises
    # ProxyRateLimitError against a bucket a real client only asked to use
    # once.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_next_hops_admission_refunds_a_request_increment_even_after_the_first_hops_own_failure_event_fires(
    time_controller,
):
    """
    Tighter regression than the test above: this reproduces the exact live
    failure this fix first shipped with. litellm's has_logged_async_failure
    dedup allows exactly the *first* failing hop's own async_log_failure_event
    through -- unlike a hop after that one, hop 1 here genuinely gets a real
    failure event, not silence. An earlier version of this fix popped
    _PENDING_REQUEST_INCREMENTS_FIELD in async_log_failure_event "for
    hygiene", discarding hop 1's entry before hop 2's own admission
    (_release_stale_hop_reservations) ever got a chance to refund it --
    silently and permanently stranding the charge, so hop 2 was rejected
    against a bucket a real client only asked to use once, live-confirmed
    against a real proxy. async_log_failure_event must leave this field
    completely untouched.
    """
    limiter = _make_limiter(time_controller)
    router = _request_limit_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    request_kwargs, kwargs = _call_context(["end_user_id:u1"])

    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy

    # Hop 1's own, real failure event -- the one has_logged_async_failure
    # lets through.
    await limiter.async_log_failure_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)

    # Hop 2's own admission must still refund hop 1's now-stale "requests"
    # increment before checking its own.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_successful_hops_own_request_increment_is_not_refunded(time_controller):
    """
    The fix above must not swing the other way and refund every hop's
    "requests" increment unconditionally -- exactly one unit must survive
    per logical request, or the limit stops limiting anything. Simulates the
    full lifecycle (admission, then the success event a real request would
    fire) and confirms a second, unrelated logical request against the same
    tag is correctly rejected: the first request's own successful hop
    already spent the only unit for this period.
    """
    limiter = _make_limiter(time_controller)
    router = _request_limit_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    request_kwargs, kwargs = _call_context(["end_user_id:u1"])

    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)

    fresh_request_kwargs, _fresh_kwargs = _call_context(["end_user_id:u1"])
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs=fresh_request_kwargs
        )


@pytest.mark.asyncio
async def test_a_hops_own_rejection_on_a_different_check_does_not_undercount_the_prior_hops_request_charge(
    time_controller,
):
    """
    Regression test for Cursor Bugbot's follow-up finding on this exact fix:
    an earlier version refunded the prior hop's "requests" charge
    unconditionally at the top of the next hop's admission, before knowing
    whether that next hop would itself be admitted. If the next hop then
    failed a *different* check (here, concurrency) before ever reaching its
    own requests renewal, the refund had already committed with nothing to
    replace it -- a logical request that genuinely made one real attempt
    (hop 1) would end up charged zero, letting a caller bypass the requests
    cap simply by having a later hop collide with someone else's
    concurrency slot.

    Fixed by folding the renewal into the same all-or-nothing atomic batch
    as every other check on that hop: a "requests" key matching an earlier
    hop's charge renews at zero net cost instead of being refunded first,
    so a batch-wide rollback (concurrency's own rejection here) refunds that
    zero-cost renewal -- a genuine no-op -- leaving hop 1's real charge
    exactly as it was.
    """
    # Two independent tag identities: "requests" is scoped to end_user_id
    # (private to our own request, never shared with the unrelated
    # contender below), "concurrency" is scoped to a separate shared_pool
    # tag that both our request and the unrelated contender carry, so they
    # compete for the same slot without also colliding on the requests cap.
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_period", "tag_id": "end_user_id", "limit": 1, "period_seconds": 300}]
                    },
                    "concurrency_limits": {
                        "limits": [{"name": "inflight", "tag_id": "shared_pool", "limit": 1, "period_seconds": 300}]
                    },
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # Hop 1 of our request admits (claiming both the requests unit and the
    # only concurrency slot), then fails for real -- its concurrency
    # reservation is released the normal way, but its requests charge is
    # left queued as this-hop's-own-charge, not refunded.
    request_kwargs, kwargs = _call_context(["end_user_id:u1", "shared_pool:pool-a"])
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    await limiter.async_log_failure_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)

    # A second, unrelated request -- no end_user_id tag at all, so it never
    # touches the requests bucket -- now claims the concurrency slot our
    # hop 1 just released, and holds it.
    other_request_kwargs, other_kwargs = _call_context(["shared_pool:pool-a"])
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=other_request_kwargs
    )

    # Hop 2 of our original request: its own "requests" renewal would
    # trivially succeed alone (net zero cost), but the concurrency slot is
    # now held by the unrelated request above, so the whole atomic batch
    # must reject -- and must NOT leave hop 1's requests charge refunded.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
        )

    # The unrelated request finishes, freeing the concurrency slot again.
    await limiter.async_log_success_event(kwargs=other_kwargs, response_obj=None, start_time=0, end_time=0)

    # A fresh probe against the same end_user_id tag, with concurrency now
    # free, must still be rejected by the requests cap: hop 1's real attempt
    # already spent the only unit for this period, and it must not have
    # been silently erased by hop 2's unrelated, different-check rejection.
    probe_request_kwargs, _probe_kwargs = _call_context(["end_user_id:u1"])
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs=probe_request_kwargs
        )


@pytest.mark.asyncio
async def test_own_rejection_does_not_release_a_live_reservation(time_controller):
    """
    Security regression test: Router.async_callback_filter_deployments fires
    async_log_failure_event for an exception raised from inside
    async_filter_deployments itself (its own except block calls
    logging_obj.async_failure_handler before re-raising) -- not only for an
    actual provider-call failure. A rejection this hook raises for being
    over its own limit never reserved anything for that specific attempt
    (_atomic_check_and_increment already refunds any of its own earlier
    admissions synchronously whenever it rejects), so releasing anyway would
    decrement a live reservation belonging to a different, genuinely
    in-flight request sharing the same tag -- letting a caller free up
    capacity simply by retrying against an already-full bucket, no
    coordination with another request required.

    The holder and the rejected attempt are modeled as two separate tasks
    (matching how two independent real requests are always isolated in
    production, each in its own asyncio task) so this actually exercises the
    explicit ProxyRateLimitError guard rather than the ContextVar's own
    per-task isolation, which would otherwise mask the same bug: two
    admissions made directly in one shared coroutine would (correctly, but
    for the wrong reason) never be able to explain away the bug this test is
    for.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # One legitimate request, in its own task, holds the only slot.
    async def _admit():
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )

    await asyncio.create_task(_admit())

    # A second, unrelated request (its own task) is rejected, and Router's
    # own exception handling fires async_log_failure_event for it, exactly
    # as Router.async_callback_filter_deployments does.
    async def _reject_and_fire_failure_event():
        with pytest.raises(ProxyRateLimitError) as exc_info:
            await limiter.async_filter_deployments(
                model="grp",
                healthy_deployments=healthy,
                messages=None,
                request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
            )
        await limiter.async_log_failure_event(
            kwargs={
                "exception": exc_info.value,
                "standard_logging_object": {"model_group": "grp"},
                "metadata": {"tags": ["end_user_id:u1"]},
            },
            response_obj=None,
            start_time=0,
            end_time=0,
        )

    await asyncio.create_task(_reject_and_fire_failure_event())

    # The first request's reservation must still be held: a third attempt is
    # still rejected. If the rejection's failure event had wrongly released
    # it, this would wrongly admit instead.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )


# ---------------------------------------------------------------------------
# tokens / dollars -- read-then-account-on-success rejection path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_limit_rejects_once_bucket_is_seeded_at_limit(time_controller):
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "token_limits": {
                        "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1000, "period_seconds": 86400}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    now = time_controller.now().timestamp()
    key = _expected_bucket_key("grp", "tokens", "daily", "end_user_id", "u1", 86400, now, limit=1000)
    await limiter.internal_usage_cache.async_set_cache(key=key, value=1000, ttl=86400, litellm_parent_otel_span=None)

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
    assert exc_info.value.detail["type"] == "tokens"
    assert exc_info.value.detail["limit_name"] == "daily"


@pytest.mark.asyncio
async def test_dollar_limit_rejects_once_bucket_is_seeded_at_limit(time_controller):
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "dollar_limits": {
                        "limits": [{"name": "monthly", "tag_id": "team_id", "limit": 50.0, "period_seconds": 2592000}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    now = time_controller.now().timestamp()
    key = _expected_bucket_key("grp", "dollars", "monthly", "team_id", "t1", 2592000, now, limit=50.0)
    await limiter.internal_usage_cache.async_set_cache(key=key, value=50.0, ttl=2592000, litellm_parent_otel_span=None)

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["team_id:t1"]}},
        )
    assert exc_info.value.detail["type"] == "dollars"
    assert exc_info.value.detail["tag_value"] == "t1"


# ---------------------------------------------------------------------------
# Real Redis -- the atomic Lua path, not just the in-memory fallback
# ---------------------------------------------------------------------------


def _redis_limiter(time_controller: TimeController):
    import os

    from litellm.caching.redis_cache import RedisCache

    redis_host = os.getenv("REDIS_HOST")
    redis_port = os.getenv("REDIS_PORT")
    if not redis_host or not redis_port:
        pytest.skip("Redis environment variables (REDIS_HOST, REDIS_PORT) not set")
    redis_cache = RedisCache(host=redis_host, port=int(redis_port), password=os.getenv("REDIS_PASSWORD"))
    dual_cache = DualCache(redis_cache=redis_cache)
    return _PROXY_ModelBasedTagRateLimitsHook(internal_usage_cache=dual_cache, time_provider=time_controller.now), redis_cache


@pytest.mark.asyncio
async def test_redis_backed_requests_admission_is_race_free_under_genuine_concurrency(time_controller):
    """
    Same race-freedom guarantee as the in-memory test, but against a real
    Redis instance so the Lua script path (not just the asyncio.Lock
    fallback) is exercised -- this is the code path every multi-instance
    proxy deployment actually runs.
    """
    limiter, redis_cache = _redis_limiter(time_controller)
    try:
        await redis_cache.ping()
    except Exception as e:
        pytest.skip(f"Redis connection failed: {e!s}")

    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 5, "period_seconds": 60}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    tag = f"redis-race-{uuid.uuid4().hex}"

    async def attempt():
        try:
            await limiter.async_filter_deployments(
                model="grp",
                healthy_deployments=healthy,
                messages=None,
                request_kwargs={"metadata": {"tags": [f"end_user_id:{tag}"]}},
            )
            return True
        except ProxyRateLimitError:
            return False

    results = await asyncio.gather(*(attempt() for _ in range(20)))
    assert sum(results) == 5


@pytest.mark.asyncio
async def test_redis_backed_cross_unit_rejection_does_not_leave_a_phantom_increment(time_controller):
    """Redis-Lua-script equivalent of the in-memory phantom-increment regression test."""
    limiter, redis_cache = _redis_limiter(time_controller)
    try:
        await redis_cache.ping()
    except Exception as e:
        pytest.skip(f"Redis connection failed: {e!s}")

    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 10, "period_seconds": 60}]
                    },
                    "concurrency_limits": {
                        "limits": [{"name": "inflight", "tag_id": "end_user_id", "limit": 1, "period_seconds": 300}]
                    },
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    tag = f"redis-phantom-check-{uuid.uuid4().hex}"

    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": [f"end_user_id:{tag}"]}},
    )
    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": [f"end_user_id:{tag}"]}},
        )
    assert exc_info.value.detail["type"] == "concurrency"

    now = time_controller.now().timestamp()
    request_key = _expected_bucket_key("grp", "requests", "per_minute", "end_user_id", tag, 60, now, limit=10)
    requests_value = await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None)
    assert (float(requests_value) if requests_value is not None else 0.0) == 1.0

    # cleanup: this key persists in the shared scratch Redis instance beyond the test's TTL otherwise
    await redis_cache.async_delete_cache(key=request_key)


@pytest.mark.asyncio
async def test_redis_backed_token_admission_sees_increments_the_in_memory_cache_missed(time_controller):
    """
    Success accounting increments a token bucket straight through a Lua
    script on redis_cache, bypassing DualCache/InternalUsageCache entirely --
    that write never touches the in-memory layer. Once an earlier read has
    backfilled that same key into the in-memory cache, DualCache's own
    async_batch_get_cache treats that non-None in-memory hit as authoritative
    and never re-checks Redis, so every later admission would see the same
    frozen snapshot while the real Redis counter keeps climbing underneath
    it, silently admitting traffic well past the configured token limit.
    """
    limiter, redis_cache = _redis_limiter(time_controller)
    try:
        await redis_cache.ping()
    except Exception as e:
        pytest.skip(f"Redis connection failed: {e!s}")

    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {"token_limits": {"limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 100, "period_seconds": 60}]}},
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    tag = f"redis-stale-check-{uuid.uuid4().hex}"
    request_kwargs = {"metadata": {"tags": [f"end_user_id:{tag}"]}}

    async def _charge(tokens: float) -> None:
        await limiter.async_log_success_event(
            kwargs={
                "metadata": {"tags": [f"end_user_id:{tag}"]},
                "standard_logging_object": {
                    "model_group": "grp",
                    "model_id": "dep-1",
                    "total_tokens": tokens,
                    "response_cost": 0,
                },
            },
            response_obj=None,
            start_time=0,
            end_time=0,
        )
        # The actual Redis increment is dispatched as a background task (see
        # _BACKGROUND_TASKS), so it needs a beat to actually run.
        await asyncio.sleep(0.05)

    # First admission: bucket doesn't exist in Redis yet, so this read finds
    # nothing to backfill into the in-memory cache either.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy
    await _charge(90)

    # Second admission: this read is the one that backfills the in-memory
    # cache with the real (90) value read from Redis.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy
    await _charge(90)  # real Redis total is now 180, well past the limit of 100

    # Third admission must see the real (180) total and reject -- not the
    # frozen 90 the in-memory cache captured on the previous read.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
        )

    now = time_controller.now().timestamp()
    token_key = _expected_bucket_key("grp", "tokens", "per_minute", "end_user_id", tag, 60, now)
    await redis_cache.async_delete_cache(key=token_key)


# ---------------------------------------------------------------------------
# team_public_model_name alias -- index lookup must not miss
# ---------------------------------------------------------------------------


def test_build_limits_index_is_also_keyed_by_team_public_model_name():
    """
    Router threads a team's public alias, not the deployment's own
    model_name, into async_filter_deployments's `model` param when a caller
    requests via that alias (Router never rewrites it for this path, unlike
    model_group_alias). The index must resolve either name to the same
    configured limits, or a team-aliased chain's limits are silently never
    checked.
    """
    deployment = _deployment(
        "real-model-name",
        "dep-1",
        {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}},
    )
    deployment["model_info"]["team_id"] = "team-1"
    deployment["model_info"]["team_public_model_name"] = "team-alias-name"
    index = _build_limits_index([deployment])
    by_name = index.resolve("real-model-name", team_id=None)
    by_alias = index.resolve("team-alias-name", team_id="team-1")
    assert by_name != ()
    assert [c.entry for c in by_name] == [c.entry for c in by_alias]
    # The alias resolution must carry the team_id into the bucket scope --
    # see test_build_limits_index_keeps_different_teams_same_alias_separate
    # for why (two teams can publish the identical alias string).
    assert by_name[0].team_scope is None
    assert by_alias[0].team_scope == "team-1"


def test_build_limits_index_preserves_key_ttl_seconds_and_max_in_memory_cache_size():
    """
    Regression test: _configured_limit_for_signature used to reconstruct a
    fresh TagRateLimitEntry from a 5-field dedup signature that didn't
    include key_ttl_seconds or max_in_memory_cache_size, silently resetting
    both to None for every entry that went through the real indexing path
    (which is every entry reachable from async_filter_deployments /
    async_log_success_event) -- only entries built directly in a test, never
    through _build_limits_index, kept their configured values.
    """
    deployment = _deployment(
        "grp",
        "dep-1",
        {
            "request_limits": {
                "limits": [
                    {
                        "name": "user_cap",
                        "tag_id": "end_user_id",
                        "limit": 5,
                        "period_seconds": 60,
                        "key_ttl_seconds": 120,
                        "max_in_memory_cache_size": 500,
                    }
                ]
            }
        },
    )
    index = _build_limits_index([deployment])
    configured = index.resolve("grp", team_id=None)
    assert len(configured) == 1
    assert configured[0].entry.key_ttl_seconds == 120
    assert configured[0].entry.max_in_memory_cache_size == 500


def test_build_limits_index_treats_a_duplicated_entry_on_one_deployment_as_chain_wide():
    """
    Regression test: a single deployment declaring the identical
    concurrency_limits entry twice (a config duplicate) used to append that
    deployment's id twice, inflating len(declaring_ids) past
    total_deployments. That made is_chain_wide false even though every
    deployment (there's only one) actually agreed on the entry, and for
    concurrency a non-chain-wide entry is silently dropped entirely --
    disabling enforcement rather than degrading it.
    """
    deployment = _deployment(
        "grp",
        "dep-1",
        {
            "concurrency_limits": {
                "limits": [
                    {"name": "inflight", "tag_id": "end_user_id", "limit": 5, "period_seconds": 300},
                    {"name": "inflight", "tag_id": "end_user_id", "limit": 5, "period_seconds": 300},
                ]
            }
        },
    )
    index = _build_limits_index([deployment])
    configured = index.resolve("grp", team_id=None)
    assert len(configured) == 1
    assert configured[0].deployment_scope is None  # chain-wide, not dropped


def test_build_limits_index_keeps_different_teams_same_alias_separate():
    """
    `team_public_model_name` is only unique per team: Router itself lets two
    different teams publish the identical alias string for different
    deployments, resolving each caller's own team's deployment by
    `(team_id, name)` rather than by name alone. Keying the limits index by
    name alone would let one team's config silently overwrite another's.
    """
    team_a = _deployment(
        "model-a", "dep-a", {"token_limits": {"limits": [{"name": "daily", "limit": 100, "period_seconds": 86400}]}}
    )
    team_a["model_info"]["team_id"] = "team-a"
    team_a["model_info"]["team_public_model_name"] = "shared-alias"

    team_b = _deployment(
        "model-b", "dep-b", {"token_limits": {"limits": [{"name": "daily", "limit": 999, "period_seconds": 86400}]}}
    )
    team_b["model_info"]["team_id"] = "team-b"
    team_b["model_info"]["team_public_model_name"] = "shared-alias"

    index = _build_limits_index([team_a, team_b])
    resolved_a = index.resolve("shared-alias", team_id="team-a")
    resolved_b = index.resolve("shared-alias", team_id="team-b")
    assert resolved_a[0].entry.limit == 100
    assert resolved_b[0].entry.limit == 999


def test_bucket_key_differs_across_teams_sharing_an_alias_and_identical_limit_config():
    """
    Two teams that happen to publish the identical team_public_model_name
    AND configure an identically-named, identically-valued limit must not
    land on the same Redis bucket -- team_public_model_name is only unique
    per team, so this is a realistic collision, not a contrived one.
    """
    team_a = _deployment(
        "model-a", "dep-a", {"request_limits": {"limits": [{"name": "per_minute", "limit": 5, "period_seconds": 60}]}}
    )
    team_a["model_info"]["team_id"] = "team-a"
    team_a["model_info"]["team_public_model_name"] = "shared-alias"

    team_b = _deployment(
        "model-b", "dep-b", {"request_limits": {"limits": [{"name": "per_minute", "limit": 5, "period_seconds": 60}]}}
    )
    team_b["model_info"]["team_id"] = "team-b"
    team_b["model_info"]["team_public_model_name"] = "shared-alias"

    index = _build_limits_index([team_a, team_b])
    limit_a = index.resolve("shared-alias", team_id="team-a")[0]
    limit_b = index.resolve("shared-alias", team_id="team-b")[0]
    assert limit_a.entry == limit_b.entry  # identical configuration, by construction

    key_a = _bucket_key("shared-alias", limit_a, tag_value="same-caller-tag", bucket_id=0)
    key_b = _bucket_key("shared-alias", limit_b, tag_value="same-caller-tag", bucket_id=0)
    assert key_a != key_b

    inflight_a = _inflight_key("shared-alias", limit_a, tag_value="same-caller-tag")
    inflight_b = _inflight_key("shared-alias", limit_b, tag_value="same-caller-tag")
    assert inflight_a != inflight_b


def test_build_limits_index_merges_alias_limits_across_different_model_names():
    """
    litellm auto-generates each team-added deployment's own internal
    model_name as model_name_{team_id}_{uuid}, so multiple deployments
    sharing one team_public_model_name alias routinely have different
    model_name values -- Router's own team_model_to_deployment_indices
    aggregates them by (team_id, alias) regardless of that. Computing alias
    limits once per model_name group and keying the alias to whichever
    group happened to declare it would silently drop every other same-alias
    group's limits: with two deployments under different model_names but
    the same alias, only the entry declared by whichever model_name group
    is processed last would survive.
    """
    dep_a = _deployment(
        "model_name_team1_aaa",
        "dep-a",
        {"token_limits": {"limits": [{"name": "daily", "limit": 100, "period_seconds": 86400}]}},
    )
    dep_a["model_info"]["team_id"] = "team-1"
    dep_a["model_info"]["team_public_model_name"] = "shared-alias"

    dep_b = _deployment(
        "model_name_team1_bbb",
        "dep-b",
        {"dollar_limits": {"limits": [{"name": "monthly", "limit": 50.0, "period_seconds": 2592000}]}},
    )
    dep_b["model_info"]["team_id"] = "team-1"
    dep_b["model_info"]["team_public_model_name"] = "shared-alias"

    index = _build_limits_index([dep_a, dep_b])
    resolved = index.resolve("shared-alias", team_id="team-1")
    units = {c.unit for c in resolved}
    assert units == {"tokens", "dollars"}


@pytest.mark.asyncio
async def test_filter_deployments_enforces_limit_when_called_with_team_alias(time_controller):
    limiter = _make_limiter(time_controller)
    deployment = _deployment(
        "real-model-name",
        "dep-1",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    deployment["model_info"]["team_id"] = "team-1"
    deployment["model_info"]["team_public_model_name"] = "team-alias-name"
    router = litellm.Router(model_list=[deployment])
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # Router passes the alias as `model`, not "real-model-name", and threads
    # the caller's team_id through request metadata.
    request_kwargs = {"metadata": {"tags": ["end_user_id:u1"], "user_api_key_team_id": "team-1"}}
    await limiter.async_filter_deployments(
        model="team-alias-name", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="team-alias-name", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
        )


@pytest.mark.asyncio
async def test_filter_deployments_does_not_cross_team_alias_boundary(time_controller):
    """
    Two teams sharing the same team_public_model_name must not share a
    counter: a caller on team-b hitting the alias must not be limited (or
    counted) by team-a's configured limit and usage.
    """
    limiter = _make_limiter(time_controller)
    team_a = _deployment(
        "model-a",
        "dep-a",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    team_a["model_info"]["team_id"] = "team-a"
    team_a["model_info"]["team_public_model_name"] = "shared-alias"

    team_b = _deployment(
        "model-b",
        "dep-b",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 5, "period_seconds": 86400}]
            }
        },
    )
    team_b["model_info"]["team_id"] = "team-b"
    team_b["model_info"]["team_public_model_name"] = "shared-alias"

    router = litellm.Router(model_list=[team_a, team_b])
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # team-a exhausts its own limit of 1.
    await limiter.async_filter_deployments(
        model="shared-alias",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key_team_id": "team-a"}},
    )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="shared-alias",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key_team_id": "team-a"}},
        )

    # team-b, same alias string and same tag value, is unaffected by team-a's exhausted limit.
    await limiter.async_filter_deployments(
        model="shared-alias",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key_team_id": "team-b"}},
    )


# ---------------------------------------------------------------------------
# concurrency_limits -- chain-wide only, divergent config dropped not leaked
# ---------------------------------------------------------------------------


def test_concurrency_divergent_config_is_dropped_not_scoped_per_deployment():
    """
    Unlike tokens/requests/dollars, a concurrency entry declared with
    different values per deployment must not become a per-deployment-scoped
    reservation -- that shape leaks (see the regression tests below this
    plan superseded). It should be dropped entirely, with the group left
    with no concurrency entry at all.
    """
    deployments = [
        _deployment(
            "grp", "dep-1", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 2, "period_seconds": 60}]}}
        ),
        _deployment(
            "grp", "dep-2", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 5, "period_seconds": 60}]}}
        ),
    ]
    configured = _build_group_limits(deployments, "concurrency")
    assert configured == ()


def test_concurrency_partial_declaration_is_dropped_not_scoped_per_deployment():
    deployments = [
        _deployment(
            "grp", "dep-1", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 2, "period_seconds": 60}]}}
        ),
        _deployment("grp", "dep-2", {}),
    ]
    configured = _build_group_limits(deployments, "concurrency")
    assert configured == ()


def test_concurrency_identical_across_all_deployments_is_still_chain_wide():
    deployments = [
        _deployment(
            "grp", "dep-1", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 2, "period_seconds": 60}]}}
        ),
        _deployment(
            "grp", "dep-2", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 2, "period_seconds": 60}]}}
        ),
    ]
    configured = _build_group_limits(deployments, "concurrency")
    assert len(configured) == 1
    assert configured[0].deployment_scope is None


# ---------------------------------------------------------------------------
# concurrency TTL floor -- a short period_seconds must not shorten the
# self-heal safety TTL below the floor
# ---------------------------------------------------------------------------


def test_concurrency_ttl_floor_overrides_a_too_short_period_seconds():
    entry = TagRateLimitEntry(name="inflight", tag_id="end_user_id", limit=1, period_seconds=5)
    configured_limit = _ConfiguredLimit(unit="concurrency", entry=entry, deployment_scope=None)
    assert _PROXY_ModelBasedTagRateLimitsHook._ttl_for(configured_limit) == _CONCURRENCY_MIN_SAFETY_TTL_SECONDS


def test_concurrency_ttl_floor_does_not_shorten_a_longer_period_seconds():
    entry = TagRateLimitEntry(
        name="inflight", tag_id="end_user_id", limit=1, period_seconds=_CONCURRENCY_MIN_SAFETY_TTL_SECONDS + 100
    )
    configured_limit = _ConfiguredLimit(unit="concurrency", entry=entry, deployment_scope=None)
    assert _PROXY_ModelBasedTagRateLimitsHook._ttl_for(configured_limit) == _CONCURRENCY_MIN_SAFETY_TTL_SECONDS + 100


# ---------------------------------------------------------------------------
# pending-concurrency-key field on model_call_details must survive a detached
# asyncio.create_task fork (e.g. litellm's own failure-logging dispatch),
# and a release must never sweep up a key a still-live sibling hop appended
# in the meantime. This dict-on-a-shared-object design is what replaced a
# contextvars.ContextVar-based holder that silently failed to release
# anything once release ran in a task that wasn't a descendant of admission's
# own task -- exactly what happens in the real proxy request pipeline (see
# _PENDING_CONCURRENCY_KEYS_FIELD's docstring).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_in_a_forked_task_is_visible_to_the_parent_context(time_controller):
    limiter = _make_limiter(time_controller)
    model_call_details: dict = {_PENDING_CONCURRENCY_KEYS_FIELD: ["key1"]}

    async def detached_release():
        return await limiter._pop_pending_concurrency_keys(model_call_details)

    released = await asyncio.create_task(detached_release())
    assert released == ("key1",)

    # The parent's own view of the same dict must see the release too.
    assert model_call_details[_PENDING_CONCURRENCY_KEYS_FIELD] == []


@pytest.mark.asyncio
async def test_release_does_not_sweep_up_a_key_appended_after_its_snapshot(time_controller):
    limiter = _make_limiter(time_controller)
    model_call_details: dict = {_PENDING_CONCURRENCY_KEYS_FIELD: ["key1"]}

    async def detached_release_then_sibling_admits():
        released = await limiter._pop_pending_concurrency_keys(model_call_details)
        # A sibling hop's admission, appending to the same shared dict,
        # interleaved right after this release's snapshot was taken.
        model_call_details[_PENDING_CONCURRENCY_KEYS_FIELD].append("key2")
        return released

    released = await asyncio.create_task(detached_release_then_sibling_admits())
    assert released == ("key1",)
    # key2 must still be pending for its own hop's eventual release.
    assert model_call_details[_PENDING_CONCURRENCY_KEYS_FIELD] == ["key2"]


@pytest.mark.asyncio
async def test_release_is_not_repeated_for_the_same_snapshot(time_controller):
    limiter = _make_limiter(time_controller)
    model_call_details: dict = {_PENDING_CONCURRENCY_KEYS_FIELD: ["key1"]}
    first = await limiter._pop_pending_concurrency_keys(model_call_details)
    second = await limiter._pop_pending_concurrency_keys(model_call_details)
    assert first == ("key1",)
    assert second == ()


# ---------------------------------------------------------------------------
# refund-on-rollback across differently-hash-tagged keys (Redis Cluster safety)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_unit_refund_leaves_no_phantom_increment_in_memory(time_controller):
    """
    In-memory equivalent of the Redis Cluster cross-slot fix: requests and
    concurrency keys carry different hash tags by construction, so the
    all-or-nothing guarantee across them must come from a refund, not a
    single multi-key atomic call. Confirms the refund path itself (not just
    the end observable behavior already covered by
    test_cross_unit_rejection_does_not_leave_a_phantom_increment).
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 10, "period_seconds": 60}]
                    },
                    "concurrency_limits": {
                        "limits": [{"name": "inflight", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
                    },
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:refund-check"]}},
    )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:refund-check"]}},
        )

    now = time_controller.now().timestamp()
    request_key = _expected_bucket_key("grp", "requests", "per_minute", "end_user_id", "refund-check", 60, now, limit=10)
    value = await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None)
    assert (float(value) if value is not None else 0.0) == 1.0


# ---------------------------------------------------------------------------
# release floors at zero -- never goes negative
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_floors_at_zero_instead_of_going_negative(time_controller):
    limiter = _make_limiter(time_controller)
    key = "{tag_rl:test:concurrency:floor:chain:u1}:inflight"
    await limiter._decrement_floor_zero(limiter.internal_usage_cache, key, -1.0)
    value = await limiter.internal_usage_cache.async_get_cache(key=key, litellm_parent_otel_span=None)
    assert (float(value) if value is not None else 0.0) == 0.0


# ---------------------------------------------------------------------------
# a failed refund must not block refunding the rest of the batch or raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refund_failure_on_one_key_does_not_block_others_or_raise(time_controller):
    """
    If `_decrement_floor_zero` fails for one key mid-rollback (e.g. a
    transient Redis error), the failure must be logged and swallowed, not
    raised: otherwise it would surface as an unhandled exception in place of
    the clean rejection the caller expects, and would abort the loop before
    refunding every other already-committed key in the same batch.
    """
    failing_key = "{tag_rl:test:refund-fail:a}:requests"
    other_key = "{tag_rl:test:refund-fail:b}:requests"
    rejecting_key = "{tag_rl:test:refund-fail:c}:requests"

    class _FlakyLimiter(_PROXY_ModelBasedTagRateLimitsHook):
        async def _decrement_floor_zero(self, cache, key: str, delta: float) -> None:
            if key == failing_key:
                raise RuntimeError("simulated transient redis failure")
            await super()._decrement_floor_zero(cache, key, delta)

    flaky = _FlakyLimiter(internal_usage_cache=DualCache(), time_provider=time_controller.now)

    failing_index, values = await flaky._atomic_check_and_increment(
        [
            (flaky.internal_usage_cache, failing_key, 10.0, 1.0, 60),
            (flaky.internal_usage_cache, other_key, 10.0, 1.0, 60),
            (flaky.internal_usage_cache, rejecting_key, 0.0, 1.0, 60),
        ]
    )

    assert failing_index == 2

    other_value = await flaky.internal_usage_cache.async_get_cache(key=other_key, litellm_parent_otel_span=None)
    assert (float(other_value) if other_value is not None else 0.0) == 0.0


@pytest.mark.asyncio
async def test_exception_mid_batch_refunds_every_earlier_admission_before_propagating(time_controller):
    """
    Regression test: a later key's own admission raising (a transient Redis
    error, or this coroutine being cancelled mid-call) used to skip the
    refund loop entirely, since it only ran on a normal rejection return.
    An earlier admission in the same batch would then stay permanently
    charged -- for concurrency, a leaked reservation the caller never gets
    to release, incorrectly throttling that tag until the 1-hour safety TTL
    expires.
    """
    admitted_key = "{tag_rl:test:exception-refund:a}:requests"
    raising_key = "{tag_rl:test:exception-refund:b}:requests"

    class _FlakyLimiter(_PROXY_ModelBasedTagRateLimitsHook):
        async def _check_and_increment_one(self, cache, key: str, limit: float, increment: float, ttl: int):
            if key == raising_key:
                raise RuntimeError("simulated transient redis failure")
            return await super()._check_and_increment_one(cache, key, limit, increment, ttl)

    flaky = _FlakyLimiter(internal_usage_cache=DualCache(), time_provider=time_controller.now)

    with pytest.raises(RuntimeError):
        await flaky._atomic_check_and_increment(
            [
                (flaky.internal_usage_cache, admitted_key, 10.0, 1.0, 60),
                (flaky.internal_usage_cache, raising_key, 10.0, 1.0, 60),
            ]
        )

    admitted_value = await flaky.internal_usage_cache.async_get_cache(key=admitted_key, litellm_parent_otel_span=None)
    assert (float(admitted_value) if admitted_value is not None else 0.0) == 0.0


@pytest.mark.asyncio
async def test_a_raising_keys_own_ambiguous_outcome_is_never_refunded(time_controller):
    """
    Regression test for a bug this exact fix briefly introduced: a key can
    commit its own increment (e.g. Redis runs the INCRBY) and still have
    the call raise if the response back to us is lost, so a raise never
    proves that key's own attempt didn't commit. But these are shared,
    chain-wide buckets with no per-request ownership tracking, so
    decrementing on that guess is just as likely to erase a *different*,
    legitimately-admitted concurrent request's charge on the same key as it
    is to undo our own -- an attacker could repeatedly cancel requests to
    erase other callers' charges and exceed the configured limit. The
    raising key's own outcome must never be refunded, only strictly earlier
    (confirmed-safe) admissions in the same batch.
    """
    admitted_key = "{tag_rl:test:ambiguous-no-refund:a}:requests"
    raising_key = "{tag_rl:test:ambiguous-no-refund:b}:requests"

    class _FlakyLimiter(_PROXY_ModelBasedTagRateLimitsHook):
        async def _check_and_increment_one(self, cache, key: str, limit: float, increment: float, ttl: int):
            if key == raising_key:
                # Simulate Redis committing the increment before the
                # response is lost: the write actually happens...
                await super()._check_and_increment_one(cache, key, limit, increment, ttl)
                # ...but the caller never finds out.
                raise RuntimeError("simulated lost response after a committed redis write")
            return await super()._check_and_increment_one(cache, key, limit, increment, ttl)

    flaky = _FlakyLimiter(internal_usage_cache=DualCache(), time_provider=time_controller.now)

    with pytest.raises(RuntimeError):
        await flaky._atomic_check_and_increment(
            [
                (flaky.internal_usage_cache, admitted_key, 10.0, 1.0, 60),
                (flaky.internal_usage_cache, raising_key, 10.0, 1.0, 60),
            ]
        )

    # The earlier, confirmed-successful admission in this same batch is
    # always safe to refund.
    admitted_value = await flaky.internal_usage_cache.async_get_cache(key=admitted_key, litellm_parent_otel_span=None)
    assert (float(admitted_value) if admitted_value is not None else 0.0) == 0.0

    # The raising key's own committed increment must survive -- refunding
    # it would be indistinguishable from erasing a different request's
    # legitimate charge on the same shared bucket.
    raising_key_value = await flaky.internal_usage_cache.async_get_cache(key=raising_key, litellm_parent_otel_span=None)
    assert float(raising_key_value) == 1.0


# ---------------------------------------------------------------------------
# scope_by_key_hash -- opt-in per-calling-key bucket separation
# ---------------------------------------------------------------------------


def _concurrency_router_scoped_by_key(limit: int) -> "litellm.Router":
    return litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "concurrency_limits": {
                        "limits": [
                            {
                                "name": "inflight",
                                "tag_id": "end_user_id",
                                "limit": limit,
                                "period_seconds": 300,
                                "scope_by_key_hash": True,
                            }
                        ]
                    }
                },
            )
        ]
    )


@pytest.mark.asyncio
async def test_request_limit_scope_by_key_hash_gives_independent_counters_per_key(time_controller):
    """
    scope_by_key_hash=True: the identical tag value sent by two different
    calling keys must get independent request counters -- exhausting keyA's
    limit must not affect keyB's admission for the same tag.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {
                                "name": "per_minute",
                                "tag_id": "end_user_id",
                                "limit": 2,
                                "period_seconds": 60,
                                "scope_by_key_hash": True,
                            }
                        ]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    for _ in range(2):
        result = await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"}},
        )
        assert result == healthy

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"}},
        )

    # keyB, identical tag value, is unaffected -- it gets its own bucket and
    # can admit up to the same limit independently of keyA's exhausted one.
    for _ in range(2):
        result = await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyB"}},
        )
        assert result == healthy

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyB"}},
        )


def test_partition_key_distinguishes_entries_that_differ_only_by_scope_by_key_hash():
    """
    scope_by_key_hash is part of the partition-key signature: two entries
    identical in every other field but differing only on this flag are
    different rate limits (different bucket keys per _hash_tag) and must
    never be routed to the same cache partition.
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
    assert _partition_key(unscoped) != _partition_key(scoped)


def test_partition_key_distinguishes_entries_that_differ_only_by_scoping_fields():
    """
    A plain, unscoped entry and a scoped override can legitimately share
    name/tag_id/limit/period_seconds/scope_by_key_hash (see
    test_bucket_key_differs_for_same_named_entries_with_different_scoping_only)
    while disagreeing on enabled_for/disabled_for/apply_to_key_alias/
    apply_to_models -- _DedupSignature and _policy_fingerprint already treat
    that as two distinct policies, so a shared max_in_memory_cache_size must
    not route them onto the same in-memory partition either, or one entry's
    high-cardinality traffic can evict the other's active counters from a
    cache neither entry asked to share.
    """
    base_kwargs = {"name": "daily", "tag_id": "end_user_id", "limit": 100, "period_seconds": 86400, "max_in_memory_cache_size": 50}
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
        _partition_key(unscoped),
        _partition_key(enabled_for_scoped),
        _partition_key(disabled_for_scoped),
        _partition_key(alias_scoped),
        _partition_key(models_scoped),
    }
    assert len(keys) == 5


@pytest.mark.asyncio
async def test_scope_by_key_hash_composes_with_max_in_memory_cache_size_and_key_ttl_seconds_overrides(time_controller):
    """
    scope_by_key_hash must keep working when combined with the two other
    per-entry overrides, going through the real _build_limits_index path
    (not a hand-built _ConfiguredLimit) -- this is exactly the path the
    signature-reconstruction bug silently broke key_ttl_seconds and
    max_in_memory_cache_size on, so it's worth covering in combination
    rather than trusting the fields compose correctly in isolation.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {
                                "name": "per_minute",
                                "tag_id": "end_user_id",
                                "limit": 1,
                                "period_seconds": 60,
                                "scope_by_key_hash": True,
                                "max_in_memory_cache_size": 5,
                                "key_ttl_seconds": 120,
                            }
                        ]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    configured = limiter._index.get(router).resolve("grp", team_id=None)
    assert configured[0].entry.max_in_memory_cache_size == 5
    assert configured[0].entry.key_ttl_seconds == 120

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"}},
    )
    assert result == healthy

    # keyA is now at its per-key limit of 1.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"}},
        )

    # keyB, identical tag value, still gets its own independent bucket on
    # the same (overridden) cache partition.
    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyB"}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_request_limit_without_scope_by_key_hash_still_shares_one_counter(time_controller):
    """
    Regression guard: scope_by_key_hash defaults to False, so today's
    existing behavior -- the bucket is shared across every key sending the
    same tag value -- must be unchanged. Two different keys sending the
    identical tag value must still share one counter.
    """
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 2, "period_seconds": 60}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # keyA and keyB share the same bucket -- one call each exhausts the
    # shared limit of 2.
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"}},
    )
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyB"}},
    )

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"}},
        )


@pytest.mark.asyncio
async def test_concurrency_scope_by_key_hash_gives_independent_reservations_per_key(time_controller):
    """
    scope_by_key_hash=True on a concurrency_limits entry: two different
    calling keys sending the identical tag value must not share one
    reservation bucket -- keyA exhausting its own single slot must not
    block keyB's admission, and releasing keyA's reservation (via the
    standard_logging_object.metadata.user_api_key_hash channel) must free
    keyA's capacity, not keyB's. Each key is modeled as its own logical
    request with its own model_call_details, and keyA's release is spawned
    as a genuinely separate child task (mirroring litellm's real dispatch)
    to prove release survives that task boundary via the shared
    model_call_details object, not via which task happens to run it.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router_scoped_by_key(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    async def _admit(key: str, request_kwargs: dict):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs=request_kwargs,
        )

    async def _release(key: str, kwargs: dict):
        kwargs["standard_logging_object"] = {
            "model_group": "grp",
            "model_id": "dep-1",
            "total_tokens": 0,
            "response_cost": 0,
            "metadata": {"user_api_key_hash": key},
        }
        await limiter.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=0,
            end_time=0,
        )

    ready_to_release = asyncio.Event()
    key_a_request, key_a_kwargs = _call_context(["end_user_id:u1"])
    key_a_request["metadata"]["user_api_key"] = "keyA"
    key_b_request, _key_b_kwargs = _call_context(["end_user_id:u1"])
    key_b_request["metadata"]["user_api_key"] = "keyB"

    async def _key_a_admits_then_waits_then_releases_from_the_same_context_chain():
        await _admit("keyA", key_a_request)
        await ready_to_release.wait()
        await asyncio.create_task(_release("keyA", key_a_kwargs))

    # keyA occupies its own single slot; keyB, same tag value, different
    # key, still admits since it has its own bucket.
    key_a_task = asyncio.create_task(_key_a_admits_then_waits_then_releases_from_the_same_context_chain())
    key_b_task = asyncio.create_task(_admit("keyB", key_b_request))
    await key_b_task
    # Let key_a_task's admission run up to (but not past) `ready_to_release.wait()`.
    await asyncio.sleep(0)

    # keyA is now at its own capacity -- a second keyA request is rejected.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"}},
        )

    # Let keyA's task proceed to its own child-task release.
    ready_to_release.set()
    await key_a_task
    await asyncio.sleep(0)

    # keyA's capacity is freed -- a fresh keyA request now admits.
    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyA"}},
    )
    assert result == healthy

    # keyB's own reservation is untouched by keyA's release -- a second keyB
    # request is still rejected. If task isolation were broken, keyA's
    # release would have drained keyB's reservation too.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": "keyB"}},
        )


# ---------------------------------------------------------------------------
# in-memory cache isolation -- caller-controlled tag buckets must never evict
# the shared cache's other, authentication-bound counters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flooding_tag_buckets_does_not_evict_the_shared_cache_authentication_bound_counter(
    time_controller,
):
    """
    The proxy-wide internal_usage_cache passed into this limiter is also
    used by the key/team parallel-request limiter for its own,
    authentication-bound counters, and its default InMemoryCache evicts at
    200 items. Without a dedicated in-memory layer for this hook's own
    caller-controlled tag buckets, an attacker sending 200+ distinct tag
    values could evict an unrelated authentication-bound counter and let
    some other caller exceed a limit nothing here configured.
    """
    shared_cache = DualCache()
    await shared_cache.async_set_cache(key="authentication_bound_counter", value="do-not-evict")

    limiter = _PROXY_ModelBasedTagRateLimitsHook(internal_usage_cache=shared_cache, time_provider=time_controller.now)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 1000, "period_seconds": 60}]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    for i in range(250):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": [f"end_user_id:flood-{i}"]}},
        )

    assert await shared_cache.async_get_cache(key="authentication_bound_counter") == "do-not-evict"


def _single_request_per_minute_router() -> "litellm.Router":
    return litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
                    }
                },
            )
        ]
    )


@pytest.mark.asyncio
async def test_max_in_memory_cache_size_setting_lets_high_cardinality_tags_avoid_early_eviction(
    time_controller, monkeypatch
):
    """
    This hook's own isolated cache still defaults to 200 items, shared across
    every distinct tag value it sees. A deployment rate-limiting on a
    high-cardinality tag_id (e.g. per end user) without Redis can raise
    `litellm_settings.model_based_tag_rate_limits_max_in_memory_cache_size` so an
    earlier bucket survives churn from later, unrelated tag values: with
    limit=1, a still-live bucket rejects a second request instead of having
    been evicted back to a fresh count of 0.
    """
    monkeypatch.setattr(litellm, "model_based_tag_rate_limits_max_in_memory_cache_size", 500)

    limiter = _PROXY_ModelBasedTagRateLimitsHook(internal_usage_cache=DualCache(), time_provider=time_controller.now)
    router = _single_request_per_minute_router()
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:early-user"]}},
    )

    for i in range(250):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": [f"end_user_id:flood-{i}"]}},
        )

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:early-user"]}},
        )


@pytest.mark.parametrize(
    "invalid_configured_size",
    [
        0,  # would hit InMemoryCache.set_cache's `max_size_in_memory == 0` short-circuit, disabling the cache
        -1,  # would loop `heapq.heappop` on an empty heap in InMemoryCache.evict_cache and raise IndexError
        "500",  # an unresolved os.environ/ substitution or config typo; `len(...) >= "500"` raises TypeError
        True,  # bool is an int subclass; must not be misread as the positive integer 1
    ],
)
@pytest.mark.asyncio
async def test_invalid_max_in_memory_cache_size_falls_back_to_the_safe_default(
    time_controller, monkeypatch, invalid_configured_size
):
    """
    DualCache.async_set_cache swallows any exception raised while writing, so an
    invalid configured size would otherwise silently disable every counter write
    for this hook (every read then sees an empty counter and is admitted) instead
    of failing loudly. Each of these must be rejected in favor of the safe
    default: a limit=1 bucket must still reject a second, immediate request.
    """
    monkeypatch.setattr(litellm, "model_based_tag_rate_limits_max_in_memory_cache_size", invalid_configured_size)

    limiter = _PROXY_ModelBasedTagRateLimitsHook(internal_usage_cache=DualCache(), time_provider=time_controller.now)
    router = _single_request_per_minute_router()
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )


# ---------------------------------------------------------------------------
# per-tag Redis/bucket key TTL override
# ---------------------------------------------------------------------------


def _request_limit(period_seconds: int, key_ttl_seconds: int | None = None) -> _ConfiguredLimit:
    return _ConfiguredLimit(
        unit="requests",
        entry=TagRateLimitEntry(
            name="per_minute",
            tag_id="end_user_id",
            limit=1,
            period_seconds=period_seconds,
            key_ttl_seconds=key_ttl_seconds,
        ),
        deployment_scope=None,
    )


def _concurrency_limit(period_seconds: int, key_ttl_seconds: int | None = None) -> _ConfiguredLimit:
    return _ConfiguredLimit(
        unit="concurrency",
        entry=TagRateLimitEntry(
            name="active", tag_id="end_user_id", limit=1, period_seconds=period_seconds, key_ttl_seconds=key_ttl_seconds
        ),
        deployment_scope=None,
    )


def test_bucket_ttl_seconds_defaults_to_period_plus_one_hour_when_unset():
    assert _bucket_ttl_seconds(_request_limit(period_seconds=60).entry) == 60 + 3600


def test_bucket_ttl_seconds_honors_key_ttl_seconds_override():
    assert _bucket_ttl_seconds(_request_limit(period_seconds=60, key_ttl_seconds=120).entry) == 120


def test_ttl_for_concurrency_honors_key_ttl_seconds_above_the_safety_floor():
    above_floor: Final = _CONCURRENCY_MIN_SAFETY_TTL_SECONDS + 100
    assert (
        _PROXY_ModelBasedTagRateLimitsHook._ttl_for(_concurrency_limit(period_seconds=60, key_ttl_seconds=above_floor))
        == above_floor
    )


def test_ttl_for_concurrency_never_drops_below_the_safety_floor_even_with_a_lower_override():
    """
    A reservation's TTL must comfortably outlast any real in-flight request, so
    an operator-set override below _CONCURRENCY_MIN_SAFETY_TTL_SECONDS must not
    be honored as-is -- a slow request's reservation would otherwise self-heal
    (expire) while still genuinely running, silently admitting extra requests.
    """
    below_floor: Final = 10
    assert (
        _PROXY_ModelBasedTagRateLimitsHook._ttl_for(_concurrency_limit(period_seconds=5, key_ttl_seconds=below_floor))
        == _CONCURRENCY_MIN_SAFETY_TTL_SECONDS
    )


def test_tag_rate_limit_entry_rejects_non_positive_key_ttl_seconds():
    with pytest.raises(ValidationError, match="key_ttl_seconds must be a positive integer"):
        TagRateLimitEntry(name="per_minute", limit=1, period_seconds=60, key_ttl_seconds=0)


def test_tag_rate_limit_entry_rejects_key_ttl_seconds_shorter_than_period_seconds():
    """
    Regression test for a real bug: a key_ttl_seconds shorter than
    period_seconds expires the bucket key before its window rolls over,
    resetting the counter to zero mid-window and letting tagged traffic
    exceed the configured limit.
    """
    with pytest.raises(ValidationError, match="key_ttl_seconds must be at least period_seconds"):
        TagRateLimitEntry(name="per_minute", limit=1, period_seconds=60, key_ttl_seconds=59)


# ---------------------------------------------------------------------------
# per-tag max_in_memory_cache_size override -- dedicated cache partitions
# ---------------------------------------------------------------------------


def test_tag_rate_limit_entry_rejects_non_positive_max_in_memory_cache_size():
    with pytest.raises(ValidationError, match="max_in_memory_cache_size must be a positive integer"):
        TagRateLimitEntry(name="per_minute", limit=1, period_seconds=60, max_in_memory_cache_size=0)


def _two_request_limit_router(team_limit: int, user_limit: int, user_cache_size: int | None) -> "litellm.Router":
    return litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {"name": "team_cap", "tag_id": "team_id", "limit": team_limit, "period_seconds": 60},
                            {
                                "name": "user_cap",
                                "tag_id": "end_user_id",
                                "limit": user_limit,
                                "period_seconds": 60,
                                "max_in_memory_cache_size": user_cache_size,
                            },
                        ]
                    }
                },
            )
        ]
    )


@pytest.mark.asyncio
async def test_max_in_memory_cache_size_override_isolates_a_flood_on_that_entry_from_a_default_partition_entry(
    time_controller,
):
    """
    An entry with its own max_in_memory_cache_size gets a dedicated cache
    partition. Flooding that entry's own high-cardinality tag values must
    never evict a *different* entry's bucket that was never given an
    override and still lives on the hook's single default partition.
    """
    limiter = _make_limiter(time_controller)
    router = _two_request_limit_router(team_limit=1, user_limit=1000, user_cache_size=5)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # team_cap's bucket (default partition) is created and admitted once.
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["team_id:t1"]}}
    )

    # Flood user_cap's own dedicated partition (cap=5) past its own capacity
    # many times over -- this must stay fully confined to user_cap's own
    # partition and never touch team_cap's default-partition bucket.
    for i in range(250):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": [f"end_user_id:flood-{i}"]}},
        )

    # team_cap's bucket must still be at its limit (1) -- a second team_id:t1
    # request is rejected. If it had been evicted by user_cap's flood, this
    # would instead admit (a fresh, zeroed counter).
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["team_id:t1"]}},
        )


@pytest.mark.asyncio
async def test_two_entries_sharing_the_identical_max_in_memory_cache_size_still_get_separate_partitions(
    time_controller,
):
    """
    Partitions are keyed by the entry's full signature, not the override
    value alone: two unrelated entries that happen to choose the identical
    max_in_memory_cache_size must not be merged into one shared cache, or
    flooding one would evict the other's bucket exactly like the bug this
    override exists to fix.
    """
    limiter = _make_limiter(time_controller)
    # Both team_cap and user_cap set the identical max_in_memory_cache_size (5).
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {
                                "name": "team_cap",
                                "tag_id": "team_id",
                                "limit": 1,
                                "period_seconds": 60,
                                "max_in_memory_cache_size": 5,
                            },
                            {
                                "name": "user_cap",
                                "tag_id": "end_user_id",
                                "limit": 1000,
                                "period_seconds": 60,
                                "max_in_memory_cache_size": 5,
                            },
                        ]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["team_id:t1"]}}
    )

    for i in range(250):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": [f"end_user_id:flood-{i}"]}},
        )

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["team_id:t1"]}},
        )


@pytest.mark.asyncio
async def test_concurrency_slot_with_a_cache_size_override_is_released_against_the_same_partition(time_controller):
    """
    A concurrency reservation on an entry with its own max_in_memory_cache_size
    must be released against that same dedicated partition. If the release
    path fell back to the default partition instead, it would silently no-op
    (nothing to decrement there) and the reservation would leak forever.
    """
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "concurrency_limits": {
                        "limits": [
                            {
                                "name": "inflight",
                                "tag_id": "end_user_id",
                                "limit": 1,
                                "period_seconds": 300,
                                "max_in_memory_cache_size": 10,
                            }
                        ]
                    }
                },
            )
        ]
    )
    limiter = _make_limiter(time_controller)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    request_kwargs, kwargs = _call_context(["end_user_id:u1"])
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )

    # At capacity: a second concurrent reservation for the same tag is rejected.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )

    # The first request completes -- its slot is released against the
    # overridden partition -- freeing capacity again.
    kwargs["standard_logging_object"] = {
        "model_group": "grp",
        "model_id": "dep-1",
        "total_tokens": 0,
        "response_cost": 0,
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_token_accounting_with_a_cache_size_override_lands_on_that_entrys_own_partition(time_controller):
    """
    tokens/dollars increments go through a per-partition v3 handler (grouped
    in async_log_success_event), not always the default one -- an entry with
    its own max_in_memory_cache_size must have its usage actually accounted,
    not silently dropped or misrouted to the default partition's handler.
    """
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "token_limits": {
                        "limits": [
                            {
                                "name": "daily",
                                "tag_id": "end_user_id",
                                "limit": 100,
                                "period_seconds": 86400,
                                "max_in_memory_cache_size": 10,
                            }
                        ]
                    }
                },
            )
        ]
    )
    limiter = _make_limiter(time_controller)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    kwargs = {
        "metadata": {"tags": ["end_user_id:u1"]},
        "standard_logging_object": {
            "model_group": "grp",
            "model_id": "dep-1",
            "total_tokens": 150,
            "response_cost": 0,
        },
    }
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    # 150 tokens already used, over the limit of 100 -- the next admission
    # check must reject. If the increment had been silently dropped (never
    # reaching the overridden partition), this would incorrectly admit.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )


# ---------------------------------------------------------------------------
# apply_to_key_alias -- shared TagRateLimitEntry field, also usable on a
# per-model entry (the global_tag_rate_limits_hook is its primary motivation,
# but the field composes with async_filter_deployments unmodified)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_to_key_alias_restricts_a_per_model_entry_to_the_listed_key(time_controller):
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {
                                "name": "per_minute",
                                "tag_id": "end_user_id",
                                "limit": 1,
                                "period_seconds": 60,
                                "apply_to_key_alias": ["premium-key"],
                            }
                        ]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # A key with no matching alias is entirely unaffected -- the entry never
    # applies to it, so it can call repeatedly with no rejection.
    for _ in range(3):
        result = await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key_alias": "other-key"}},
        )
        assert result == healthy

    # The listed key alias is admitted once, then rejected on its 2nd call.
    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key_alias": "premium-key"}},
    )
    assert result == healthy

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key_alias": "premium-key"}},
        )


# ---------------------------------------------------------------------------
# apply_to_models -- shared TagRateLimitEntry field, also usable on a
# per-model entry (expected to be rarely useful there, since a per-deployment
# entry is already implicitly scoped to whichever model_name declares it, but
# it must compose identically to every other shared scoping field)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_to_models_ignores_a_non_matching_model_group(time_controller):
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {
                                "name": "per_minute",
                                "tag_id": "end_user_id",
                                "limit": 1,
                                "period_seconds": 60,
                                "apply_to_models": ["other-group"],
                            }
                        ]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    for _ in range(3):
        result = await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
        assert result == healthy


@pytest.mark.asyncio
async def test_apply_to_models_restricts_a_per_model_entry_to_the_listed_model_group(time_controller):
    limiter = _make_limiter(time_controller)
    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {
                    "request_limits": {
                        "limits": [
                            {
                                "name": "per_minute",
                                "tag_id": "end_user_id",
                                "limit": 1,
                                "period_seconds": 60,
                                "apply_to_models": ["grp"],
                            }
                        ]
                    }
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    result = await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    assert result == healthy

    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
