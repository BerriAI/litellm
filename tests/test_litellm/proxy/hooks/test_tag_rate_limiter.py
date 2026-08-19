"""
Unit tests for tag-scoped token/request/dollar rate limiting.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Final

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.tag_rate_limiter import (
    _CONCURRENCY_MIN_SAFETY_TTL_SECONDS,
    _bucket_key,
    _bucket_ttl_seconds,
    _build_group_limits,
    _build_limits_index,
    _ConfiguredLimit,
    _extract_identity,
    _extract_key_hash,
    _extract_team_id,
    _fixed_length_identity,
    _inflight_key,
    _partition_key,
    _pending_concurrency_holder,
    _PROXY_TagRateLimiter,
)
from litellm.types.router import TagRateLimitEntry


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


def _make_limiter(time_controller: TimeController) -> _PROXY_TagRateLimiter:
    return _PROXY_TagRateLimiter(
        internal_usage_cache=DualCache(),
        time_provider=time_controller.now,
    )


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
) -> str:
    """
    Builds the exact key the real code would compute (via _hash_tag's
    fixed-length hashing of tag_value), instead of hand-writing the raw
    tag value into a literal string -- the internal key format (hashed or
    not) is an implementation detail these tests shouldn't hardcode.
    """
    configured = _ConfiguredLimit(
        unit=unit,
        entry=TagRateLimitEntry(name=name, tag_id=tag_id, limit=1, period_seconds=period_seconds),
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
# TagRateLimitEntry -- period_seconds validation
# ---------------------------------------------------------------------------


def test_tag_rate_limit_entry_rejects_zero_period_seconds():
    with pytest.raises(Exception):
        TagRateLimitEntry(name="n", limit=1, period_seconds=0)


def test_tag_rate_limit_entry_rejects_negative_period_seconds():
    with pytest.raises(Exception):
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
    token_key = _expected_bucket_key("grp", "tokens", "daily", "end_user_id", "u1", 86400, now)
    dollar_key = _expected_bucket_key("grp", "dollars", "monthly", "end_user_id", "u1", 2592000, now)

    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    )
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=dollar_key, litellm_parent_otel_span=None)) == 0.01
    )

    # "requests" is accounted atomically at admission (async_filter_deployments),
    # not here -- async_log_success_event must not touch its bucket at all.
    request_key = _expected_bucket_key("grp", "requests", "daily", "end_user_id", "u1", 86400, now)
    assert await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None) is None


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
    token_key = _expected_bucket_key("grp", "tokens", "daily", "end_user_id", "u1", 86400, now)
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
    token_key = _expected_bucket_key("backend-a", "tokens", "daily", "end_user_id", "u1", 86400, now)
    assert (
        float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    )


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
    request_key = _expected_bucket_key("grp", "requests", "per_minute", "end_user_id", "u1", 60, now)
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

    kwargs = {"metadata": {"tags": ["end_user_id:u1"]}}
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs
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

    kwargs = {"metadata": {"tags": ["end_user_id:u1"]}}
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs
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
    await limiter.async_release_disconnect_state_hook()

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

    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )

    await limiter.async_log_failure_event(
        kwargs={"standard_logging_object": {"model_group": "grp"}, "metadata": {"tags": ["end_user_id:u1"]}},
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

    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
    )
    await limiter.async_log_failure_event(
        kwargs={
            "standard_logging_object": {"model_group": "grp", "model_id": "dep-1"},
            "metadata": {"tags": ["end_user_id:u1"]},
        },
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
async def test_pending_concurrency_context_does_not_leak_across_concurrent_tasks(time_controller):
    """
    Security regression test, current design: `_pending_concurrency_keys` is
    a `contextvars.ContextVar`, isolated per asyncio task/context rather
    than a plain shared dict or list -- which matters because two genuinely
    concurrent, unrelated requests each get their own task in production (a
    hard ASGI guarantee, not something litellm or this hook controls), so
    they can never share a context regardless of what identifiers (tags,
    keys, litellm_call_id) they happen to reuse. Prove this directly: if
    this were a shared collection instead of a real `ContextVar`, one task's
    own release would incorrectly drain the other task's still-pending
    reservation too, since nothing would distinguish which task accumulated
    which key. An earlier design correlated reservations using
    litellm_call_id specifically -- caller-controlled via the
    x-litellm-call-id header -- as a registry key; that's what let two
    unrelated concurrent requests merge reservations in the first place, and
    is why this test isolates via real tasks rather than a shared id at all.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=2)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    async def _admit(tag_value):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": [f"end_user_id:{tag_value}"]}},
        )

    async def _release(tag_value):
        await limiter.async_log_success_event(
            kwargs={
                "standard_logging_object": {
                    "model_group": "grp",
                    "model_id": "dep-1",
                    "total_tokens": 0,
                    "response_cost": 0,
                },
                "metadata": {"tags": [f"end_user_id:{tag_value}"]},
            },
            response_obj=None,
            start_time=0,
            end_time=0,
        )

    # Two separate, genuinely concurrent tasks admit -- reaching capacity.
    task_a = asyncio.create_task(_admit("a"))
    task_b = asyncio.create_task(_admit("b"))
    await task_a
    await task_b

    # Task A releases its own reservation, in its own task -- this must not
    # also release task B's still-pending one.
    await asyncio.create_task(_release("a"))
    await asyncio.sleep(0)

    # Exactly one slot was freed: a fresh request is admitted (back to 2 in flight)...
    await limiter.async_filter_deployments(
        model="grp",
        healthy_deployments=healthy,
        messages=None,
        request_kwargs={"metadata": {"tags": ["end_user_id:a"]}},
    )
    # ...but a second one does not, since B's reservation is genuinely still
    # held. If task isolation were broken, task A's release would have
    # drained B's reservation too, and this would wrongly admit.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:a"]}},
        )


@pytest.mark.asyncio
async def test_concurrency_released_for_every_hop_across_a_real_task_boundary(time_controller):
    """
    litellm dedupes `async_log_failure_event` to fire once per logical
    request: only the first failed hop's failure reaches it (see
    `Logging.has_run_logging`'s `has_logged_async_failure` guard); a later
    failed hop (a retry or a further fallback) never gets its own failure
    event at all. Reservations still accumulate at admission for every hop
    regardless (onto `_pending_concurrency_keys`), so whichever event fires
    next must release everything accumulated since the last release, not
    just its own hop's key.

    Hop 3's eventual success is fired as a child task of the same admission
    chain -- exactly like litellm's real dispatch, where `wrapper_async`
    create_task's the success path and `LoggingWorker.enqueue` explicitly
    propagates the calling context -- to prove the fix survives the actual
    task boundary a real success event crosses in production, not just a
    same-coroutine call that would pass regardless of whether
    `_pending_concurrency_keys` were a real `ContextVar` or an ordinary
    variable.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=2)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    async def _one_logical_request():
        # Hop 1 admits and fails; its failure event is the one that fires
        # (dedup allows exactly the first failure through), releasing its
        # own key immediately.
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )
        await limiter.async_log_failure_event(
            kwargs={"standard_logging_object": {"model_group": "grp"}, "metadata": {"tags": ["end_user_id:u1"]}},
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
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )

        # Hop 3 admits and succeeds. Its success event, dispatched as a
        # child task (mirroring the real worker hop), must release both
        # hop 2's still-pending reservation and its own.
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}},
        )

        async def _hop_3_success_event():
            await limiter.async_log_success_event(
                kwargs={
                    "standard_logging_object": {
                        "model_group": "grp",
                        "model_id": "dep-1",
                        "total_tokens": 0,
                        "response_cost": 0,
                    }
                },
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
    key = _expected_bucket_key("grp", "tokens", "daily", "end_user_id", "u1", 86400, now)
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
    key = _expected_bucket_key("grp", "dollars", "monthly", "team_id", "t1", 2592000, now)
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
    return _PROXY_TagRateLimiter(internal_usage_cache=dual_cache, time_provider=time_controller.now), redis_cache


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
    request_key = _expected_bucket_key("grp", "requests", "per_minute", "end_user_id", tag, 60, now)
    requests_value = await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None)
    assert (float(requests_value) if requests_value is not None else 0.0) == 1.0

    # cleanup: this key persists in the shared scratch Redis instance beyond the test's TTL otherwise
    await redis_cache.async_delete_cache(key=request_key)


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
    assert _PROXY_TagRateLimiter._ttl_for(configured_limit) == _CONCURRENCY_MIN_SAFETY_TTL_SECONDS


def test_concurrency_ttl_floor_does_not_shorten_a_longer_period_seconds():
    entry = TagRateLimitEntry(
        name="inflight", tag_id="end_user_id", limit=1, period_seconds=_CONCURRENCY_MIN_SAFETY_TTL_SECONDS + 100
    )
    configured_limit = _ConfiguredLimit(unit="concurrency", entry=entry, deployment_scope=None)
    assert _PROXY_TagRateLimiter._ttl_for(configured_limit) == _CONCURRENCY_MIN_SAFETY_TTL_SECONDS + 100


# ---------------------------------------------------------------------------
# pending-concurrency-key holder must survive a detached asyncio.create_task
# fork (e.g. litellm's own failure-logging dispatch) without a rebind in that
# forked task hiding the release from the parent, and a release must never
# sweep up a key a still-live sibling hop appended in the meantime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_in_a_forked_task_is_visible_to_the_parent_context():
    _pending_concurrency_holder().keys.clear()
    _pending_concurrency_holder().keys.append("key1")

    async def detached_release():
        return _PROXY_TagRateLimiter._pop_pending_concurrency_keys()

    released = await asyncio.create_task(detached_release())
    assert released == ("key1",)

    # The parent's own binding must see the same, now-empty holder --
    # not a stale copy still holding "key1".
    assert _pending_concurrency_holder().keys == []


@pytest.mark.asyncio
async def test_release_does_not_sweep_up_a_key_appended_after_its_snapshot():
    _pending_concurrency_holder().keys.clear()
    _pending_concurrency_holder().keys.append("key1")

    async def detached_release_then_sibling_admits():
        released = _PROXY_TagRateLimiter._pop_pending_concurrency_keys()
        # A sibling hop's admission, appending to the same shared holder,
        # interleaved right after this release's snapshot was taken.
        _pending_concurrency_holder().keys.append("key2")
        return released

    released = await asyncio.create_task(detached_release_then_sibling_admits())
    assert released == ("key1",)
    # key2 must still be pending for its own hop's eventual release.
    assert _pending_concurrency_holder().keys == ["key2"]


@pytest.mark.asyncio
async def test_release_is_not_repeated_for_the_same_snapshot():
    _pending_concurrency_holder().keys.clear()
    _pending_concurrency_holder().keys.append("key1")
    first = _PROXY_TagRateLimiter._pop_pending_concurrency_keys()
    second = _PROXY_TagRateLimiter._pop_pending_concurrency_keys()
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
    request_key = _expected_bucket_key("grp", "requests", "per_minute", "end_user_id", "refund-check", 60, now)
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

    class _FlakyLimiter(_PROXY_TagRateLimiter):
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

    class _FlakyLimiter(_PROXY_TagRateLimiter):
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

    class _FlakyLimiter(_PROXY_TagRateLimiter):
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
    request: one task does admission and then spawns its own release as a
    child task, exactly like litellm's real dispatch (`wrapper_async`
    create_task's the success path, itself a descendant of the same
    admission-time task/context chain) -- release must never be spawned as
    an unrelated sibling task from the test's own top level, which would
    start from a fresh context that never saw the admission's `ContextVar`
    write at all, an artifact of this test's own construction rather than a
    real bug.
    """
    limiter = _make_limiter(time_controller)
    router = _concurrency_router_scoped_by_key(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    async def _admit(key: str):
        await limiter.async_filter_deployments(
            model="grp",
            healthy_deployments=healthy,
            messages=None,
            request_kwargs={"metadata": {"tags": ["end_user_id:u1"], "user_api_key": key}},
        )

    async def _release(key: str):
        await limiter.async_log_success_event(
            kwargs={
                "standard_logging_object": {
                    "model_group": "grp",
                    "model_id": "dep-1",
                    "total_tokens": 0,
                    "response_cost": 0,
                    "metadata": {"user_api_key_hash": key},
                },
                "metadata": {"tags": ["end_user_id:u1"]},
            },
            response_obj=None,
            start_time=0,
            end_time=0,
        )

    ready_to_release = asyncio.Event()

    async def _key_a_admits_then_waits_then_releases_from_the_same_context_chain():
        await _admit("keyA")
        await ready_to_release.wait()
        await asyncio.create_task(_release("keyA"))

    # keyA occupies its own single slot; keyB, same tag value, different
    # key, still admits since it has its own bucket.
    key_a_task = asyncio.create_task(_key_a_admits_then_waits_then_releases_from_the_same_context_chain())
    key_b_task = asyncio.create_task(_admit("keyB"))
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

    limiter = _PROXY_TagRateLimiter(internal_usage_cache=shared_cache, time_provider=time_controller.now)
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
    `litellm_settings.tag_rate_limiter_max_in_memory_cache_size` so an
    earlier bucket survives churn from later, unrelated tag values: with
    limit=1, a still-live bucket rejects a second request instead of having
    been evicted back to a fresh count of 0.
    """
    monkeypatch.setattr(litellm, "tag_rate_limiter_max_in_memory_cache_size", 500)

    limiter = _PROXY_TagRateLimiter(internal_usage_cache=DualCache(), time_provider=time_controller.now)
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
    monkeypatch.setattr(litellm, "tag_rate_limiter_max_in_memory_cache_size", invalid_configured_size)

    limiter = _PROXY_TagRateLimiter(internal_usage_cache=DualCache(), time_provider=time_controller.now)
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
        _PROXY_TagRateLimiter._ttl_for(_concurrency_limit(period_seconds=60, key_ttl_seconds=above_floor))
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
        _PROXY_TagRateLimiter._ttl_for(_concurrency_limit(period_seconds=5, key_ttl_seconds=below_floor))
        == _CONCURRENCY_MIN_SAFETY_TTL_SECONDS
    )


def test_tag_rate_limit_entry_rejects_non_positive_key_ttl_seconds():
    with pytest.raises(ValueError):
        TagRateLimitEntry(name="per_minute", limit=1, period_seconds=60, key_ttl_seconds=0)


def test_tag_rate_limit_entry_rejects_key_ttl_seconds_shorter_than_period_seconds():
    """
    Regression test for a real bug: a key_ttl_seconds shorter than
    period_seconds expires the bucket key before its window rolls over,
    resetting the counter to zero mid-window and letting tagged traffic
    exceed the configured limit.
    """
    with pytest.raises(ValueError):
        TagRateLimitEntry(name="per_minute", limit=1, period_seconds=60, key_ttl_seconds=59)


# ---------------------------------------------------------------------------
# per-tag max_in_memory_cache_size override -- dedicated cache partitions
# ---------------------------------------------------------------------------


def test_tag_rate_limit_entry_rejects_non_positive_max_in_memory_cache_size():
    with pytest.raises(ValueError):
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

    kwargs = {"metadata": {"tags": ["end_user_id:u1"]}}
    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs
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
