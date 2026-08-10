"""
Unit tests for tag-scoped token/request/dollar rate limiting.
"""

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.tag_rate_limiter import (
    _build_group_limits,
    _build_limits_index,
    _ConfiguredLimit,
    _CONCURRENCY_MIN_SAFETY_TTL_SECONDS,
    _extract_identity,
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
# _build_group_limits -- chain-wide vs per-deployment scoping
# ---------------------------------------------------------------------------


def test_build_group_limits_chain_wide_when_all_deployments_agree():
    deployments = [
        _deployment("grp", "dep-1", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}}),
        _deployment("grp", "dep-2", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}}),
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
        _deployment("grp", "dep-1", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}}),
        _deployment("grp", "dep-2", {"token_limits": {"limits": [{"name": "daily", "limit": 999, "period_seconds": 86400}]}}),
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
        _deployment("grp", "dep-1", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}}),
        _deployment("grp", "dep-2", {}),
    ]
    configured = _build_group_limits(deployments, "tokens")
    assert len(configured) == 1
    assert configured[0].deployment_scope == ("dep-1",)


def test_build_group_limits_empty_when_no_deployment_configures_unit():
    deployments = [_deployment("grp", "dep-1", {}), _deployment("grp", "dep-2", {})]
    assert _build_group_limits(deployments, "tokens") == []


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
                {"request_limits": {"limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 2, "period_seconds": 60}]}},
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
    team_key = f"{{tag_rl:grp:requests:monthly:chain:whatever}}:{team_bucket_id}"
    assert await limiter.internal_usage_cache.async_get_cache(key=team_key, litellm_parent_otel_span=None) is None


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
                {"request_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]}},
            ),
            _deployment(
                "grp",
                "dep-2",
                {"request_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 999, "period_seconds": 86400}]}},
            ),
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    request_kwargs = {"metadata": {"tags": ["end_user_id:u1"]}}

    now = time_controller.now().timestamp()
    bucket_id = int(now) // 86400
    dep1_key = f"{{tag_rl:grp:requests:daily:dep:dep-1:u1}}:{bucket_id}"
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
                    "token_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 500000, "period_seconds": 86400}]},
                    "request_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 100, "period_seconds": 86400}]},
                    "dollar_limits": {"limits": [{"name": "monthly", "tag_id": "end_user_id", "limit": 50.0, "period_seconds": 2592000}]},
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
    token_key = f"{{tag_rl:grp:tokens:daily:chain:u1}}:{int(now) // 86400}"
    dollar_key = f"{{tag_rl:grp:dollars:monthly:chain:u1}}:{int(now) // 2592000}"

    assert float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    assert float(await limiter.internal_usage_cache.async_get_cache(key=dollar_key, litellm_parent_otel_span=None)) == 0.01

    # "requests" is accounted atomically at admission (async_filter_deployments),
    # not here -- async_log_success_event must not touch its bucket at all.
    request_key = f"{{tag_rl:grp:requests:daily:chain:u1}}:{int(now) // 86400}"
    assert await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None) is None


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
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
    )

    # A second attempt: requests-unit alone would admit (well under 10), but
    # concurrency is exhausted, so the whole hop must reject -- and the
    # requests counter must remain untouched by this rejected attempt.
    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
        )
    assert exc_info.value.detail["type"] == "concurrency"

    now = time_controller.now().timestamp()
    request_key = f"{{tag_rl:grp:requests:per_minute:chain:u1}}:{int(now) // 60}"
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
    await limiter.async_filter_deployments(model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs_1)
    await limiter.async_filter_deployments(model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs_2)

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
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
                {"request_limits": {"limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 5, "period_seconds": 60}]}},
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
                {"request_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]}},
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
    )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
        )

    # Same length, deployment mutated in place -- raise the limit to 100.
    router.model_list[0]["model_info"]["tag_rate_limits"] = {
        "request_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 100, "period_seconds": 86400}]}
    }

    time_controller.advance(6)  # past _INDEX_TTL_SECONDS

    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=router.model_list, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
    )
    assert result == router.model_list


@pytest.mark.asyncio
async def test_concurrency_slot_released_on_success_frees_capacity(time_controller):
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    kwargs = {"metadata": {"tags": ["end_user_id:u1"]}}
    await limiter.async_filter_deployments(model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs)

    # At capacity: a second concurrent request is rejected.
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
        )

    # The first request completes -- its slot is released -- freeing capacity again.
    kwargs["standard_logging_object"] = {"model_group": "grp", "model_id": "dep-1", "total_tokens": 0, "response_cost": 0}
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
    )
    assert result == healthy


@pytest.mark.asyncio
async def test_concurrency_slot_released_on_failure_frees_capacity(time_controller):
    limiter = _make_limiter(time_controller)
    router = _concurrency_router(limit=1)
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    kwargs = {"model": "grp", "metadata": {"tags": ["end_user_id:u1"]}}
    await limiter.async_filter_deployments(model="grp", healthy_deployments=healthy, messages=None, request_kwargs=kwargs)

    await limiter.async_post_call_failure_hook(
        request_data=kwargs,
        original_exception=Exception("provider error"),
        user_api_key_dict=UserAPIKeyAuth(),
    )

    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
    )
    assert result == healthy


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
                {"token_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1000, "period_seconds": 86400}]}},
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    now = time_controller.now().timestamp()
    key = f"{{tag_rl:grp:tokens:daily:chain:u1}}:{int(now) // 86400}"
    await limiter.internal_usage_cache.async_set_cache(key=key, value=1000, ttl=86400, litellm_parent_otel_span=None)

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
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
                {"dollar_limits": {"limits": [{"name": "monthly", "tag_id": "team_id", "limit": 50.0, "period_seconds": 2592000}]}},
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    now = time_controller.now().timestamp()
    key = f"{{tag_rl:grp:dollars:monthly:chain:t1}}:{int(now) // 2592000}"
    await limiter.internal_usage_cache.async_set_cache(key=key, value=50.0, ttl=2592000, litellm_parent_otel_span=None)

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["team_id:t1"]}}
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
        pytest.skip(f"Redis connection failed: {str(e)}")

    router = litellm.Router(
        model_list=[
            _deployment(
                "grp",
                "dep-1",
                {"request_limits": {"limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 5, "period_seconds": 60}]}},
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list
    tag = f"redis-race-{uuid.uuid4().hex}"

    async def attempt():
        try:
            await limiter.async_filter_deployments(
                model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": [f"end_user_id:{tag}"]}}
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
        pytest.skip(f"Redis connection failed: {str(e)}")

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
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": [f"end_user_id:{tag}"]}}
    )
    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": [f"end_user_id:{tag}"]}}
        )
    assert exc_info.value.detail["type"] == "concurrency"

    now = time_controller.now().timestamp()
    request_key = f"{{tag_rl:grp:requests:per_minute:chain:{tag}}}:{int(now) // 60}"
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
    deployment = _deployment("real-model-name", "dep-1", {"token_limits": {"limits": [{"name": "daily", "limit": 500, "period_seconds": 86400}]}})
    deployment["model_info"]["team_public_model_name"] = "team-alias-name"
    index = _build_limits_index([deployment])
    assert "real-model-name" in index
    assert "team-alias-name" in index
    assert index["real-model-name"] == index["team-alias-name"]


@pytest.mark.asyncio
async def test_filter_deployments_enforces_limit_when_called_with_team_alias(time_controller):
    limiter = _make_limiter(time_controller)
    deployment = _deployment("real-model-name", "dep-1", {"request_limits": {"limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]}})
    deployment["model_info"]["team_public_model_name"] = "team-alias-name"
    router = litellm.Router(model_list=[deployment])
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    # Router passes the alias as `model`, not "real-model-name".
    await limiter.async_filter_deployments(
        model="team-alias-name", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
    )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="team-alias-name", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:u1"]}}
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
        _deployment("grp", "dep-1", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 2, "period_seconds": 60}]}}),
        _deployment("grp", "dep-2", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 5, "period_seconds": 60}]}}),
    ]
    configured = _build_group_limits(deployments, "concurrency")
    assert configured == []


def test_concurrency_partial_declaration_is_dropped_not_scoped_per_deployment():
    deployments = [
        _deployment("grp", "dep-1", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 2, "period_seconds": 60}]}}),
        _deployment("grp", "dep-2", {}),
    ]
    configured = _build_group_limits(deployments, "concurrency")
    assert configured == []


def test_concurrency_identical_across_all_deployments_is_still_chain_wide():
    deployments = [
        _deployment("grp", "dep-1", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 2, "period_seconds": 60}]}}),
        _deployment("grp", "dep-2", {"concurrency_limits": {"limits": [{"name": "inflight", "limit": 2, "period_seconds": 60}]}}),
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
    entry = TagRateLimitEntry(name="inflight", tag_id="end_user_id", limit=1, period_seconds=_CONCURRENCY_MIN_SAFETY_TTL_SECONDS + 100)
    configured_limit = _ConfiguredLimit(unit="concurrency", entry=entry, deployment_scope=None)
    assert _PROXY_TagRateLimiter._ttl_for(configured_limit) == _CONCURRENCY_MIN_SAFETY_TTL_SECONDS + 100


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
                    "request_limits": {"limits": [{"name": "per_minute", "tag_id": "end_user_id", "limit": 10, "period_seconds": 60}]},
                    "concurrency_limits": {"limits": [{"name": "inflight", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]},
                },
            )
        ]
    )
    limiter.update_variables(llm_router=router)
    healthy = router.model_list

    await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:refund-check"]}}
    )
    with pytest.raises(ProxyRateLimitError):
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs={"metadata": {"tags": ["end_user_id:refund-check"]}}
        )

    now = time_controller.now().timestamp()
    request_key = f"{{tag_rl:grp:requests:per_minute:chain:refund-check}}:{int(now) // 60}"
    value = await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None)
    assert (float(value) if value is not None else 0.0) == 1.0


# ---------------------------------------------------------------------------
# release floors at zero -- never goes negative
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_floors_at_zero_instead_of_going_negative(time_controller):
    limiter = _make_limiter(time_controller)
    key = "{tag_rl:test:concurrency:floor:chain:u1}:inflight"
    await limiter._decrement_floor_zero(key, -1.0)
    value = await limiter.internal_usage_cache.async_get_cache(key=key, litellm_parent_otel_span=None)
    assert (float(value) if value is not None else 0.0) == 0.0
