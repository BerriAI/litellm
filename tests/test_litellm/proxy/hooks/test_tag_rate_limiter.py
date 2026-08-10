"""
Unit tests for tag-scoped token/request/dollar rate limiting.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.tag_rate_limiter import (
    _build_group_limits,
    _extract_identity,
    _PROXY_TagRateLimiter,
)


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
    request_kwargs = {"metadata": {"tags": ["end_user_id:u1"]}}

    # Under limit: allowed, no bucket seeded yet.
    result = await limiter.async_filter_deployments(
        model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
    )
    assert result == healthy

    # Seed the bucket at the limit (2 requests already accounted).
    now = time_controller.now().timestamp()
    bucket_id = int(now) // 60
    key = f"{{tag_rl:grp:requests:per_minute:chain:u1}}:{bucket_id}"
    await limiter.internal_usage_cache.async_set_cache(key=key, value=2, ttl=60, litellm_parent_otel_span=None)

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await limiter.async_filter_deployments(
            model="grp", healthy_deployments=healthy, messages=None, request_kwargs=request_kwargs
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
    request_key = f"{{tag_rl:grp:requests:daily:chain:u1}}:{int(now) // 86400}"
    dollar_key = f"{{tag_rl:grp:dollars:monthly:chain:u1}}:{int(now) // 2592000}"

    assert float(await limiter.internal_usage_cache.async_get_cache(key=token_key, litellm_parent_otel_span=None)) == 42.0
    assert float(await limiter.internal_usage_cache.async_get_cache(key=request_key, litellm_parent_otel_span=None)) == 1.0
    assert float(await limiter.internal_usage_cache.async_get_cache(key=dollar_key, litellm_parent_otel_span=None)) == 0.01
