"""
Tests for V2 health-check-driven failover.

Validates the production routing behavior:
- Fail over on qualifying 5xx-style deployment failures
- Do NOT fail over on 429 rate limits
- Do NOT fail over on timeout errors
- Background health checks provide visibility only, not routing exclusion
"""

import time

import pytest

from litellm.caching.caching import DualCache
from litellm.router_utils.cooldown_handlers import (
    _is_cooldown_required,
    _is_qualifying_failure_for_v2,
)
from litellm.router_utils.health_state_cache import DeploymentHealthCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deployment(model_id: str, model_name: str = "gpt-4") -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model_name, "api_key": "fake"},
        "model_info": {"id": model_id},
    }


class _FakeRouterV2:
    """Minimal router stub with enable_health_check_routing=True."""

    def __init__(self, enable_health_check_routing: bool = True):
        self.enable_health_check_routing = enable_health_check_routing


class _FakeRouterV1:
    """Minimal router stub with V1 behavior (health check routing off)."""

    def __init__(self):
        self.enable_health_check_routing = False


# ---------------------------------------------------------------------------
# 1. Failure qualification unit tests
# ---------------------------------------------------------------------------


class TestIsQualifyingFailureForV2:
    """Direct unit tests for the V2 failure qualification gate."""

    def test_5xx_qualifies(self):
        assert _is_qualifying_failure_for_v2(500) is True
        assert _is_qualifying_failure_for_v2(502) is True
        assert _is_qualifying_failure_for_v2(503) is True
        assert _is_qualifying_failure_for_v2(504) is True

    def test_401_qualifies(self):
        assert _is_qualifying_failure_for_v2(401) is True

    def test_404_qualifies(self):
        assert _is_qualifying_failure_for_v2(404) is True

    def test_429_does_not_qualify(self):
        assert _is_qualifying_failure_for_v2(429) is False

    def test_408_timeout_does_not_qualify(self):
        assert _is_qualifying_failure_for_v2(408) is False

    def test_other_4xx_do_not_qualify(self):
        assert _is_qualifying_failure_for_v2(400) is False
        assert _is_qualifying_failure_for_v2(403) is False
        assert _is_qualifying_failure_for_v2(422) is False

    def test_string_status_codes(self):
        assert _is_qualifying_failure_for_v2("503") is True
        assert _is_qualifying_failure_for_v2("429") is False
        assert _is_qualifying_failure_for_v2("") is False


# ---------------------------------------------------------------------------
# 2. _is_cooldown_required branching on V2
# ---------------------------------------------------------------------------


class TestIsCooldownRequiredV2Branch:
    """Test that _is_cooldown_required uses V2 qualification when enabled."""

    def test_429_does_not_trigger_cooldown_in_v2(self):
        """Scenario: 429 handling — no failover."""
        router = _FakeRouterV2(enable_health_check_routing=True)
        result = _is_cooldown_required(
            litellm_router_instance=router,
            model_id="deploy-1",
            exception_status=429,
        )
        assert result is False

    def test_408_timeout_does_not_trigger_cooldown_in_v2(self):
        """Scenario: Timeout handling — no failover."""
        router = _FakeRouterV2(enable_health_check_routing=True)
        result = _is_cooldown_required(
            litellm_router_instance=router,
            model_id="deploy-1",
            exception_status=408,
        )
        assert result is False

    def test_503_triggers_cooldown_in_v2(self):
        """Scenario: True 5xx failover — 503 qualifies."""
        router = _FakeRouterV2(enable_health_check_routing=True)
        result = _is_cooldown_required(
            litellm_router_instance=router,
            model_id="deploy-1",
            exception_status=503,
        )
        assert result is True

    def test_401_auth_failure_triggers_cooldown_in_v2(self):
        """Scenario: Hard auth failure qualifies."""
        router = _FakeRouterV2(enable_health_check_routing=True)
        result = _is_cooldown_required(
            litellm_router_instance=router,
            model_id="deploy-1",
            exception_status=401,
        )
        assert result is True

    def test_404_deleted_deployment_triggers_cooldown_in_v2(self):
        """Scenario: Hard upstream failure (deleted deployment) — 404 qualifies."""
        router = _FakeRouterV2(enable_health_check_routing=True)
        result = _is_cooldown_required(
            litellm_router_instance=router,
            model_id="deploy-1",
            exception_status=404,
        )
        assert result is True

    def test_v1_behavior_preserved_when_v2_disabled(self):
        """When enable_health_check_routing=False, V1 behavior: 429 DOES trigger cooldown."""
        router = _FakeRouterV1()
        result = _is_cooldown_required(
            litellm_router_instance=router,
            model_id="deploy-1",
            exception_status=429,
        )
        assert result is True  # V1 cools down on 429


# ---------------------------------------------------------------------------
# 3. Health check filter is visibility-only
# ---------------------------------------------------------------------------


def _make_health_cache(
    unhealthy_ids: set = None, staleness_threshold: float = 60.0
) -> DeploymentHealthCache:
    cache = DualCache()
    health_cache = DeploymentHealthCache(
        cache=cache, staleness_threshold=staleness_threshold
    )
    if unhealthy_ids:
        now = time.time()
        states = {
            uid: {"is_healthy": False, "timestamp": now, "reason": "test"}
            for uid in unhealthy_ids
        }
        health_cache.set_deployment_health_states(states)
    return health_cache


class TestHealthCheckFilterVisibilityOnly:
    """Test that the health check filter is visibility-only in V2."""

    def _make_router_like(self, enable: bool, health_cache: DeploymentHealthCache):
        from litellm.router import Router

        class FakeRouter:
            def __init__(self):
                self.enable_health_check_routing = enable
                self.health_state_cache = health_cache

        fake = FakeRouter()
        fake._filter_health_check_unhealthy_deployments = (
            Router._filter_health_check_unhealthy_deployments.__get__(fake, FakeRouter)
        )
        fake._async_filter_health_check_unhealthy_deployments = (
            Router._async_filter_health_check_unhealthy_deployments.__get__(
                fake, FakeRouter
            )
        )
        return fake

    def test_health_check_filter_visibility_only(self):
        """Scenario: Background health visibility — no routing impact."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-2"})
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
            _make_deployment("deploy-3"),
        ]
        result = router._filter_health_check_unhealthy_deployments(deployments)
        # All deployments returned — filter is visibility-only
        assert len(result) == 3
        ids = {d["model_info"]["id"] for d in result}
        assert "deploy-2" in ids

    @pytest.mark.asyncio
    async def test_async_health_check_filter_visibility_only(self):
        """Async version: visibility-only, no routing impact."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-2"})
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
        ]
        result = await router._async_filter_health_check_unhealthy_deployments(
            healthy_deployments=deployments
        )
        assert len(result) == 2  # all returned

    def test_filter_noop_when_disabled(self):
        """When disabled, filter is a no-op regardless."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-1"})
        router = self._make_router_like(enable=False, health_cache=health_cache)

        deployments = [_make_deployment("deploy-1"), _make_deployment("deploy-2")]
        result = router._filter_health_check_unhealthy_deployments(deployments)
        assert len(result) == 2
