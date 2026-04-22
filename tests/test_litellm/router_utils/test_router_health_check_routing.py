"""
Tests for health-check-driven routing filter in the Router.
"""

import time

import pytest

from litellm.caching.caching import DualCache
from litellm.router_utils.health_state_cache import DeploymentHealthCache


def _make_deployment(model_id: str, model_name: str = "gpt-4") -> dict:
    """Helper to create a deployment dict for testing."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": model_name, "api_key": "fake"},
        "model_info": {"id": model_id},
    }


def _make_health_cache(
    unhealthy_ids: set = None, staleness_threshold: float = 60.0
) -> DeploymentHealthCache:
    """Create a health cache pre-populated with unhealthy deployment IDs."""
    cache = DualCache()
    health_cache = DeploymentHealthCache(
        cache=cache, staleness_threshold=staleness_threshold
    )
    if unhealthy_ids:
        now = time.time()
        states = {}
        for uid in unhealthy_ids:
            states[uid] = {
                "is_healthy": False,
                "timestamp": now,
                "reason": "test_unhealthy",
            }
        health_cache.set_deployment_health_states(states)
    return health_cache


class TestFilterHealthCheckUnhealthyDeployments:
    """Test the sync filter method."""

    def _make_router_like(self, enable: bool, health_cache: DeploymentHealthCache):
        """Create a minimal object that behaves like Router for filter testing."""

        class FakeRouter:
            def __init__(self):
                self.enable_health_check_routing = enable
                self.health_state_cache = health_cache
                self.allowed_fails_policy = None

        # Import the actual method and bind it
        from litellm.router import Router

        fake = FakeRouter()
        # Use the unbound method
        fake._filter_health_check_unhealthy_deployments = (
            Router._filter_health_check_unhealthy_deployments.__get__(fake, FakeRouter)
        )
        return fake

    def test_filter_removes_unhealthy_deployments(self):
        """Unhealthy deployments should be removed from candidates."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-2"})
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
            _make_deployment("deploy-3"),
        ]
        result = router._filter_health_check_unhealthy_deployments(deployments)
        assert len(result) == 2
        assert all(d["model_info"]["id"] != "deploy-2" for d in result)

    def test_filter_noop_when_disabled(self):
        """When enable_health_check_routing=False, filter should be a no-op."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-1"})
        router = self._make_router_like(enable=False, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
        ]
        result = router._filter_health_check_unhealthy_deployments(deployments)
        assert len(result) == 2  # no filtering

    def test_filter_returns_all_when_all_unhealthy(self):
        """Safety net: if ALL deployments are unhealthy, return all (don't cause outage)."""
        health_cache = _make_health_cache(
            unhealthy_ids={"deploy-1", "deploy-2", "deploy-3"}
        )
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
            _make_deployment("deploy-3"),
        ]
        result = router._filter_health_check_unhealthy_deployments(deployments)
        assert len(result) == 3  # all returned, safety net

    def test_filter_returns_all_when_cache_empty(self):
        """When cache is empty, all deployments should pass through."""
        health_cache = _make_health_cache()  # empty
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
        ]
        result = router._filter_health_check_unhealthy_deployments(deployments)
        assert len(result) == 2


class TestAsyncFilterHealthCheckUnhealthyDeployments:
    """Test the async filter method."""

    def _make_router_like(self, enable: bool, health_cache: DeploymentHealthCache):
        from litellm.router import Router

        class FakeRouter:
            def __init__(self):
                self.enable_health_check_routing = enable
                self.health_state_cache = health_cache
                self.allowed_fails_policy = None

        fake = FakeRouter()
        fake._async_filter_health_check_unhealthy_deployments = (
            Router._async_filter_health_check_unhealthy_deployments.__get__(
                fake, FakeRouter
            )
        )
        return fake

    @pytest.mark.asyncio
    async def test_async_filter_removes_unhealthy(self):
        """Async version: unhealthy deployments removed."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-2"})
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
            _make_deployment("deploy-3"),
        ]
        result = await router._async_filter_health_check_unhealthy_deployments(
            healthy_deployments=deployments
        )
        assert len(result) == 2
        assert all(d["model_info"]["id"] != "deploy-2" for d in result)

    @pytest.mark.asyncio
    async def test_async_filter_safety_net(self):
        """Async version: safety net when all unhealthy."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-1", "deploy-2"})
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
        ]
        result = await router._async_filter_health_check_unhealthy_deployments(
            healthy_deployments=deployments
        )
        assert len(result) == 2  # safety net


class TestBuildDeploymentHealthStates:
    """Test the build_deployment_health_states function."""

    def test_builds_states_from_endpoints(self):
        from litellm.proxy.health_check import build_deployment_health_states

        healthy = [{"model": "gpt-4", "model_id": "deploy-1"}]
        unhealthy = [{"model": "gpt-4", "model_id": "deploy-2", "error": "timeout"}]

        states = build_deployment_health_states(healthy, unhealthy)
        assert states["deploy-1"]["is_healthy"] is True
        assert states["deploy-2"]["is_healthy"] is False

    def test_no_model_id_skipped(self):
        from litellm.proxy.health_check import build_deployment_health_states

        healthy = [{"model": "gpt-4"}]  # no model_id
        unhealthy = [{"model": "gpt-4", "model_id": "deploy-2"}]

        states = build_deployment_health_states(healthy, unhealthy)
        assert "deploy-1" not in states
        assert states["deploy-2"]["is_healthy"] is False

    def test_empty_endpoints(self):
        from litellm.proxy.health_check import build_deployment_health_states

        states = build_deployment_health_states([], [])
        assert states == {}
