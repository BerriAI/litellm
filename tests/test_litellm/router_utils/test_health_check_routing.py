"""
Tests for health-check-driven routing filter in the Router.
"""

import time
from typing import Any, Optional

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
    unhealthy_ids: Optional[set] = None, staleness_threshold: float = 60.0
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

    def _make_router_like(
        self, enable: bool, health_cache: DeploymentHealthCache
    ) -> Any:
        """Create a minimal object that behaves like Router for filter testing."""
        from litellm.router import Router

        class FakeRouter:
            def __init__(self) -> None:
                self.enable_health_check_routing = enable
                self.health_state_cache = health_cache
                setattr(
                    self,
                    "_filter_health_check_unhealthy_deployments",
                    Router._filter_health_check_unhealthy_deployments.__get__(
                        self, FakeRouter
                    ),
                )

        return FakeRouter()

    def test_filter_is_visibility_only_when_enabled(self):
        """V2: unhealthy deployments are logged but NOT filtered from candidates."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-2"})
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
            _make_deployment("deploy-3"),
        ]
        result = router._filter_health_check_unhealthy_deployments(deployments)
        assert len(result) == 3  # all returned, visibility only

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
        """V2: all deployments returned even when all unhealthy (visibility only)."""
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
        assert len(result) == 3  # all returned, visibility only

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

    def _make_router_like(
        self, enable: bool, health_cache: DeploymentHealthCache
    ) -> Any:
        from litellm.router import Router

        class FakeRouter:
            def __init__(self) -> None:
                self.enable_health_check_routing = enable
                self.health_state_cache = health_cache
                setattr(
                    self,
                    "_async_filter_health_check_unhealthy_deployments",
                    Router._async_filter_health_check_unhealthy_deployments.__get__(
                        self, FakeRouter
                    ),
                )

        return FakeRouter()

    @pytest.mark.asyncio
    async def test_async_filter_is_visibility_only(self):
        """V2: async filter logs but does not remove unhealthy deployments."""
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
        assert len(result) == 3  # all returned, visibility only

    @pytest.mark.asyncio
    async def test_async_filter_all_unhealthy_returns_all(self):
        """V2: all deployments returned even when all unhealthy."""
        health_cache = _make_health_cache(unhealthy_ids={"deploy-1", "deploy-2"})
        router = self._make_router_like(enable=True, health_cache=health_cache)

        deployments = [
            _make_deployment("deploy-1"),
            _make_deployment("deploy-2"),
        ]
        result = await router._async_filter_health_check_unhealthy_deployments(
            healthy_deployments=deployments
        )
        assert len(result) == 2  # all returned, visibility only


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
