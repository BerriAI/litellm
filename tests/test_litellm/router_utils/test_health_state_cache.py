"""
Tests for DeploymentHealthCache - the cache layer for health-check-driven routing.
"""

import time

import pytest

from litellm.caching.caching import DualCache
from litellm.router_utils.health_state_cache import DeploymentHealthCache


@pytest.fixture
def cache():
    return DualCache()


@pytest.fixture
def health_cache(cache):
    return DeploymentHealthCache(cache=cache, staleness_threshold=60.0)


def test_set_and_get_unhealthy_ids(health_cache):
    """Write states, verify unhealthy set is returned correctly."""
    now = time.time()
    states = {
        "deploy-1": {"is_healthy": True, "timestamp": now, "reason": ""},
        "deploy-2": {"is_healthy": False, "timestamp": now, "reason": "check_failed"},
        "deploy-3": {"is_healthy": False, "timestamp": now, "reason": "timeout"},
    }
    health_cache.set_deployment_health_states(states)
    result = health_cache.get_unhealthy_deployment_ids()
    assert result == {"deploy-2", "deploy-3"}


@pytest.mark.asyncio
async def test_async_get_unhealthy_ids(health_cache):
    """Async version of set and get."""
    now = time.time()
    states = {
        "deploy-1": {"is_healthy": True, "timestamp": now, "reason": ""},
        "deploy-2": {"is_healthy": False, "timestamp": now, "reason": "check_failed"},
    }
    health_cache.set_deployment_health_states(states)
    result = await health_cache.async_get_unhealthy_deployment_ids()
    assert result == {"deploy-2"}


def test_staleness_filtering(health_cache):
    """Entries older than staleness_threshold should be ignored."""
    old_time = time.time() - 120  # 2 minutes ago, threshold is 60s
    states = {
        "deploy-1": {
            "is_healthy": False,
            "timestamp": old_time,
            "reason": "check_failed",
        },
    }
    health_cache.set_deployment_health_states(states)
    result = health_cache.get_unhealthy_deployment_ids()
    assert result == set()  # stale entry should be ignored


def test_empty_cache_returns_empty_set(health_cache):
    """No data in cache should return empty set."""
    result = health_cache.get_unhealthy_deployment_ids()
    assert result == set()


def test_all_healthy_returns_empty_set(health_cache):
    """All healthy deployments should return empty set."""
    now = time.time()
    states = {
        "deploy-1": {"is_healthy": True, "timestamp": now, "reason": ""},
        "deploy-2": {"is_healthy": True, "timestamp": now, "reason": ""},
    }
    health_cache.set_deployment_health_states(states)
    result = health_cache.get_unhealthy_deployment_ids()
    assert result == set()


def test_mixed_stale_and_fresh(health_cache):
    """Only fresh unhealthy entries should be returned."""
    now = time.time()
    old_time = now - 120  # stale
    states = {
        "deploy-1": {
            "is_healthy": False,
            "timestamp": old_time,
            "reason": "stale",
        },
        "deploy-2": {
            "is_healthy": False,
            "timestamp": now,
            "reason": "fresh",
        },
    }
    health_cache.set_deployment_health_states(states)
    result = health_cache.get_unhealthy_deployment_ids()
    assert result == {"deploy-2"}


def test_malformed_state_entries_are_skipped(health_cache):
    """Non-dict entries in the cache should be skipped safely."""
    now = time.time()
    states = {
        "deploy-1": {"is_healthy": False, "timestamp": now, "reason": "bad"},
        "deploy-2": "not_a_dict",  # malformed
        "deploy-3": None,  # malformed
    }
    health_cache.set_deployment_health_states(states)
    result = health_cache.get_unhealthy_deployment_ids()
    assert result == {"deploy-1"}
