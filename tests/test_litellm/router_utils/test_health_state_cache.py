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


def test_set_merges_states_from_scoped_writers(health_cache):
    """A writer covering one scope must not erase another scope's fresh states."""
    now = time.time()
    health_cache.set_deployment_health_states(
        {"listed-bad": {"is_healthy": False, "timestamp": now, "reason": "check_failed"}}
    )
    health_cache.set_deployment_health_states(
        {"other-ok": {"is_healthy": True, "timestamp": now, "reason": ""}}
    )
    assert health_cache.get_unhealthy_deployment_ids() == {"listed-bad"}


def test_set_prunes_expired_entries(health_cache, cache):
    """Entries older than 1.5x the staleness threshold are dropped on write."""
    expired_time = time.time() - 100  # threshold 60s, prune horizon 90s
    health_cache.set_deployment_health_states(
        {"gone": {"is_healthy": False, "timestamp": expired_time, "reason": "check_failed"}}
    )
    now = time.time()
    health_cache.set_deployment_health_states(
        {"fresh": {"is_healthy": False, "timestamp": now, "reason": "check_failed"}}
    )
    stored = cache.get_cache(key=DeploymentHealthCache.CACHE_KEY)
    assert set(stored.keys()) == {"fresh"}


class _SharedRedisFake:
    """Shared get/set key-value store standing in for the Redis layer of a DualCache."""

    def __init__(self):
        self.store = {}
        self.fail_get = False

    def get_cache(self, key, parent_otel_span=None, **kwargs):
        if self.fail_get:
            return None  # RedisCache.get_cache swallows connection errors and returns None
        return self.store.get(key)

    def set_cache(self, key, value, **kwargs):
        self.store[key] = value


def test_scoped_writers_on_shared_redis_preserve_each_other():
    """Pods with different allowlists share one Redis entry; each merge must keep the peer's scope."""
    redis_fake = _SharedRedisFake()
    pod_a = DeploymentHealthCache(cache=DualCache(redis_cache=redis_fake), staleness_threshold=60.0)
    pod_b = DeploymentHealthCache(cache=DualCache(redis_cache=redis_fake), staleness_threshold=60.0)
    pod_a.set_deployment_health_states(
        {"prod-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "check_failed"}}
    )
    pod_b.set_deployment_health_states(
        {"internal-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "timeout"}}
    )
    pod_a.set_deployment_health_states(
        {"prod-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "check_failed"}}
    )
    assert set(redis_fake.store[DeploymentHealthCache.CACHE_KEY]) == {"prod-bad", "internal-bad"}
    assert pod_a.get_unhealthy_deployment_ids() == {"prod-bad", "internal-bad"}


def test_failed_redis_read_falls_back_to_local_copy():
    """A swallowed Redis GET error must not make a writer erase peer scopes it already saw."""
    redis_fake = _SharedRedisFake()
    pod_a = DeploymentHealthCache(cache=DualCache(redis_cache=redis_fake), staleness_threshold=60.0)
    pod_b = DeploymentHealthCache(cache=DualCache(redis_cache=redis_fake), staleness_threshold=60.0)
    pod_a.set_deployment_health_states(
        {"prod-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "check_failed"}}
    )
    pod_b.set_deployment_health_states(
        {"internal-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "timeout"}}
    )
    pod_a.set_deployment_health_states(
        {"prod-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "check_failed"}}
    )
    redis_fake.fail_get = True
    pod_a.set_deployment_health_states(
        {"prod-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "check_failed"}}
    )
    assert set(redis_fake.store[DeploymentHealthCache.CACHE_KEY]) == {"prod-bad", "internal-bad"}
