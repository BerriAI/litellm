"""
Tests for DeploymentHealthCache - the cache layer for health-check-driven routing.
"""

import time

import pytest

from litellm.redis.stores.deployment_health import DeploymentHealthStates, DeploymentHealthStore
from litellm.router_utils.health_state_cache import DeploymentHealthCache
from litellm.rust_bridge._native import NativeDeploymentHealth, NativeRedisClient


@pytest.fixture
def handle() -> NativeDeploymentHealth:
    return NativeDeploymentHealth(None)


@pytest.fixture
def store(handle: NativeDeploymentHealth) -> DeploymentHealthStore:
    return DeploymentHealthStore(handle)


@pytest.fixture
def health_cache(store):
    return DeploymentHealthCache(store=store, staleness_threshold=60.0)


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


def test_malformed_state_entries_are_skipped(
    health_cache: DeploymentHealthCache, handle: NativeDeploymentHealth
) -> None:
    """Records another writer left that are no dict are dropped alone when the store reads them."""
    now = time.time()
    handle.set_sync(
        {
            "deploy-1": {"is_healthy": False, "timestamp": now, "reason": "bad"},
            "deploy-2": "not_a_dict",
            "deploy-3": None,
        },
        90.0,
    )
    result = health_cache.get_unhealthy_deployment_ids()
    assert result == {"deploy-1"}


def test_set_merges_states_from_scoped_writers(health_cache):
    """A writer covering one scope must not erase another scope's fresh states."""
    now = time.time()
    health_cache.set_deployment_health_states(
        {"listed-bad": {"is_healthy": False, "timestamp": now, "reason": "check_failed"}}
    )
    health_cache.set_deployment_health_states({"other-ok": {"is_healthy": True, "timestamp": now, "reason": ""}})
    assert health_cache.get_unhealthy_deployment_ids() == {"listed-bad"}


def test_set_prunes_expired_entries(health_cache, store):
    """Entries older than 1.5x the staleness threshold are dropped on write."""
    expired_time = time.time() - 100  # threshold 60s, prune horizon 90s
    health_cache.set_deployment_health_states(
        {"gone": {"is_healthy": False, "timestamp": expired_time, "reason": "check_failed"}}
    )
    now = time.time()
    health_cache.set_deployment_health_states(
        {"fresh": {"is_healthy": False, "timestamp": now, "reason": "check_failed"}}
    )
    stored = store.get_sync()
    assert set(stored.keys()) == {"fresh"}


class _SharedRedis:
    """The Redis entry every pod shares; a failed read reads as a miss, as the store's does."""

    def __init__(self):
        self.states: DeploymentHealthStates | None = None
        self.fail_get = False


class _PodHandle:
    """One pod's ``DeploymentHealthHandle``: its own memory in front of the shared Redis entry."""

    def __init__(self, redis: _SharedRedis):
        self._redis = redis
        self._memory: DeploymentHealthStates | None = None

    @property
    def has_redis(self) -> bool:
        return True

    def attach(self, client: NativeRedisClient) -> bool:
        return False

    async def get(self) -> DeploymentHealthStates | None:
        return self.get_sync()

    def get_sync(self) -> DeploymentHealthStates | None:
        return self._memory if self._memory is not None else self.get_shared_sync()

    def get_shared_sync(self) -> DeploymentHealthStates | None:
        return None if self._redis.fail_get else self._redis.states

    def set_sync(self, states: DeploymentHealthStates, ttl_seconds: float) -> None:
        self._memory = states
        self._redis.states = states


def _pod(redis: _SharedRedis) -> DeploymentHealthCache:
    return DeploymentHealthCache(store=DeploymentHealthStore(_PodHandle(redis)), staleness_threshold=60.0)


def test_scoped_writers_on_shared_redis_preserve_each_other():
    """Pods with different allowlists share one Redis entry; each merge must keep the peer's scope."""
    redis_fake = _SharedRedis()
    pod_a = _pod(redis_fake)
    pod_b = _pod(redis_fake)
    pod_a.set_deployment_health_states(
        {"prod-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "check_failed"}}
    )
    pod_b.set_deployment_health_states(
        {"internal-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "timeout"}}
    )
    pod_a.set_deployment_health_states(
        {"prod-bad": {"is_healthy": False, "timestamp": time.time(), "reason": "check_failed"}}
    )
    assert set(redis_fake.states or {}) == {"prod-bad", "internal-bad"}
    assert pod_a.get_unhealthy_deployment_ids() == {"prod-bad", "internal-bad"}


def test_failed_redis_read_falls_back_to_local_copy():
    """A swallowed Redis GET error must not make a writer erase peer scopes it already saw."""
    redis_fake = _SharedRedis()
    pod_a = _pod(redis_fake)
    pod_b = _pod(redis_fake)
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
    assert set(redis_fake.states or {}) == {"prod-bad", "internal-bad"}


class _UnreachableHealthHandle:
    has_redis: bool = True

    def attach(self, client: NativeRedisClient) -> bool:
        return True

    async def get(self) -> DeploymentHealthStates | None:
        raise ConnectionError("redis down")

    def get_sync(self) -> DeploymentHealthStates | None:
        raise ConnectionError("redis down")

    def get_shared_sync(self) -> DeploymentHealthStates | None:
        raise ConnectionError("redis down")

    def set_sync(self, states: DeploymentHealthStates, ttl_seconds: float) -> None:
        raise ConnectionError("redis down")


@pytest.mark.asyncio
async def test_an_unreachable_store_reads_as_no_health_state_and_drops_writes():
    cache = DeploymentHealthCache(store=DeploymentHealthStore(_UnreachableHealthHandle()), staleness_threshold=60.0)

    cache.set_deployment_health_states({"deploy-1": {"is_healthy": False, "timestamp": time.time(), "reason": "x"}})

    assert cache.get_states() is None
    assert await cache.async_get_states() is None
    assert cache.get_unhealthy_deployment_ids() == set()
    assert await cache.async_get_unhealthy_deployment_ids() == set()
