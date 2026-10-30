import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Final
from unittest.mock import patch

import fakeredis
import pytest
import pytest_asyncio
from prometheus_client import REGISTRY
from pydantic import TypeAdapter
from redis.exceptions import TimeoutError as RedisTimeoutError

import litellm
from litellm.caching.redis_cache import RedisCache
from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.health_check_utils.shared_health_check_manager import SharedHealthCheckManager
from tests._support.fake_redis import fake_redis_cache

NOW: Final = 1_700_000_000.25
LOCK: Final = "shared:health_check_lock"
RESULTS: Final = "shared:health_check_results"
_VALUE: Final = TypeAdapter(object)


@dataclass(frozen=True)
class HealthRedis:
    server: fakeredis.FakeServer
    raw: fakeredis.FakeAsyncRedis
    cache: RedisCache
    manager: SharedHealthCheckManager


@pytest_asyncio.fixture(loop_scope="function")
async def health(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[HealthRedis]:
    monkeypatch.setattr(time, "time", lambda: NOW)
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    cache: Final = fake_redis_cache(server, namespace="shared")
    manager: Final = SharedHealthCheckManager(cache, health_check_ttl=300, lock_ttl=60)
    manager.pod_id = 'pod_雪"'
    yield HealthRedis(server, raw, cache, manager)


def test_public_key_helpers_keep_global_and_model_keys() -> None:
    assert SharedHealthCheckManager.get_health_check_lock_key() == "health_check_lock"
    assert SharedHealthCheckManager.get_health_check_cache_key() == "health_check_results"
    assert SharedHealthCheckManager.get_model_health_check_lock_key("a:雪") == "health_check_lock:a:雪"
    assert SharedHealthCheckManager.get_model_health_check_cache_key("a:雪") == "health_check_results:a:雪"


@pytest.mark.asyncio
async def test_old_lock_writer_new_status_and_owner_checked_release(health: HealthRedis) -> None:
    await health.cache.async_set_cache("health_check_lock", "other_雪", ttl=60, nx=True)
    assert await health.manager.acquire_health_check_lock() is False
    assert await health.manager.is_health_check_in_progress() is True
    assert await health.manager.get_health_check_status() == {
        "pod_id": health.manager.pod_id,
        "redis_available": True,
        "lock_ttl": 60,
        "cache_ttl": 300,
        "lock_owner": "other_雪",
        "lock_in_progress": True,
        "cache_available": False,
    }
    await health.manager.release_health_check_lock()
    assert await health.raw.get(LOCK) == b'"other_\\u96ea"'
    health.manager.pod_id = "other_雪"
    assert await health.manager.is_health_check_in_progress() is False
    await health.manager.release_health_check_lock()
    assert await health.raw.get(LOCK) is None


@pytest.mark.asyncio
async def test_new_lock_writer_old_reader_exact_bytes_namespace_and_expiry(health: HealthRedis) -> None:
    assert await health.manager.acquire_health_check_lock() is True
    assert await health.raw.get(LOCK) == b'"pod_\\u96ea\\""'
    assert await health.cache.async_get_cache("health_check_lock") == health.manager.pod_id
    assert await health.raw.pttl(LOCK) == 60_000
    assert await health.raw.keys("*") == [LOCK.encode()]
    assert await health.manager.acquire_health_check_lock() is False
    await health.manager.release_health_check_lock()
    assert await health.raw.get(LOCK) is None
    assert await health.manager.is_health_check_in_progress() is False


@pytest.mark.asyncio
async def test_snapshot_new_writer_old_reader_exact_double_encoding_and_ttl(health: HealthRedis) -> None:
    healthy: Final = [{"model": "雪", "latency": 1.25, "count": 2, "text": "a\x00b"}]
    unhealthy: Final = [{"model": "bad", "exception_status": 503}]
    expected: Final = {
        "healthy_endpoints": [{"model": "雪", "latency": 1.25, "count": 2, "text": "ab"}],
        "unhealthy_endpoints": unhealthy,
        "healthy_count": 1,
        "unhealthy_count": 1,
        "timestamp": NOW,
        "checked_by": health.manager.pod_id,
    }
    await health.manager.cache_health_check_results(healthy, unhealthy)
    await health.cache.async_set_cache("old_writer", safe_dumps(expected), ttl=300)
    assert await health.raw.get(RESULTS) == await health.raw.get("shared:old_writer")
    assert await health.raw.get(RESULTS) == json.dumps(json.dumps(expected)).encode()
    assert await health.raw.pttl(RESULTS) == 300_000
    assert await health.raw.get("shared:shared:health_check_results") is None
    assert _VALUE.validate_json(await health.cache.async_get_cache("health_check_results")) == expected
    assert await health.manager.get_cached_health_check_results() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["double", "single", "literal"])
async def test_old_snapshot_formats_preserve_extra_fields_and_cached_flow(health: HealthRedis, encoding: str) -> None:
    snapshot: Final = {
        "healthy_endpoints": [{"model": "雪", "latency": 1.25}],
        "unhealthy_endpoints": [],
        "timestamp": NOW - 300,
        "checked_by": {"legacy": "pod"},
        "extra": [0, False, None],
    }
    if encoding == "double":
        await health.cache.async_set_cache("health_check_results", safe_dumps(snapshot), ttl=300)
    elif encoding == "single":
        await health.cache.async_set_cache("health_check_results", snapshot, ttl=300)
    else:
        await health.raw.set(RESULTS, repr(snapshot), ex=300)
    assert await health.manager.get_cached_health_check_results() == snapshot
    assert await health.manager.perform_shared_health_check([]) == (snapshot["healthy_endpoints"], [], {})
    assert await health.manager.get_health_check_status() == {
        "pod_id": health.manager.pod_id,
        "redis_available": True,
        "lock_ttl": 60,
        "cache_ttl": 300,
        "lock_owner": None,
        "lock_in_progress": False,
        "cache_available": True,
        "cache_age_seconds": 300,
        "last_checked_by": {"legacy": "pod"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        {"timestamp": NOW - 301},
        {"timestamp": str(NOW)},
        {"timestamp": True},
        {"timestamp": None},
        {"timestamp": NOW, "healthy_endpoints": None},
        {"timestamp": NOW, "unhealthy_endpoints": [1]},
    ],
)
async def test_stale_or_schema_invalid_snapshot_is_a_miss(health: HealthRedis, snapshot: Mapping[str, object]) -> None:
    await health.cache.async_set_cache("health_check_results", safe_dumps(snapshot), ttl=300)
    assert await health.manager.get_cached_health_check_results() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [b"broken", b'"broken"', b"[]", b"null", b"\xff", b'"null"'])
async def test_malformed_snapshot_and_null_are_misses(health: HealthRedis, raw: bytes) -> None:
    await health.raw.set(RESULTS, raw)
    assert await health.manager.get_cached_health_check_results() is None


@pytest.mark.asyncio
async def test_missing_endpoints_and_future_timestamp_remain_valid(health: HealthRedis) -> None:
    snapshot: Final = {"timestamp": NOW + 1}
    await health.cache.async_set_cache("health_check_results", safe_dumps(snapshot), ttl=300)
    assert await health.manager.get_cached_health_check_results() == snapshot
    assert await health.manager.perform_shared_health_check([]) == ([], [], {})


@pytest.mark.asyncio
async def test_lock_holder_executes_health_check_caches_and_releases(health: HealthRedis) -> None:
    assert await health.manager.perform_shared_health_check([]) == ([], [], {})
    assert await health.manager.get_cached_health_check_results() == {
        "healthy_endpoints": [],
        "unhealthy_endpoints": [],
        "healthy_count": 0,
        "unhealthy_count": 0,
        "timestamp": NOW,
        "checked_by": health.manager.pod_id,
    }
    assert await health.raw.get(LOCK) is None
    assert await health.raw.pttl(RESULTS) == 300_000


@pytest.mark.asyncio
async def test_waiter_reads_snapshot_published_during_poll(health: HealthRedis) -> None:
    await health.cache.async_set_cache("health_check_lock", "other", ttl=60)

    async def publish(delay: float) -> None:
        assert delay == 5
        await health.manager.cache_health_check_results([{"model": "published"}], [])

    with patch("asyncio.sleep", new=publish):
        result: Final = await health.manager.perform_shared_health_check([])
    assert result == ([{"model": "published"}], [], {})
    assert await health.raw.get(LOCK) == b'"other"'


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["orphaned", "outage", "held"])
async def test_wait_policy_falls_back_locally_without_writing_snapshot(health: HealthRedis, change: str) -> None:
    health.manager.lock_ttl = 10
    await health.cache.async_set_cache("health_check_lock", "other", ttl=60)

    async def during_wait(delay: float) -> None:
        assert delay == 5
        if change == "orphaned":
            await health.raw.delete(LOCK)
        elif change == "outage":
            health.server.connected = False

    with patch("asyncio.sleep", wraps=during_wait) as sleep:
        assert await health.manager.perform_shared_health_check([]) == ([], [], {})
    assert sleep.await_count == (2 if change == "held" else 1)
    health.server.connected = True
    assert await health.raw.get(RESULTS) is None


@pytest.mark.asyncio
async def test_outage_is_false_or_miss_then_breaker_refuses_after_recovery(health: HealthRedis) -> None:
    health.server.connected = False
    assert await health.manager.acquire_health_check_lock() is False
    assert await health.manager.get_cached_health_check_results() is None
    assert await health.manager.is_health_check_in_progress() is False
    assert await health.manager.release_health_check_lock() is None
    assert await health.manager.cache_health_check_results([], []) is None
    health.server.connected = True
    assert await health.manager.acquire_health_check_lock() is False
    assert await health.raw.get(LOCK) is None
    status: Final = await health.manager.get_health_check_status()
    assert "error" in status
    assert "lock_owner" not in status


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, -1])
async def test_nonpositive_ttl_does_not_write_a_short_lived_lock_or_snapshot(health: HealthRedis, ttl: int) -> None:
    health.manager.lock_ttl = ttl
    health.manager.health_check_ttl = ttl
    assert await health.manager.acquire_health_check_lock() is False
    await health.manager.cache_health_check_results([], [])
    assert await health.raw.get(LOCK) is None
    assert await health.raw.get(RESULTS) is None


@pytest.mark.asyncio
async def test_lost_write_replies_remain_unknown_applied(health: HealthRedis) -> None:
    async def applied_set(
        name: str, value: str, *, px: int | None = None, ex: int | None = None, nx: bool = False
    ) -> None:
        await health.raw.set(name, value, px=px, ex=ex, nx=nx)
        raise RedisTimeoutError("reply lost after set")

    with patch.object(health.cache.init_async_client(), "set", new=applied_set):
        assert await health.manager.acquire_health_check_lock() is False
        await health.manager.cache_health_check_results([], [])
    assert await health.raw.get(LOCK) == json.dumps(health.manager.pod_id).encode()
    assert await health.raw.pttl(LOCK) == 60_000
    assert await health.manager.get_cached_health_check_results() == {
        "healthy_endpoints": [],
        "unhealthy_endpoints": [],
        "healthy_count": 0,
        "unhealthy_count": 0,
        "timestamp": NOW,
        "checked_by": health.manager.pod_id,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("applied", [False, True])
async def test_lost_delete_reply_is_best_effort(health: HealthRedis, applied: bool) -> None:
    assert await health.manager.acquire_health_check_lock() is True

    async def delete_timeout(name: str) -> None:
        if applied:
            await health.raw.delete(name)
        raise RedisTimeoutError("reply lost during delete")

    with patch.object(health.cache.init_async_client(), "delete", new=delete_timeout):
        assert await health.manager.release_health_check_lock() is None
    assert await health.raw.get(LOCK) == (None if applied else json.dumps(health.manager.pod_id).encode())


@pytest.mark.asyncio
async def test_release_keeps_existing_get_then_delete_race(health: HealthRedis) -> None:
    assert await health.manager.acquire_health_check_lock() is True

    async def replace_after_read(name: str) -> bytes | None:
        previous: Final = await health.raw.get(name)
        await health.raw.set(name, json.dumps("replacement"), ex=60)
        return previous

    with patch.object(health.cache.init_async_client(), "get", new=replace_after_read):
        await health.manager.release_health_check_lock()
    assert await health.raw.get(LOCK) is None


@pytest.mark.asyncio
async def test_no_redis_runs_locally_without_polling() -> None:
    manager: Final = SharedHealthCheckManager()

    async def no_wait(delay: float) -> None:
        pytest.fail(f"unexpected local polling delay {delay}")

    with patch("asyncio.sleep", new=no_wait):
        assert await manager.perform_shared_health_check([]) == ([], [], {})
    assert await manager.acquire_health_check_lock() is False
    assert await manager.is_health_check_in_progress() is False
    assert await manager.get_cached_health_check_results() is None
    assert await manager.release_health_check_lock() is None
    assert await manager.cache_health_check_results([], []) is None
    assert await manager.get_health_check_status() == {
        "pod_id": manager.pod_id,
        "redis_available": False,
        "lock_ttl": 60,
        "cache_ttl": 300,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw, expected",
    [
        (b"42", 42),
        (b"[]", []),
        (b"false", False),
        (b'""', ""),
        (b"'legacy'", "legacy"),
        (b"null", None),
        (b"broken", None),
        (b"\xff", None),
    ],
)
async def test_legacy_owner_decoding_keeps_lock_presence(health: HealthRedis, raw: bytes, expected: object) -> None:
    await health.raw.set(LOCK, raw)
    assert await health.manager.is_health_check_in_progress() is (expected is not None)
    status: Final = await health.manager.get_health_check_status()
    assert status["lock_owner"] == expected
    assert status["lock_in_progress"] is (expected is not None)
    await health.manager.release_health_check_lock()
    assert await health.raw.get(LOCK) == raw


@pytest.mark.asyncio
async def test_keys_expire_at_caller_selected_seconds(health: HealthRedis, monkeypatch: pytest.MonkeyPatch) -> None:
    assert await health.manager.acquire_health_check_lock() is True
    await health.manager.cache_health_check_results([], [])
    monkeypatch.setattr(time, "time", lambda: NOW + 61)
    assert await health.raw.pttl(LOCK) == -2
    assert await health.raw.pttl(RESULTS) == 239_000
    assert await health.manager.is_health_check_in_progress() is False
    monkeypatch.setattr(time, "time", lambda: NOW + 301)
    assert await health.manager.get_cached_health_check_results() is None
    assert await health.raw.pttl(RESULTS) == -2


def _redis_metric(name: str) -> float:
    samples: Final = chain.from_iterable(metric.samples for metric in REGISTRY.collect())
    return sum(sample.value for sample in samples if sample.name == name)


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [1.9, 0.5, 0.9999999, 1.9999999, -0.5])
async def test_fractional_ttl_matches_legacy_seconds(health: HealthRedis, ttl: float) -> None:
    setattr(health.manager, "lock_ttl", ttl)
    setattr(health.manager, "health_check_ttl", ttl)
    assert await health.manager.acquire_health_check_lock() is (int(ttl) > 0)
    await health.manager.cache_health_check_results([], [])
    expected: Final = int(ttl) * 1000 if int(ttl) > 0 else -2
    assert await health.raw.pttl(LOCK) == expected
    assert await health.raw.pttl(RESULTS) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, -1])
async def test_invalid_ttl_emits_failure_events(health: HealthRedis, monkeypatch: pytest.MonkeyPatch, ttl: int) -> None:
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    health.manager.lock_ttl = ttl
    health.manager.health_check_ttl = ttl
    pending_before: Final = asyncio.all_tasks()
    before: Final = _redis_metric("litellm_redis_failed_requests_total")
    breaker_before: Final = _redis_metric("litellm_redis_circuit_breaker_failures_total")
    assert await health.manager.acquire_health_check_lock() is False
    await health.manager.cache_health_check_results([], [])
    await asyncio.gather(*(asyncio.all_tasks() - pending_before))
    assert _redis_metric("litellm_redis_failed_requests_total") - before == 2
    assert _redis_metric("litellm_redis_circuit_breaker_failures_total") == breaker_before


@pytest.mark.asyncio
async def test_zero_ttl_outage_still_opens_breaker(health: HealthRedis) -> None:
    health.server.connected = False
    health.manager.lock_ttl = 0
    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        assert await health.manager.acquire_health_check_lock() is False
    health.server.connected = True
    health.manager.lock_ttl = 60
    assert await health.manager.acquire_health_check_lock() is False
    assert await health.raw.get(LOCK) is None
