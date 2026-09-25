"""Tests for the Redis lock: acquire NX/PX with a token, owner-only extend and release, namespacing.

The happy paths run against a real ``redis-server``, once standalone and once as a one-node cluster,
because every operation is a Lua script and the cluster client registers scripts on its own path; the
degrade paths use a fake ``RedisCache`` whose scripts fail.
"""

import asyncio
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import redis

from litellm.caching.redis_cache import RedisCache
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisDistributedLock,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)

_ALL_HASH_SLOTS: Final = (0, 16383)


@dataclass(frozen=True, slots=True)
class _Server:
    port: int
    cluster: bool


def _wait_until(ready: Callable[[], bool], failure: str, log_path: Path) -> None:
    for _ in range(100):
        if ready():
            return
        time.sleep(0.1)
    pytest.fail(f"{failure}: {log_path.read_text()}")


def _pings(admin: redis.Redis) -> bool:
    try:
        return bool(admin.ping())
    except redis.ConnectionError:
        return False


def _cluster_is_ok(admin: redis.Redis) -> bool:
    return b"cluster_state:ok" in admin.execute_command("CLUSTER", "INFO")


@pytest.fixture(params=("standalone", "cluster"))
def redis_server(
    request: pytest.FixtureRequest, tmp_path: Path, unused_tcp_port_factory: Callable[[], int]
) -> Iterator[_Server]:
    server: Final = shutil.which("redis-server")
    if server is None:
        pytest.skip("redis-server is required for the lock's Lua scripts")
    cluster: Final = request.param == "cluster"
    port: Final = unused_tcp_port_factory()
    log_path: Final = tmp_path / "redis.log"
    config: Final = tmp_path / "redis.conf"
    config.write_text(
        f'bind 127.0.0.1\nport {port}\ndir "{tmp_path}"\nsave ""\nappendonly no\n'
        + (
            f"cluster-enabled yes\ncluster-config-file nodes.conf\ncluster-port {unused_tcp_port_factory()}\n"
            if cluster
            else ""
        )
    )
    with log_path.open("w") as log:
        process: Final = subprocess.Popen((server, str(config)), stdout=log, stderr=subprocess.STDOUT)
        try:
            with redis.Redis(host="127.0.0.1", port=port, socket_timeout=1, socket_connect_timeout=1) as admin:
                _wait_until(lambda: _pings(admin), "Redis did not start", log_path)
                if cluster:
                    admin.execute_command("CLUSTER", "ADDSLOTSRANGE", *_ALL_HASH_SLOTS)
                    _wait_until(lambda: _cluster_is_ok(admin), "Cluster never became ok", log_path)
            yield _Server(port=port, cluster=cluster)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture
def redis_cache(redis_server: _Server, monkeypatch: pytest.MonkeyPatch) -> RedisCache:
    for name in (
        "REDIS_URL",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_PASSWORD",
        "REDIS_CLUSTER_NODES",
        "REDIS_SENTINEL_NODES",
    ):
        monkeypatch.delenv(name, raising=False)
    if redis_server.cluster:
        return RedisCache(startup_nodes=[{"host": "127.0.0.1", "port": redis_server.port}], namespace="tenant")
    return RedisCache(host="127.0.0.1", port=redis_server.port, namespace="tenant")


@pytest.fixture
def raw(redis_server: _Server) -> Iterator[redis.Redis | redis.RedisCluster]:
    if redis_server.cluster:
        with redis.RedisCluster(host="127.0.0.1", port=redis_server.port) as cluster_client:
            yield cluster_client
        return
    with redis.Redis(host="127.0.0.1", port=redis_server.port) as client:
        yield client


class _FailingScriptsCache:
    """A RedisCache whose every registered script fails at run time, as a dead Redis does."""

    def async_register_script(self, script: str) -> Callable[..., object]:
        async def run(keys: Sequence[str], args: Sequence[str]) -> object:
            raise ConnectionError("redis down")

        return run


async def test_acquire_wins_once_and_reports_held_to_the_next_caller(redis_cache: RedisCache, raw: redis.Redis | redis.RedisCluster) -> None:
    lock = RedisDistributedLock(redis_cache)

    assert await lock.acquire("k", "tok-1", 10.0) is LockAcquisition.ACQUIRED
    assert await lock.acquire("k", "tok-2", 10.0) is LockAcquisition.HELD
    assert raw.get("tenant:k") == b"tok-1", "the key carries the cache namespace and the winner's token"
    assert 0 < raw.pttl("tenant:k") <= 10_000


async def test_acquire_reports_error_on_redis_error_distinct_from_held() -> None:
    lock = RedisDistributedLock(_FailingScriptsCache())  # pyright: ignore[reportArgumentType]  # duck-typed fake

    assert await lock.acquire("k", "tok", 10.0) is LockAcquisition.ERROR


async def test_release_deletes_only_when_the_token_matches(redis_cache: RedisCache) -> None:
    lock = RedisDistributedLock(redis_cache)
    await lock.acquire("k", "owner-B", 10.0)

    await lock.release("k", "owner-A")
    assert await lock.is_held("k") is True, "a stale token must not delete another worker's lock"

    await lock.release("k", "owner-B")
    assert await lock.is_held("k") is False


async def test_extend_refreshes_ttl_only_when_the_token_matches(redis_cache: RedisCache, raw: redis.Redis | redis.RedisCluster) -> None:
    lock = RedisDistributedLock(redis_cache)
    await lock.acquire("k", "owner-B", 1.0)

    assert await lock.extend("k", "owner-A", 30.0) is False
    assert raw.pttl("tenant:k") <= 1_000
    assert await lock.extend("k", "owner-B", 30.0) is True
    assert raw.pttl("tenant:k") > 1_000


async def test_extend_and_release_degrade_on_redis_error() -> None:
    lock = RedisDistributedLock(_FailingScriptsCache())  # pyright: ignore[reportArgumentType]  # duck-typed fake

    assert await lock.extend("k", "tok", 10.0) is False
    await lock.release("k", "tok")


async def test_lock_expires_on_its_own_after_the_ttl(redis_cache: RedisCache) -> None:
    lock = RedisDistributedLock(redis_cache)
    await lock.acquire("k", "tok", 0.05)
    assert await lock.is_held("k") is True

    await asyncio.sleep(0.2)

    assert await lock.is_held("k") is False
    assert await lock.acquire("k", "tok-2", 10.0) is LockAcquisition.ACQUIRED


async def test_is_held_degrades_to_false_on_redis_error() -> None:
    lock = RedisDistributedLock(_FailingScriptsCache())  # pyright: ignore[reportArgumentType]  # duck-typed fake

    assert await lock.is_held("k") is False
