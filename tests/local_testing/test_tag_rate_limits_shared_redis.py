"""
Real-Redis execution coverage for tag_rate_limits_shared's two Lua scripts.

Lives outside tests/test_litellm/ (which can only contain mocked tests, see
tests/test_litellm/readme.md) because fakeredis has no EVALSHA support without
the optional lupa dependency; these run against a throwaway local redis-server
instead (same idiom as tests/test_litellm/proxy/hooks/test_batch_enqueued_tokens.py's
test_redis_lua_path_full_lifecycle).
"""

import shutil
import socket
import subprocess
import time
from collections.abc import Iterator

import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.hooks.tag_rate_limits_shared import (
    TAG_RL_CHECK_AND_INCR_SCRIPT,
    TAG_RL_DECR_FLOOR_ZERO_SCRIPT,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def redis_port() -> Iterator[int]:
    if shutil.which("redis-server") is None:
        pytest.skip("requires a local redis-server binary to exercise the Lua script path")
    port = _free_port()
    proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
        else:
            proc.terminate()
            pytest.skip("local redis-server did not become ready in time")
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _register(port: int, script: str):
    return RedisCache(host="127.0.0.1", port=port).async_register_script(script)


@pytest.mark.asyncio
async def test_check_and_incr_admits_under_limit_and_sets_ttl(redis_port):
    run = _register(redis_port, TAG_RL_CHECK_AND_INCR_SCRIPT)
    admitted, new_value = await run(keys=["bucket1"], args=[5, 1, 60, 0])
    assert (admitted, new_value) == (1, 1)


@pytest.mark.asyncio
async def test_check_and_incr_rejects_over_limit_without_incrementing(redis_port):
    run = _register(redis_port, TAG_RL_CHECK_AND_INCR_SCRIPT)
    await run(keys=["bucket2"], args=[1, 1, 60, 0])
    rejected, current = await run(keys=["bucket2"], args=[1, 1, 60, 0])
    assert (rejected, current) == (0, 1)


@pytest.mark.asyncio
async def test_check_and_incr_requests_ttl_is_set_once_and_not_refreshed(redis_port):
    """refresh_ttl=0 (the `requests` fixed-window semantics): the epoch-bucketed
    TTL must be set on the first write and left alone after, or the bucket
    outlives the epoch it's meant to reset at."""
    run = _register(redis_port, TAG_RL_CHECK_AND_INCR_SCRIPT)
    await run(keys=["bucket3"], args=[5, 1, 100, 0])
    raw = RedisCache(host="127.0.0.1", port=redis_port)
    client = raw.init_async_client()
    first_ttl = await client.ttl("bucket3")
    await run(keys=["bucket3"], args=[5, 1, 5, 0])
    second_ttl = await client.ttl("bucket3")
    assert first_ttl > 5
    assert second_ttl > 5  # unchanged by the second call's much shorter ttl arg


@pytest.mark.asyncio
async def test_check_and_incr_concurrency_ttl_refreshes_on_every_admission(redis_port):
    """refresh_ttl=1 (the `concurrency` reservation semantics): every admission
    must push the crash-safety TTL back out, or a long-lived burst of traffic
    expires the whole counter mid-flight."""
    run = _register(redis_port, TAG_RL_CHECK_AND_INCR_SCRIPT)
    await run(keys=["bucket4"], args=[5, 1, 5, 1])
    await run(keys=["bucket4"], args=[5, 1, 100, 1])
    raw = RedisCache(host="127.0.0.1", port=redis_port)
    client = raw.init_async_client()
    ttl = await client.ttl("bucket4")
    assert ttl > 5


@pytest.mark.asyncio
async def test_decr_floor_zero_floors_at_zero_and_deletes_the_key(redis_port):
    run_incr = _register(redis_port, TAG_RL_CHECK_AND_INCR_SCRIPT)
    run_decr = _register(redis_port, TAG_RL_DECR_FLOOR_ZERO_SCRIPT)
    await run_incr(keys=["bucket5"], args=[5, 1, 60, 1])

    floored = await run_decr(keys=["bucket5"], args=[-5])
    assert floored == 0

    raw = RedisCache(host="127.0.0.1", port=redis_port)
    client = raw.init_async_client()
    assert await client.get("bucket5") is None  # floored via DEL, not a TTL-less `SET 0`


@pytest.mark.asyncio
async def test_decr_floor_zero_decrements_normally_when_result_stays_non_negative(redis_port):
    run_incr = _register(redis_port, TAG_RL_CHECK_AND_INCR_SCRIPT)
    run_decr = _register(redis_port, TAG_RL_DECR_FLOOR_ZERO_SCRIPT)
    await run_incr(keys=["bucket6"], args=[5, 3, 60, 1])

    remaining = await run_decr(keys=["bucket6"], args=[-1])
    assert remaining == 2
