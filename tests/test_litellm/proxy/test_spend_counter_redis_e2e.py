"""
Live Redis E2E tests for spend-counter read amplification (LIT-3263).

Two layers:
1. **Hypothesis** — raw RedisCache sequential vs batched MGET (passes on any branch).
2. **Auth fix** — real ``get_current_spend`` / ``prefetch_spend_counters`` path:
   fails when the fix is not applied (shows amplification), passes after ``git stash pop``.

Default target: litellm_redis from docker compose (localhost:6379).

Run locally:
  docker compose up -d redis
  uv run pytest tests/test_litellm/proxy/test_spend_counter_redis_e2e.py -vv
"""

from __future__ import annotations

import asyncio
import importlib
import os
import socket
import uuid
from typing import Awaitable, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
import redis.asyncio as aioredis

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache

# Mirrors a heavy team-key auth path: team + 2 windows + key window + team member.
NUM_SPEND_COUNTERS = 6
CONCURRENT_AUTH_REQUESTS = 24


def _redis_host_port() -> Tuple[str, int]:
    return os.getenv("REDIS_HOST", "localhost"), int(os.getenv("REDIS_PORT", "6379"))


def _optional_redis_password() -> str | None:
    password = os.getenv("REDIS_PASSWORD")
    if password and password.strip():
        return password
    return None


def _redis_reachable() -> bool:
    host, port = _redis_host_port()
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(autouse=True)
def _passwordless_redis_for_e2e(monkeypatch):
    """litellm_redis from docker compose has no AUTH."""
    if os.getenv("REDIS_E2E_USE_PASSWORD", "").lower() != "true":
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)


pytestmark = [
    pytest.mark.skipif(
        not _redis_reachable(),
        reason=(
            "Redis not reachable — start litellm_redis via "
            "`docker compose up -d redis` (defaults to localhost:6379)"
        ),
    ),
]


def _build_counter_keys(run_id: str) -> Dict[str, float]:
    return {
        f"spend:team:team-{run_id}": 10.0,
        f"spend:team:team-{run_id}:window:24h": 5.0,
        f"spend:team:team-{run_id}:window:30d": 5.0,
        f"spend:key:key-{run_id}": 1.0,
        f"spend:key:key-{run_id}:window:24h": 0.5,
        f"spend:team_member:user-{run_id}:team-{run_id}": 2.0,
    }


def _make_redis_cache(*, max_connections: int) -> RedisCache:
    host, port = _redis_host_port()
    kwargs = {
        "host": host,
        "port": port,
        "max_connections": max_connections,
        "socket_timeout": 5.0,
    }
    password = _optional_redis_password()
    if password:
        kwargs["password"] = password

    with patch("litellm.caching.redis_cache.RedisCache._setup_health_pings"):
        return RedisCache(**kwargs)


async def _seed_counters(redis_cache: RedisCache, counter_values: Dict[str, float]) -> None:
    for key, value in counter_values.items():
        await redis_cache.async_set_cache(key=key, value=value, ttl=300)


async def _delete_counters(redis_cache: RedisCache, counter_keys: List[str]) -> None:
    for key in counter_keys:
        await redis_cache.async_delete_cache(key=key)


async def _admin_client() -> aioredis.Redis:
    host, port = _redis_host_port()
    password = _optional_redis_password()
    url = (
        f"redis://:{password}@{host}:{port}/0"
        if password
        else f"redis://{host}:{port}/0"
    )
    return aioredis.from_url(url)


async def _redis_command_delta(admin: aioredis.Redis, coro: Awaitable[None]) -> int:
    before = int((await admin.info("stats"))["total_commands_processed"])
    await coro
    after = int((await admin.info("stats"))["total_commands_processed"])
    return after - before


async def _sequential_mget_reads(
    redis_cache: RedisCache,
    counter_requests: Dict[str, float],
) -> None:
    """
    Pre-fix auth pattern: each budget check resolves one counter independently,
    issuing one Redis MGET per key (via async_batch_get_cache with a single key).
    """
    for counter_key in counter_requests:
        values = await redis_cache.async_batch_get_cache([counter_key])
        assert values.get(counter_key) == counter_requests[counter_key]


async def _batched_mget_reads(
    redis_cache: RedisCache,
    counter_requests: Dict[str, float],
) -> None:
    """
    Post-fix auth pattern: resolve all counters in one Redis MGET.
    """
    counter_keys = list(counter_requests.keys())
    values = await redis_cache.async_batch_get_cache(counter_keys)
    for counter_key, expected in counter_requests.items():
        assert values.get(counter_key) == expected


def _auth_prefetch_fix_available() -> bool:
    proxy_server = importlib.import_module("litellm.proxy.proxy_server")
    return callable(getattr(proxy_server, "prefetch_spend_counters", None))


def _load_auth_spend_api() -> Tuple[
    Callable[..., Awaitable[float]],
    Optional[Callable[..., Awaitable[None]]],
    Optional[Callable[[], None]],
]:
    proxy_server = importlib.import_module("litellm.proxy.proxy_server")
    get_current_spend = proxy_server.get_current_spend
    prefetch = getattr(proxy_server, "prefetch_spend_counters", None)
    clear_prefetch = getattr(proxy_server, "clear_spend_counter_prefetch", None)
    return get_current_spend, prefetch, clear_prefetch


async def _auth_before_path(
    get_current_spend: Callable[..., Awaitable[float]],
    clear_prefetch: Optional[Callable[[], None]],
    counter_requests: Dict[str, float],
) -> None:
    """Pre-fix auth: each budget check calls get_current_spend independently."""
    if clear_prefetch is not None:
        clear_prefetch()
    for counter_key, fallback in counter_requests.items():
        spend = await get_current_spend(
            counter_key=counter_key,
            fallback_spend=fallback,
        )
        assert spend == counter_requests[counter_key]


async def _auth_after_path(
    get_current_spend: Callable[..., Awaitable[float]],
    prefetch_spend_counters: Callable[..., Awaitable[None]],
    clear_prefetch: Callable[[], None],
    counter_requests: Dict[str, float],
) -> None:
    """Post-fix auth: one prefetch MGET, then cache hits for each counter."""
    clear_prefetch()
    await prefetch_spend_counters(counter_requests)
    for counter_key, fallback in counter_requests.items():
        spend = await get_current_spend(
            counter_key=counter_key,
            fallback_spend=fallback,
        )
        assert spend == counter_requests[counter_key]


@pytest.mark.asyncio
async def test_auth_prefetch_fix_reduces_read_amplification():
    """
    End-to-end auth path against real Redis.

    - Without the LIT-3263 fix: fails after measuring high read amplification.
    - With prefetch_spend_counters applied: passes (before >> after command count).
    """
    import litellm.proxy.proxy_server as ps

    get_current_spend, prefetch_spend_counters, clear_prefetch = _load_auth_spend_api()

    run_id = uuid.uuid4().hex[:8]
    counter_requests = _build_counter_keys(run_id)
    counter_keys = list(counter_requests.keys())

    redis_cache = _make_redis_cache(max_connections=20)
    await _seed_counters(redis_cache, counter_requests)
    counter_cache = DualCache(redis_cache=redis_cache)
    admin = await _admin_client()

    orig_counter = ps.spend_counter_cache
    ps.spend_counter_cache = counter_cache

    try:
        before_commands = await _redis_command_delta(
            admin,
            _auth_before_path(
                get_current_spend,
                clear_prefetch,
                counter_requests,
            ),
        )

        assert before_commands >= NUM_SPEND_COUNTERS, (
            f"pre-fix auth path expected at least {NUM_SPEND_COUNTERS} Redis commands, "
            f"got {before_commands}"
        )

        if not _auth_prefetch_fix_available():
            pytest.fail(
                f"Read amplification confirmed: pre-fix auth path issued "
                f"{before_commands} Redis commands for {NUM_SPEND_COUNTERS} spend "
                f"counters (expected ~{NUM_SPEND_COUNTERS} sequential reads). "
                f"Apply the LIT-3263 prefetch fix (`git stash pop`) to batch into ~1 MGET."
            )

        assert prefetch_spend_counters is not None
        assert clear_prefetch is not None

        after_commands = await _redis_command_delta(
            admin,
            _auth_after_path(
                get_current_spend,
                prefetch_spend_counters,
                clear_prefetch,
                counter_requests,
            ),
        )

        assert after_commands <= 2, (
            f"prefetch auth path should batch into one MGET (+/- overhead), "
            f"got {after_commands}"
        )
        assert before_commands >= after_commands * 3, (
            f"prefetch fix must cut read amplification: before={before_commands}, "
            f"after={after_commands}"
        )
    finally:
        if clear_prefetch is not None:
            clear_prefetch()
        ps.spend_counter_cache = orig_counter
        await _delete_counters(redis_cache, counter_keys)
        await redis_cache.disconnect()
        await admin.aclose()


@pytest.mark.asyncio
async def test_auth_prefetch_fix_reduces_concurrent_read_amplification():
    """
    Concurrent auth-shaped load: prefetch fix must keep Redis commands ~N× lower
    than the sequential get_current_spend path.
    """
    import litellm.proxy.proxy_server as ps

    get_current_spend, prefetch_spend_counters, clear_prefetch = _load_auth_spend_api()

    run_id = uuid.uuid4().hex[:8]
    counter_requests = _build_counter_keys(run_id)
    counter_keys = list(counter_requests.keys())
    admin = await _admin_client()

    async def _run_concurrent(workload: Callable[[], Awaitable[None]]) -> None:
        await asyncio.gather(*[workload() for _ in range(CONCURRENT_AUTH_REQUESTS)])

    async def _run_phase(workload: Callable[[], Awaitable[None]]) -> int:
        redis_cache = _make_redis_cache(max_connections=20)
        await _seed_counters(redis_cache, counter_requests)
        counter_cache = DualCache(redis_cache=redis_cache)
        ps.spend_counter_cache = counter_cache
        try:
            return await _redis_command_delta(admin, _run_concurrent(workload))
        finally:
            if clear_prefetch is not None:
                clear_prefetch()
            await _delete_counters(redis_cache, counter_keys)
            await redis_cache.disconnect()

    orig_counter = ps.spend_counter_cache

    async def before_workload() -> None:
        await _auth_before_path(get_current_spend, clear_prefetch, counter_requests)

    async def after_workload() -> None:
        assert prefetch_spend_counters is not None
        assert clear_prefetch is not None
        await _auth_after_path(
            get_current_spend,
            prefetch_spend_counters,
            clear_prefetch,
            counter_requests,
        )

    try:
        before_commands = await _run_phase(before_workload)

        expected_before = CONCURRENT_AUTH_REQUESTS * NUM_SPEND_COUNTERS
        assert before_commands >= expected_before * 0.8, (
            f"concurrent pre-fix auth path expected ~{expected_before} Redis commands, "
            f"got {before_commands}"
        )

        if not _auth_prefetch_fix_available():
            pytest.fail(
                f"Concurrent read amplification confirmed: pre-fix auth path issued "
                f"{before_commands} Redis commands for {CONCURRENT_AUTH_REQUESTS} "
                f"requests × {NUM_SPEND_COUNTERS} counters (expected "
                f"~{expected_before}). Apply the LIT-3263 prefetch fix (`git stash pop`)."
            )

        after_commands = await _run_phase(after_workload)
    finally:
        if clear_prefetch is not None:
            clear_prefetch()
        ps.spend_counter_cache = orig_counter
        await admin.aclose()

    expected_after = CONCURRENT_AUTH_REQUESTS

    assert after_commands <= expected_after * 3, (
        f"concurrent prefetch auth path expected ~{expected_after} Redis commands "
        f"(+ connection overhead), got {after_commands}"
    )
    assert before_commands >= after_commands * 2, (
        f"prefetch fix must cut concurrent read amplification: "
        f"before={before_commands}, after={after_commands} "
        f"(expected ~{expected_before} vs ~{expected_after})"
    )


@pytest.mark.asyncio
async def test_live_redis_sequential_mget_uses_more_commands_than_batched_mget():
    """
    Single auth-shaped request against real Redis: sequential per-key MGET vs one batched MGET.
    """
    run_id = uuid.uuid4().hex[:8]
    counter_requests = _build_counter_keys(run_id)
    counter_keys = list(counter_requests.keys())

    redis_cache = _make_redis_cache(max_connections=20)
    await _seed_counters(redis_cache, counter_requests)
    admin = await _admin_client()

    try:
        sequential_commands = await _redis_command_delta(
            admin,
            _sequential_mget_reads(redis_cache, counter_requests),
        )
        batched_commands = await _redis_command_delta(
            admin,
            _batched_mget_reads(redis_cache, counter_requests),
        )

        assert sequential_commands >= NUM_SPEND_COUNTERS, (
            f"expected at least {NUM_SPEND_COUNTERS} Redis commands on sequential path, "
            f"got {sequential_commands}"
        )
        assert batched_commands <= 2, (
            f"batched path should use one MGET (+/- overhead), got {batched_commands}"
        )
        assert sequential_commands >= batched_commands * 3, (
            f"sequential ({sequential_commands} cmds) should be >> batched "
            f"({batched_commands} cmds) — MGET batching reduces read amplification"
        )
    finally:
        await _delete_counters(redis_cache, counter_keys)
        await redis_cache.disconnect()
        await admin.aclose()


@pytest.mark.asyncio
async def test_live_redis_concurrent_sequential_mget_amplifies_commands_vs_batched():
    """
    Under concurrent auth-shaped load, sequential per-key MGET must generate many more
    Redis commands than a single batched MGET per request.
    """
    run_id = uuid.uuid4().hex[:8]
    counter_requests = _build_counter_keys(run_id)
    counter_keys = list(counter_requests.keys())
    admin = await _admin_client()

    async def _run_concurrent(
        redis_cache: RedisCache,
        workload: Callable[[RedisCache], Awaitable[None]],
    ) -> None:
        await asyncio.gather(
            *[workload(redis_cache) for _ in range(CONCURRENT_AUTH_REQUESTS)]
        )

    async def _run_phase(
        workload: Callable[[RedisCache], Awaitable[None]],
    ) -> int:
        redis_cache = _make_redis_cache(max_connections=20)
        await _seed_counters(redis_cache, counter_requests)
        try:
            return await _redis_command_delta(
                admin,
                _run_concurrent(redis_cache, workload),
            )
        finally:
            await _delete_counters(redis_cache, counter_keys)
            await redis_cache.disconnect()

    try:
        sequential_commands = await _run_phase(
            lambda cache: _sequential_mget_reads(cache, counter_requests)
        )
        batched_commands = await _run_phase(
            lambda cache: _batched_mget_reads(cache, counter_requests)
        )
    finally:
        await admin.aclose()

    expected_sequential = CONCURRENT_AUTH_REQUESTS * NUM_SPEND_COUNTERS
    expected_batched = CONCURRENT_AUTH_REQUESTS

    assert sequential_commands >= expected_sequential * 0.8, (
        f"concurrent sequential path expected ~{expected_sequential} Redis commands, "
        f"got {sequential_commands}"
    )
    assert batched_commands <= expected_batched * 3, (
        f"concurrent batched path expected ~{expected_batched} Redis commands "
        f"(+ connection overhead), got {batched_commands}"
    )
    assert sequential_commands >= batched_commands * 2, (
        f"under concurrent load, sequential ({sequential_commands} cmds) must be "
        f"at least 2× batched ({batched_commands} cmds) — "
        f"expected ~{expected_sequential} vs ~{expected_batched}"
    )
