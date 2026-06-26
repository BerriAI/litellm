"""Tests for the cross-replica refresh coordinator: winner refreshes, losers wait then re-read."""

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    RedisRefreshCoordinator,
)

_KEY = "mcp:refresh_lock:u:s"


class _FakeLock:
    def __init__(self, acquired, held_sequence=()):
        self._acquired = acquired
        self._held = list(held_sequence)
        self.acquired_keys = []
        self.released = []

    async def acquire(self, key, ttl_seconds):
        self.acquired_keys.append((key, ttl_seconds))
        return self._acquired

    async def release(self, key):
        self.released.append(key)

    async def is_held(self, key):
        return self._held.pop(0) if self._held else False


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _advancing_sleep(clock, step=0.5):
    async def sleep(_seconds):
        clock.t += step

    return sleep


@pytest.mark.asyncio
async def test_winner_refreshes_then_releases_and_never_rereads():
    lock = _FakeLock(acquired=True)
    refreshed = OAuthToken(access_token="new")
    reread_calls = []

    async def refresh():
        return refreshed

    async def reread():
        reread_calls.append(1)
        return None

    result = await RedisRefreshCoordinator(lock).run("u", "s", refresh, reread)
    assert result is refreshed
    assert lock.acquired_keys == [(_KEY, 10.0)]
    assert lock.released == [_KEY]
    assert reread_calls == []


@pytest.mark.asyncio
async def test_winner_releases_even_when_refresh_raises():
    lock = _FakeLock(acquired=True)

    async def refresh():
        raise RuntimeError("boom")

    async def reread():
        return None

    with pytest.raises(RuntimeError):
        await RedisRefreshCoordinator(lock).run("u", "s", refresh, reread)
    assert lock.released == [_KEY]  # the lock is freed even on failure


@pytest.mark.asyncio
async def test_loser_waits_for_the_holder_then_rereads_persisted_token():
    clock = _Clock()
    lock = _FakeLock(acquired=False, held_sequence=[True, True, False])
    refresh_calls = []

    async def refresh():
        refresh_calls.append(1)
        return None

    async def reread():
        return OAuthToken(access_token="persisted-by-winner")

    coord = RedisRefreshCoordinator(
        lock, clock=clock, sleep=_advancing_sleep(clock), wait_timeout_seconds=100.0
    )
    result = await coord.run("u", "s", refresh, reread)
    assert result is not None and result.access_token == "persisted-by-winner"
    assert (
        refresh_calls == []
    )  # the loser never refreshes - it reads the winner's result
    assert lock.released == []  # ...and never holds the lock


@pytest.mark.asyncio
async def test_loser_rereads_after_timeout_if_holder_never_releases():
    clock = _Clock()
    lock = _FakeLock(
        acquired=False, held_sequence=[True] * 100
    )  # holder never releases

    async def refresh():
        return None

    async def reread():
        return OAuthToken(access_token="whatever-is-there")

    coord = RedisRefreshCoordinator(
        lock,
        clock=clock,
        sleep=_advancing_sleep(clock, step=0.5),
        wait_timeout_seconds=1.0,
        poll_interval_seconds=0.1,
    )
    result = await coord.run("u", "s", refresh, reread)
    # Gave up waiting (bounded) and returned what's persisted rather than blocking forever.
    assert result is not None and result.access_token == "whatever-is-there"
