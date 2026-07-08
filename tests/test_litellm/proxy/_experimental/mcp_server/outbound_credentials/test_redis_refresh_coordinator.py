"""Tests for the cross-replica refresh coordinator: winner refreshes, losers wait then re-read."""

import asyncio

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
    RedisRefreshCoordinator,
)

_KEY = "mcp:refresh_lock:u:s"


class _FakeLock:
    def __init__(self, acquired, held_sequence=()):
        self._acquired = acquired
        self._held = list(held_sequence)
        self.acquire_calls = []
        self.extend_calls = []
        self.released = []
        self.extended = asyncio.Event()

    async def acquire(self, key, token, ttl_seconds):
        self.acquire_calls.append((key, token, ttl_seconds))
        return self._acquired

    async def release(self, key, token):
        self.released.append((key, token))

    async def extend(self, key, token, ttl_seconds):
        self.extend_calls.append((key, token, ttl_seconds))
        self.extended.set()
        return True

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
async def test_winner_acquires_and_releases_with_the_same_token_and_never_rereads():
    lock = _FakeLock(acquired=LockAcquisition.ACQUIRED)
    refreshed = OAuthToken(access_token="new")
    reread_calls = []

    async def refresh():
        return refreshed

    async def reread():
        reread_calls.append(1)
        return None

    coord = RedisRefreshCoordinator(lock, new_token=lambda: "tok")
    result = await coord.run("u", "s", refresh, reread)
    assert result is refreshed
    assert lock.acquire_calls == [(_KEY, "tok", 10.0)]
    # released with the SAME token it acquired with, so it can only delete its own lock
    assert lock.released == [(_KEY, "tok")]
    assert reread_calls == []


@pytest.mark.asyncio
async def test_winner_releases_even_when_refresh_raises():
    lock = _FakeLock(acquired=LockAcquisition.ACQUIRED)

    async def refresh():
        raise RuntimeError("boom")

    async def reread():
        return None

    coord = RedisRefreshCoordinator(lock, new_token=lambda: "tok")
    with pytest.raises(RuntimeError):
        await coord.run("u", "s", refresh, reread)
    assert lock.released == [(_KEY, "tok")]  # the lock is freed even on failure


@pytest.mark.asyncio
async def test_winner_renews_the_lock_while_refresh_runs():
    lock = _FakeLock(acquired=LockAcquisition.ACQUIRED)
    refresh_finished = asyncio.Event()
    sleep_started = asyncio.Event()
    sleep_can_finish = asyncio.Event()
    sleep_count = 0

    async def sleep(seconds):
        nonlocal sleep_count
        sleep_count += 1
        assert seconds == 5.0
        sleep_started.set()
        if sleep_count > 1:
            await asyncio.Event().wait()
            return
        await sleep_can_finish.wait()

    async def refresh():
        await refresh_finished.wait()
        return OAuthToken(access_token="new")

    async def reread():
        return None

    coord = RedisRefreshCoordinator(lock, new_token=lambda: "tok", sleep=sleep)
    task = asyncio.create_task(coord.run("u", "s", refresh, reread))
    await sleep_started.wait()
    sleep_can_finish.set()
    await lock.extended.wait()
    refresh_finished.set()

    result = await task
    assert result is not None and result.access_token == "new"
    assert lock.extend_calls == [(_KEY, "tok", 10.0)]
    assert lock.released == [(_KEY, "tok")]


@pytest.mark.asyncio
async def test_loser_waits_for_the_holder_then_rereads_persisted_token():
    clock = _Clock()
    lock = _FakeLock(acquired=LockAcquisition.HELD, held_sequence=[True, True, False])
    refresh_calls = []

    async def refresh():
        refresh_calls.append(1)
        return None

    async def reread():
        return OAuthToken(access_token="persisted-by-winner")

    coord = RedisRefreshCoordinator(lock, clock=clock, sleep=_advancing_sleep(clock), wait_timeout_seconds=100.0)
    result = await coord.run("u", "s", refresh, reread)
    assert result is not None and result.access_token == "persisted-by-winner"
    assert refresh_calls == []  # the loser never refreshes - it reads the winner's result
    assert lock.released == []  # ...and never releases a lock it does not own


@pytest.mark.asyncio
async def test_loser_rereads_after_timeout_if_holder_never_releases():
    clock = _Clock()
    lock = _FakeLock(acquired=LockAcquisition.HELD, held_sequence=[True] * 100)

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


@pytest.mark.asyncio
async def test_lock_backend_error_refreshes_anyway_instead_of_serving_stale():
    # Regression: on a total lock-backend outage every worker gets ERROR (not HELD). If ERROR were
    # treated as "someone else holds it", no worker would refresh and all would re-read the still-
    # expired token and serve a stale bearer upstream. ERROR must instead refresh anyway.
    lock = _FakeLock(acquired=LockAcquisition.ERROR)
    refreshed = OAuthToken(access_token="refreshed-despite-redis-down")
    refresh_calls = []
    reread_calls = []

    async def refresh():
        refresh_calls.append(1)
        return refreshed

    async def reread():
        reread_calls.append(1)
        return OAuthToken(access_token="stale-expired-token")

    result = await RedisRefreshCoordinator(lock).run("u", "s", refresh, reread)
    assert result is refreshed  # served the fresh token, not the stale re-read
    assert refresh_calls == [1]
    assert reread_calls == []  # never fell back to re-reading the expired token
    assert lock.released == []  # nothing was acquired, so nothing is released


def test_wait_timeout_outlasts_the_holders_max_lock_hold():
    # Regression: a loser's wait must exceed the holder's max lock-hold - the renewal budget plus one
    # lease tail. When wait_timeout_seconds was merely == lock_ttl_seconds, a loser bailed at the lease
    # TTL while the holder was still renewing mid-refresh and re-read the still-expired token.
    coord = RedisRefreshCoordinator(_FakeLock(acquired=LockAcquisition.HELD))
    assert coord.wait_timeout_seconds > coord.refresh_budget_seconds + coord.lock_ttl_seconds


@pytest.mark.asyncio
async def test_loser_with_default_timeout_waits_past_the_lease_ttl_for_the_holder():
    # With the DEFAULT wait_timeout (not an inflated test value), a loser keeps waiting while the holder
    # still holds the lock past lock_ttl_seconds, then re-reads what the holder persisted - rather than
    # giving up at the lease TTL mid-refresh. At 0.5s/poll, 20 polls reach the 10s lease TTL.
    clock = _Clock()
    lock = _FakeLock(acquired=LockAcquisition.HELD, held_sequence=[True] * 30 + [False])
    refresh_calls = []

    async def refresh():
        refresh_calls.append(1)
        return None

    async def reread():
        return OAuthToken(access_token="persisted-by-winner")

    coord = RedisRefreshCoordinator(lock, clock=clock, sleep=_advancing_sleep(clock, step=0.5))
    result = await coord.run("u", "s", refresh, reread)
    assert result is not None and result.access_token == "persisted-by-winner"
    assert refresh_calls == []  # the loser never refreshes - it waits for the holder
    assert lock._held == []  # waited until the holder released (whole sequence drained)
    assert clock.t > coord.lock_ttl_seconds  # ...past the lease TTL, not bailing at it


@pytest.mark.asyncio
async def test_winner_stops_renewing_after_the_refresh_budget():
    # A holder whose refresh outruns the budget stops renewing - the lock then lapses for losers -
    # instead of extending the lease forever and wedging the single-flight on one slow request.
    clock = _Clock()
    lock = _FakeLock(acquired=LockAcquisition.ACQUIRED)
    release_refresh = asyncio.Event()

    async def sleep(seconds):
        clock.t += seconds  # renewal sleeps lock_ttl/2 each cycle

    async def refresh():
        await release_refresh.wait()  # outlives the budget
        return OAuthToken(access_token="new")

    async def reread():
        return None

    coord = RedisRefreshCoordinator(
        lock,
        new_token=lambda: "tok",
        clock=clock,
        sleep=sleep,
        lock_ttl_seconds=10.0,
        refresh_budget_seconds=20.0,
    )
    task = asyncio.create_task(coord.run("u", "s", refresh, reread))
    for _ in range(200):  # let the bounded renewal drain to its budget (clock 5..20)
        if clock.t >= 20.0:
            break
        await asyncio.sleep(0)
    extends_during_budget = len(lock.extend_calls)
    release_refresh.set()
    result = await task
    assert result is not None and result.access_token == "new"
    assert extends_during_budget == 4  # renewed at clock 5, 10, 15, 20 then stopped
    assert len(lock.extend_calls) == 4  # no further renewals after the budget
