"""Tests for the cross-replica refresh coordinator: winner refreshes, losers wait then re-read."""

import asyncio
from typing import Final

import fakeredis
import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy.mcp_server.outbound_credentials.redis_distributed_lock import RedisDistributedLock
from litellm.proxy.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
    RedisRefreshCoordinator,
)
from litellm.redis.stores.mcp_credential_refresh import RefreshLockTarget
from tests.unit.proxy.mcp_server.outbound_credentials.conftest import RedisClock, lost_write_reply

_TARGET = RefreshLockTarget(user_id="u", server_id="s")


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

    coord = RedisRefreshCoordinator(lock, new_token=lambda: "tok")
    result = await coord.run("u", "s", refresh, reread)
    assert result is refreshed
    assert lock.acquire_calls == [(_TARGET, "tok", 10.0)]
    # released with the SAME token it acquired with, so it can only delete its own lock
    assert lock.released == [(_TARGET, "tok")]
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
    assert lock.released == [(_TARGET, "tok")]  # the lock is freed even on failure


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
    assert lock.extend_calls == [(_TARGET, "tok", 10.0)]
    assert lock.released == [(_TARGET, "tok")]


@pytest.mark.asyncio
async def test_loser_waits_for_the_holder_then_rereads_persisted_token():
    clock = _Clock()
    lock = _FakeLock(acquired=LockAcquisition.HELD, held_sequence=[True, True, False])
    refresh_calls = []

    async def refresh():
        refresh_calls.append(1)

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


_KEY: Final = "ns:mcp:refresh_lock:u:s"


async def _never_refresh() -> OAuthToken | None:
    pytest.fail("a loser must not refresh")


async def _never_reread() -> OAuthToken | None:
    pytest.fail("a winner must not reread")


@pytest.mark.asyncio
async def test_winner_acquires_and_releases_with_the_same_token_and_never_rereads_with_redis_store(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis
) -> None:
    refreshed: Final = OAuthToken(access_token="new")

    async def refresh() -> OAuthToken:
        assert await refresh_raw.get(_KEY) == b"tok"
        assert await refresh_raw.pttl(_KEY) == 10235
        return refreshed

    coord: Final = RedisRefreshCoordinator(refresh_lock, new_token=lambda: "tok", lock_ttl_seconds=10.2345)
    assert await coord.run("u", "s", refresh, _never_reread) is refreshed
    assert await refresh_raw.get(_KEY) is None


@pytest.mark.asyncio
async def test_winner_releases_even_when_refresh_raises_with_redis_store(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis
) -> None:
    async def refresh() -> OAuthToken | None:
        assert await refresh_raw.get(_KEY) == b"tok"
        raise RuntimeError("boom")

    coord: Final = RedisRefreshCoordinator(refresh_lock, new_token=lambda: "tok")
    with pytest.raises(RuntimeError, match="boom"):
        await coord.run("u", "s", refresh, _never_reread)
    assert await refresh_raw.get(_KEY) is None


@pytest.mark.asyncio
async def test_winner_renews_the_lock_while_refresh_runs_with_redis_store(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis, redis_clock: RedisClock
) -> None:
    refreshed: Final = OAuthToken(access_token="new")
    start: Final = redis_clock()
    renewed: Final = asyncio.Event()
    finish: Final = asyncio.Event()

    async def sleep(seconds: float) -> None:
        assert seconds == 5.0
        redis_clock.advance(seconds)
        if redis_clock() >= start + 10:
            renewed.set()
            await asyncio.Event().wait()

    async def refresh() -> OAuthToken:
        await finish.wait()
        return refreshed

    coord: Final = RedisRefreshCoordinator(refresh_lock, new_token=lambda: "tok", sleep=sleep, clock=redis_clock)
    task: Final = asyncio.create_task(coord.run("u", "s", refresh, _never_reread))
    try:
        await renewed.wait()
        assert await refresh_raw.get(_KEY) == b"tok"
        assert await refresh_raw.pttl(_KEY) == 5000
    finally:
        finish.set()
        assert await task is refreshed
    assert await refresh_raw.get(_KEY) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("holder_seconds", (1.0, 15.0))
async def test_loser_waits_for_the_holder_then_rereads_persisted_token_with_redis_store(
    refresh_lock: RedisDistributedLock,
    refresh_raw: fakeredis.FakeAsyncRedis,
    redis_clock: RedisClock,
    holder_seconds: float,
) -> None:
    await refresh_raw.set(_KEY, "other-worker", px=30_000)
    start: Final = redis_clock()
    persisted: Final = OAuthToken(access_token="persisted-by-winner")

    async def sleep(seconds: float) -> None:
        assert seconds == 0.05
        redis_clock.advance(0.5)
        if redis_clock() >= start + holder_seconds:
            await refresh_raw.delete(_KEY)

    async def reread() -> OAuthToken:
        assert redis_clock() == start + holder_seconds
        assert await refresh_raw.get(_KEY) is None
        return persisted

    coord: Final = RedisRefreshCoordinator(refresh_lock, sleep=sleep, clock=redis_clock)
    assert await coord.run("u", "s", _never_refresh, reread) is persisted


@pytest.mark.asyncio
async def test_loser_rereads_after_timeout_without_releasing_the_holder(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis, redis_clock: RedisClock
) -> None:
    await refresh_raw.set(_KEY, "other-worker", px=10_000)
    start: Final = redis_clock()
    persisted: Final = OAuthToken(access_token="persisted")

    async def sleep(seconds: float) -> None:
        redis_clock.advance(0.5)

    async def reread() -> OAuthToken:
        assert redis_clock() == start + 1.0
        return persisted

    coord: Final = RedisRefreshCoordinator(refresh_lock, sleep=sleep, clock=redis_clock, wait_timeout_seconds=1.0)
    assert await coord.run("u", "s", _never_refresh, reread) is persisted
    assert await refresh_raw.get(_KEY) == b"other-worker"
    assert await refresh_raw.pttl(_KEY) == 9000


@pytest.mark.asyncio
async def test_lock_backend_error_refreshes_anyway_instead_of_serving_stale_with_redis_store(
    refresh_server: fakeredis.FakeServer, refresh_lock: RedisDistributedLock
) -> None:
    refresh_server.connected = False
    refreshed: Final = OAuthToken(access_token="refreshed-despite-redis-down")

    async def refresh() -> OAuthToken:
        return refreshed

    assert await RedisRefreshCoordinator(refresh_lock).run("u", "s", refresh, _never_reread) is refreshed


@pytest.mark.asyncio
async def test_unknown_applied_acquire_refreshes_without_releasing_an_unconfirmed_lease(
    refresh_cache: RedisCache, refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis
) -> None:
    client: Final = refresh_cache.init_async_client()
    assert isinstance(client, fakeredis.FakeAsyncRedis)
    client.set_response_callback("SET", lost_write_reply)
    refreshed: Final = OAuthToken(access_token="new")

    async def refresh() -> OAuthToken:
        assert await refresh_raw.get(_KEY) == b"tok"
        return refreshed

    coord: Final = RedisRefreshCoordinator(refresh_lock, new_token=lambda: "tok")
    assert await coord.run("u", "s", refresh, _never_reread) is refreshed
    assert await refresh_raw.get(_KEY) == b"tok"
    assert await refresh_raw.pttl(_KEY) == 10000


@pytest.mark.asyncio
async def test_expired_owner_does_not_release_a_replacement_lease(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis, redis_clock: RedisClock
) -> None:
    refreshed: Final = OAuthToken(access_token="new")

    async def refresh() -> OAuthToken:
        assert await refresh_raw.get(_KEY) == b"tok"
        redis_clock.advance(11.0)
        assert await refresh_raw.set(_KEY, "replacement", nx=True, px=30_000) is True
        return refreshed

    coord: Final = RedisRefreshCoordinator(refresh_lock, new_token=lambda: "tok", clock=redis_clock)
    assert await coord.run("u", "s", refresh, _never_reread) is refreshed
    assert await refresh_raw.get(_KEY) == b"replacement"
    assert await refresh_raw.pttl(_KEY) == 30000


@pytest.mark.asyncio
async def test_winner_stops_renewing_after_the_refresh_budget_with_redis_store(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis, redis_clock: RedisClock
) -> None:
    start: Final = redis_clock()
    refreshed: Final = OAuthToken(access_token="new")
    budget_reached: Final = asyncio.Event()

    async def sleep(seconds: float) -> None:
        assert seconds == 5.0
        redis_clock.advance(seconds)
        assert redis_clock() <= start + 20
        if redis_clock() == start + 20:
            budget_reached.set()

    async def refresh() -> OAuthToken:
        await budget_reached.wait()
        for _ in range(100):
            if await refresh_raw.pttl(_KEY) == 10000:
                break
            await asyncio.sleep(0)
        else:
            pytest.fail("the final renewal did not retain its lease tail")
        redis_clock.advance(11.0)
        assert await refresh_raw.get(_KEY) is None
        return refreshed

    coord: Final = RedisRefreshCoordinator(refresh_lock, new_token=lambda: "tok", clock=redis_clock, sleep=sleep)
    assert coord.wait_timeout_seconds > coord.refresh_budget_seconds + coord.lock_ttl_seconds
    assert await coord.run("u", "s", refresh, _never_reread) is refreshed
    assert await refresh_raw.get(_KEY) is None


@pytest.mark.asyncio
async def test_loser_rereads_if_redis_fails_while_waiting(
    refresh_server: fakeredis.FakeServer,
    refresh_lock: RedisDistributedLock,
    refresh_raw: fakeredis.FakeAsyncRedis,
    redis_clock: RedisClock,
) -> None:
    await refresh_raw.set(_KEY, "other-worker", px=10_000)
    persisted: Final = OAuthToken(access_token="persisted")

    async def sleep(seconds: float) -> None:
        refresh_server.connected = False

    async def reread() -> OAuthToken:
        return persisted

    coord: Final = RedisRefreshCoordinator(refresh_lock, sleep=sleep, clock=redis_clock)
    assert await coord.run("u", "s", _never_refresh, reread) is persisted
    refresh_server.connected = True
    assert await refresh_raw.get(_KEY) == b"other-worker"


@pytest.mark.asyncio
async def test_runtime_coordinator_uses_the_auth_cache_redis_connection(
    refresh_cache: RedisCache, refresh_raw: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server.outbound_credentials.runtime_refresh_coordinator import runtime_refresh_coordinator

    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache(redis_cache=refresh_cache))
    coord: Final = runtime_refresh_coordinator()
    assert coord is not None
    refreshed: Final = OAuthToken(access_token="new")

    async def refresh() -> OAuthToken:
        assert await refresh_raw.get(_KEY) is not None
        assert await refresh_raw.pttl(_KEY) == 10000
        return refreshed

    assert await coord.run("u", "s", refresh, _never_reread) is refreshed
    assert await refresh_raw.get(_KEY) is None
