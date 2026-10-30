import asyncio
from typing import Final

import fakeredis
import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken
from litellm.proxy.mcp_server.outbound_credentials.redis_distributed_lock import RedisDistributedLock
from litellm.proxy.mcp_server.outbound_credentials.redis_refresh_coordinator import RedisRefreshCoordinator
from tests.unit.proxy.mcp_server.outbound_credentials.conftest import RedisClock, lost_write_reply

_KEY: Final = "ns:mcp:refresh_lock:u:s"


async def _never_refresh() -> OAuthToken | None:
    pytest.fail("a loser must not refresh")


async def _never_reread() -> OAuthToken | None:
    pytest.fail("a winner must not reread")


@pytest.mark.asyncio
async def test_winner_acquires_and_releases_with_the_same_token_and_never_rereads(
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
async def test_winner_releases_even_when_refresh_raises(
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
async def test_winner_renews_the_lock_while_refresh_runs(
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
async def test_loser_waits_for_the_holder_then_rereads_persisted_token(
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
async def test_lock_backend_error_refreshes_anyway_instead_of_serving_stale(
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
async def test_winner_stops_renewing_after_the_refresh_budget(
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
