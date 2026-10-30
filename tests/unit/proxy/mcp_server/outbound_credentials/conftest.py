from typing import Final, NoReturn

import fakeredis
import pytest
import pytest_asyncio
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.mcp_server.outbound_credentials.redis_distributed_lock import RedisDistributedLock
from litellm.redis.stores.mcp_credential_refresh import TransportMcpRefreshLockStore
from tests._support.fake_redis import fake_redis_cache


def lost_write_reply(response: object, **kwargs: object) -> NoReturn:
    raise RedisTimeoutError("reply lost after applying the write")


class RedisClock:
    def __init__(self) -> None:
        self.now: float = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def redis_clock(monkeypatch: pytest.MonkeyPatch) -> RedisClock:
    clock: Final = RedisClock()
    monkeypatch.setattr("time.time", clock)
    return clock


@pytest.fixture
def refresh_server(redis_clock: RedisClock) -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def refresh_raw(refresh_server: fakeredis.FakeServer) -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=refresh_server)


@pytest_asyncio.fixture(loop_scope="function")
async def refresh_cache(refresh_server: fakeredis.FakeServer) -> RedisCache:
    return fake_redis_cache(refresh_server, namespace="ns")


@pytest.fixture
def refresh_lock(refresh_cache: RedisCache) -> RedisDistributedLock:
    return RedisDistributedLock(TransportMcpRefreshLockStore(refresh_cache.transport()))
