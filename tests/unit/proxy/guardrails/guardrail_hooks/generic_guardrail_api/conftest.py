from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final
from unittest.mock import patch
from uuid import uuid4

import pytest
from fakeredis import FakeRedis, FakeServer
from fakeredis.aioredis import FakeConnection
from redis.asyncio import BlockingConnectionPool
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.redis_cache import RedisCache


class _SetReplyLostConnection(FakeConnection):
    async def send_command(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: Redis transport signature
        self.sent_set = bool(args) and str(args[0]).upper() == "SET"
        await super().send_command(*args, **kwargs)

    async def read_response(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: Redis transport signature
        response: Final = await super().read_response(*args, **kwargs)
        if getattr(self, "sent_set", False):
            raise RedisTimeoutError("the SET was applied but its reply never arrived")
        return response


@asynccontextmanager
async def _fake_redis(connection_class: type[FakeConnection]) -> AsyncIterator[tuple[RedisCache, FakeServer]]:
    server: Final = FakeServer()
    client: Final = FakeRedis(server=server)
    pool: Final = BlockingConnectionPool(connection_class=connection_class, server=server)
    try:
        with (
            patch("redis.Redis.from_url", return_value=client),
            patch("redis.asyncio.BlockingConnectionPool.from_url", return_value=pool),
        ):
            yield RedisCache(url=f"redis://guardrail-scope-{uuid4().hex}.example:6379/0"), server
    finally:
        await pool.aclose()
        client.close()


@pytest.fixture
async def shared_redis() -> AsyncIterator[tuple[RedisCache, FakeServer]]:
    async with _fake_redis(FakeConnection) as redis:
        yield redis


@pytest.fixture
async def redis_losing_set_replies() -> AsyncIterator[tuple[RedisCache, FakeServer]]:
    async with _fake_redis(_SetReplyLostConnection) as redis:
        yield redis
