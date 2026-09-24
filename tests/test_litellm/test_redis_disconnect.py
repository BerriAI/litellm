from unittest.mock import AsyncMock, Mock

import pytest

from litellm.caching.redis_cache import RedisCache


@pytest.mark.asyncio
async def test_disconnect_allows_missing_async_pool():
    cache = RedisCache.__new__(RedisCache)
    cache.async_redis_conn_pool = None
    cache.redis_client = Mock()

    await cache.disconnect()

    cache.redis_client.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_disconnect_closes_present_async_pool():
    cache = RedisCache.__new__(RedisCache)
    cache.async_redis_conn_pool = Mock(disconnect=AsyncMock())
    cache.redis_client = Mock()

    await cache.disconnect()

    cache.async_redis_conn_pool.disconnect.assert_awaited_once_with(inuse_connections=True)
