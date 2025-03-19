import asyncio
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock

from litellm.caching.redis_cache import RedisCache


@pytest.mark.parametrize("namespace", [None, "test"])
@pytest.mark.asyncio
async def test_redis_cache_async_increment(namespace, monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()

    # Make sure the mock can be used as an async context manager
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    assert redis_cache is not None

    expected_key = "test:test" if namespace else "test"

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        # Call async_set_cache
        await redis_cache.async_increment(key=expected_key, value=1)

        # Verify that the set method was called on the mock Redis instance
        mock_redis_instance.incrbyfloat.assert_called_once_with(
            name=expected_key, amount=1
        )


@pytest.mark.asyncio
async def test_redis_client_init_with_socket_timeout(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "my-fake-host")
    redis_cache = RedisCache(socket_timeout=1.0)
    assert redis_cache.redis_kwargs["socket_timeout"] == 1.0
    client = redis_cache.init_async_client()
    assert client is not None
    assert client.connection_pool.connection_kwargs["socket_timeout"] == 1.0
