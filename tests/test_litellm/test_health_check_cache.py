import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.health_endpoints._health_endpoints import (
    _REDIS_CACHE_INDEX_INFO_TTL_SECONDS,
    _get_redis_cache_index_info_with_ttl,
    _redis_cache_index_info,
)


@pytest.mark.asyncio
async def test_redis_cache_index_info_caches_result():
    """Test that _get_redis_cache_index_info_with_ttl caches results within TTL."""
    global _redis_cache_index_info
    _redis_cache_index_info.clear()

    call_count = 0

    async def mock_index_info():
        nonlocal call_count
        call_count += 1
        return {"index": "test_index", "status": "ok"}

    with patch(
        "litellm.cache.cache._index_info",
        new_callable=AsyncMock,
        side_effect=mock_index_info,
    ):
        result1 = await _get_redis_cache_index_info_with_ttl()
        result2 = await _get_redis_cache_index_info_with_ttl()

        assert result1 == {"index": "test_index", "status": "ok"}
        assert result2 == result1
        assert call_count == 1, "Should only call _index_info once within TTL"


@pytest.mark.asyncio
async def test_redis_cache_index_info_refreshes_after_ttl():
    """Test that cache is refreshed after TTL expires."""
    global _redis_cache_index_info
    _redis_cache_index_info.clear()

    call_count = 0

    async def mock_index_info():
        nonlocal call_count
        call_count += 1
        return {"index": f"test_index_{call_count}"}

    with patch(
        "litellm.cache.cache._index_info",
        new_callable=AsyncMock,
        side_effect=mock_index_info,
    ):
        result1 = await _get_redis_cache_index_info_with_ttl()
        assert call_count == 1

        _redis_cache_index_info["timestamp"] = time.time() - (_REDIS_CACHE_INDEX_INFO_TTL_SECONDS + 1)

        result2 = await _get_redis_cache_index_info_with_ttl()
        assert call_count == 2
        assert result1["index"] == "test_index_1"
        assert result2["index"] == "test_index_2"


@pytest.mark.asyncio
async def test_redis_cache_index_info_handles_errors():
    """Test that errors are cached with the same TTL."""
    global _redis_cache_index_info
    _redis_cache_index_info.clear()

    async def mock_index_info_error():
        raise ValueError("Connection failed")

    with patch(
        "litellm.cache.cache._index_info",
        new_callable=AsyncMock,
        side_effect=mock_index_info_error,
    ):
        result = await _get_redis_cache_index_info_with_ttl()
        assert "index does not exist - error" in result
        assert "Connection failed" in result

        result2 = await _get_redis_cache_index_info_with_ttl()
        assert result == result2
