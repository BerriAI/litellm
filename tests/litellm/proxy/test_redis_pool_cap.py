"""
Regression tests for Redis connection pool max_connections cap.

Validates that:
1. get_redis_connection_pool applies DEFAULT_REDIS_MAX_CONNECTIONS when not specified
2. User-specified max_connections is respected
3. Both URL-based and kwargs-based paths are covered
"""

from unittest.mock import patch

import pytest


def test_redis_pool_default_max_connections():
    """should apply default max_connections when not explicitly set"""
    from litellm._redis import get_redis_connection_pool
    from litellm.constants import DEFAULT_REDIS_MAX_CONNECTIONS

    with patch.dict(
        "os.environ",
        {"REDIS_HOST": "localhost", "REDIS_PORT": "6379", "REDIS_PASSWORD": "test"},
    ):
        pool = get_redis_connection_pool(
            host="localhost", port=6379, password="test"
        )
        assert pool.max_connections == DEFAULT_REDIS_MAX_CONNECTIONS


def test_redis_pool_custom_max_connections():
    """should respect user-specified max_connections"""
    from litellm._redis import get_redis_connection_pool

    pool = get_redis_connection_pool(
        host="localhost", port=6379, password="test", max_connections=50
    )
    assert pool.max_connections == 50


def test_redis_pool_url_default_max_connections():
    """should apply default max_connections for URL-based pools"""
    from litellm._redis import get_redis_connection_pool
    from litellm.constants import DEFAULT_REDIS_MAX_CONNECTIONS

    pool = get_redis_connection_pool(url="redis://localhost:6379/0")
    assert pool.max_connections == DEFAULT_REDIS_MAX_CONNECTIONS


def test_redis_pool_url_custom_max_connections():
    """should respect user-specified max_connections for URL-based pools"""
    from litellm._redis import get_redis_connection_pool

    pool = get_redis_connection_pool(
        url="redis://localhost:6379/0", max_connections=200
    )
    assert pool.max_connections == 200
