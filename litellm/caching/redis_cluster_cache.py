"""
Redis Cluster Cache implementation

Key differences:
- RedisClient NEEDs to be re-used across requests, adds 3000ms latency if it's re-created
"""

from typing import TYPE_CHECKING, Any, Optional

from litellm.caching.redis_cache import RedisCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span
    from redis.asyncio import Redis, RedisCluster
    from redis.asyncio.client import Pipeline

    pipeline = Pipeline
    async_redis_client = Redis
    Span = _Span
else:
    pipeline = Any
    async_redis_client = Any
    Span = Any


class RedisClusterCache(RedisCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_cluster_client: Optional[RedisCluster] = None

    def init_async_client(self):
        from redis.asyncio import RedisCluster

        from .._redis import get_redis_async_client

        if self.redis_cluster_client:
            return self.redis_cluster_client

        _redis_client = get_redis_async_client(
            connection_pool=self.async_redis_conn_pool, **self.redis_kwargs
        )
        if isinstance(_redis_client, RedisCluster):
            self.redis_cluster_client = _redis_client
        return _redis_client
