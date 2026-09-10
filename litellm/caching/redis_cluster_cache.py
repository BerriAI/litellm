"""
Redis Cluster Cache implementation

Key differences:
- RedisClient NEEDs to be re-used across requests, adds 3000ms latency if it's re-created
"""

from typing import TYPE_CHECKING, Any, Final

from litellm.caching.redis_cache import RedisCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span
    from redis.asyncio import Redis, RedisCluster
    from redis.asyncio.client import Pipeline

    pipeline = Pipeline
    async_redis_client = Redis
    Span = _Span | Any
else:
    pipeline = Any
    async_redis_client = Any
    Span = Any


class RedisClusterCache(RedisCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_async_redis_cluster_client: RedisCluster | None = None
        self.redis_sync_redis_cluster_client: RedisCluster | None = None

    def init_async_client(self):
        from redis.asyncio import RedisCluster

        from .._redis import get_redis_async_client

        if self.redis_async_redis_cluster_client:
            return self.redis_async_redis_cluster_client

        _redis_client: Final = get_redis_async_client(connection_pool=self.async_redis_conn_pool, **self.redis_kwargs)
        if isinstance(_redis_client, RedisCluster):
            self.redis_async_redis_cluster_client = _redis_client

        return _redis_client

    def _run_redis_mget_operation(self, keys: list[str]) -> list[Any]:
        """
        Overrides `_run_redis_mget_operation` in redis_cache.py
        """
        return self.redis_client.mget_nonatomic(keys=keys)

    async def _async_run_redis_mget_operation(self, keys: list[str]) -> list[Any]:
        """
        Overrides `_async_run_redis_mget_operation` in redis_cache.py
        """
        async_redis_cluster_client: Final = self.init_async_client()
        return await async_redis_cluster_client.mget_nonatomic(keys=keys)

    async def test_connection(self) -> dict:
        """
        Test the Redis Cluster connection.

        Returns:
            dict: {"status": "success" | "failed", "message": str, "error": Optional[str]}
        """
        try:
            from .._redis import get_redis_async_client

            redis_client: Final = get_redis_async_client(**self.redis_kwargs)

            # Test the connection
            ping_result: Final = await redis_client.ping()

            # Close the connection
            await redis_client.aclose()

            if ping_result:
                return {
                    "status": "success",
                    "message": "Redis Cluster connection test successful",
                }
            else:
                return {
                    "status": "failed",
                    "message": "Redis Cluster ping returned False",
                }
        except Exception as e:
            from litellm._logging import verbose_logger

            verbose_logger.error("Redis Cluster connection test failed: %s", e)
            return {
                "status": "failed",
                "message": f"Redis Cluster connection failed: {e}",
                "error": str(e),
            }
