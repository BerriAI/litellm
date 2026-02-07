"""
Redis Cluster Cache implementation

Key differences:
- RedisClient NEEDs to be re-used across requests, adds 3000ms latency if it's re-created
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

from litellm.caching.redis_cache import RedisCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span
    from redis.asyncio import Redis, RedisCluster
    from redis.asyncio.client import Pipeline

    pipeline = Pipeline
    async_redis_client = Redis
    Span = Union[_Span, Any]
else:
    pipeline = Any
    async_redis_client = Any
    Span = Any


class RedisClusterCache(RedisCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_async_redis_cluster_client: Optional[RedisCluster] = None
        self.redis_sync_redis_cluster_client: Optional[RedisCluster] = None

    def init_async_client(self):
        from redis.asyncio import RedisCluster

        from .._redis import get_redis_async_client

        if self.redis_async_redis_cluster_client:
            return self.redis_async_redis_cluster_client

        _redis_client = get_redis_async_client(
            connection_pool=self.async_redis_conn_pool, **self.redis_kwargs
        )
        if isinstance(_redis_client, RedisCluster):
            self.redis_async_redis_cluster_client = _redis_client

        return _redis_client

    def _run_redis_mget_operation(self, keys: List[str]) -> List[Any]:
        """
        Overrides `_run_redis_mget_operation` in redis_cache.py
        """
        return self.redis_client.mget_nonatomic(keys=keys)  # type: ignore

    async def _async_run_redis_mget_operation(self, keys: List[str]) -> List[Any]:
        """
        Overrides `_async_run_redis_mget_operation` in redis_cache.py
        """
        async_redis_cluster_client = self.init_async_client()
        return await async_redis_cluster_client.mget_nonatomic(keys=keys)  # type: ignore
    
    async def test_connection(self) -> dict:
        """
        Test the Redis Cluster connection.
        
        Returns:
            dict: {"status": "success" | "failed", "message": str, "error": Optional[str]}
        """
        try:
            import redis.asyncio as redis_async
            from redis.cluster import ClusterNode

            # Create ClusterNode objects from startup_nodes
            cluster_kwargs = self.redis_kwargs.copy()
            startup_nodes = cluster_kwargs.pop("startup_nodes", [])
            
            new_startup_nodes: List[ClusterNode] = []
            for item in startup_nodes:
                new_startup_nodes.append(ClusterNode(**item))
            
            # Create a fresh Redis Cluster client with current settings
            redis_client = redis_async.RedisCluster(
                startup_nodes=new_startup_nodes, **cluster_kwargs  # type: ignore
            )
            
            # Test the connection
            ping_result = await redis_client.ping()  # type: ignore[attr-defined, misc]

            # Close the connection
            await redis_client.aclose()  # type: ignore[attr-defined]
            
            if ping_result:
                return {
                    "status": "success",
                    "message": "Redis Cluster connection test successful"
                }
            else:
                return {
                    "status": "failed",
                    "message": "Redis Cluster ping returned False"
                }
        except Exception as e:
            from litellm._logging import verbose_logger
            verbose_logger.error(f"Redis Cluster connection test failed: {str(e)}")
            return {
                "status": "failed",
                "message": f"Redis Cluster connection failed: {str(e)}",
                "error": str(e)
            }