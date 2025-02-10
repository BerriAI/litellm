"""
Custom stub for redis.asyncio.cluster to add missing attributes for RedisCluster and ClusterPipeline.
"""

from redis.asyncio.client import Redis

class RedisCluster(Redis):
    pass
