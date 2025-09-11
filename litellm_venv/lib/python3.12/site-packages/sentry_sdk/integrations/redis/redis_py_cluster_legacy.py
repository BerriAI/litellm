"""
Instrumentation for redis-py-cluster
The project redis-py-cluster is EOL and was integrated into redis-py starting from version 4.1.0 (Dec 26, 2021).

https://github.com/grokzen/redis-py-cluster
"""

from sentry_sdk.integrations.redis._sync_common import (
    patch_redis_client,
    patch_redis_pipeline,
)
from sentry_sdk.integrations.redis.modules.queries import _set_db_data
from sentry_sdk.integrations.redis.utils import _parse_rediscluster_command


def _patch_rediscluster():
    # type: () -> None
    try:
        import rediscluster  # type: ignore
    except ImportError:
        return

    patch_redis_client(
        rediscluster.RedisCluster,
        is_cluster=True,
        set_db_data_fn=_set_db_data,
    )

    # up to v1.3.6, __version__ attribute is a tuple
    # from v2.0.0, __version__ is a string and VERSION a tuple
    version = getattr(rediscluster, "VERSION", rediscluster.__version__)

    # StrictRedisCluster was introduced in v0.2.0 and removed in v2.0.0
    # https://github.com/Grokzen/redis-py-cluster/blob/master/docs/release-notes.rst
    if (0, 2, 0) < version < (2, 0, 0):
        pipeline_cls = rediscluster.pipeline.StrictClusterPipeline
        patch_redis_client(
            rediscluster.StrictRedisCluster,
            is_cluster=True,
            set_db_data_fn=_set_db_data,
        )
    else:
        pipeline_cls = rediscluster.pipeline.ClusterPipeline

    patch_redis_pipeline(
        pipeline_cls,
        is_cluster=True,
        get_command_args_fn=_parse_rediscluster_command,
        set_db_data_fn=_set_db_data,
    )
