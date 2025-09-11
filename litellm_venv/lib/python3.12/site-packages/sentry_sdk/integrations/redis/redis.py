"""
Instrumentation for Redis

https://github.com/redis/redis-py
"""

from sentry_sdk.integrations.redis._sync_common import (
    patch_redis_client,
    patch_redis_pipeline,
)
from sentry_sdk.integrations.redis.modules.queries import _set_db_data

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Sequence


def _get_redis_command_args(command):
    # type: (Any) -> Sequence[Any]
    return command[0]


def _patch_redis(StrictRedis, client):  # noqa: N803
    # type: (Any, Any) -> None
    patch_redis_client(
        StrictRedis,
        is_cluster=False,
        set_db_data_fn=_set_db_data,
    )
    patch_redis_pipeline(
        client.Pipeline,
        is_cluster=False,
        get_command_args_fn=_get_redis_command_args,
        set_db_data_fn=_set_db_data,
    )
    try:
        strict_pipeline = client.StrictPipeline
    except AttributeError:
        pass
    else:
        patch_redis_pipeline(
            strict_pipeline,
            is_cluster=False,
            get_command_args_fn=_get_redis_command_args,
            set_db_data_fn=_set_db_data,
        )

    try:
        import redis.asyncio
    except ImportError:
        pass
    else:
        from sentry_sdk.integrations.redis._async_common import (
            patch_redis_async_client,
            patch_redis_async_pipeline,
        )

        patch_redis_async_client(
            redis.asyncio.client.StrictRedis,
            is_cluster=False,
            set_db_data_fn=_set_db_data,
        )
        patch_redis_async_pipeline(
            redis.asyncio.client.Pipeline,
            False,
            _get_redis_command_args,
            set_db_data_fn=_set_db_data,
        )
