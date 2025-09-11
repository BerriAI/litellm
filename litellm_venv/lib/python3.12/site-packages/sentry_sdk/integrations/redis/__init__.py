from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.redis.consts import _DEFAULT_MAX_DATA_SIZE
from sentry_sdk.integrations.redis.rb import _patch_rb
from sentry_sdk.integrations.redis.redis import _patch_redis
from sentry_sdk.integrations.redis.redis_cluster import _patch_redis_cluster
from sentry_sdk.integrations.redis.redis_py_cluster_legacy import _patch_rediscluster
from sentry_sdk.utils import logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


class RedisIntegration(Integration):
    identifier = "redis"

    def __init__(self, max_data_size=_DEFAULT_MAX_DATA_SIZE, cache_prefixes=None):
        # type: (int, Optional[list[str]]) -> None
        self.max_data_size = max_data_size
        self.cache_prefixes = cache_prefixes if cache_prefixes is not None else []

    @staticmethod
    def setup_once():
        # type: () -> None
        try:
            from redis import StrictRedis, client
        except ImportError:
            raise DidNotEnable("Redis client not installed")

        _patch_redis(StrictRedis, client)
        _patch_redis_cluster()
        _patch_rb()

        try:
            _patch_rediscluster()
        except Exception:
            logger.exception("Error occurred while patching `rediscluster` library")
