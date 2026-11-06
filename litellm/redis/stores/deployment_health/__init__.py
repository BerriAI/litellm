from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm.constants import HEALTH_CHECKS_TARGET
from litellm.redis.stores.deployment_health.handle import DeploymentHealthHandle

if TYPE_CHECKING:
    from litellm.redis.stores.connection import RedisStores

__all__ = ["DeploymentHealthStore"]


class DeploymentHealthStore:
    """The background health checks' per-deployment states: memory per worker, Redis shared by every pod.
    The store owns the key and its encoding; merging and staleness stay with ``DeploymentHealthCache``."""

    def __init__(self, handle: DeploymentHealthHandle) -> None:
        self._handle: Final = handle

    @property
    def has_redis(self) -> bool:
        return self._handle.has_redis

    def attach_redis(self, redis: RedisStores) -> None:
        """Adds the Redis tier once, keeping memory, as ``DualCache.attach_redis_cache`` does."""
        if redis.client is not None:
            self._handle.attach(redis.client)

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def get(self) -> object:
        return await self._handle.get()

    @with_service_target(HEALTH_CHECKS_TARGET)
    def get_sync(self) -> object:
        return self._handle.get_sync()

    @with_service_target(HEALTH_CHECKS_TARGET)
    def get_shared_sync(self) -> object:
        """What every pod merged so far, from Redis alone. No Redis, a failed read and an open breaker read as None."""
        return self._handle.get_shared_sync()

    @with_service_target(HEALTH_CHECKS_TARGET)
    def set_sync(self, states: object, ttl: float) -> None:
        self._handle.set_sync(states, ttl)
