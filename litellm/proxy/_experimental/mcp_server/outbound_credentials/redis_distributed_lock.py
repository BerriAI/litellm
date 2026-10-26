"""Map cache lease outcomes to the refresh coordinator's best-effort lock policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)
from litellm.types.caching import CacheFailure


@dataclass(frozen=True, slots=True)
class RedisDistributedLock:
    cache: RedisCache

    async def acquire(self, key: str, token: str, ttl_seconds: float) -> LockAcquisition:
        result: Final = await self.cache.async_acquire_lease(key=key, token=token, ttl_seconds=ttl_seconds)
        match result:
            case "acquired":
                return LockAcquisition.ACQUIRED
            case "held":
                return LockAcquisition.HELD
            case CacheFailure():
                verbose_logger.warning("RedisDistributedLock.acquire failed: %s", result.message)
                return LockAcquisition.ERROR
            case _:
                assert_never(result)

    async def extend(self, key: str, token: str, ttl_seconds: float) -> bool:
        result: Final = await self.cache.async_extend_lease(key=key, token=token, ttl_seconds=ttl_seconds)
        if isinstance(result, CacheFailure):
            verbose_logger.warning("RedisDistributedLock.extend failed: %s", result.message)
            return False
        return result

    async def release(self, key: str, token: str) -> None:
        result: Final = await self.cache.async_release_lease(key=key, token=token)
        if isinstance(result, CacheFailure):
            verbose_logger.warning("RedisDistributedLock.release failed: %s", result.message)

    async def is_held(self, key: str) -> bool:
        result: Final = await self.cache.async_lease_held(key=key)
        if isinstance(result, CacheFailure):
            verbose_logger.warning("RedisDistributedLock.is_held failed: %s", result.message)
            return False
        return result
