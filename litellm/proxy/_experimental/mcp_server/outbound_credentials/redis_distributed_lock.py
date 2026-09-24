"""Concrete ``DistributedLock`` over ``RedisCache`` scripts: ``SET NX PX`` / owner-only renew / delete.

The cross-replica lock the ``RedisRefreshCoordinator`` elects refreshers with. ``acquire`` is an
atomic ``SET key token NX PX ttl`` (only the first caller wins; the entry self-expires so a crashed
holder can't wedge refresh). ``extend`` renews the lease only when the token still matches, and
``release`` deletes the key only when it still holds this caller's token, so a holder whose lock already
PX-expired and was re-acquired by another worker cannot delete the new holder's lock. ``is_held`` is
``EXISTS``. Every operation runs through ``RedisCache.async_register_script``, which prefixes the key
with the cache namespace, so lock keys carry the same namespace as cache keys and cannot collide with
another deployment sharing Redis.

A transport error on ``acquire`` returns ``LockAcquisition.ERROR`` - distinct from ``HELD`` - so the
coordinator refreshes anyway instead of mistaking a dead backend for a busy holder; a Redis blip
degrades to an extra refresh, never a stale bearer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache

_ACQUIRE_IF_ABSENT: Final = "return redis.call('set', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2])"
_EXTEND_IF_OWNER: Final = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
)
_RELEASE_IF_OWNER: Final = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
)
_IS_HELD: Final = "return redis.call('exists', KEYS[1])"
_SCRIPT_REPLY: Final[TypeAdapter[int | bytes | str | None]] = TypeAdapter(int | bytes | str | None)


def _milliseconds(seconds: float) -> str:
    return str(int(seconds * 1000))


@dataclass(frozen=True, slots=True)
class RedisDistributedLock:
    redis_cache: RedisCache

    async def _run(self, script: str, key: str, *args: str) -> int | bytes | str | None:
        return _SCRIPT_REPLY.validate_python(
            await self.redis_cache.async_register_script(script)(keys=(key,), args=args)
        )

    async def acquire(self, key: str, token: str, ttl_seconds: float) -> LockAcquisition:
        try:
            result: Final = await self._run(_ACQUIRE_IF_ABSENT, key, token, _milliseconds(ttl_seconds))
        except Exception as exc:  # noqa: BLE001  # the lock must never crash the resolve path
            verbose_logger.warning("RedisDistributedLock.acquire failed: %s", exc)
            return LockAcquisition.ERROR
        return LockAcquisition.ACQUIRED if result is not None else LockAcquisition.HELD

    async def extend(self, key: str, token: str, ttl_seconds: float) -> bool:
        try:
            result: Final = await self._run(_EXTEND_IF_OWNER, key, token, _milliseconds(ttl_seconds))
        except Exception as exc:  # noqa: BLE001  # the lock must never crash the resolve path
            verbose_logger.warning("RedisDistributedLock.extend failed: %s", exc)
            return False
        return result == 1

    async def release(self, key: str, token: str) -> None:
        try:
            await self._run(_RELEASE_IF_OWNER, key, token)
        except Exception as exc:  # noqa: BLE001  # the lock must never crash the resolve path
            verbose_logger.warning("RedisDistributedLock.release failed: %s", exc)

    async def is_held(self, key: str) -> bool:
        try:
            result: Final = await self._run(_IS_HELD, key)
        except Exception as exc:  # noqa: BLE001  # a waiter stops waiting and re-reads rather than blocking
            verbose_logger.warning("RedisDistributedLock.is_held failed: %s", exc)
            return False
        return isinstance(result, int) and result > 0
