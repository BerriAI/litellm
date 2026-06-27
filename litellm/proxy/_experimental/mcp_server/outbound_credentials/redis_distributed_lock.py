"""Concrete ``DistributedLock`` over a Redis client: ``SET NX PX`` / owner-only renew / delete.

The cross-replica lock the ``RedisRefreshCoordinator`` elects refreshers with. ``acquire`` is an
atomic ``SET key token NX PX ttl`` (only the first caller wins; the entry self-expires so a crashed
holder can't wedge refresh). ``extend`` renews the lease only when the token still matches, and
``release`` deletes the key only when it still holds this caller's token, so a holder whose lock already
PX-expired and was re-acquired by another worker cannot delete the new holder's lock. ``is_held`` is
``EXISTS``. Every key is run through the injected ``namespace_key`` before it reaches Redis, so lock
keys carry the same namespace as cache keys and cannot collide with another deployment sharing Redis.

The Redis client is injected (in production the async client from LiteLLM's ``RedisCache``), so the
lock is unit-testable with a fake. A transport error on ``acquire`` returns ``LockAcquisition.ERROR`` -
distinct from ``HELD`` - so the coordinator refreshes anyway instead of mistaking a dead backend for a
busy holder; a Redis blip degrades to an extra refresh, never a stale bearer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import Protocol

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)

# Delete the key only if it still holds this caller's token, so a holder whose lock already expired
# (PX) and was re-acquired by another worker cannot delete the new holder's lock.
_RELEASE_IF_OWNER = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
_EXTEND_IF_OWNER = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
)


class RedisCommands(Protocol):
    """The slice of the async Redis client this lock needs."""

    async def set(self, name: str, value: str, *, nx: bool = False, px: int | None = None) -> object | None: ...

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object: ...

    async def exists(self, *names: str) -> int: ...


@dataclass(frozen=True, slots=True)
class RedisDistributedLock:
    client: RedisCommands
    _: KW_ONLY
    namespace_key: Callable[[str], str] = lambda key: key

    async def acquire(self, key: str, token: str, ttl_seconds: float) -> LockAcquisition:
        try:
            result = await self.client.set(self.namespace_key(key), token, nx=True, px=int(ttl_seconds * 1000))
        # Degrade on any Redis client error: redis.exceptions narrows only via an import that
        # is Unknown under basedpyright, and the lock must never crash the resolve path.
        except Exception as exc:  # noqa: BLE001
            verbose_logger.warning("RedisDistributedLock.acquire failed: %s", exc)
            return LockAcquisition.ERROR
        return LockAcquisition.ACQUIRED if result is not None else LockAcquisition.HELD

    async def extend(self, key: str, token: str, ttl_seconds: float) -> bool:
        try:
            result = await self.client.eval(
                _EXTEND_IF_OWNER,
                1,
                self.namespace_key(key),
                token,
                str(int(ttl_seconds * 1000)),
            )
        # Degrade on any Redis client error: redis.exceptions narrows only via an import that
        # is Unknown under basedpyright, and the lock must never crash the resolve path.
        except Exception as exc:  # noqa: BLE001
            verbose_logger.warning("RedisDistributedLock.extend failed: %s", exc)
            return False
        return result == 1

    async def release(self, key: str, token: str) -> None:
        try:
            await self.client.eval(_RELEASE_IF_OWNER, 1, self.namespace_key(key), token)
        # Degrade on any Redis client error: redis.exceptions narrows only via an import that
        # is Unknown under basedpyright, and the lock must never crash the resolve path.
        except Exception as exc:  # noqa: BLE001
            verbose_logger.warning("RedisDistributedLock.release failed: %s", exc)

    async def is_held(self, key: str) -> bool:
        try:
            return await self.client.exists(self.namespace_key(key)) > 0
        # Degrade on any Redis client error: redis.exceptions narrows only via an import that
        # is Unknown under basedpyright, and the lock must never crash the resolve path.
        except Exception as exc:  # noqa: BLE001
            # On error, report "not held" so a waiter stops waiting and re-reads rather than blocking.
            verbose_logger.warning("RedisDistributedLock.is_held failed: %s", exc)
            return False
