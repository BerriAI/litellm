"""Concrete ``DistributedLock`` over a Redis client: ``SET NX PX`` / ``DEL`` / ``EXISTS``.

The cross-replica lock the ``RedisRefreshCoordinator`` elects refreshers with. ``acquire`` is an
atomic ``SET key NX PX ttl`` (only the first caller wins; the entry self-expires so a crashed holder
can't wedge refresh), ``release`` is ``DEL``, ``is_held`` is ``EXISTS``. The Redis client is injected
(in production the async client from LiteLLM's ``RedisCache``), so the lock is unit-testable with a
fake. A transport error on ``acquire`` returns ``LockAcquisition.ERROR`` - distinct from ``HELD`` - so
the coordinator refreshes anyway instead of mistaking a dead backend for a busy holder; a Redis blip
degrades to an extra refresh, never a stale bearer.
"""

from __future__ import annotations

from typing import Protocol

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)


class RedisCommands(Protocol):
    """The slice of the async Redis client this lock needs."""

    async def set(
        self, name: str, value: str, *, nx: bool = False, px: int | None = None
    ) -> object | None: ...

    async def delete(self, *names: str) -> int: ...

    async def exists(self, *names: str) -> int: ...


class RedisDistributedLock:
    def __init__(self, client: RedisCommands) -> None:
        self._client = client

    async def acquire(self, key: str, ttl_seconds: float) -> LockAcquisition:
        try:
            result = await self._client.set(
                key, "1", nx=True, px=int(ttl_seconds * 1000)
            )
        # Degrade on any Redis client error: redis.exceptions narrows only via an import that
        # is Unknown under basedpyright, and the lock must never crash the resolve path.
        except Exception as exc:  # noqa: BLE001
            verbose_logger.warning("RedisDistributedLock.acquire failed: %s", exc)
            return LockAcquisition.ERROR
        return LockAcquisition.ACQUIRED if result is not None else LockAcquisition.HELD

    async def release(self, key: str) -> None:
        try:
            await self._client.delete(key)
        # Degrade on any Redis client error: redis.exceptions narrows only via an import that
        # is Unknown under basedpyright, and the lock must never crash the resolve path.
        except Exception as exc:  # noqa: BLE001
            verbose_logger.warning("RedisDistributedLock.release failed: %s", exc)

    async def is_held(self, key: str) -> bool:
        try:
            return await self._client.exists(key) > 0
        # Degrade on any Redis client error: redis.exceptions narrows only via an import that
        # is Unknown under basedpyright, and the lock must never crash the resolve path.
        except Exception as exc:  # noqa: BLE001
            # On error, report "not held" so a waiter stops waiting and re-reads rather than blocking.
            verbose_logger.warning("RedisDistributedLock.is_held failed: %s", exc)
            return False
