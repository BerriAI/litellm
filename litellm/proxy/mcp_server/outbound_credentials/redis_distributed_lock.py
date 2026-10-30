"""Concrete ``DistributedLock`` over ``McpRefreshLockStore``: ``SET NX PX`` / owner-only renew / delete.

The cross-replica lock the ``RedisRefreshCoordinator`` elects refreshers with. ``acquire`` is an
atomic ``SET key token NX PX ttl`` (only the first caller wins; the entry self-expires so a crashed
holder can't wedge refresh). ``extend`` renews the lease only when the token still matches, and
``release`` deletes the key only when it still holds this caller's token, so a holder whose lock already
PX-expired and was re-acquired by another worker cannot delete the new holder's lock.

A Redis failure on ``acquire`` returns ``LockAcquisition.ERROR`` - distinct from ``HELD`` - so the
coordinator refreshes anyway instead of mistaking a dead backend for a busy holder; a Redis blip
degrades to an extra refresh, never a stale bearer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from litellm.proxy.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)
from litellm.redis.stores.mcp_credential_refresh import McpRefreshLockStore, RefreshLockTarget
from litellm.redis.transport.contract import RedisFailure


@dataclass(frozen=True, slots=True)
class RedisDistributedLock:
    store: McpRefreshLockStore

    async def acquire(self, target: RefreshLockTarget, token: str, ttl_seconds: float) -> LockAcquisition:
        acquired: Final = await self.store.acquire(target, token, ttl=timedelta(seconds=ttl_seconds))
        if isinstance(acquired, RedisFailure):
            return LockAcquisition.ERROR
        return LockAcquisition.ACQUIRED if acquired else LockAcquisition.HELD

    async def extend(self, target: RefreshLockTarget, token: str, ttl_seconds: float) -> bool:
        return await self.store.extend(target, token, ttl=timedelta(seconds=ttl_seconds)) is True

    async def release(self, target: RefreshLockTarget, token: str) -> None:
        _ = await self.store.release(target, token)

    async def is_held(self, target: RefreshLockTarget) -> bool:
        return await self.store.is_held(target) is True
