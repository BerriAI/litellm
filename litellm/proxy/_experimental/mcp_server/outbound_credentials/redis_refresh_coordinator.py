"""Cross-replica ``RefreshCoordinator``: one refresh per ``(user, server)`` across all workers.

Plugs into the foundation's ``RefreshingTokenStore`` via the ``RefreshCoordinator`` seam. A ``SET NX
PX`` lock elects one worker to run the refresh while the rest wait for it and re-read the token it
persisted - so a rotating refresh_token is used once across the fleet, not once per worker. The lock
auto-expires (``PX``), so a crashed holder can't wedge refresh; a loser that times out (or whose
holder crashed mid-refresh) falls back to a re-read, and the surrounding store re-checks expiry on the
next fetch, so a crash self-heals rather than serving stale forever. Reading needs no lock, so losers
don't serialize behind each other. The lock is injected (a thin Redis ``SET NX``/``DEL``/``EXISTS``
wrapper in production, a fake in tests).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)


class DistributedLock(Protocol):
    """A best-effort cross-replica lock. ``acquire`` is ``SET key NX PX ttl`` (only the first caller
    wins, the entry self-expires); ``release`` is ``DEL``; ``is_held`` is ``EXISTS`` (so a waiter can
    poll without taking the lock)."""

    async def acquire(self, key: str, ttl_seconds: float) -> bool: ...

    async def release(self, key: str) -> None: ...

    async def is_held(self, key: str) -> bool: ...


class RedisRefreshCoordinator:
    def __init__(
        self,
        lock: DistributedLock,
        *,
        key_prefix: str = "mcp:refresh_lock:",
        lock_ttl_seconds: float = 10.0,
        wait_timeout_seconds: float = 10.0,
        poll_interval_seconds: float = 0.05,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = lock
        self._key_prefix = key_prefix
        self._lock_ttl_seconds = lock_ttl_seconds
        self._wait_timeout_seconds = wait_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._sleep = sleep
        self._clock = clock

    def _key(self, user_id: str, server_id: str) -> str:
        return f"{self._key_prefix}{user_id}:{server_id}"

    async def run(
        self,
        user_id: str,
        server_id: str,
        refresh: Callable[[], Awaitable[OAuthToken | None]],
        reread: Callable[[], Awaitable[OAuthToken | None]],
    ) -> OAuthToken | None:
        key = self._key(user_id, server_id)
        if await self._lock.acquire(key, self._lock_ttl_seconds):
            try:
                return await refresh()
            finally:
                await self._lock.release(key)

        # Another worker holds the lock; wait for it to finish (release or PX-expiry), then read the
        # token it persisted - the winner wrote the fresh token to the store, so a plain re-read sees
        # it without us refreshing again.
        deadline = self._clock() + self._wait_timeout_seconds
        while self._clock() < deadline and await self._lock.is_held(key):
            await self._sleep(self._poll_interval_seconds)
        return await reread()
