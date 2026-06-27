"""Cross-replica ``RefreshCoordinator``: one refresh per ``(user, server)`` across all workers.

Plugs into the foundation's ``RefreshingTokenStore`` via the ``RefreshCoordinator`` seam. A ``SET NX
PX`` lock elects one worker to run the refresh while the rest wait for it and re-read the token it
persisted - so a rotating refresh_token is used once across the fleet, not once per worker. The lock
auto-expires (``PX``), so a crashed holder can't wedge refresh; a loser that times out (or whose
holder crashed mid-refresh) falls back to a re-read, and the surrounding store re-checks expiry on the
next fetch, so a crash self-heals rather than serving stale forever. Reading needs no lock, so losers
don't serialize behind each other. The lock is injected (a thin Redis ``SET NX``/``DEL``/``EXISTS``
wrapper in production, a fake in tests).

The lock is a single-flight optimization, not a correctness mutex, so it fails open: when the lock
backend is unreachable, ``acquire`` reports ``ERROR`` (distinct from ``HELD``) and this coordinator
refreshes anyway rather than wait on a holder that may not exist and then serve a still-expired token.
That degrades a Redis outage to the no-coordinator behavior (each worker may refresh), never a stale
bearer the upstream would 401.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Protocol

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)


class LockAcquisition(Enum):
    """Outcome of a best-effort ``acquire``. ``ERROR`` is kept distinct from ``HELD`` so a caller can
    tell "someone else is refreshing" (wait and re-read) from "the lock backend is down" (no election
    happened, so refresh anyway) instead of conflating both into a single ``False``."""

    ACQUIRED = "acquired"  # won the election; this worker refreshes
    HELD = "held"  # another worker holds it; wait then re-read
    ERROR = "error"  # lock backend unreachable; holder unknown, so refresh anyway


class DistributedLock(Protocol):
    """A best-effort cross-replica lock. ``acquire`` is ``SET key NX PX ttl`` reported as a
    ``LockAcquisition`` (won / held by another / backend error); ``release`` is ``DEL``; ``is_held``
    is ``EXISTS`` (so a waiter can poll without taking the lock)."""

    async def acquire(self, key: str, ttl_seconds: float) -> LockAcquisition: ...

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
        match await self._lock.acquire(key, self._lock_ttl_seconds):
            case LockAcquisition.ACQUIRED:
                try:
                    return await refresh()
                finally:
                    await self._lock.release(key)
            case LockAcquisition.ERROR:
                # No election happened (lock backend down), so waiting would just re-read the
                # still-expired token. Refresh anyway; worst case is an extra refresh, not a stale bearer.
                return await refresh()
            case LockAcquisition.HELD:
                # Another worker holds the lock; wait for it to finish (release or PX-expiry), then read
                # the token it persisted - the winner wrote the fresh token to the store, so a plain
                # re-read sees it without us refreshing again.
                deadline = self._clock() + self._wait_timeout_seconds
                while self._clock() < deadline and await self._lock.is_held(key):
                    await self._sleep(self._poll_interval_seconds)
                return await reread()
