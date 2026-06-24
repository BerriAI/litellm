"""v2-native BYOK credential stores.

``CachedByokStore`` is a per-process TTL cache in front of any ``ByokCredentialStore``, so
v2 owns the BYOK credential lifecycle rather than borrowing v1's cache. It composes over the
backing store (the v1-backed lookup today, a v2-native source later) without either side
knowing about the other.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

from litellm.proxy._experimental.mcp_server.outbound_credentials.seams import (
    ByokCredentialStore,
)


class CachedByokStore:
    """Caches an inner ``ByokCredentialStore`` by ``(user_id, server_id)`` for ``ttl_seconds``.

    The clock is injected so expiry is deterministic in tests. A missing credential (``None``)
    is cached too, so a not-yet-provisioned user does not re-hit the backing store on every
    call. The cache is bounded and cleared wholesale when full. Distributed single-flight and
    proactive refresh are the later step-1b hardening, not this in-process cache.
    """

    def __init__(
        self,
        inner: ByokCredentialStore,
        *,
        ttl_seconds: float,
        max_size: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._clock = clock
        self._cache: Dict[Tuple[str, str], Tuple[Optional[str], float]] = {}

    async def fetch(self, user_id: str, server_id: str) -> Optional[str]:
        key = (user_id, server_id)
        hit = self._cache.get(key)
        if hit is not None:
            value, stored_at = hit
            if self._clock() - stored_at < self._ttl_seconds:
                return value

        value = await self._inner.fetch(user_id, server_id)
        if len(self._cache) >= self._max_size:
            self._cache.clear()
        self._cache[key] = (value, self._clock())
        return value
