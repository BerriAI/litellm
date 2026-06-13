from __future__ import annotations

import time
from typing import Dict, Generic, Optional, Tuple

from litellm.proxy.auth_v2.sessions.base import SessionValue


class InMemorySessionStore(Generic[SessionValue]):
    """Process-local fallback when Redis is unavailable. Single-process only."""

    def __init__(self, namespace: str, default_ttl: int, max_size: int = 10000) -> None:
        self._namespace = namespace
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._entries: Dict[str, Tuple[float, Optional[SessionValue]]] = {}

    def _key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def _expiry(self, ttl_seconds: Optional[int]) -> float:
        return time.time() + (self._default_ttl if ttl_seconds is None else ttl_seconds)

    def _live(
        self, key: str, now: float
    ) -> Optional[Tuple[float, Optional[SessionValue]]]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry[0] < now:
            self._entries.pop(key, None)
            return None
        return entry

    async def get(self, key: str) -> Optional[SessionValue]:
        entry = self._live(self._key(key), time.time())
        return entry[1] if entry is not None else None

    async def set(
        self, key: str, value: SessionValue, ttl_seconds: Optional[int] = None
    ) -> None:
        self._evict(time.time())
        self._entries[self._key(key)] = (self._expiry(ttl_seconds), value)

    async def pop(self, key: str) -> Optional[SessionValue]:
        entry = self._entries.pop(self._key(key), None)
        if entry is None or entry[0] < time.time():
            return None
        return entry[1]

    async def delete(self, key: str) -> None:
        self._entries.pop(self._key(key), None)

    async def add_if_absent(self, key: str, ttl_seconds: Optional[int] = None) -> bool:
        now = time.time()
        self._evict(now)
        namespaced = self._key(key)
        if self._live(namespaced, now) is not None:
            return False
        self._entries[namespaced] = (self._expiry(ttl_seconds), None)
        return True

    def _evict(self, now: float) -> None:
        for key in [k for k, (exp, _) in self._entries.items() if exp < now]:
            self._entries.pop(key, None)
        overflow = len(self._entries) - self._max_size + 1
        if overflow > 0:
            oldest = sorted(self._entries, key=lambda k: self._entries[k][0])
            for key in oldest[:overflow]:
                self._entries.pop(key, None)
