"""L1 per-pod bounded LRU. Hot read path for header hints only.

Never authoritative for admission (see R3): budget and concurrent policies always
take the L2 write path. L1 holds the last value read from L2 plus the monotonic
timestamp it was read, so a reader can decide staleness without a clock import.
A capacity miss falls through to L2 cleanly; L1 can never fail in a way that
matters.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class L1Entry:
    value: float
    read_at_monotonic_s: float


class BoundedLRUCounterCache:
    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._store: "OrderedDict[str, L1Entry]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> L1Entry | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)
            return entry

    async def set(self, key: str, value: float, read_at_monotonic_s: float) -> None:
        async with self._lock:
            self._store[key] = L1Entry(
                value=value, read_at_monotonic_s=read_at_monotonic_s
            )
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)
