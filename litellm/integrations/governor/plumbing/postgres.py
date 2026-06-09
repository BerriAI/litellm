"""L3 durable store skeleton (Prisma-backed).

The canonical historical spend and the idempotent flush live here, but the
``LiteLLM_GovernanceLedger`` / ``LiteLLM_GovernanceAuditLog`` Prisma models land
in a follow-up. For now the interface is fixed and the flusher is a no-op when
``prisma_client`` is None, so the engine can wire it without a database.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional

from litellm._logging import verbose_proxy_logger


@dataclass(frozen=True)
class FlushEntry:
    """One reconciled delta awaiting durable persistence, keyed for idempotency
    by ``(request_id, policy_id)``."""

    request_id: str
    policy_id: str
    subject_kind: str
    subject_id: str
    bucket_key: str
    delta: float


class PendingFlushQueue:
    """Bounded in-memory queue of deltas not yet persisted to L3.

    On overflow the oldest entries are dropped and the loss is logged at WARNING;
    per R10 there is no on-disk WAL. L2 counters survive a pod restart anyway, so
    a dropped flush costs historical precision, not live enforcement.
    """

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._queue: Deque[FlushEntry] = deque()
        self._lock = asyncio.Lock()
        self._dropped = 0

    async def append(self, entry: FlushEntry) -> None:
        async with self._lock:
            if len(self._queue) >= self._max_entries:
                self._queue.popleft()
                self._dropped += 1
                verbose_proxy_logger.warning(
                    "governor L3 PendingFlushQueue overflow; dropped %s entries",
                    self._dropped,
                )
            self._queue.append(entry)

    async def drain(self) -> List[FlushEntry]:
        async with self._lock:
            drained = list(self._queue)
            self._queue.clear()
            return drained

    async def requeue(self, entries: List[FlushEntry]) -> None:
        async with self._lock:
            self._queue.extendleft(reversed(entries))

    @property
    def dropped(self) -> int:
        return self._dropped

    async def size(self) -> int:
        async with self._lock:
            return len(self._queue)


class PostgresCounterStore:
    def __init__(self, prisma_client: Optional[Any] = None) -> None:
        self._prisma = prisma_client

    @property
    def available(self) -> bool:
        return self._prisma is not None

    async def read_historical(
        self, policy_id: str, subject_kind: str, subject_id: str, bucket_key: str
    ) -> float:
        if self._prisma is None:
            return 0.0
        raise NotImplementedError(
            "historical read lands with the LiteLLM_GovernanceLedger model"
        )

    async def flush(self, entries: List[FlushEntry]) -> None:
        if self._prisma is None or not entries:
            return
        raise NotImplementedError(
            "idempotent upsert lands with the LiteLLM_GovernanceLedger model"
        )
