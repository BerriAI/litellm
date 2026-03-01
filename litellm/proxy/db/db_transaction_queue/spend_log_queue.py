"""Bounded in-memory queue for spend-log transactions.

Wraps a ``collections.deque`` with a configurable ``maxlen`` so that the
buffer can never grow without bound.  When full, the **oldest** entries
are silently dropped — this is the safest behaviour because:

* The spend-log is an *eventual-consistency* reporting artefact, not a
  transactional record (losing a few oldest entries when the DB is down
  is far less harmful than OOM-killing the proxy).
* A warning is emitted every time items are dropped so operators can
  tune the limit via ``SPEND_LOG_TRANSACTIONS_MAX_SIZE``.

The class deliberately exposes the same slice/len/append API that
``PrismaClient.spend_log_transactions`` already relies on, so the
migration is purely mechanical.
"""

from collections import deque
from typing import Any, Dict, List

from litellm._logging import verbose_proxy_logger


class SpendLogQueue:
    """Thread-safe (via external asyncio.Lock) bounded FIFO buffer."""

    __slots__ = ("_buf", "_dropped")

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._dropped: int = 0

    # -- mutators ----------------------------------------------------------

    def append(self, item: Dict[str, Any]) -> None:
        if len(self._buf) == self._buf.maxlen:
            self._dropped += 1
            if self._dropped % 100 == 1:
                verbose_proxy_logger.warning(
                    "SpendLogQueue full (maxlen=%d) — dropping oldest entry "
                    "(%d total dropped since last drain)",
                    self._buf.maxlen,
                    self._dropped,
                )
        self._buf.append(item)

    def extend(self, items: List[Dict[str, Any]]) -> None:
        """Append multiple items. Each item that causes overflow drops the oldest."""
        for item in items:
            self.append(item)

    def drain(self, n: int) -> List[Dict[str, Any]]:
        """Remove and return up to *n* items from the front."""
        actual = min(n, len(self._buf))
        batch = [self._buf.popleft() for _ in range(actual)]
        if batch:
            self._dropped = 0
        return batch

    # -- read-only helpers -------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, index: slice) -> List[Dict[str, Any]]:
        """Support ``queue[:N]`` slicing (read-only, non-destructive)."""
        return list(self._buf)[index]
