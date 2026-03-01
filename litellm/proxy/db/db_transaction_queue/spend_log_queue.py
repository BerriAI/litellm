"""Bounded in-memory queue for spend-log transactions.

Wraps a ``collections.deque`` with a configurable ``maxlen`` so that the
buffer can never grow without bound.  When full, the **oldest** entries
are silently dropped — this is the safest behaviour because:

* The spend-log is an *eventual-consistency* reporting artefact, not a
  transactional record (losing a few oldest entries when the DB is down
  is far less harmful than OOM-killing the proxy).
* A warning is emitted when items are dropped (every 100th drop) so
  operators can tune the limit via ``SPEND_LOG_TRANSACTIONS_MAX_SIZE``.

The class deliberately exposes the same slice/len/append API that
``PrismaClient.spend_log_transactions`` already relies on, so the
migration is purely mechanical.
"""

import itertools
from collections import deque
from typing import Any, Dict, List, Union, overload

from litellm._logging import verbose_proxy_logger


class SpendLogQueue:
    """Thread-safe (via external asyncio.Lock) bounded FIFO buffer."""

    __slots__ = ("_buf", "_dropped")

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._dropped: int = 0

    # -- mutators ----------------------------------------------------------

    def append(self, item: Dict[str, Any]) -> None:
        # Check *before* appending: deque.append() auto-evicts the oldest
        # element when at maxlen, so we must count the drop before it happens.
        will_drop = len(self._buf) == self._buf.maxlen
        self._buf.append(item)
        if will_drop:
            self._dropped += 1
            if self._dropped % 100 == 1:
                verbose_proxy_logger.warning(
                    "SpendLogQueue full (maxlen=%d) — dropping oldest entry "
                    "(%d total dropped since last drain)",
                    self._buf.maxlen,
                    self._dropped,
                )

    def extend(self, items: List[Dict[str, Any]]) -> None:
        """Append multiple items, dropping oldest when overflow occurs."""
        before = len(self._buf)
        maxlen = self._buf.maxlen or 0
        self._buf.extend(items)
        overflow = max(0, (before + len(items)) - maxlen)
        if overflow > 0:
            self._dropped += overflow
            if self._dropped % 100 < overflow or self._dropped <= overflow:
                verbose_proxy_logger.warning(
                    "SpendLogQueue full (maxlen=%d) — dropped %d entries "
                    "(%d total dropped since last drain)",
                    maxlen,
                    overflow,
                    self._dropped,
                )

    def drain(self, n: int) -> List[Dict[str, Any]]:
        """Remove and return up to *n* items from the front."""
        actual = min(n, len(self._buf))
        batch = [self._buf.popleft() for _ in range(actual)]
        if batch and self._dropped > 0:
            # Log the accumulated drop count *before* resetting so operators
            # know how many entries were lost during this trouble window.
            verbose_proxy_logger.warning(
                "SpendLogQueue: draining %d items; %d entries were dropped "
                "since the last successful drain",
                len(batch),
                self._dropped,
            )
            self._dropped = 0
        return batch

    # -- read-only helpers -------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)

    @overload
    def __getitem__(self, index: int) -> Dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> List[Dict[str, Any]]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Support ``queue[i]`` and ``queue[:N]`` access (read-only)."""
        if isinstance(index, int):
            return self._buf[index]
        # For slices, use islice to avoid copying the entire deque
        start, stop, step = index.indices(len(self._buf))
        result = list(itertools.islice(self._buf, start, stop, step))
        return result
