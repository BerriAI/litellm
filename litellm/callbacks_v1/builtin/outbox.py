"""Where a sink's handler leaves its work, so that the handler never does I/O.

The v1 contract runs an observer inline in the call it observes. A handler that posted to
a vendor would put that vendor's latency into every call, so a sink's handler only builds
a value and `put`s it here; one daemon thread per outbox delivers it, alone or in batches.
"""

import atexit
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Final, Generic, TypeVar

from litellm._logging import verbose_logger

T = TypeVar("T")  # rebind-ok: a TypeVar must bind to a bare name to be recognised as one


class Outbox(Generic[T]):
    """Delivers items off the call path, `batch_size` at a time or every `flush_interval`.

    `deliver` gets a non-empty batch and may raise: the failure is logged and the batch is
    dropped, as the legacy batch loggers do. A full outbox drops the new item rather than
    block the call that produced it.
    """

    def __init__(
        self,
        name: str,
        deliver: Callable[[tuple[T, ...]], None],
        batch_size: int = 1,
        flush_interval: float = 5.0,
        capacity: int = 50_000,
    ) -> None:
        self._name: Final = name
        self._deliver: Final = deliver
        self._batch_size: Final = max(1, batch_size)
        self._flush_interval: Final = flush_interval
        self._items: Final[queue.Queue[T]] = queue.Queue(maxsize=capacity)
        self._lock: Final = threading.Lock()
        self._flush_now: Final = threading.Event()
        self._worker_pid: int | None = None
        atexit.register(self.flush)

    def put(self, item: T) -> None:
        self._ensure_worker()
        try:
            self._items.put_nowait(item)
        except queue.Full:
            verbose_logger.warning("callback %s outbox is full; dropping one item", self._name)

    def flush(self, timeout: float = 5.0) -> bool:
        """Delivers everything already put; returns whether it drained within `timeout`."""
        deadline: Final = time.monotonic() + timeout
        self._flush_now.set()
        while self._items.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return not self._items.unfinished_tasks

    def _ensure_worker(self) -> None:
        # A forked child inherits the queue but not the thread, so the worker is per process.
        with self._lock:
            if self._worker_pid == os.getpid():
                return
            self._worker_pid = os.getpid()
            threading.Thread(target=self._run, name=f"callback-outbox-{self._name}", daemon=True).start()

    def _next_batch(self) -> tuple[T, ...]:
        first: Final = self._items.get()
        deadline: Final = time.monotonic() + self._flush_interval
        batch: Final[list[T]] = [first]  # mutable-ok: a batch is gathered from a blocking queue one item at a time
        while len(batch) < self._batch_size:
            remaining = deadline - time.monotonic()  # rebind-ok: the wait shrinks on every pass of the gather loop
            if remaining <= 0 or (self._flush_now.is_set() and self._items.empty()):
                break
            try:
                batch.append(self._items.get(timeout=min(remaining, 0.01)))
            except queue.Empty:
                continue
        return tuple(batch)

    def _run(self) -> None:
        while True:
            batch = self._next_batch()  # rebind-ok: the worker loop takes a new batch on every pass
            try:
                self._deliver(batch)
            except Exception:  # noqa: BLE001  # a vendor failure must never end the delivery thread
                verbose_logger.exception("callback %s failed to deliver %d item(s)", self._name, len(batch))
            finally:
                for _ in batch:
                    self._items.task_done()
                if self._items.empty():
                    self._flush_now.clear()
