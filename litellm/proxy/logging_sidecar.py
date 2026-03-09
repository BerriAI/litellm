"""
Logging Sidecar — offloads heavy per-request logging work to a separate process.

The main proxy process puts lightweight event dicts on a multiprocessing queue.
A dedicated worker process drains the queue, builds full StandardLoggingPayloads
and SpendLogsPayloads, and executes callbacks — all in its own process with its
own GIL, so none of this work competes with request handling.
"""

import multiprocessing
import queue
from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger


class LoggingSidecar:
    """
    Manages a background process that handles logging work.

    Usage:
        sidecar = LoggingSidecar()
        sidecar.start()

        # In hot path (per-request):
        sidecar.enqueue(event_dict)

        # On shutdown:
        sidecar.stop()
    """

    def __init__(self, max_queue_size: int = 50000):
        self._queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=max_queue_size)
        self._process: Optional[multiprocessing.Process] = None
        self._started = False
        self._dropped = 0

    def start(self):
        if self._started:
            return
        self._process = multiprocessing.Process(
            target=_worker_loop,
            args=(self._queue,),
            daemon=True,
            name="litellm-logging-sidecar",
        )
        self._process.start()
        self._started = True
        verbose_proxy_logger.info(
            "Logging sidecar started (pid=%s, queue_size=%s)",
            self._process.pid,
            self._queue._maxsize,
        )

    def enqueue(self, event: Dict[str, Any]) -> bool:
        """
        Put a logging event on the queue. Returns False if queue is full
        (event is dropped to protect the hot path).
        """
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            self._dropped += 1
            if self._dropped % 1000 == 1:
                verbose_proxy_logger.warning(
                    "Logging sidecar queue full, dropped %d events", self._dropped
                )
            return False

    def stop(self, timeout: float = 5.0):
        if not self._started:
            return
        self._queue.put_nowait(None)  # sentinel
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
        self._started = False
        verbose_proxy_logger.info(
            "Logging sidecar stopped (dropped=%d)", self._dropped
        )

    @property
    def is_running(self) -> bool:
        return self._started and self._process is not None and self._process.is_alive()


def _worker_loop(q: multiprocessing.Queue):
    """
    Worker process main loop. Drains the queue and processes logging events.
    Runs in a completely separate process with its own GIL.
    """
    batch = []
    BATCH_SIZE = 64
    BATCH_TIMEOUT = 0.05  # 50ms

    while True:
        try:
            event = q.get(timeout=BATCH_TIMEOUT)
            if event is None:  # sentinel
                if batch:
                    _process_batch(batch)
                break
            batch.append(event)
            if len(batch) >= BATCH_SIZE:
                _process_batch(batch)
                batch = []
        except queue.Empty:
            if batch:
                _process_batch(batch)
                batch = []


def _process_batch(batch):
    """
    Process a batch of logging events. This is where the heavy work happens,
    safely in a separate process.

    For now this is a minimal implementation that just processes the events.
    In production, this would build StandardLoggingPayloads, write spend logs
    to the database, and execute callbacks.
    """
    for event in batch:
        _process_single_event(event)


def _process_single_event(event: Dict[str, Any]):
    """
    Process a single logging event. Simulates the work that would normally
    happen in the main process.
    """
    event_type = event.get("type", "unknown")
    if event_type == "success":
        _handle_success_event(event)
    elif event_type == "failure":
        _handle_failure_event(event)


def _handle_success_event(event: Dict[str, Any]):
    """Handle a success logging event in the worker process."""
    pass  # Placeholder — in production this builds payloads and writes to DB


def _handle_failure_event(event: Dict[str, Any]):
    """Handle a failure logging event in the worker process."""
    pass  # Placeholder


# Global singleton
_logging_sidecar: Optional[LoggingSidecar] = None


def get_logging_sidecar() -> Optional[LoggingSidecar]:
    return _logging_sidecar


def init_logging_sidecar(max_queue_size: int = 50000) -> LoggingSidecar:
    global _logging_sidecar
    _logging_sidecar = LoggingSidecar(max_queue_size=max_queue_size)
    _logging_sidecar.start()
    return _logging_sidecar
