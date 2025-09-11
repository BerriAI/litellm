import os
import threading

from time import sleep, time
from sentry_sdk._queue import Queue, FullError
from sentry_sdk.utils import logger
from sentry_sdk.consts import DEFAULT_QUEUE_SIZE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Optional
    from typing import Callable


_TERMINATOR = object()


class BackgroundWorker:
    def __init__(self, queue_size=DEFAULT_QUEUE_SIZE):
        # type: (int) -> None
        self._queue = Queue(queue_size)  # type: Queue
        self._lock = threading.Lock()
        self._thread = None  # type: Optional[threading.Thread]
        self._thread_for_pid = None  # type: Optional[int]

    @property
    def is_alive(self):
        # type: () -> bool
        if self._thread_for_pid != os.getpid():
            return False
        if not self._thread:
            return False
        return self._thread.is_alive()

    def _ensure_thread(self):
        # type: () -> None
        if not self.is_alive:
            self.start()

    def _timed_queue_join(self, timeout):
        # type: (float) -> bool
        deadline = time() + timeout
        queue = self._queue

        queue.all_tasks_done.acquire()

        try:
            while queue.unfinished_tasks:
                delay = deadline - time()
                if delay <= 0:
                    return False
                queue.all_tasks_done.wait(timeout=delay)

            return True
        finally:
            queue.all_tasks_done.release()

    def start(self):
        # type: () -> None
        with self._lock:
            if not self.is_alive:
                self._thread = threading.Thread(
                    target=self._target, name="sentry-sdk.BackgroundWorker"
                )
                self._thread.daemon = True
                try:
                    self._thread.start()
                    self._thread_for_pid = os.getpid()
                except RuntimeError:
                    # At this point we can no longer start because the interpreter
                    # is already shutting down.  Sadly at this point we can no longer
                    # send out events.
                    self._thread = None

    def kill(self):
        # type: () -> None
        """
        Kill worker thread. Returns immediately. Not useful for
        waiting on shutdown for events, use `flush` for that.
        """
        logger.debug("background worker got kill request")
        with self._lock:
            if self._thread:
                try:
                    self._queue.put_nowait(_TERMINATOR)
                except FullError:
                    logger.debug("background worker queue full, kill failed")

                self._thread = None
                self._thread_for_pid = None

    def flush(self, timeout, callback=None):
        # type: (float, Optional[Any]) -> None
        logger.debug("background worker got flush request")
        with self._lock:
            if self.is_alive and timeout > 0.0:
                self._wait_flush(timeout, callback)
        logger.debug("background worker flushed")

    def full(self):
        # type: () -> bool
        return self._queue.full()

    def _wait_flush(self, timeout, callback):
        # type: (float, Optional[Any]) -> None
        initial_timeout = min(0.1, timeout)
        if not self._timed_queue_join(initial_timeout):
            pending = self._queue.qsize() + 1
            logger.debug("%d event(s) pending on flush", pending)
            if callback is not None:
                callback(pending, timeout)

            if not self._timed_queue_join(timeout - initial_timeout):
                pending = self._queue.qsize() + 1
                logger.error("flush timed out, dropped %s events", pending)

    def submit(self, callback):
        # type: (Callable[[], None]) -> bool
        self._ensure_thread()
        try:
            self._queue.put_nowait(callback)
            return True
        except FullError:
            return False

    def _target(self):
        # type: () -> None
        while True:
            callback = self._queue.get()
            try:
                if callback is _TERMINATOR:
                    break
                try:
                    callback()
                except Exception:
                    logger.error("Failed processing job", exc_info=True)
            finally:
                self._queue.task_done()
            sleep(0)
