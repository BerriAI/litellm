import contextvars
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class CallbackJob:
    callback: Callable[[], None]
    context: contextvars.Context
    finished: Future[None]


class KafkaCallbackWorker:
    def __init__(self, capacity: int, drop: Callable[[str], None]) -> None:
        self._queue: Final[queue.Queue[CallbackJob]] = queue.Queue(maxsize=capacity)
        self._drop: Final = drop
        self._lock: Final = threading.Lock()
        self._finished: Final[Future[None]] = Future()
        self._thread: threading.Thread | None = None
        self._pending: tuple[Future[None], ...] = ()
        self._stopping: bool = False
        self._deadline: float = 0.0

    def submit(self, callback: Callable[[], None]) -> Future[None] | None:
        job: Final = CallbackJob(callback, contextvars.copy_context(), Future())
        reason: Final = self._enqueue(job)
        if reason is None:
            return job.finished
        self._drop(reason)
        return None

    def _enqueue(self, job: CallbackJob) -> str | None:
        with self._lock:
            if self._stopping:
                return "closed"
            try:
                self._queue.put_nowait(job)
                if self._thread is None:
                    thread: Final = threading.Thread(target=self._run, name="litellm-kafka-prepare", daemon=True)
                    thread.start()
                    self._thread = thread
                self._pending = (*self._pending, job.finished)
            except queue.Full:
                return "callback_queue_full"
            except RuntimeError:
                _ = self._queue.get_nowait()
                return "callback_worker_unavailable"
        return None

    def pending(self) -> tuple[Future[None], ...]:
        with self._lock:
            return tuple(future for future in self._pending if not future.cancelled())

    def _forget(self, job: CallbackJob) -> None:
        with self._lock:
            self._pending = tuple(future for future in self._pending if future is not job.finished)

    def stop(self, deadline: float) -> Future[None]:
        with self._lock:
            if not self._stopping:
                self._deadline = deadline
                self._stopping = True
            if self._thread is None and not self._finished.done():
                self._finished.set_result(None)
        return self._finished

    def discard_pending(self) -> None:
        for job in self._queued_jobs():
            self._discard(job)

    def _queued_jobs(self) -> Iterator[CallbackJob]:
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                return

    def _discard(self, job: CallbackJob) -> None:
        self._drop("shutdown_callback")
        if job.finished.set_running_or_notify_cancel():
            job.finished.set_result(None)
        self._forget(job)

    def _jobs(self) -> Iterator[CallbackJob]:
        while not self._stopping or (not self._queue.empty() and time.monotonic() < self._deadline):
            try:
                yield self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

    def _run(self) -> None:
        try:
            for job in self._jobs():
                if self._stopping and time.monotonic() >= self._deadline:
                    self._discard(job)
                    continue
                if not job.finished.set_running_or_notify_cancel():
                    self._drop("cancelled_callback")
                    self._forget(job)
                    continue
                try:
                    job.context.run(job.callback)
                except Exception:  # noqa: BLE001  # a failed log job must not stop the destination worker
                    self._drop("callback_worker_error")
                finally:
                    job.finished.set_result(None)
                    self._forget(job)
        finally:
            self.discard_pending()
            self._finished.set_result(None)
