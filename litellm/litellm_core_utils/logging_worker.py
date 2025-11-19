# This file may be a good candidate to be the first one to be refactored into a separate process,
# for the sake of performance and scalability.

import asyncio
import contextvars
from typing import Coroutine, Optional

from typing_extensions import TypedDict

from litellm._logging import verbose_logger
from litellm.constants import LOGGING_WORKER_CONCURRENCY


class LoggingTask(TypedDict):
    """
    A logging task with its associated context to ensure logging is executed in
    the original task's context.
    """

    coroutine: Coroutine
    context: contextvars.Context


class LoggingWorker:
    """
    A simple, async logging worker that processes log coroutines in the background.
    Designed to be best-effort with bounded queues to prevent backpressure.

    This leads to a +200 RPS performance improvement when using LiteLLM Python SDK or Proxy Server.
    - Use this to queue coroutine tasks that are not critical to the main flow of the application. e.g Success/Error callbacks, logging, etc.
    """

    LOGGING_WORKER_MAX_QUEUE_SIZE = 50_000
    LOGGING_WORKER_MAX_TIME_PER_COROUTINE = 20.0
    MAX_ITERATIONS_TO_CLEAR_QUEUE = 200
    MAX_TIME_TO_CLEAR_QUEUE = 5.0

    def __init__(
        self,
        timeout: float = LOGGING_WORKER_MAX_TIME_PER_COROUTINE,
        max_queue_size: int = LOGGING_WORKER_MAX_QUEUE_SIZE,
        concurrency: int = LOGGING_WORKER_CONCURRENCY,
    ):
        self.timeout = timeout
        self.max_queue_size = max_queue_size
        self.concurrency = concurrency
        self._queue: Optional[asyncio.Queue[LoggingTask]] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._running_tasks: set[asyncio.Task] = set()
        self._sem: Optional[asyncio.Semaphore] = None

    def _ensure_queue(self) -> None:
        """Initialize the queue if it doesn't exist."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)

    def start(self) -> None:
        """Start the logging worker. Idempotent - safe to call multiple times."""
        self._ensure_queue()
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _process_log_task(self, task: LoggingTask, sem: asyncio.Semaphore):
        """Runs the logging task and handles cleanup. Releases semaphore when done."""
        try:
            if self._queue is not None:
                try:
                    # Run the coroutine in its original context
                    await asyncio.wait_for(
                        task["context"].run(asyncio.create_task, task["coroutine"]),
                        timeout=self.timeout,
                    )
                except Exception as e:
                    verbose_logger.exception(f"LoggingWorker error: {e}")
                finally:
                    self._queue.task_done()
        finally:
            # Always release semaphore, even if queue is None
            sem.release()

    async def _worker_loop(self) -> None:
        """Main worker loop that gets tasks and schedules them to run concurrently."""
        try:
            if self._queue is None or self._sem is None:
                return

            while True:
                # Acquire semaphore before removing task from queue to prevent
                # unbounded growth of waiting tasks
                await self._sem.acquire()
                try:
                    task = await self._queue.get()
                    # Track each spawned coroutine so we can cancel on shutdown.
                    processing_task = asyncio.create_task(
                        self._process_log_task(task, self._sem)
                    )
                    self._running_tasks.add(processing_task)
                    processing_task.add_done_callback(self._running_tasks.discard)
                except Exception:
                    # If task creation fails, release semaphore to prevent deadlock
                    self._sem.release()
                    raise

        except asyncio.CancelledError:
            verbose_logger.debug("LoggingWorker cancelled during shutdown")
            # Attempt to clear remaining items to prevent "never awaited" warnings
            await self.clear_queue()

    def enqueue(self, coroutine: Coroutine) -> None:
        """
        Add a coroutine to the logging queue.
        Hot path: never blocks, drops logs if queue is full.
        """
        if self._queue is None:
            return

        try:
            # Capture the current context when enqueueing
            task = LoggingTask(coroutine=coroutine, context=contextvars.copy_context())
            self._queue.put_nowait(task)
        except asyncio.QueueFull as e:
            verbose_logger.exception(f"LoggingWorker queue is full: {e}")
            # Drop logs on overload to protect request throughput
            pass

    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine):
        """
        Ensure the logging worker is initialized and enqueue the coroutine.
        """
        self.start()
        self.enqueue(async_coroutine)

    async def stop(self) -> None:
        """Stop the logging worker and clean up resources."""
        if self._worker_task is None and not self._running_tasks:
            # No worker launched and no in-flight tasks to drain.
            return

        tasks_to_cancel: list[asyncio.Task] = list(self._running_tasks)
        if self._worker_task:
            # Include the main worker loop so it stops fetching work.
            tasks_to_cancel.append(self._worker_task)

        for task in tasks_to_cancel:
            # Propagate cancellation to every pending task.
            task.cancel()

        # Wait for cancellation to settle; ignore errors raised during shutdown.
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._worker_task = None
        # Drop references to completed tasks so we can restart cleanly.
        self._running_tasks.clear()

    async def flush(self) -> None:
        """Flush the logging queue."""
        if self._queue is None:
            return
        while not self._queue.empty():
            await self._queue.join()

    async def clear_queue(self):
        """
        Clear the queue with a maximum time limit.
        """
        if self._queue is None:
            return

        start_time = asyncio.get_event_loop().time()

        for _ in range(self.MAX_ITERATIONS_TO_CLEAR_QUEUE):
            # Check if we've exceeded the maximum time
            if (
                asyncio.get_event_loop().time() - start_time
                >= self.MAX_TIME_TO_CLEAR_QUEUE
            ):
                verbose_logger.warning(
                    f"clear_queue exceeded max_time of {self.MAX_TIME_TO_CLEAR_QUEUE}s, stopping early"
                )
                break

            try:
                task = self._queue.get_nowait()
                # Await the coroutine to properly execute and avoid "never awaited" warnings
                try:
                    await asyncio.wait_for(
                        task["context"].run(asyncio.create_task, task["coroutine"]),
                        timeout=self.timeout,
                    )
                except Exception:
                    # Suppress errors during cleanup
                    pass
                self._queue.task_done()  # If you're using join() elsewhere
            except asyncio.QueueEmpty:
                break


# Global instance for backward compatibility
GLOBAL_LOGGING_WORKER = LoggingWorker()
