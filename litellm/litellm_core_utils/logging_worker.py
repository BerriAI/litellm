# This file may be a good candidate to be the first one to be refactored into a separate process,
# for the sake of performance and scalability.

import asyncio
import contextvars
from typing import Coroutine, Optional
import atexit
from typing_extensions import TypedDict

from litellm._logging import verbose_logger
from litellm.constants import (
    LOGGING_WORKER_CONCURRENCY,
    LOGGING_WORKER_MAX_QUEUE_SIZE,
    LOGGING_WORKER_MAX_TIME_PER_COROUTINE,
    LOGGING_WORKER_CLEAR_PERCENTAGE,
    LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS,
    MAX_ITERATIONS_TO_CLEAR_QUEUE,
    MAX_TIME_TO_CLEAR_QUEUE,
)


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
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_aggressive_clear_time: float = 0.0
        self._aggressive_clear_in_progress: bool = False

        # Register cleanup handler to flush remaining events on exit
        atexit.register(self._flush_on_exit)

    def _ensure_queue(self) -> None:
        """Initialize the queue if it doesn't exist or if event loop has changed."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, can't initialize
            return

        # Check if we need to reinitialize due to event loop change
        if self._queue is not None and self._bound_loop is not current_loop:
            verbose_logger.debug(
                "LoggingWorker: Event loop changed, reinitializing queue and worker"
            )
            # Clear old state - these are bound to the old loop
            self._queue = None
            self._sem = None
            self._worker_task = None
            self._running_tasks.clear()

        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
            self._bound_loop = current_loop

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
        Hot path: never blocks, aggressively clears queue if full.
        """
        if self._queue is None:
            return

        # Capture the current context when enqueueing
        task = LoggingTask(coroutine=coroutine, context=contextvars.copy_context())

        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            # Queue is full - handle it appropriately
            verbose_logger.exception("LoggingWorker queue is full")
            self._handle_queue_full(task)

    def _should_start_aggressive_clear(self) -> bool:
        """
        Check if we should start a new aggressive clear operation.
        Returns True if cooldown period has passed and no clear is in progress.
        """
        if self._aggressive_clear_in_progress:
            return False

        try:
            loop = asyncio.get_running_loop()
            current_time = loop.time()
            time_since_last_clear = current_time - self._last_aggressive_clear_time

            if time_since_last_clear < LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS:
                return False

            return True
        except RuntimeError:
            # No event loop running, drop the task
            return False

    def _mark_aggressive_clear_started(self) -> None:
        """
        Mark that an aggressive clear operation has started.

        Note: This should only be called after _should_start_aggressive_clear()
        returns True, which guarantees an event loop exists.
        """
        loop = asyncio.get_running_loop()
        self._last_aggressive_clear_time = loop.time()
        self._aggressive_clear_in_progress = True

    def _handle_queue_full(self, task: LoggingTask) -> None:
        """
        Handle queue full condition by either starting an aggressive clear
        or scheduling a delayed retry.
        """

        if self._should_start_aggressive_clear():
            self._mark_aggressive_clear_started()
            # Schedule clearing as async task so enqueue returns immediately (non-blocking)
            asyncio.create_task(self._aggressively_clear_queue_async(task))
        else:
            # Cooldown active or clear in progress, schedule a delayed retry
            self._schedule_delayed_enqueue_retry(task)

    def _calculate_retry_delay(self) -> float:
        """
        Calculate the delay before retrying an enqueue operation.
        Returns the delay in seconds.
        """
        try:
            loop = asyncio.get_running_loop()
            current_time = loop.time()
            time_since_last_clear = current_time - self._last_aggressive_clear_time
            remaining_cooldown = max(
                0.0,
                LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS
                - time_since_last_clear,
            )
            # Add a small buffer (10% of cooldown or 50ms, whichever is larger) to ensure
            # cooldown has expired and aggressive clear has completed
            return remaining_cooldown + max(
                0.05, LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS * 0.1
            )
        except RuntimeError:
            # No event loop, return minimum delay
            return 0.1

    def _schedule_delayed_enqueue_retry(self, task: LoggingTask) -> None:
        """
        Schedule a delayed retry to enqueue the task after cooldown expires.
        This prevents dropping tasks when the queue is full during cooldown.
        Preserves the original task context.
        """
        try:
            # Check that we have a running event loop (will raise RuntimeError if not)
            asyncio.get_running_loop()
            delay = self._calculate_retry_delay()

            # Schedule the retry as a background task
            asyncio.create_task(self._retry_enqueue_task(task, delay))
        except RuntimeError:
            # No event loop, drop the task as we can't schedule a retry
            pass

    async def _retry_enqueue_task(self, task: LoggingTask, delay: float) -> None:
        """
        Retry enqueueing the task after delay, preserving original context.
        This is called as a background task from _schedule_delayed_enqueue_retry.
        """
        await asyncio.sleep(delay)

        # Try to enqueue the task directly, preserving its original context
        if self._queue is None:
            return

        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            # Still full - handle it appropriately (clear or retry again)
            self._handle_queue_full(task)

    def _extract_tasks_from_queue(self) -> list[LoggingTask]:
        """
        Extract tasks from the queue to make room.
        Returns a list of extracted tasks based on percentage of queue size.
        """
        if self._queue is None:
            return []

        # Calculate items based on percentage of queue size
        items_to_extract = (
            self.max_queue_size * LOGGING_WORKER_CLEAR_PERCENTAGE
        ) // 100
        # Use actual queue size to avoid unnecessary iterations
        actual_size = self._queue.qsize()
        if actual_size == 0:
            return []
        items_to_extract = min(items_to_extract, actual_size)

        # Extract tasks from queue (using list comprehension would require wrapping in try/except)
        extracted_tasks = []
        for _ in range(items_to_extract):
            try:
                extracted_tasks.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        return extracted_tasks

    async def _aggressively_clear_queue_async(
        self, new_task: Optional[LoggingTask] = None
    ) -> None:
        """
        Aggressively clear the queue by extracting and processing items.
        This is called when the queue is full to prevent dropping logs.
        Fully async and non-blocking - runs in background task.
        """
        try:
            if self._queue is None:
                return

            extracted_tasks = self._extract_tasks_from_queue()

            # Add new task to extracted tasks to process directly
            if new_task is not None:
                extracted_tasks.append(new_task)

            # Process extracted tasks directly
            if extracted_tasks:
                await self._process_extracted_tasks(extracted_tasks)
        except Exception as e:
            verbose_logger.exception(
                f"LoggingWorker error during aggressive clear: {e}"
            )
        finally:
            # Always reset the flag even if an error occurs
            self._aggressive_clear_in_progress = False

    async def _process_single_task(self, task: LoggingTask) -> None:
        """Process a single task and mark it done."""
        if self._queue is None:
            return

        try:
            await asyncio.wait_for(
                task["context"].run(asyncio.create_task, task["coroutine"]),
                timeout=self.timeout,
            )
        except Exception:
            # Suppress errors during processing to ensure we keep going
            pass
        finally:
            self._queue.task_done()

    async def _process_extracted_tasks(self, tasks: list[LoggingTask]) -> None:
        """
        Process tasks that were extracted from the queue to make room.
        Processes them concurrently without semaphore limits for maximum speed.
        """
        if not tasks or self._queue is None:
            return

        # Process all tasks concurrently for maximum speed
        await asyncio.gather(*[self._process_single_task(task) for task in tasks])

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

        for _ in range(MAX_ITERATIONS_TO_CLEAR_QUEUE):
            # Check if we've exceeded the maximum time
            if asyncio.get_event_loop().time() - start_time >= MAX_TIME_TO_CLEAR_QUEUE:
                verbose_logger.warning(
                    f"clear_queue exceeded max_time of {MAX_TIME_TO_CLEAR_QUEUE}s, stopping early"
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
                finally:
                    # Clear reference to prevent memory leaks
                    task = None
                self._queue.task_done()  # If you're using join() elsewhere
            except asyncio.QueueEmpty:
                break

    def _safe_log(self, level: str, message: str) -> None:
        """
        Safely log a message during shutdown, suppressing errors if logging is closed.
        """
        # Check if logger has valid handlers before attempting to log
        # During shutdown, handlers may be closed, causing ValueError when writing
        if not hasattr(verbose_logger, 'handlers') or not verbose_logger.handlers:
            return
        
        # Check if any handler has a valid stream
        has_valid_handler = False
        for handler in verbose_logger.handlers:
            try:
                if hasattr(handler, 'stream') and handler.stream and not handler.stream.closed:
                    has_valid_handler = True
                    break
                elif not hasattr(handler, 'stream'):
                    # Non-stream handlers (like NullHandler) are always valid
                    has_valid_handler = True
                    break
            except (AttributeError, ValueError):
                continue
        
        if not has_valid_handler:
            return
        
        try:
            if level == "debug":
                verbose_logger.debug(message)
            elif level == "info":
                verbose_logger.info(message)
            elif level == "warning":
                verbose_logger.warning(message)
            elif level == "error":
                verbose_logger.error(message)
        except (ValueError, OSError, AttributeError):
            # Logging handlers may be closed during shutdown
            # Silently ignore logging errors to prevent breaking shutdown
            pass

    def _flush_on_exit(self):
        """
        Flush remaining events synchronously before process exit.
        Called automatically via atexit handler.

        This ensures callbacks queued by async completions are processed
        even when the script exits before the worker loop can handle them.

        Note: All logging in this method is wrapped to handle cases where
        logging handlers are closed during shutdown.
        """
        if self._queue is None:
            self._safe_log("debug", "[LoggingWorker] atexit: No queue initialized")
            return

        if self._queue.empty():
            self._safe_log("debug", "[LoggingWorker] atexit: Queue is empty")
            return

        queue_size = self._queue.qsize()
        self._safe_log(
            "info", f"[LoggingWorker] atexit: Flushing {queue_size} remaining events..."
        )

        # Create a new event loop since the original is closed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Process remaining queue items with time limit
            processed = 0
            start_time = loop.time()

            while not self._queue.empty() and processed < MAX_ITERATIONS_TO_CLEAR_QUEUE:
                if loop.time() - start_time >= MAX_TIME_TO_CLEAR_QUEUE:
                    self._safe_log(
                        "warning",
                        f"[LoggingWorker] atexit: Reached time limit ({MAX_TIME_TO_CLEAR_QUEUE}s), stopping flush",
                    )
                    break

                try:
                    task = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                # Run the coroutine synchronously in new loop
                # Note: We run the coroutine directly, not via create_task,
                # since we're in a new event loop context
                try:
                    loop.run_until_complete(task["coroutine"])
                    processed += 1
                except Exception:
                    # Silent failure to not break user's program
                    pass
                finally:
                    # Clear reference to prevent memory leaks
                    task = None

            self._safe_log(
                "info",
                f"[LoggingWorker] atexit: Successfully flushed {processed} events!",
            )

        finally:
            loop.close()


# Global instance for backward compatibility
GLOBAL_LOGGING_WORKER = LoggingWorker()
