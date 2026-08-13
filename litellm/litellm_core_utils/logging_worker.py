# This file may be a good candidate to be the first one to be refactored into a separate process,
# for the sake of performance and scalability.

import asyncio
import atexit
import contextvars
import logging
import threading
import weakref
from collections.abc import Coroutine
from typing import Final

from typing_extensions import TypedDict

from litellm._logging import verbose_logger
from litellm.constants import (
    LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS,
    LOGGING_WORKER_CLEAR_PERCENTAGE,
    LOGGING_WORKER_CONCURRENCY,
    LOGGING_WORKER_MAX_QUEUE_SIZE,
    LOGGING_WORKER_MAX_TIME_PER_COROUTINE,
    MAX_ITERATIONS_TO_CLEAR_QUEUE,
    MAX_TIME_TO_CLEAR_QUEUE,
)


class LoggingTask(TypedDict):
    """
    A logging task with its associated context to ensure logging is executed in
    the original task's context.
    """

    coroutine: Coroutine[object, object, object]
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
        register_atexit: bool = True,
    ):
        self.timeout = timeout
        self.max_queue_size = max_queue_size
        self.concurrency = concurrency
        self._queue: asyncio.Queue[LoggingTask] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._running_tasks: set[asyncio.Task[None]] = set()  # mutable-ok: tracks live tasks for cancellation
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._stopping: bool = False
        self._last_aggressive_clear_time: float = 0.0
        self._aggressive_clear_in_progress: bool = False

        if register_atexit:
            atexit.register(self._flush_on_exit)

    def _ensure_queue(self) -> None:
        """Initialize the queue if it doesn't exist or if event loop has changed."""
        try:
            current_loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, can't initialize
            return

        # Check if we need to reinitialize due to event loop change
        if self._queue is not None and self._bound_loop is not current_loop:
            verbose_logger.debug("LoggingWorker: Event loop changed, reinitializing queue and worker")
            # Clear old state - these are bound to the old loop
            self._queue = None
            self._worker_task = None
            self._running_tasks.clear()

        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
            self._bound_loop = current_loop

    def start(self) -> None:
        """Start the logging worker. Idempotent - safe to call multiple times."""
        self._ensure_queue()

    async def _process_log_task(self, task: LoggingTask) -> None:
        """Run one logging task and update the queue completion counter."""
        try:
            await asyncio.wait_for(task["coroutine"], timeout=self.timeout)
        except Exception as e:
            verbose_logger.exception("LoggingWorker error: %s", e)
        finally:
            if self._queue is not None:
                self._queue.task_done()

    def _start_queued_tasks(self) -> None:
        if self._queue is None or self._stopping:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        while len(self._running_tasks) < self.concurrency:
            if not self._start_one_queued_task():
                return

    def _start_one_queued_task(self) -> bool:
        if self._queue is None:
            return False
        try:
            logging_task: Final = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False

        processing_task: Final = logging_task["context"].run(
            asyncio.create_task,
            self._process_log_task(logging_task),
        )
        self._running_tasks.add(processing_task)
        self._worker_task = processing_task
        processing_task.add_done_callback(self._processing_task_done)
        return True

    def _processing_task_done(self, task: asyncio.Task[None]) -> None:
        self._running_tasks.discard(task)
        if self._worker_task is task:
            self._worker_task = next(iter(self._running_tasks), None)
        self._start_queued_tasks()

    def enqueue(self, coroutine: Coroutine[object, object, object]) -> None:
        """
        Add a coroutine to the logging queue.
        Hot path: never blocks, aggressively clears queue if full.
        """
        if self._queue is None:
            return

        # Capture the current context when enqueueing
        task: Final = LoggingTask(coroutine=coroutine, context=contextvars.copy_context())

        try:
            self._queue.put_nowait(task)
            self._start_queued_tasks()
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
            loop: Final = asyncio.get_running_loop()
            current_time: Final = loop.time()
            time_since_last_clear: Final = current_time - self._last_aggressive_clear_time

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
        loop: Final = asyncio.get_running_loop()
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
            loop: Final = asyncio.get_running_loop()
            current_time: Final = loop.time()
            time_since_last_clear: Final = current_time - self._last_aggressive_clear_time
            remaining_cooldown: Final = max(
                0.0,
                LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS - time_since_last_clear,
            )
            # Add a small buffer (10% of cooldown or 50ms, whichever is larger) to ensure
            # cooldown has expired and aggressive clear has completed
            return remaining_cooldown + max(0.05, LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS * 0.1)
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
            delay: Final = self._calculate_retry_delay()

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
        items_to_extract = (self.max_queue_size * LOGGING_WORKER_CLEAR_PERCENTAGE) // 100
        # Use actual queue size to avoid unnecessary iterations
        actual_size: Final = self._queue.qsize()
        if actual_size == 0:
            return []
        items_to_extract = min(items_to_extract, actual_size)

        # Extract tasks from queue (using list comprehension would require wrapping in try/except)
        extracted_tasks: Final = []
        for _ in range(items_to_extract):
            try:
                extracted_tasks.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        return extracted_tasks

    async def _aggressively_clear_queue_async(self, new_task: LoggingTask | None = None) -> None:
        """
        Aggressively clear the queue by extracting and processing items.
        This is called when the queue is full to prevent dropping logs.
        Fully async and non-blocking - runs in background task.
        """
        try:
            if self._queue is None:
                return

            extracted_tasks: Final = self._extract_tasks_from_queue()

            # Add new task to extracted tasks to process directly
            if new_task is not None:
                extracted_tasks.append(new_task)

            # Process extracted tasks directly
            if extracted_tasks:
                await self._process_extracted_tasks(extracted_tasks)
        except Exception as e:
            verbose_logger.exception("LoggingWorker error during aggressive clear: %s", e)
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

    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine[object, object, object]) -> None:
        """
        Ensure the logging worker is initialized and enqueue the coroutine.
        """
        self.start()
        self.enqueue(async_coroutine)

    async def stop(self) -> None:
        """Stop the logging worker and clean up resources."""
        if self._worker_task is None and not self._running_tasks:
            await self.clear_queue()
            return

        self._stopping = True
        tasks_to_cancel: Final[list[asyncio.Task[None]]] = list(self._running_tasks)

        for task in tasks_to_cancel:
            # Propagate cancellation to every pending task.
            task.cancel()

        # Wait for cancellation to settle; ignore errors raised during shutdown.
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._worker_task = None
        # Drop references to completed tasks so we can restart cleanly.
        self._running_tasks.clear()
        await self.clear_queue()
        self._stopping = False

    async def flush(self) -> None:
        """Flush the logging queue.

        Waits until every enqueued task has completed. ``queue.join()`` blocks
        on the queue's unfinished-task counter (decremented by ``task_done()``),
        so it correctly handles items that have been dequeued but whose
        callback hasn't finished yet — ``queue.empty()`` would return True in
        that window and cause us to skip the wait.
        """
        if self._queue is None:
            return
        await self._queue.join()

    async def clear_queue(self):
        """
        Clear the queue with a maximum time limit.
        """
        if self._queue is None:
            return

        start_time: Final = asyncio.get_event_loop().time()

        for _ in range(MAX_ITERATIONS_TO_CLEAR_QUEUE):
            # Check if we've exceeded the maximum time
            if asyncio.get_event_loop().time() - start_time >= MAX_TIME_TO_CLEAR_QUEUE:
                verbose_logger.warning("clear_queue exceeded max_time of %ss, stopping early", MAX_TIME_TO_CLEAR_QUEUE)
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
        if not hasattr(verbose_logger, "handlers") or not verbose_logger.handlers:
            return

        # Check if any handler has a valid stream
        has_valid_handler = False
        for handler in verbose_logger.handlers:
            try:
                if hasattr(handler, "stream") and handler.stream and not handler.stream.closed:
                    has_valid_handler = True
                    break
                elif not hasattr(handler, "stream"):
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

        queue_size: Final = self._queue.qsize()
        self._safe_log("info", f"[LoggingWorker] atexit: Flushing {queue_size} remaining events...")

        # Create a new event loop since the original is closed
        loop: Final = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Process remaining queue items with time limit
            processed = 0
            start_time: Final = loop.time()

            # logging.raiseExceptions is a process-wide global; scope the
            # suppression to just the drain loop, where shutdown callbacks may
            # log to already-closed handler streams, so other threads keep their
            # logging error reporting for as little of the window as possible.
            previous_raise_exceptions: Final = logging.raiseExceptions
            logging.raiseExceptions = False
            try:
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
            finally:
                logging.raiseExceptions = previous_raise_exceptions

            self._safe_log(
                "info",
                f"[LoggingWorker] atexit: Successfully flushed {processed} events!",
            )

        finally:
            loop.close()

    def flush_on_exit(self) -> None:
        self._flush_on_exit()


class LoopAwareLoggingWorker:
    def __init__(
        self,
        timeout: float = LOGGING_WORKER_MAX_TIME_PER_COROUTINE,
        max_queue_size: int = LOGGING_WORKER_MAX_QUEUE_SIZE,
        concurrency: int = LOGGING_WORKER_CONCURRENCY,
    ) -> None:
        self.timeout = timeout
        self.max_queue_size = max_queue_size
        self.concurrency = concurrency
        self._workers: dict[  # mutable-ok: registry is updated as event loops appear and close
            int,
            tuple[weakref.ReferenceType[asyncio.AbstractEventLoop], LoggingWorker],
        ] = {}
        self._lock = threading.Lock()
        atexit.register(self._flush_on_exit)

    def _worker_for_current_loop(self, *, create: bool = True) -> LoggingWorker | None:
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return None

        with self._lock:
            stale_loop_ids: Final = tuple(
                loop_id
                for loop_id, (loop_ref, _worker) in self._workers.items()
                if (tracked_loop := loop_ref()) is None or tracked_loop.is_closed()
            )
            stale_workers: Final = tuple(self._workers.pop(loop_id)[1] for loop_id in stale_loop_ids)
            worker_entry: Final = self._workers.get(id(loop))
            worker: LoggingWorker | None = (
                worker_entry[1] if worker_entry is not None and worker_entry[0]() is loop else None
            )
            if worker is None and create:
                worker = LoggingWorker(
                    timeout=self.timeout,
                    max_queue_size=self.max_queue_size,
                    concurrency=self.concurrency,
                    register_atexit=False,
                )
                self._workers[id(loop)] = (weakref.ref(loop), worker)

        for stale_worker in stale_workers:
            stale_worker.flush_on_exit()
        return worker

    def start(self) -> None:
        worker: Final = self._worker_for_current_loop()
        if worker is not None:
            worker.start()

    def enqueue(self, coroutine: Coroutine[object, object, object]) -> None:
        worker: Final = self._worker_for_current_loop()
        if worker is not None:
            worker.enqueue(coroutine)

    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine[object, object, object]) -> None:
        worker: Final = self._worker_for_current_loop()
        if worker is not None:
            worker.ensure_initialized_and_enqueue(async_coroutine)

    async def stop(self) -> None:
        loop: Final = asyncio.get_running_loop()
        worker: Final = self._worker_for_current_loop(create=False)
        if worker is None:
            return

        await worker.stop()
        with self._lock:
            worker_entry: Final = self._workers.get(id(loop))
            if worker_entry is not None and worker_entry[0]() is loop:
                self._workers.pop(id(loop))

    async def flush(self) -> None:
        worker: Final = self._worker_for_current_loop(create=False)
        if worker is not None:
            await worker.flush()

    async def clear_queue(self) -> None:
        worker: Final = self._worker_for_current_loop(create=False)
        if worker is not None:
            await worker.clear_queue()

    def _flush_on_exit(self) -> None:
        with self._lock:
            workers: Final = tuple(worker for _loop_ref, worker in self._workers.values())
        for worker in workers:
            worker.flush_on_exit()


# Global instance for backward compatibility
GLOBAL_LOGGING_WORKER: Final = LoopAwareLoggingWorker()
