import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObject,
    )
else:
    LiteLLMLoggingObject = Any


class LoggingTaskManager:
    """
    Manages logging tasks for async and sync LLM calls.

    We create:
      - a dedicated event loop (in a separate thread) for async logging
      - a ThreadPoolExecutor (single worker) for sync logging
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.queue = asyncio.Queue()
        self.worker_task = None
        self._initialized = False

    async def _initialize(self):
        """Initialize the worker task in an async context"""
        if not self._initialized:
            self.worker_task = asyncio.create_task(self._queue_worker())
            self._initialized = True

    async def _queue_worker(self):
        while True:
            try:
                # Get the next logging task from the queue
                logging_obj, result, start_time, end_time, kwargs = (
                    await self.queue.get()
                )

                # Process the logging
                await logging_obj.async_success_handler(
                    result, start_time, end_time, **kwargs
                )

                # Mark task as done
                self.queue.task_done()
            except Exception as e:
                continue

    async def submit_logging_tasks_for_async_llm_call(
        self,
        logging_obj: LiteLLMLoggingObject,
        result: Any,
        start_time: datetime,
        end_time: datetime,
        is_completion_with_fallbacks: bool = False,
        **kwargs,
    ) -> None:
        """
        - Schedule the async_success_handler(...) to run on the dedicated async loop.
        - Schedule any synchronous logging via the ThreadPoolExecutor.

        Args:
            logging_obj: LiteLLMLoggingObject containing callback handlers
            result: The result from the LLM API call
            start_time: Unix timestamp of when the call started
            end_time: Unix timestamp of when the call ended
            cache_hit: Whether the call was a cache hit
            is_completion_with_fallbacks: Whether this is a completion with fallbacks call. If it is true, we will not run the async_success_handler.

        """
        if not self._initialized:
            await self._initialize()

        if not is_completion_with_fallbacks:
            # Add to queue instead of executing directly
            await self.queue.put((logging_obj, result, start_time, end_time, kwargs))

            if logging_obj._should_run_sync_callbacks_for_async_calls():
                # Schedule any synchronous callbacks in the executor
                self.executor.submit(
                    self.submit_logging_tasks_for_sync_llm_call,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    **kwargs,
                )

    def submit_logging_tasks_for_sync_llm_call(
        self,
        logging_obj: LiteLLMLoggingObject,
        result: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Submit logging tasks to be executed in background thread for sync LLM calls

        Args:
            logging_obj: LiteLLMLoggingObject containing callback handlers
            result: The result from the LLM API call
            start_time: Unix timestamp of when the call started
            end_time: Unix timestamp of when the call ended
            cache_hit: Whether the call was a cache hit
        """
        self.executor.submit(
            logging_obj.success_handler,
            result,
            start_time,
            end_time,
        )

    def shutdown(self):
        """
        Optional: Graceful shutdown of the ThreadPoolExecutor and the worker task.
        """

        # Shut down the ThreadPoolExecutor
        self.executor.shutdown(wait=True)
