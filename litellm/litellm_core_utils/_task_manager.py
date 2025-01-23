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
        # 1) Create an event loop specifically for our async logging
        self._loop = asyncio.new_event_loop()
        # Start that loop in its own thread
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

        # 2) Create a single-worker ThreadPoolExecutor for sync logging
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit_logging_tasks_for_async_llm_call(
        self,
        logging_obj: LiteLLMLoggingObject,
        result: Any,
        start_time: datetime,
        end_time: datetime,
        is_completion_with_fallbacks: bool = False,
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
            is_completion_with_fallbacks: Whether this is a completion with fallbacks call

        """
        if not is_completion_with_fallbacks:
            # Schedule the async callback in the dedicated event-loop thread
            coro = logging_obj.async_success_handler(result, start_time, end_time)
            asyncio.run_coroutine_threadsafe(coro, self._loop)

            # Schedule any synchronous callbacks in the executor
            self.executor.submit(
                logging_obj.handle_sync_success_callbacks_for_async_calls,
                result=result,
                start_time=start_time,
                end_time=end_time,
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
        Optional: Graceful shutdown of the ThreadPoolExecutor and the event loop.
        """
        # Shut down the ThreadPoolExecutor
        self.executor.shutdown(wait=True)

        # Stop the event loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        # Join the event loop thread
        self._thread.join()
