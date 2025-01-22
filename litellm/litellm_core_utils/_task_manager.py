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
    def __init__(self):
        """
        Initialize TaskManager with a ThreadPoolExecutor with a single worker
        """
        self.executor = ThreadPoolExecutor(max_workers=1)

    def submit_logging_tasks_for_async_llm_call(
        self,
        logging_obj: LiteLLMLoggingObject,
        result: Any,
        start_time: datetime,
        end_time: datetime,
        is_completion_with_fallbacks: bool = False,
    ) -> None:
        """
        Submit logging tasks to be executed in background thread for async LLM calls

        Args:
            logging_obj: LiteLLMLoggingObject containing callback handlers
            result: The result from the LLM API call
            start_time: Unix timestamp of when the call started
            end_time: Unix timestamp of when the call ended
            is_completion_with_fallbacks: Whether this is a completion with fallbacks call
        """
        if not is_completion_with_fallbacks:  # Avoid double logging for fallback calls
            if logging_obj.async_success_handler:
                self.executor.submit(
                    logging_obj.async_success_handler,
                    result,
                    start_time,
                    end_time,
                )

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
        """
        self.executor.submit(
            logging_obj.success_handler,
            result,
            start_time,
            end_time,
        )
