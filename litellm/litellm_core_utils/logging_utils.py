import asyncio
import functools
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm import ModelResponse as _ModelResponse
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObject,
    )

    LiteLLMModelResponse = _ModelResponse
    Span = Union[_Span, Any]
else:
    LiteLLMModelResponse = Any
    LiteLLMLoggingObject = Any
    Span = Any


import litellm

"""
Helper utils used for logging callbacks
"""

# Global service logger instance to avoid recreating it
_service_logger = None


def _get_service_logger():
    """Get or create the global ServiceLogging instance"""
    global _service_logger
    if _service_logger is None:
        from litellm._service_logger import ServiceLogging

        _service_logger = ServiceLogging()
    return _service_logger


def _get_parent_otel_span_from_logging_obj(
    logging_obj: Optional[LiteLLMLoggingObject] = None,
) -> Optional[Span]:
    """
    Extract the parent OTEL span from the logging object using existing helper.

    Args:
        logging_obj: The LiteLLM logging object containing model call details

    Returns:
        The parent OTEL span if found, None otherwise
    """
    try:
        if logging_obj is None or not hasattr(logging_obj, "model_call_details"):
            return None

        # Reuse existing function by passing model_call_details as kwargs
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )

        return _get_parent_otel_span_from_kwargs(logging_obj.model_call_details)

    except Exception as e:
        verbose_logger.exception(
            f"Error in _get_parent_otel_span_from_logging_obj: {str(e)}"
        )
        return None


def convert_litellm_response_object_to_str(
    response_obj: Union[Any, LiteLLMModelResponse],
) -> Optional[str]:
    """
    Get the string of the response object from LiteLLM

    """
    if isinstance(response_obj, litellm.ModelResponse):
        response_str = ""
        for choice in response_obj.choices:
            if isinstance(choice, litellm.Choices):
                if choice.message.content and isinstance(choice.message.content, str):
                    response_str += choice.message.content
        return response_str

    return None


def _assemble_complete_response_from_streaming_chunks(
    result: Union[ModelResponse, TextCompletionResponse, ModelResponseStream],
    start_time: datetime,
    end_time: datetime,
    request_kwargs: dict,
    streaming_chunks: List[Any],
    is_async: bool,
):
    """
    Assemble a complete response from a streaming chunks

    - assemble a complete streaming response if result.choices[0].finish_reason is not None
    - else append the chunk to the streaming_chunks


    Args:
        result: ModelResponse
        start_time: datetime
        end_time: datetime
        request_kwargs: dict
        streaming_chunks: List[Any]
        is_async: bool

    Returns:
        Optional[Union[ModelResponse, TextCompletionResponse]]: Complete streaming response

    """
    complete_streaming_response: Optional[
        Union[ModelResponse, TextCompletionResponse]
    ] = None

    if isinstance(result, ModelResponse):
        return result

    if result.choices[0].finish_reason is not None:  # if it's the last chunk
        streaming_chunks.append(result)
        try:
            complete_streaming_response = litellm.stream_chunk_builder(
                chunks=streaming_chunks,
                messages=request_kwargs.get("messages", None),
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            log_message = (
                "Error occurred building stream chunk in {} success logging: {}".format(
                    "async" if is_async else "sync", str(e)
                )
            )
            verbose_logger.exception(log_message)
            complete_streaming_response = None
    else:
        streaming_chunks.append(result)
    return complete_streaming_response


def _set_duration_in_model_call_details(
    logging_obj: Any,  # we're not guaranteed this will be `LiteLLMLoggingObject`
    start_time: datetime,
    end_time: datetime,
):
    """Helper to set duration in model_call_details, with error handling"""
    try:
        duration_ms = (end_time - start_time).total_seconds() * 1000
        if logging_obj and hasattr(logging_obj, "model_call_details"):
            logging_obj.model_call_details["llm_api_duration_ms"] = duration_ms
        else:
            verbose_logger.debug(
                "`logging_obj` not found - unable to track `llm_api_duration_ms"
            )
    except Exception as e:
        verbose_logger.warning(f"Error setting `llm_api_duration_ms`: {str(e)}")


def track_llm_api_timing():
    """
    Decorator to track LLM API call timing for both sync and async functions.
    The logging_obj is expected to be passed as an argument to the decorated function.
    Logs timing using ServiceLogging similar to Redis cache.
    """

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            start_time_float = time.time()
            logging_obj = kwargs.get("logging_obj", None)

            # Extract parent OTEL span from logging object
            parent_otel_span = _get_parent_otel_span_from_logging_obj(logging_obj)

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = datetime.now()
                end_time_float = time.time()
                duration = end_time_float - start_time_float

                # Set duration in model call details
                _set_duration_in_model_call_details(
                    logging_obj=logging_obj,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Log timing using ServiceLogging (like Redis cache)
                try:
                    from litellm.types.services import ServiceTypes

                    service_logger = _get_service_logger()

                    # Get function name for call_type
                    call_type = f"{func.__name__} <- track_llm_api_timing"

                    # Create async task for service logging (similar to Redis cache pattern)
                    asyncio.create_task(
                        service_logger.async_service_success_hook(
                            service=ServiceTypes.LITELLM,
                            duration=duration,
                            call_type=call_type,
                            start_time=start_time_float,
                            end_time=end_time_float,
                            parent_otel_span=parent_otel_span,
                        )
                    )
                except Exception as e:
                    verbose_logger.debug(f"Error in service logging: {str(e)}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            start_time_float = time.time()
            logging_obj = kwargs.get("logging_obj", None)

            # Extract parent OTEL span from logging object
            parent_otel_span = _get_parent_otel_span_from_logging_obj(logging_obj)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = datetime.now()
                end_time_float = time.time()
                duration = end_time_float - start_time_float

                # Set duration in model call details
                _set_duration_in_model_call_details(
                    logging_obj=logging_obj,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Log timing using ServiceLogging (like Redis cache)
                try:
                    from litellm.types.services import ServiceTypes

                    service_logger = _get_service_logger()

                    # Get function name for call_type
                    call_type = f"{func.__name__} <- track_llm_api_timing"

                    # Use sync service logging for sync functions
                    service_logger.service_success_hook(
                        service=ServiceTypes.LITELLM,
                        duration=duration,
                        call_type=call_type,
                        start_time=start_time_float,
                        end_time=end_time_float,
                        parent_otel_span=parent_otel_span,
                    )
                except Exception as e:
                    verbose_logger.debug(f"Error in service logging: {str(e)}")

        # Check if the function is async or sync
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
