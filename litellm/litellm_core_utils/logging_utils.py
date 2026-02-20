import asyncio
import functools
import inspect
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.constants import MAX_BASE64_LENGTH_FOR_LOGGING
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

_BYTES_PER_KIB = 1024
_BYTES_PER_MIB = 1024 * 1024

# Regex matching data-URI base64 content: "data:<mime>;base64,<payload>"
# Captures: group(1)=mime_type, group(2)=base64_payload
_DATA_URI_RE = re.compile(r"data:([^;]+);base64,([A-Za-z0-9+/=]+)")

# Maximum nesting depth for _truncate_base64_in_value to guard against
# pathological payloads. OpenAI message format is typically 3-4 levels deep.
_MAX_TRUNCATION_DEPTH = 20


def _format_base64_size(num_chars: int) -> str:
    """Return a human-readable byte-size estimate from a base64 character count."""
    num_bytes = num_chars * 3 / 4
    if num_bytes >= _BYTES_PER_MIB:
        return f"{num_bytes / _BYTES_PER_MIB:.2f}MB"
    if num_bytes >= _BYTES_PER_KIB:
        return f"{num_bytes / _BYTES_PER_KIB:.1f}KB"
    return f"{int(num_bytes)}B"


def _base64_data_uri_replacer(match: re.Match) -> str:
    """Replace a single base64 data-URI match with a size placeholder if too long."""
    mime_type = match.group(1)
    payload = match.group(2)
    if len(payload) <= MAX_BASE64_LENGTH_FOR_LOGGING:
        return match.group(0)
    size_str = _format_base64_size(len(payload))
    return f"data:{mime_type};base64,[base64_data truncated: {size_str}]"


def _truncate_base64_in_string(value: str) -> str:
    """Replace long base64 data-URI payloads in a string with a size placeholder."""
    if MAX_BASE64_LENGTH_FOR_LOGGING <= 0:
        return value
    return _DATA_URI_RE.sub(_base64_data_uri_replacer, value)


def _truncate_base64_in_value(value: Any) -> Any:
    """Iteratively truncate base64 data URIs in a JSON-like value (str/list/dict).

    Uses an explicit stack instead of recursion to satisfy the project's
    recursive-function detector and avoid stack-overflow on deep payloads.
    """
    # Stack entries: (source_value, depth, parent_container, key_or_index)
    # We mutate *copies* of dicts/lists in-place via parent references.
    if isinstance(value, str):
        return _truncate_base64_in_string(value)
    if not isinstance(value, (dict, list)):
        return value

    # Shallow-copy the root so we don't mutate the caller's data.
    root = {k: v for k, v in value.items()} if isinstance(value, dict) else list(value)
    stack: list = [(root, 0)]

    while stack:
        container, depth = stack.pop()
        if depth > _MAX_TRUNCATION_DEPTH:
            continue
        if isinstance(container, dict):
            for k, v in container.items():
                if isinstance(v, str):
                    container[k] = _truncate_base64_in_string(v)
                elif isinstance(v, dict):
                    copy: Union[dict, list] = {ck: cv for ck, cv in v.items()}
                    container[k] = copy
                    stack.append((copy, depth + 1))
                elif isinstance(v, list):
                    copy = list(v)
                    container[k] = copy
                    stack.append((copy, depth + 1))
        elif isinstance(container, list):
            for i, v in enumerate(container):
                if isinstance(v, str):
                    container[i] = _truncate_base64_in_string(v)
                elif isinstance(v, dict):
                    copy = {ck: cv for ck, cv in v.items()}
                    container[i] = copy
                    stack.append((copy, depth + 1))
                elif isinstance(v, list):
                    copy = list(v)
                    container[i] = copy
                    stack.append((copy, depth + 1))

    return root


def truncate_base64_in_messages(
    messages: Optional[Union[str, list, dict]],
) -> Optional[Union[str, list, dict]]:
    """
    Return a copy of *messages* with long base64 data-URI payloads replaced
    by human-readable size placeholders.
    """
    if messages is None or MAX_BASE64_LENGTH_FOR_LOGGING <= 0:
        return messages
    try:
        return _truncate_base64_in_value(messages)
    except Exception as e:
        verbose_logger.debug("Failed to truncate base64 in messages: %s", e)
        return messages


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
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
