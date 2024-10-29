import logging
import threading
from typing import Optional

from litellm.integrations.custom_logger import CustomLogger

_logger = logging.getLogger(__name__)


class MlflowLogger(CustomLogger):
    def __init__(self):
        from mlflow.tracking import MlflowClient

        self._client = MlflowClient()

        self._stream_id_to_span = {}
        self._lock = threading.Lock()  # lock for _stream_id_to_span

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Log the success event as an MLflow span.
        Note that this method is called asynchronously in the background thread.
        """
        from mlflow.entities import SpanStatusCode

        try:
            span = self._start_span_or_trace(kwargs, start_time)
            end_time_ns = int(end_time.timestamp() * 1e9)
            self._end_span_or_trace(
                span=span,
                outputs=response_obj,
                status=SpanStatusCode.OK,
                end_time_ns=end_time_ns,
            )
        except Exception as e:
            _logger.debug(f"Failed to log success event for litellm call: {e}", exc_info=True)


    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Log the failure event as an MLflow span.
        Note that this method is called *synchronously* unlike the success handler.
        """
        from mlflow.entities import SpanEvent, SpanStatusCode

        try:
            span = self._start_span_or_trace(kwargs, start_time)

            end_time_ns = int(end_time.timestamp() * 1e9)

            # Record exception info as event
            exception_attributes = {
                "exception.message": kwargs.get("exception"),
                "exception.type": None,
                "exception.stacktrace": kwargs.get("traceback_exception"),
            }
            event = SpanEvent(
                name="exception", timestamp=end_time_ns, attributes=exception_attributes
            )
            span.add_event(event)

            self._end_span_or_trace(
                span=span,
                outputs=response_obj,
                status=SpanStatusCode.ERROR,
                end_time_ns=end_time_ns,
            )

        except Exception as e:
            _logger.debug(f"Failed to log failure event for litellm call: {e}", exc_info=True)

    def _construct_input(self, kwargs):
        """Construct span inputs with optional parameters"""
        inputs = {"messages": kwargs.get("messages")}
        for key in ["functions", "tools", "stream", "tool_choice", "user"]:
            if value := kwargs.get("optional_params", {}).pop(key, None):
                inputs[key] = value
        return inputs

    def _extract_attributes(self, kwargs):
        """
        Extract span attributes from kwargs.

        With the latest version of litellm, the standard_logging_object contains
        canonical information for logging. If it is not present, we extract
        subset of attributes from other kwargs.
        """
        attributes = {
            "litellm_call_id": kwargs.get("litellm_call_id"),
            "call_type": kwargs.get("call_type"),
            "model": kwargs.get("model"),
        }
        standard_obj = kwargs.get("standard_logging_object")
        if standard_obj:
            attributes.update(
                {
                    "api_base": standard_obj.get("api_base"),
                    "cache_hit": standard_obj.get("cache_hit"),
                    "usage": {
                        "completion_tokens": standard_obj.get("completion_token"),
                        "prompt_tokens": standard_obj.get("prompt_token"),
                        "total_tokens": standard_obj.get("total_token"),
                    },
                    "raw_llm_response": standard_obj.get("response"),
                    "response_cost": standard_obj.get("response_cost"),
                    "saved_cache_cost": standard_obj.get("saved_cache_cost"),
                }
            )
        else:
            litellm_params = kwargs.get("litellm_params", {})
            attributes.update(
                {
                    "model": kwargs.get("model"),
                    "cache_hit": kwargs.get("cache_hit"),
                    "custom_llm_provider": kwargs.get("custom_llm_provider"),
                    "api_base": litellm_params.get("api_base"),
                    "response_cost": kwargs.get("response_cost"),
                }
            )
        return attributes

    def _get_span_type(self, call_type: Optional[str]) -> str:
        from mlflow.entities import SpanType

        if call_type in ["completion", "acompletion"]:
            return SpanType.LLM
        elif call_type == "embeddings":
            return SpanType.EMBEDDING
        else:
            return SpanType.LLM

    def _start_span_or_trace(self, kwargs, start_time):
        """
        Start an MLflow span or a trace.

        If there is an active span, we start a new span as a child of
        that span. Otherwise, we start a new trace.
        """
        import mlflow

        call_type = kwargs.get("call_type", "completion")
        span_name = f"litellm-{call_type}"
        span_type = self._get_span_type(call_type)
        start_time_ns = int(start_time.timestamp() * 1e9)

        inputs = self._construct_input(kwargs)
        attributes = self._extract_attributes(kwargs)

        if active_span := mlflow.get_current_active_span():
            return self._client.start_span(
                name=span_name,
                request_id=active_span.request_id,
                parent_id=active_span.span_id,
                span_type=span_type,
                inputs=inputs,
                attributes=attributes,
                start_time_ns=start_time_ns,
            )
        else:
            return self._client.start_trace(
                name=span_name,
                span_type=span_type,
                inputs=inputs,
                attributes=attributes,
                start_time_ns=start_time_ns,
            )

    def _end_span_or_trace(self, span, outputs, end_time_ns, status):
        """End an MLflow span or a trace."""
        if span.parent_id is None:
            self._client.end_trace(
                request_id=span.request_id,
                outputs=outputs,
                status=status,
                end_time_ns=end_time_ns,
            )
        else:
            self._client.end_span(
                request_id=span.request_id,
                span_id=span.span_id,
                outputs=outputs,
                status=status,
                end_time_ns=end_time_ns,
            )
