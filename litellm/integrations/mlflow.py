import json
import threading
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.types.utils import StandardLoggingPayload
else:
    StandardLoggingPayload = Any


class MlflowLogger(CustomLogger):
    def __init__(self):
        from mlflow.tracking import MlflowClient

        self._client = MlflowClient()

        self._stream_id_to_span = {}
        self._lock = threading.Lock()  # lock for _stream_id_to_span

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success(kwargs, response_obj, start_time, end_time)

    def _handle_success(self, kwargs, response_obj, start_time, end_time):
        """
        Log the success event as an MLflow span.
        Note that this method is called asynchronously in the background thread.
        """
        from mlflow.entities import SpanStatusCode

        try:
            verbose_logger.debug("MLflow logging start for success event")

            if kwargs.get("stream"):
                self._handle_stream_event(kwargs, response_obj, start_time, end_time)
            else:
                span = self._start_span_or_trace(kwargs, start_time)
                end_time_ns = int(end_time.timestamp() * 1e9)
                self._extract_and_set_chat_attributes(span, kwargs, response_obj)
                self._end_span_or_trace(
                    span=span,
                    outputs=response_obj,
                    status=SpanStatusCode.OK,
                    end_time_ns=end_time_ns,
                )
        except Exception:
            verbose_logger.debug("MLflow Logging Error", stack_info=True)

    def _extract_and_set_chat_attributes(self, span, kwargs, response_obj):
        try:
            from mlflow.tracing.utils import set_span_chat_messages  # type: ignore
            from mlflow.tracing.utils import set_span_chat_tools  # type: ignore
        except ImportError:
            return

        inputs = self._construct_input(kwargs)
        input_messages = inputs.get("messages", [])
        output_messages = [
            c.message.model_dump(exclude_none=True)
            for c in getattr(response_obj, "choices", [])
        ]
        if messages := [*input_messages, *output_messages]:
            set_span_chat_messages(span, messages)
        if tools := inputs.get("tools"):
            set_span_chat_tools(span, tools)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        """
        Log the failure event as an MLflow span.
        Note that this method is called *synchronously* unlike the success handler.
        """
        from mlflow.entities import SpanEvent, SpanStatusCode

        try:
            span = self._start_span_or_trace(kwargs, start_time)

            end_time_ns = int(end_time.timestamp() * 1e9)

            # Record exception info as event
            if exception := kwargs.get("exception"):
                span.add_event(SpanEvent.from_exception(exception))  # type: ignore

            self._extract_and_set_chat_attributes(span, kwargs, response_obj)
            self._end_span_or_trace(
                span=span,
                outputs=response_obj,
                status=SpanStatusCode.ERROR,
                end_time_ns=end_time_ns,
            )

        except Exception as e:
            verbose_logger.debug(f"MLflow Logging Error - {e}", stack_info=True)

    def _handle_stream_event(self, kwargs, response_obj, start_time, end_time):
        """
        Handle the success event for a streaming response. For streaming calls,
        log_success_event handle is triggered for every chunk of the stream.
        We create a single span for the entire stream request as follows:

        1. For the first chunk, start a new span and store it in the map.
        2. For subsequent chunks, add the chunk as an event to the span.
        3. For the final chunk, end the span and remove the span from the map.
        """
        from mlflow.entities import SpanStatusCode

        litellm_call_id = kwargs.get("litellm_call_id")

        if litellm_call_id not in self._stream_id_to_span:
            with self._lock:
                # Check again after acquiring lock
                if litellm_call_id not in self._stream_id_to_span:
                    # Start a new span for the first chunk of the stream
                    span = self._start_span_or_trace(kwargs, start_time)
                    self._stream_id_to_span[litellm_call_id] = span

        # Add chunk as event to the span
        span = self._stream_id_to_span[litellm_call_id]
        self._add_chunk_events(span, response_obj)

        # If this is the final chunk, end the span. The final chunk
        # has complete_streaming_response that gathers the full response.
        if final_response := kwargs.get("complete_streaming_response"):
            end_time_ns = int(end_time.timestamp() * 1e9)

            self._extract_and_set_chat_attributes(span, kwargs, final_response)
            self._end_span_or_trace(
                span=span,
                outputs=final_response,
                status=SpanStatusCode.OK,
                end_time_ns=end_time_ns,
            )

            # Remove the stream_id from the map
            with self._lock:
                self._stream_id_to_span.pop(litellm_call_id)

    def _add_chunk_events(self, span, response_obj):
        from mlflow.entities import SpanEvent

        try:
            for choice in response_obj.choices:
                span.add_event(
                    SpanEvent(
                        name="streaming_chunk",
                        attributes={"delta": json.dumps(choice.delta.model_dump())},
                    )
                )
        except Exception:
            verbose_logger.debug("Error adding chunk events to span", stack_info=True)

    def _construct_input(self, kwargs):
        """Construct span inputs with optional parameters"""
        inputs = {"messages": kwargs.get("messages")}
        if tools := kwargs.get("tools"):
            inputs["tools"] = tools

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
        standard_obj: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")
        if standard_obj:
            attributes.update(
                {
                    "api_base": standard_obj.get("api_base"),
                    "cache_hit": standard_obj.get("cache_hit"),
                    "mlflow.chat.tokenUsage": {
                        "input_tokens": standard_obj.get("prompt_tokens"),
                        "output_tokens": standard_obj.get("completion_tokens"),
                        "total_tokens": standard_obj.get("total_tokens"),
                    },
                    "raw_llm_response": standard_obj.get("response"),
                    "response_cost": standard_obj.get("response_cost"),
                    "saved_cache_cost": standard_obj.get("saved_cache_cost"),
                    "request_tags": standard_obj.get("request_tags"),
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

        if active_span := mlflow.get_current_active_span():  # type: ignore
            return self._client.start_span(
                name=span_name,
                trace_id=active_span.request_id,
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
                tags=self._transform_tag_list_to_dict(attributes.get("request_tags", [])),
                start_time_ns=start_time_ns,
            )
    def _transform_tag_list_to_dict(self, tag_list: list) -> dict:
        return {tag: "" for tag in tag_list}

    def _end_span_or_trace(self, span, outputs, end_time_ns, status):
        """End an MLflow span or a trace."""
        if span.parent_id is None:
            self._client.end_trace(
                trace_id=span.request_id,
                outputs=outputs,
                status=status,
                end_time_ns=end_time_ns,
            )
        else:
            self._client.end_span(
                trace_id=span.request_id,
                span_id=span.span_id,
                outputs=outputs,
                status=status,
                end_time_ns=end_time_ns,
            )
