"""
New Relic AI Monitoring Integration for LiteLLM

This module provides integration with New Relic's AI Monitoring feature to track
LLM requests, responses, and usage metrics.

Environment Variables:
    NEW_RELIC_LICENSE_KEY: Your New Relic license key (required)
    NEW_RELIC_APP_NAME: Your application name (required)
    NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED: Whether to record message content (optional, default: false)

Configuration:
    Message logging can be controlled via:
    1. turn_off_message_logging parameter (takes priority) - pass via callback initialization
    2. NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED env var (fallback)

Usage:
    import litellm
    litellm.callbacks = ["newrelic"]

    # Or with explicit configuration:
    from litellm.integrations.newrelic import NewRelicLogger
    litellm.callbacks = [NewRelicLogger(turn_off_message_logging=True)]

    # Ensure New Relic agent is initialized (use newrelic-admin or initialize manually)
    # newrelic-admin run-program python your_app.py
"""

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse, Message

import newrelic.agent


# Global state for supportability metric emission
# Protected by _metric_lock to ensure thread-safe access
_last_metric_emission_time: float = 0.0
_metric_lock = threading.Lock()


class NewRelicLogger(CustomLogger):
    """
    New Relic logger for LiteLLM to send AI monitoring events.

    This logger creates two types of New Relic custom events:
    1. LlmChatCompletionSummary - One per completion request
    2. LlmChatCompletionMessage - One per message (request and response)
    """

    def __init__(self, **kwargs):
        # Check if turn_off_message_logging was explicitly provided before calling super()
        turn_off_message_logging_provided = "turn_off_message_logging" in kwargs

        # CustomLogger.__init__ will set self.turn_off_message_logging from kwargs
        super().__init__(**kwargs)

        # Check for required environment variables
        self.license_key = os.getenv("NEW_RELIC_LICENSE_KEY")
        self.app_name = os.getenv("NEW_RELIC_APP_NAME")

        # Determine if message content should be recorded
        # Priority: turn_off_message_logging param > env var
        # Note: turn_off_message_logging=True means record_content=False (inverted logic)
        if turn_off_message_logging_provided:
            # Use the parameter value set by CustomLogger.__init__ (inverted for record_content)
            self.record_content = not self.turn_off_message_logging
        else:
            # Fall back to env var when parameter not provided
            self.record_content = self._parse_bool_env(
                "NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED", False
            )

        # Validate configuration
        if not self.license_key or not self.app_name:
            verbose_logger.warning(
                "New Relic integration requires NEW_RELIC_LICENSE_KEY and "
                "NEW_RELIC_APP_NAME environment variables. Integration will be disabled."
            )
            self.enabled = False
        else:
            # Validate that newrelic package is available
            try:
                newrelic.agent.register_application()

                self.enabled = True
                verbose_logger.info(
                    f"New Relic AI Monitoring initialized for app: {self.app_name}, "
                    f"content recording: {self.record_content}"
                )
            except ImportError:
                verbose_logger.error(
                    "New Relic Python agent not installed. "
                    "Install with: pip install newrelic "
                    "Integration will be disabled."
                )
                self.enabled = False

    def _parse_bool_env(self, var_name: str, default: bool = False) -> bool:
        """Parse boolean environment variable. Accepts 'true' (case-insensitive) per spec."""
        value = os.getenv(var_name, "")
        if not value:
            return default
        # Spec requires value to be either true (bool) or 'true' (string)
        return value.lower() == "true"

    def _get_litellm_version(self) -> str:
        """
        Get litellm version for supportability metrics.

        Returns:
            Version string (e.g., "1.80.0") or "unknown" if unable to determine
        """
        try:
            from importlib.metadata import version
            return version('litellm')
        except Exception as e:
            verbose_logger.warning(f"Unable to determine litellm version: {e}")
            return "unknown"

    def _emit_supportability_metric(self):
        """
        Emit New Relic supportability metric for LiteLLM usage.

        Per spec, this metric should be emitted at least once every 27 hours
        to indicate the library is in use. Format:
        Supportability/Python/ML/LiteLLM/{version}

        This method updates the global _last_metric_emission_time and should
        be called within a lock when checking periodic emission.
        """
        global _last_metric_emission_time

        try:
            litellm_version = self._get_litellm_version()
            metric_name = f"Supportability/Python/ML/LiteLLM/{litellm_version}"

            # Record metric with value of 1 (will be aggregated by New Relic)
            app = newrelic.agent.application()

            if app and app.enabled:
                app.record_custom_metric(metric_name, 1)

                # Update last emission time
                _last_metric_emission_time = time.time()

                verbose_logger.info(
                    f"Emitted New Relic supportability metric: {metric_name}"
                )
            else:
                verbose_logger.info("New Relic application is not enabled; skipping metric recording.")


        except Exception as e:
            verbose_logger.warning(f"Failed to emit supportability metric: {e}")

    def _check_and_emit_periodic_metric(self):
        """
        Check if 23 hours have passed since last metric emission and re-emit if needed.

        Uses a mutex to ensure only one thread emits the metric even if multiple
        requests are being processed concurrently.
        """
        global _last_metric_emission_time

        # Quick check without lock to avoid unnecessary locking
        current_time = time.time()
        time_since_last_emission = current_time - _last_metric_emission_time

        # 1 hour in seconds = 3600
        if time_since_last_emission >= 3600:
            # Acquire lock to ensure only one thread emits
            with _metric_lock:
                # Double-check inside lock in case another thread just emitted
                current_time = time.time()
                time_since_last_emission = current_time - _last_metric_emission_time

                if time_since_last_emission >= 82800:
                    self._emit_supportability_metric()

    def _should_record_content(self) -> bool:
        """Check if message content should be recorded."""
        return self.record_content

    def _get_trace_context(self, kwargs: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Get current New Relic trace ID and span ID from distributed tracing headers.

        This integration runs asynchronously from the actual request (via logging worker),
        so we cannot use the New Relic agent to pull the current traceId and spanId.
        Instead, we extract from request headers if available.

        For the trace ID, we look in kwargs for:
        - litellm_params.metadata.headers.traceparent (W3C Trace Context)
        - litellm_params.metadata.headers.newrelic (New Relic proprietary)

        Returns:
            Tuple of (trace_id, span_id) - both Optional[str] (None if not available)
        """
        try:
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})
            headers = metadata.get("headers", {})
            newrelic = headers.get("newrelic", None)
            traceparent = headers.get("traceparent", None)

            trace_id = None
            span_id = None

            if traceparent:
                # Extract trace_id from traceparent header if available
                # traceparent format: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
                parts = traceparent.split("-")
                if len(parts) == 4:
                    trace_id = parts[1]

            if not trace_id:
                # attempt to pull from newrelic header
                pass

            if not trace_id:
                verbose_logger.debug(
                    "New Relic trace_id not available from distributed tracing headers. "
                    "AI monitoring events will be recorded without trace correlation."
                )
                return None, None

            return trace_id, span_id

        except ImportError:
            verbose_logger.warning(
                "New Relic Python agent not available."
            )
            return None, None
        except Exception as e:
            verbose_logger.warning(f"Unable to get New Relic trace context: {e}")
            return None, None

    def _extract_completion_id(self, kwargs: Dict, response_obj: ModelResponse) -> str:
        """
        Extract completion ID from kwargs or response_obj, or generate one.
        """
        completion_id = None

        if response_obj:
            completion_id = response_obj.get("id")

        if not completion_id:
            completion_id = kwargs.get("litellm_call_id")

        # If still not found, generate UUID and log warning per spec
        if not completion_id:
            completion_id = str(uuid.uuid4())
            verbose_logger.warning(
                "No completion ID found in request or response. Generated UUID."
            )

        return completion_id

    def _get_vendor(self, kwargs: Dict) -> str:
        """Extract vendor/provider from kwargs."""
        litellm_params = kwargs.get("litellm_params", {}) or {}
        return litellm_params.get("custom_llm_provider", "litellm")

    def _get_model_names(self, kwargs: Dict, response_obj: ModelResponse) -> Tuple[str, str]:
        """
        Extract request and response model names.

        Returns:
            Tuple of (request_model, response_model)
        """
        request_model: str = str(kwargs.get("model", "unknown"))
        response_model: str = str(response_obj.get("model", request_model))
        return request_model, response_model

    def _extract_usage(self, response_obj: ModelResponse) -> Dict[str, int]:
        """Extract usage statistics from response."""
        usage = response_obj.get("usage", None)
        if not usage:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }

    def _get_finish_reason(self, response_obj: ModelResponse) -> str:
        """
        Extract finish reason from first choice in the response.

        Returns "unknown" if choices are not present or finish_reason is not found.
        """
        choices = response_obj.get("choices", [])
        if choices and len(choices) > 0:
            return choices[0].get("finish_reason", "unknown")
        return "unknown"

    def _get_duration(self, kwargs: Dict, start_time: Optional[float], end_time: Optional[float]) -> Optional[float]:
        """
        Extract duration in milliseconds.

        First tries to get llm_api_duration_ms from kwargs, then falls back to
        calculating from start_time and end_time if available.
        """
        # Try to get pre-calculated duration from kwargs
        duration_ms = kwargs.get("llm_api_duration_ms")
        if duration_ms is not None:
            return float(duration_ms)

        # Fall back to calculating from timestamps
        if start_time is not None and end_time is not None:
            return (end_time - start_time) * 1000.0  # Convert to milliseconds

        return None

    def _get_request_params(self, kwargs: Dict) -> Dict[str, Any]:
        """
        Extract request parameters like temperature and max_tokens.

        Returns dict with available parameters, omitting those not present.
        """
        optional_params = kwargs.get("optional_params", {})
        params = {}

        temperature = optional_params.get("temperature")
        if temperature is not None:
            params["temperature"] = temperature

        max_tokens = optional_params.get("max_tokens")
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        return params

    def _extract_message_content(self, message: Union[Message, Dict]) -> str:
        """
        Extract content from a message, handling various formats.

        Handles tool calls, multimodal content (as JSON), and standard text content.
        Returns empty string if content is None or missing.
        """
        content = message.get("content")

        # Handle tool calls
        if message.get("tool_calls"):
            try:
                return json.dumps(message["tool_calls"])
            except Exception:
                return str(message["tool_calls"])

        # Handle None or missing content
        if content is None:
            return ""

        # Handle list content (multimodal)
        if isinstance(content, list):
            try:
                return json.dumps(content)
            except Exception:
                return str(content)

        # Handle non-string content
        if not isinstance(content, str):
            return str(content)

        return content

    def _extract_all_messages(
        self,
        kwargs: Dict,
        response_obj: ModelResponse,
        response_model: str,
        vendor: str
    ) -> List[Dict[str, Any]]:
        """
        Extract all messages (request + response) with sequence numbers and timestamps.

        Processes request messages from kwargs["messages"] and response messages
        from response_obj["choices"]. Assigns sequential numbers starting at 0.
        Adds timestamps from kwargs if available (converted to epoch milliseconds).
        """
        messages = []
        sequence = 0

        # Extract timestamps from kwargs and convert to milliseconds
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")

        # Extract request messages
        request_messages = kwargs.get("messages", [])
        for msg in request_messages:
            message_data = {
                "role": msg.get("role", "user"),
                "sequence": sequence,
                "response.model": response_model,
                "vendor": vendor
            }

            # Add timestamp for request message if available (convert to milliseconds)
            if start_time is not None:
                # Handle both datetime objects and float timestamps
                if hasattr(start_time, 'timestamp'):
                    message_data["timestamp"] = int(start_time.timestamp() * 1000.0)
                else:
                    message_data["timestamp"] = int(start_time * 1000.0)

            # Only add content if recording is enabled
            if self._should_record_content():
                message_data["content"] = self._extract_message_content(msg)

            messages.append(message_data)
            sequence += 1

        # Extract response messages from choices
        choices = response_obj.get("choices", [])
        if choices and len(choices) > 0:
            for choice in choices:
                message = choice.get("message", None)
                if message:
                    message_data = {
                        "role": message.get("role", "assistant"),
                        "sequence": sequence,
                        "response.model": response_model,
                        "vendor": vendor,
                        "is_response": True
                    }

                    # Add timestamp for response message if available (convert to milliseconds)
                    if end_time is not None:
                        # Handle both datetime objects and float timestamps
                        if hasattr(end_time, 'timestamp'):
                            message_data["timestamp"] = int(end_time.timestamp() * 1000.0)
                        else:
                            message_data["timestamp"] = int(end_time * 1000.0)

                    # Only add content if recording is enabled
                    if self._should_record_content():
                        message_data["content"] = self._extract_message_content(message)

                    messages.append(message_data)
                    sequence += 1

        return messages

    def _record_summary_event(
        self,
        request_id: str,
        trace_id: Optional[str],
        span_id: Optional[str],
        request_model: str,
        response_model: str,
        vendor: str,
        finish_reason: str,
        num_messages: int,
        usage: Dict[str, int],
        duration: Optional[float] = None,
        request_params: Optional[Dict[str, Any]] = None
    ):
        """Record LlmChatCompletionSummary event to New Relic."""
        try:
            import newrelic.agent

            event_data = {
                "id": request_id,
                "request_id": request_id,
                "request.model": request_model,
                "response.model": response_model,
                "response.choices.finish_reason": finish_reason,
                "response.number_of_messages": num_messages,
                "vendor": vendor,
                "ingest_source": "litellm",
                "response.usage.prompt_tokens": usage["prompt_tokens"],
                "response.usage.completion_tokens": usage["completion_tokens"],
                "response.usage.total_tokens": usage["total_tokens"]
            }

            # Add optional attributes if present
            if trace_id:
                event_data["trace_id"] = trace_id
            if span_id:
                event_data["span_id"] = span_id

            if duration is not None:
                event_data["duration"] = duration

            # Add request parameters if present
            if request_params:
                if "temperature" in request_params:
                    event_data["request.temperature"] = request_params["temperature"]
                if "max_tokens" in request_params:
                    event_data["request.max_tokens"] = request_params["max_tokens"]

            app = newrelic.agent.application()

            if app and app.enabled:
                app.record_custom_event("LlmChatCompletionSummary", event_data)
            else:
                verbose_logger.info("New Relic application is not enabled; skipping event recording.")

            import pprint
            verbose_logger.info(f"Recorded LlmChatCompletionSummary event: {pprint.pformat(event_data)}")

        except Exception as e:
            verbose_logger.warning(f"Failed to record New Relic summary event: {e}")
            self.handle_callback_failure("newrelic")

    def _record_message_events(
        self,
        request_id: str,
        llm_response_id: str,
        trace_id: Optional[str],
        span_id: Optional[str],
        messages: List[Dict[str, Any]]
    ):
        """Record LlmChatCompletionMessage events to New Relic.

        Args:
            request_id: Agent-generated UUID that links to Summary event's id
            llm_response_id: LLM's response ID (e.g., "chatcmpl-...") for message id format
            trace_id: Trace ID for distributed tracing (None if not available)
            span_id: Span ID for distributed tracing (None if not available)
            messages: List of message dicts to record
        """
        try:
            app = newrelic.agent.application()

            for message in messages:
                sequence = message["sequence"]
                event_data = {
                    "id": f"{llm_response_id}-{sequence}",
                    "request_id": request_id,
                    "completion_id": request_id,
                    "role": message["role"],
                    "sequence": sequence,
                    "response.model": message["response.model"],
                    "vendor": message["vendor"],
                    "ingest_source": "litellm",
                    "token_count": 0
                }

                # Add trace context if available
                if trace_id:
                    event_data["trace_id"] = trace_id
                if span_id:
                    event_data["span_id"] = span_id

                # Add content only if it was included in the message data
                if "content" in message:
                    event_data["content"] = message["content"]

                # Add is_response only if True (per spec, omit for request messages)
                if message.get("is_response"):
                    event_data["is_response"] = True


                if app and app.enabled:
                    app.record_custom_event("LlmChatCompletionMessage", event_data)
                else:
                    verbose_logger.info("New Relic application is not enabled; skipping event recording.")

            import pprint
            verbose_logger.info(
                f"Recorded {len(messages)} LlmChatCompletionMessage events: {pprint.pformat(messages)}"
            )

        except Exception as e:
            verbose_logger.warning(f"Failed to record New Relic message events: {e}")
            self.handle_callback_failure("newrelic")

    def _record_error_metric(self):
        """Record error metric to New Relic."""
        try:
            import newrelic.agent

            newrelic.agent.record_custom_metric("LLM/LiteLLM/Error", 1)
            verbose_logger.info("Recorded LLM/LiteLLM/Error metric")

        except Exception as e:
            verbose_logger.warning(f"Failed to record New Relic error metric: {e}")
            self.handle_callback_failure("newrelic")

    def _process_success(
        self,
        kwargs: Dict,
        response_obj: ModelResponse,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """
        Core logic for processing successful LLM calls.
        Used by both sync and async success event handlers.
        """
        # Early exit if not enabled
        if not self.enabled:
            return

        # Check and emit periodic supportability metric if 23 hours have passed
        self._check_and_emit_periodic_metric()

        import pprint
        verbose_logger.info(f"newrelic._process_success called, kwargs=\n{pprint.pformat(kwargs)}, \nresponse_obj=\n{pprint.pformat(response_obj)}")

        # Get trace context (may be empty string if not available)
        trace_id, span_id = self._get_trace_context(kwargs)

        verbose_logger.info(f"Trace ID: {trace_id or '(none)'}, Span ID: {span_id or '(none)'}")

        # Generate unique request ID for this request (used as Summary event id)
        request_id = str(uuid.uuid4())

        # Extract data from response
        llm_response_id = self._extract_completion_id(kwargs, response_obj)
        vendor = self._get_vendor(kwargs)
        request_model, response_model = self._get_model_names(kwargs, response_obj)
        usage = self._extract_usage(response_obj)
        finish_reason = self._get_finish_reason(response_obj)

        # Extract additional summary event fields
        duration = self._get_duration(kwargs, start_time, end_time)
        request_params = self._get_request_params(kwargs)

        # Extract all messages
        messages = self._extract_all_messages(
            kwargs, response_obj, response_model, vendor
        )

        # Record summary event
        self._record_summary_event(
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
            request_model=request_model,
            response_model=response_model,
            vendor=vendor,
            finish_reason=finish_reason,
            num_messages=len(messages),
            usage=usage,
            duration=duration,
            request_params=request_params
        )

        # Record message events
        self._record_message_events(
            request_id=request_id,
            llm_response_id=llm_response_id,
            trace_id=trace_id,
            span_id=span_id,
            messages=messages
        )

    # CustomLogger interface implementation

    def log_pre_api_call(self, model, messages, kwargs):
        """Unused per spec."""
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        """Unused per spec."""
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Main success path for non-streaming requests.

        Note: New Relic's record_custom_event is synchronous but non-blocking
        (in-memory operation), so it's safe to call from sync context.
        """
        try:
            self._process_success(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            verbose_logger.warning(f"Error in New Relic log_success_event: {e}")
            self.handle_callback_failure("newrelic")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Main success path for async/streaming requests.

        Note: New Relic's SDK is thread-safe and record_custom_event is fast,
        so we can call it directly without asyncio.to_thread().
        """
        try:
            self._process_success(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            verbose_logger.warning(f"Error in New Relic async_log_success_event: {e}")
            self.handle_callback_failure("newrelic")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Log error metric for failed LLM calls (sync).

        Per spec: Do not send AI events on failure, only record error metric.
        """
        try:
            if not self.enabled:
                return

            self._record_error_metric()

        except Exception as e:
            verbose_logger.warning(f"Error in New Relic log_failure_event: {e}")
            self.handle_callback_failure("newrelic")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Log error metric for failed LLM calls (async).

        Per spec: Do not send AI events on failure, only record error metric.
        """
        try:
            if not self.enabled:
                return

            self._record_error_metric()

        except Exception as e:
            verbose_logger.warning(f"Error in New Relic async_log_failure_event: {e}")
            self.handle_callback_failure("newrelic")
