"""
New Relic AI Monitoring Integration for LiteLLM

This module provides integration with New Relic's AI Monitoring feature to track
LLM requests, responses, and usage metrics.

Environment Variables:
    NEW_RELIC_LICENSE_KEY: Your New Relic license key (required)
    NEW_RELIC_APP_NAME: Your application name (required)
    NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED: Whether to record message content (optional, default: false)

Usage:
    import litellm
    litellm.callbacks = ["newrelic"]

    # Ensure New Relic agent is initialized (use newrelic-admin or initialize manually)
    # newrelic-admin run-program python your_app.py
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger


class NewRelicLogger(CustomLogger):
    """
    New Relic logger for LiteLLM to send AI monitoring events.

    This logger creates two types of New Relic custom events:
    1. LlmChatCompletionSummary - One per completion request
    2. LlmChatCompletionMessage - One per message (request and response)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Check for required environment variables
        self.license_key = os.getenv("NEW_RELIC_LICENSE_KEY")
        self.app_name = os.getenv("NEW_RELIC_APP_NAME")
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
                import newrelic.agent
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

    def _should_record_content(self) -> bool:
        """Check if message content should be recorded."""
        return self.record_content

    def _get_trace_context(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get current New Relic trace ID and span ID.

        Returns:
            Tuple of (trace_id, span_id) or (None, None) if not available
        """
        try:
            import newrelic.agent

            trace_id = newrelic.agent.current_trace_id()
            span_id = newrelic.agent.current_span_id()

            if not trace_id or not span_id:
                verbose_logger.warning(
                    "New Relic trace_id or span_id not available. "
                    "Skipping New Relic event recording."
                )
                return None, None

            return trace_id, span_id

        except ImportError:
            verbose_logger.warning(
                "New Relic Python agent not available. Skipping event recording."
            )
            return None, None
        except Exception:
            verbose_logger.warning("Unable to get New Relic trace context.")
            return None, None

    def _extract_completion_id(self, kwargs: Dict, response_obj: Dict) -> str:
        """
        Extract completion ID from kwargs or response_obj, or generate one.

        Per spec: Check kwargs first, then response_obj, then generate UUID.
        """
        # Check kwargs first per spec
        completion_id = kwargs.get("id")

        # If not in kwargs, check response_obj
        if not completion_id:
            completion_id = response_obj.get("id")

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
        return litellm_params.get("custom_llm_provider", "unknown")

    def _get_model_names(self, kwargs: Dict, response_obj: Dict) -> Tuple[str, str]:
        """
        Extract request and response model names.

        Returns:
            Tuple of (request_model, response_model)
        """
        request_model = kwargs.get("model", "unknown")
        response_model = response_obj.get("model", request_model)
        return request_model, response_model

    def _extract_usage(self, response_obj: Dict) -> Dict[str, int]:
        """Extract usage statistics from response."""
        usage = response_obj.get("usage", {})
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

    def _get_finish_reason(self, response_obj: Dict) -> str:
        """
        Extract finish reason from first choice in the response.

        Returns "unknown" if choices are not present or finish_reason is not found.
        """
        choices = response_obj.get("choices", [])
        if choices and len(choices) > 0:
            return choices[0].get("finish_reason", "unknown")
        return "unknown"

    def _extract_message_content(self, message: Dict) -> str:
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
        response_obj: Dict,
        response_model: str,
        vendor: str
    ) -> List[Dict[str, Any]]:
        """
        Extract all messages (request + response) with sequence numbers.

        Processes request messages from kwargs["messages"] and response messages
        from response_obj["choices"]. Assigns sequential numbers starting at 0.
        """
        messages = []
        sequence = 0

        # Extract request messages
        request_messages = kwargs.get("messages", [])
        for msg in request_messages:
            message_data = {
                "role": msg.get("role", "user"),
                "sequence": sequence,
                "response.model": response_model,
                "vendor": vendor
            }

            # Only add content if recording is enabled
            if self._should_record_content():
                message_data["content"] = self._extract_message_content(msg)

            messages.append(message_data)
            sequence += 1

        # Extract response messages from choices
        choices = response_obj.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            if message:
                message_data = {
                    "role": message.get("role", "assistant"),
                    "sequence": sequence,
                    "response.model": response_model,
                    "vendor": vendor
                }

                # Only add content if recording is enabled
                if self._should_record_content():
                    message_data["content"] = self._extract_message_content(message)

                messages.append(message_data)
                sequence += 1

        return messages

    def _record_summary_event(
        self,
        completion_id: str,
        trace_id: str,
        span_id: str,
        request_model: str,
        response_model: str,
        vendor: str,
        finish_reason: str,
        num_messages: int,
        usage: Dict[str, int]
    ):
        """Record LlmChatCompletionSummary event to New Relic."""
        try:
            import newrelic.agent

            event_data = {
                "id": completion_id,
                "trace_id": trace_id,
                "span_id": span_id,
                "request.model": request_model,
                "response.model": response_model,
                "response.choices.finish_reason": finish_reason,
                "response.number_of_messages": num_messages,
                "vendor": vendor,
                "response.usage.prompt_tokens": usage["prompt_tokens"],
                "response.usage.completion_tokens": usage["completion_tokens"],
                "response.usage.total_tokens": usage["total_tokens"]
            }

            newrelic.agent.record_custom_event("LlmChatCompletionSummary", event_data)
            verbose_logger.debug("Recorded LlmChatCompletionSummary event")

        except Exception as e:
            verbose_logger.warning(f"Failed to record New Relic summary event: {e}")
            self.handle_callback_failure("newrelic")

    def _record_message_events(
        self,
        completion_id: str,
        trace_id: str,
        span_id: str,
        messages: List[Dict[str, Any]]
    ):
        """Record LlmChatCompletionMessage events to New Relic."""
        try:
            import newrelic.agent

            for message in messages:
                event_data = {
                    "completion_id": completion_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "role": message["role"],
                    "sequence": message["sequence"],
                    "response.model": message["response.model"],
                    "vendor": message["vendor"]
                }

                # Add content only if it was included in the message data
                if "content" in message:
                    event_data["content"] = message["content"]

                newrelic.agent.record_custom_event("LlmChatCompletionMessage", event_data)

            verbose_logger.debug(
                f"Recorded {len(messages)} LlmChatCompletionMessage events"
            )

        except Exception as e:
            verbose_logger.warning(f"Failed to record New Relic message events: {e}")
            self.handle_callback_failure("newrelic")

    def _record_error_metric(self):
        """Record error metric to New Relic."""
        try:
            import newrelic.agent

            newrelic.agent.record_custom_metric("LLM/LiteLLM/Error", 1)
            verbose_logger.debug("Recorded LLM/LiteLLM/Error metric")

        except Exception as e:
            verbose_logger.warning(f"Failed to record New Relic error metric: {e}")
            self.handle_callback_failure("newrelic")

    def _process_success(self, kwargs: Dict, response_obj: Dict):
        """
        Core logic for processing successful LLM calls.
        Used by both sync and async success event handlers.
        """
        # Early exit if not enabled
        if not self.enabled:
            return

        # Get trace context
        trace_id, span_id = self._get_trace_context()
        if not trace_id or not span_id:
            return

        # Extract data from response
        completion_id = self._extract_completion_id(kwargs, response_obj)
        vendor = self._get_vendor(kwargs)
        request_model, response_model = self._get_model_names(kwargs, response_obj)
        usage = self._extract_usage(response_obj)
        finish_reason = self._get_finish_reason(response_obj)

        # Extract all messages
        messages = self._extract_all_messages(
            kwargs, response_obj, response_model, vendor
        )

        # Record summary event
        self._record_summary_event(
            completion_id=completion_id,
            trace_id=trace_id,
            span_id=span_id,
            request_model=request_model,
            response_model=response_model,
            vendor=vendor,
            finish_reason=finish_reason,
            num_messages=len(messages),
            usage=usage
        )

        # Record message events
        self._record_message_events(
            completion_id=completion_id,
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
            self._process_success(kwargs, response_obj)
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
            self._process_success(kwargs, response_obj)
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
