"""
PostHog Integration - sends LLM analytics events to PostHog

Follows PostHog's LLM Analytics format: https://posthog.com/docs/llm-analytics/manual-capture

async_log_success_event: stores batch of events in memory and flushes to PostHog
async_log_failure_event: logs failed LLM calls with error information

For batching specific details see CustomBatchLogger class
"""

import asyncio
import os
from typing import Any, Dict, Optional, Tuple

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.posthog import (
    POSTHOG_MAX_BATCH_SIZE,
    PostHogEventPayload,
)
from litellm.types.utils import StandardCallbackDynamicParams, StandardLoggingPayload


class PostHogLogger(CustomBatchLogger):
    def __init__(self, **kwargs):
        """
        Initializes the PostHog logger, checks if the correct env variables are set

        Required environment variables:
        `POSTHOG_API_KEY` - your PostHog API key
        `POSTHOG_API_URL` - your PostHog API URL (defaults to https://app.posthog.com)
        """
        try:
            verbose_logger.debug("PostHog: in init posthog logger")
            if os.getenv("POSTHOG_API_KEY", None) is None:
                raise Exception("POSTHOG_API_KEY is not set, set 'POSTHOG_API_KEY=<>'")

            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            self.sync_client = _get_httpx_client()
            
            self.POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")
            posthog_api_url = os.getenv("POSTHOG_API_URL", "https://us.i.posthog.com")
            self.posthog_host = posthog_api_url.rstrip('/')
            self.capture_url = f"{self.posthog_host}/batch/"

            self._async_initialized = False
            self.flush_lock = None
            self.log_queue = []
            
            super().__init__(
                **kwargs, flush_lock=None, batch_size=POSTHOG_MAX_BATCH_SIZE
            )

        except Exception as e:
            verbose_logger.exception(
                f"PostHog: Got exception on init PostHog client {str(e)}"
            )
            raise e

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "PostHog: Sync logging - Enters logging function for model %s", kwargs
            )

            api_key, api_url = self._get_credentials_for_request(kwargs)
            if api_key is None or api_url is None:
                raise Exception("PostHog credentials not found in kwargs")
            event_payload = self.create_posthog_event_payload(kwargs)

            headers = {
                "Content-Type": "application/json",
            }

            payload = self._create_posthog_payload([event_payload], api_key)
            capture_url = f"{api_url.rstrip('/')}/batch/"

            response = self.sync_client.post(
                url=capture_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

            if response.status_code != 200:
                raise Exception(
                    f"Response from PostHog API status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug("PostHog: Sync event successfully sent")

        except Exception as e:
            verbose_logger.exception(f"PostHog Sync Layer Error - {str(e)}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "PostHog: Async logging - Enters logging function for model %s", kwargs
            )
            self._ensure_async_setup()  # Lazy initialization
            await self._log_async_event(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            verbose_logger.exception(f"PostHog Layer Error - {str(e)}")
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "PostHog: Async logging - Enters logging function for model %s", kwargs
            )
            self._ensure_async_setup()  # Lazy initialization
            await self._log_async_event(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            verbose_logger.exception(f"PostHog Layer Error - {str(e)}")
            pass

    async def _log_async_event(self, kwargs, response_obj=None, start_time=0.0, end_time=0.0):
        # Note: response_obj, start_time, end_time not used - all data comes from kwargs
        api_key, api_url = self._get_credentials_for_request(kwargs)
        event_payload = self.create_posthog_event_payload(kwargs)

        # Store event with its credentials for batch sending
        self.log_queue.append({
            "event": event_payload,
            "api_key": api_key,
            "api_url": api_url
        })
        verbose_logger.debug(
            f"PostHog, event added to queue. Will flush in {self.flush_interval} seconds..."
        )

        if len(self.log_queue) >= self.batch_size:
            await self.flush_queue()

    def create_posthog_event_payload(self, kwargs: Dict[str, Any]) -> PostHogEventPayload:
        """
        Helper function to create a PostHog event payload for logging

        Args:
            kwargs (Dict[str, Any]): request kwargs containing standard_logging_object

        Returns:
            PostHogEventPayload: defined in types.py
        """
        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_object is None:
            raise ValueError("standard_logging_object not found in kwargs")

        call_type = standard_logging_object.get("call_type", "")
        event_name = "$ai_embedding" if call_type == "embedding" else "$ai_generation"

        properties = self._create_posthog_properties(
            standard_logging_object=standard_logging_object,
            kwargs=kwargs,
            event_name=event_name,
        )

        distinct_id = self._get_distinct_id(standard_logging_object, kwargs)

        return PostHogEventPayload(
            event=event_name,
            properties=properties,
            distinct_id=distinct_id,
        )

    def _create_posthog_properties(
        self,
        standard_logging_object: StandardLoggingPayload,
        kwargs: Dict[str, Any],
        event_name: str,
    ) -> Dict[str, Any]:
        """Create PostHog properties following LLM Analytics spec"""
        properties = {}

        # Core model information
        properties["$ai_model"] = self._safe_get(standard_logging_object, "model", "")
        properties["$ai_provider"] = self._safe_get(standard_logging_object, "custom_llm_provider", "")

        # Input/Output data
        messages = self._safe_get(standard_logging_object, "messages")
        if messages is not None:
            properties["$ai_input"] = messages

        if event_name == "$ai_generation":
            response = self._safe_get(standard_logging_object, "response")
            if response is not None:
                properties["$ai_output_choices"] = response

        # Token information
        properties["$ai_input_tokens"] = self._safe_get(standard_logging_object, "prompt_tokens", 0)
        if event_name == "$ai_generation":
            properties["$ai_output_tokens"] = self._safe_get(standard_logging_object, "completion_tokens", 0)

        # Cost and performance
        response_cost = self._safe_get(standard_logging_object, "response_cost")
        if response_cost is not None:
            properties["$ai_total_cost_usd"] = response_cost

        properties["$ai_latency"] = self._safe_get(standard_logging_object, "response_time", 0.0)

        # Error handling
        if self._safe_get(standard_logging_object, "status") == "failure":
            properties["$ai_is_error"] = True
            error_str = self._safe_get(standard_logging_object, "error_str")
            if error_str is not None:
                properties["$ai_error"] = error_str

        # Add trace properties
        self._add_trace_properties(properties, kwargs)

        # Add custom metadata fields
        self._add_custom_metadata_properties(properties, kwargs)

        return properties

    def _add_trace_properties(self, properties: Dict[str, Any], kwargs: Dict[str, Any]):
        standard_logging_object = self._safe_get(kwargs, "standard_logging_object", {})

        trace_id = self._safe_get(standard_logging_object, "trace_id", self._safe_uuid())
        properties["$ai_trace_id"] = trace_id

        span_id = self._safe_get(standard_logging_object, "id", self._safe_uuid())
        properties["$ai_span_id"] = span_id

        metadata = self._extract_metadata(kwargs)
        parent_id = metadata.get("parent_run_id") or metadata.get("parent_id")
        if parent_id:
            properties["$ai_parent_id"] = parent_id

    def _add_custom_metadata_properties(self, properties: Dict[str, Any], kwargs: Dict[str, Any]):
        """Add custom metadata fields to PostHog properties"""
        metadata = self._extract_metadata(kwargs)
        if not isinstance(metadata, dict):
            return

        litellm_internal_fields = {
            "endpoint", "caching_groups", "user_api_key_hash", "user_api_key_alias",
            "user_api_key_team_id", "user_api_key_user_id", "user_api_key_org_id",
            "user_api_key_team_alias", "user_api_key_end_user_id", "user_api_key_user_email",
            "user_api_key", "user_api_end_user_max_budget", "litellm_api_version",
            "global_max_parallel_requests", "user_api_key_team_max_budget", "user_api_key_team_spend",
            "user_api_key_spend", "user_api_key_max_budget", "user_api_key_model_max_budget",
            "user_api_key_metadata", "headers", "litellm_parent_otel_span", "requester_ip_address",
            "model_group", "model_group_size", "deployment", "model_info", "api_base",
            "caching_groups", "hidden_params", "parent_run_id", "parent_id", "user_id"
        }

        for key, value in metadata.items():
            if key not in litellm_internal_fields:
                properties[key] = value

    def _get_distinct_id(
        self, standard_logging_object: StandardLoggingPayload, kwargs: Dict[str, Any]
    ) -> str:
        metadata = self._extract_metadata(kwargs)
        user_id = self._safe_get(metadata, "user_id")
        if user_id:
            return str(user_id)
        end_user = self._safe_get(standard_logging_object, "end_user")
        if end_user:
            return str(end_user)
        trace_id = self._safe_get(standard_logging_object, "trace_id")
        if trace_id:
            return str(trace_id)

        return self._safe_uuid()

    def _get_credentials_for_request(self, kwargs: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """
        Get PostHog credentials for this request.

        Checks for per-request credentials in standard_callback_dynamic_params,
        falls back to instance defaults from environment variables.

        Args:
            kwargs: Request kwargs containing standard_callback_dynamic_params

        Returns:
            tuple[str, str]: (api_key, api_url)
        """
        standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = (
            kwargs.get("standard_callback_dynamic_params", None)
        )

        if standard_callback_dynamic_params is not None:
            api_key = standard_callback_dynamic_params.get("posthog_api_key") or self.POSTHOG_API_KEY
            api_url = standard_callback_dynamic_params.get("posthog_api_url") or self.posthog_host
        else:
            api_key = self.POSTHOG_API_KEY
            api_url = self.posthog_host

        return api_key, api_url

    async def async_send_batch(self):
        """
        Sends the in memory logs queue to PostHog API

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                f"PostHog: Sending batch of {len(self.log_queue)} events"
            )

            # Group events by credentials for batch sending
            batches_by_credentials: Dict[tuple[str, str], list] = {}
            for item in self.log_queue:
                key = (item["api_key"], item["api_url"])
                if key not in batches_by_credentials:
                    batches_by_credentials[key] = []
                batches_by_credentials[key].append(item["event"])

            # Send each batch to its respective PostHog instance
            for (api_key, api_url), events in batches_by_credentials.items():
                headers = {
                    "Content-Type": "application/json",
                }

                payload = self._create_posthog_payload(events, api_key)
                capture_url = f"{api_url.rstrip('/')}/batch/"

                response = await self.async_client.post(
                    url=capture_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()

                if response.status_code != 200:
                    raise Exception(
                        f"Response from PostHog API status_code: {response.status_code}, text: {response.text}"
                    )

            verbose_logger.debug(
                f"PostHog: Batch of {len(self.log_queue)} events successfully sent"
            )
        except Exception as e:
            verbose_logger.exception(f"PostHog Error sending batch API - {str(e)}")

    def _ensure_async_setup(self):
        if not self._async_initialized:
            try:
                self.flush_lock = asyncio.Lock()
                asyncio.create_task(self.periodic_flush())
                self._async_initialized = True
                verbose_logger.debug("PostHog: Async components initialized")
            except Exception as e:
                verbose_logger.error(f"PostHog: Failed to initialize async components: {str(e)}")
                raise

    def _extract_metadata(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        litellm_params = kwargs.get("litellm_params", {}) or {}
        return litellm_params.get("metadata", {}) or {}

    def _safe_uuid(self) -> str:
        return str(uuid.uuid4())

    def _create_posthog_payload(self, events: list, api_key: str) -> Dict[str, Any]:
        return {"api_key": api_key, "batch": events}

    def _safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        if obj is None or not hasattr(obj, 'get'):
            return default
        return obj.get(key, default)
