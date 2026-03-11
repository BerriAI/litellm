"""
Callback to log events to a Generic API Endpoint

- Creates a StandardLoggingPayload
- Adds to batch queue
- Flushes based on CustomBatchLogger settings
"""

import asyncio
import json
import os
import re
import traceback
from typing import Dict, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload

API_EVENT_TYPES = Literal["llm_api_success", "llm_api_failure"]
LOG_FORMAT_TYPES = Literal["json_array", "ndjson", "single"]


def load_compatible_callbacks() -> Dict:
    """
    Load the generic_api_compatible_callbacks.json file

    Returns:
        Dict: Dictionary of compatible callbacks configuration
    """
    try:
        json_path = os.path.join(
            os.path.dirname(__file__), "generic_api_compatible_callbacks.json"
        )
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        verbose_logger.warning(
            f"Error loading generic_api_compatible_callbacks.json: {str(e)}"
        )
        return {}


def is_callback_compatible(callback_name: str) -> bool:
    """
    Check if a callback_name exists in the compatible callbacks list

    Args:
        callback_name: Name of the callback to check

    Returns:
        bool: True if callback_name exists in the compatible callbacks, False otherwise
    """
    compatible_callbacks = load_compatible_callbacks()
    return callback_name in compatible_callbacks


def get_callback_config(callback_name: str) -> Optional[Dict]:
    """
    Get the configuration for a specific callback

    Args:
        callback_name: Name of the callback to get config for

    Returns:
        Optional[Dict]: Configuration dict for the callback, or None if not found
    """
    compatible_callbacks = load_compatible_callbacks()
    return compatible_callbacks.get(callback_name)


def substitute_env_variables(value: str) -> str:
    """
    Replace {{environment_variables.VAR_NAME}} patterns with actual environment variable values

    Args:
        value: String that may contain {{environment_variables.VAR_NAME}} patterns

    Returns:
        str: String with environment variables substituted
    """
    pattern = r"\{\{environment_variables\.([A-Z_]+)\}\}"

    def replace_env_var(match):
        env_var_name = match.group(1)
        return os.getenv(env_var_name, "")

    return re.sub(pattern, replace_env_var, value)


class GenericAPILogger(CustomBatchLogger):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        headers: Optional[dict] = None,
        event_types: Optional[List[API_EVENT_TYPES]] = None,
        callback_name: Optional[str] = None,
        log_format: Optional[LOG_FORMAT_TYPES] = None,
        **kwargs,
    ):
        """
        Initialize the GenericAPILogger

        Args:
            endpoint: Optional[str] = None,
            headers: Optional[dict] = None,
            event_types: Optional[List[API_EVENT_TYPES]] = None,
            callback_name: Optional[str] = None - If provided, loads config from generic_api_compatible_callbacks.json
            log_format: Optional[LOG_FORMAT_TYPES] = None - Format for log output: "json_array" (default), "ndjson", or "single"
        """
        #########################################################
        # Check if callback_name is provided and load config
        #########################################################
        if callback_name:
            if is_callback_compatible(callback_name):
                verbose_logger.debug(
                    f"Loading configuration for callback: {callback_name}"
                )
                callback_config = get_callback_config(callback_name)

                # Use config from JSON if not explicitly provided
                if callback_config:
                    if endpoint is None and "endpoint" in callback_config:
                        endpoint = substitute_env_variables(callback_config["endpoint"])

                    if "headers" in callback_config:
                        headers = headers or {}
                        for key, value in callback_config["headers"].items():
                            if key not in headers:
                                headers[key] = substitute_env_variables(value)

                    if event_types is None and "event_types" in callback_config:
                        event_types = callback_config["event_types"]

                    if log_format is None and "log_format" in callback_config:
                        log_format = callback_config["log_format"]
            else:
                verbose_logger.warning(
                    f"callback_name '{callback_name}' not found in generic_api_compatible_callbacks.json"
                )

        #########################################################
        # Init httpx client
        #########################################################
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        endpoint = endpoint or os.getenv("GENERIC_LOGGER_ENDPOINT")
        if endpoint is None:
            raise ValueError(
                "endpoint not set for GenericAPILogger, GENERIC_LOGGER_ENDPOINT not found in environment variables"
            )

        self.headers: Dict = self._get_headers(headers)
        self.endpoint: str = endpoint
        self.event_types: Optional[List[API_EVENT_TYPES]] = event_types
        self.callback_name: Optional[str] = callback_name

        # Validate and store log_format
        if log_format is not None and log_format not in ["json_array", "ndjson", "single"]:
            raise ValueError(
                f"Invalid log_format: {log_format}. Must be one of: 'json_array', 'ndjson', 'single'"
            )
        self.log_format: LOG_FORMAT_TYPES = log_format or "json_array"

        verbose_logger.debug(
            f"in init GenericAPILogger, callback_name: {self.callback_name}, endpoint {self.endpoint}, headers {self.headers}, event_types: {self.event_types}, log_format: {self.log_format}"
        )

        #########################################################
        # Init variables for batch flushing logs
        #########################################################
        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)
        asyncio.create_task(self.periodic_flush())
        self.log_queue: List[Union[Dict, StandardLoggingPayload]] = []

    def _get_headers(self, headers: Optional[dict] = None):
        """
        Get headers for the Generic API Logger

        Returns:
            Dict: Headers for the Generic API Logger

        Args:
            headers: Optional[dict] = None
        """
        # Process headers from different sources
        headers_dict = {
            "Content-Type": "application/json",
        }

        # 1. First check for headers from env var
        env_headers = os.getenv("GENERIC_LOGGER_HEADERS")
        if env_headers:
            try:
                # Parse headers in format "key1=value1,key2=value2" or "key1=value1"
                header_items = env_headers.split(",")
                for item in header_items:
                    if "=" in item:
                        key, value = item.split("=", 1)
                        headers_dict[key.strip()] = value.strip()
            except Exception as e:
                verbose_logger.warning(
                    f"Error parsing headers from environment variables: {str(e)}"
                )

        # 2. Update with litellm generic headers if available
        if litellm.generic_logger_headers:
            headers_dict.update(litellm.generic_logger_headers)

        # 3. Override with directly provided headers if any
        if headers:
            headers_dict.update(headers)

        return headers_dict

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log success events to Generic API Endpoint

        - Creates a StandardLoggingPayload
        - Adds to batch queue
        - Flushes based on CustomBatchLogger settings

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """

        if self.event_types is not None and "llm_api_success" not in self.event_types:
            return

        try:
            verbose_logger.debug(
                "Generic API Logger - Enters logging function for model %s", kwargs
            )
            standard_logging_payload = kwargs.get("standard_logging_object", None)

            # Backwards compatibility with old logging payload
            if litellm.generic_api_use_v1 is True:
                payload = self._get_v1_logging_payload(
                    kwargs=kwargs,
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                )
                self.log_queue.append(payload)
            else:
                # New logging payload, StandardLoggingPayload
                self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"Generic API Logger Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log failure events to Generic API Endpoint

        - Creates a StandardLoggingPayload
        - Adds to batch queue
        """
        if self.event_types is not None and "llm_api_failure" not in self.event_types:
            return

        try:
            verbose_logger.debug(
                "Generic API Logger - Enters logging function for model %s", kwargs
            )
            standard_logging_payload = kwargs.get("standard_logging_object", None)

            if litellm.generic_api_use_v1 is True:
                payload = self._get_v1_logging_payload(
                    kwargs=kwargs,
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                )
                self.log_queue.append(payload)
            else:
                self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"Generic API Logger Error - {str(e)}\n{traceback.format_exc()}"
            )

    async def async_send_batch(self):
        """
        Sends the batch of messages to Generic API Endpoint

        Supports three formats:
        - json_array: Sends all logs as a JSON array (default)
        - ndjson: Sends logs as newline-delimited JSON
        - single: Sends each log as individual HTTP request in parallel
        """
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                f"Generic API Logger - about to flush {len(self.log_queue)} events in '{self.log_format}' format"
            )

            if self.log_format == "single":
                # Send each log as individual HTTP request in parallel
                tasks = []
                for log_entry in self.log_queue:
                    task = self.async_httpx_client.post(
                        url=self.endpoint,
                        headers=self.headers,
                        data=safe_dumps(log_entry),
                    )
                    tasks.append(task)

                # Execute all requests in parallel
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Log results
                for idx, result in enumerate(responses):
                    if isinstance(result, Exception):
                        verbose_logger.exception(
                            f"Generic API Logger - Error sending log {idx}: {result}"
                        )
                    else:
                        # result is a Response object
                        verbose_logger.debug(
                            f"Generic API Logger - sent log {idx}, status: {result.status_code}"  # type: ignore
                        )
            else:
                # Format the payload based on log_format
                if self.log_format == "json_array":
                    data = safe_dumps(self.log_queue)
                elif self.log_format == "ndjson":
                    data = "\n".join(safe_dumps(log) for log in self.log_queue)
                else:
                    raise ValueError(f"Unknown log_format: {self.log_format}")

                # Make POST request
                response = await self.async_httpx_client.post(
                    url=self.endpoint,
                    headers=self.headers,
                    data=data,
                )

                verbose_logger.debug(
                    f"Generic API Logger - sent batch to {self.endpoint}, "
                    f"status: {response.status_code}, format: {self.log_format}"
                )

        except Exception as e:
            verbose_logger.exception(
                f"Generic API Logger Error sending batch - {str(e)}\n{traceback.format_exc()}"
            )
        finally:
            self.log_queue.clear()

    def _get_v1_logging_payload(
        self, kwargs, response_obj, start_time, end_time
    ) -> dict:
        """
        Maintained for backwards compatibility with old logging payload

        Returns a dict of the payload to send to the Generic API Endpoint
        """
        verbose_logger.debug(
            f"GenericAPILogger Logging - Enters logging function for model {kwargs}"
        )

        # construct payload to send custom logger
        # follows the same params as langfuse.py
        litellm_params = kwargs.get("litellm_params", {})
        metadata = (
            litellm_params.get("metadata", {}) or {}
        )  # if litellm_params['metadata'] == None
        messages = kwargs.get("messages")
        cost = kwargs.get("response_cost", 0.0)
        optional_params = kwargs.get("optional_params", {})
        call_type = kwargs.get("call_type", "litellm.completion")
        cache_hit = kwargs.get("cache_hit", False)
        usage = response_obj["usage"]
        id = response_obj.get("id", str(uuid.uuid4()))

        # Build the initial payload
        payload = {
            "id": id,
            "call_type": call_type,
            "cache_hit": cache_hit,
            "startTime": start_time,
            "endTime": end_time,
            "model": kwargs.get("model", ""),
            "user": kwargs.get("user", ""),
            "modelParameters": optional_params,
            "messages": messages,
            "response": response_obj,
            "usage": usage,
            "metadata": metadata,
            "cost": cost,
        }

        # Ensure everything in the payload is converted to str
        for key, value in payload.items():
            try:
                payload[key] = str(value)
            except Exception:
                # non blocking if it can't cast to a str
                pass

        return payload
