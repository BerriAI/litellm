"""
Callback to log events to a Generic API Endpoint

- Creates a StandardLoggingPayload
- Adds to batch queue
- Flushes based on CustomBatchLogger settings
"""

import asyncio
import os
import traceback
import uuid
from typing import Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload


class GenericAPILogger(CustomBatchLogger):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the GenericAPILogger

        Args:
            endpoint: Optional[str] = None,
            headers: Optional[dict] = None,
        """
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
        verbose_logger.debug(
            f"in init GenericAPILogger, endpoint {self.endpoint}, headers {self.headers}"
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
        from litellm.proxy.utils import _premium_user_check

        _premium_user_check()

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
        from litellm.proxy.utils import _premium_user_check

        _premium_user_check()

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
        """
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                f"Generic API Logger - about to flush {len(self.log_queue)} events"
            )

            # make POST request to Generic API Endpoint
            response = await self.async_httpx_client.post(
                url=self.endpoint,
                headers=self.headers,
                data=safe_dumps(self.log_queue),
            )

            verbose_logger.debug(
                f"Generic API Logger - sent batch to {self.endpoint}, status code {response.status_code}"
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
