"""
Callback to log events to a Generic API Endpoint

- Creates a StandardLoggingPayload
- Adds to batch queue
- Flushes based on CustomBatchLogger settings
"""

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
    ):
        """
        Initialize the GenericAPILogger

        Args:
            endpoint: Optional[str] = None,
            headers: Optional[dict] = None,
        """
        endpoint = endpoint or os.getenv("GENERIC_LOGGER_ENDPOINT")
        if endpoint is None:
            raise ValueError(
                "endpoint not set for GenericAPILogger, GENERIC_LOGGER_ENDPOINT not found in environment variables"
            )
        headers = headers or litellm.generic_logger_headers
        self.endpoint: str = endpoint
        self.headers: Dict = headers or {}
        self.log_queue: List[Union[Dict, StandardLoggingPayload]] = []
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        verbose_logger.debug(
            f"in init GenericAPILogger, endpoint {self.endpoint}, headers {self.headers}"
        )

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
