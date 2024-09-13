"""
DataDog Integreation - sends logs to /api/v2/log

DD Reference API: https://docs.datadoghq.com/api/latest/logs

`async_log_success_event` - used by litellm proxy to send logs to datadog
`log_success_event` - sync version of logging to DataDog, only used on litellm Python SDK, if user opts in to using sync functions
"""

import asyncio
import datetime
import os
import traceback
import uuid
from typing import Any, Dict, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .types import DatadogPayload
from .utils import make_json_serializable


class DataDogLogger(CustomBatchLogger):
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initializes the datadog logger, checks if the correct env variables are set

        Required environment variables:
        `DD_API_KEY` - your datadog api key
        `DD_SITE` - your datadog site, example = `"us5.datadoghq.com"`
        """
        try:
            verbose_logger.debug(f"Datadog: in init datadog logger")
            # check if the correct env variables are set
            if os.getenv("DD_API_KEY", None) is None:
                raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>")
            if os.getenv("DD_SITE", None) is None:
                raise Exception("DD_SITE is not set in .env, set 'DD_SITE=<>")
            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            self.DD_API_KEY = os.getenv("DD_API_KEY")
            self.intake_url = (
                f"https://http-intake.logs.{os.getenv('DD_SITE')}/api/v2/logs"
            )
            self.sync_client = _get_httpx_client()
            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            super().__init__(**kwargs, flush_lock=self.flush_lock)
        except Exception as e:
            verbose_logger.exception(
                f"Datadog: Got exception on init Datadog client {str(e)}"
            )
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log success events to Datadog

        - Creates a Datadog payload
        - Adds the Payload to the in memory logs queue
        - Payload is flushed every 10 seconds or when batch size is greater than 100


        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            verbose_logger.debug(
                "Datadog: Logging - Enters logging function for model %s", kwargs
            )
            dd_payload = self.create_datadog_logging_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )

            self.log_queue.append(dd_payload)
            verbose_logger.debug(
                f"Datadog, event added to queue. Will flush in {self.flush_interval} seconds..."
            )

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"Datadog Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    async def async_send_batch(self):
        """
        Sends the in memory logs queue to datadog api

        Logs sent to /api/v2/logs

        DD Ref: https://docs.datadoghq.com/api/latest/logs/

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            if not self.log_queue:
                verbose_logger.exception("Datadog: log_queue does not exist")
                return

            response = await self.async_client.post(
                url=self.intake_url,
                json=self.log_queue,
                headers={
                    "DD-API-KEY": self.DD_API_KEY,
                },
            )

            response.raise_for_status()
            if response.status_code != 202:
                raise Exception(
                    f"Response from datadog API status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                "Datadog: Response from datadog API status_code: %s, text: %s",
                response.status_code,
                response.text,
            )
        except Exception as e:
            verbose_logger.exception(
                f"Datadog Error sending batch API - {str(e)}\n{traceback.format_exc()}"
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync Log success events to Datadog

        - Creates a Datadog payload
        - instantly logs it on DD API
        """
        try:
            verbose_logger.debug(
                "Datadog: Logging - Enters logging function for model %s", kwargs
            )
            dd_payload = self.create_datadog_logging_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )

            response = self.sync_client.post(
                url=self.intake_url,
                json=dd_payload,
                headers={
                    "DD-API-KEY": self.DD_API_KEY,
                },
            )

            response.raise_for_status()
            if response.status_code != 202:
                raise Exception(
                    f"Response from datadog API status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                "Datadog: Response from datadog API status_code: %s, text: %s",
                response.status_code,
                response.text,
            )

        except Exception as e:
            verbose_logger.exception(
                f"Datadog Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass
        pass

    def create_datadog_logging_payload(
        self,
        kwargs: Union[dict, Any],
        response_obj: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> DatadogPayload:
        """
        Helper function to create a datadog payload for logging

        Args:
            kwargs (Union[dict, Any]): request kwargs
            response_obj (Any): llm api response
            start_time (datetime.datetime): start time of request
            end_time (datetime.datetime): end time of request

        Returns:
            DatadogPayload: defined in types.py
        """
        litellm_params = kwargs.get("litellm_params", {})
        metadata = (
            litellm_params.get("metadata", {}) or {}
        )  # if litellm_params['metadata'] == None
        messages = kwargs.get("messages")
        optional_params = kwargs.get("optional_params", {})
        call_type = kwargs.get("call_type", "litellm.completion")
        cache_hit = kwargs.get("cache_hit", False)
        usage = response_obj["usage"]
        id = response_obj.get("id", str(uuid.uuid4()))
        usage = dict(usage)
        try:
            response_time = (end_time - start_time).total_seconds() * 1000
        except:
            response_time = None

        try:
            response_obj = dict(response_obj)
        except:
            response_obj = response_obj

        # Clean Metadata before logging - never log raw metadata
        # the raw metadata can contain circular references which leads to infinite recursion
        # we clean out all extra litellm metadata params before logging
        clean_metadata = {}
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                # clean litellm metadata before logging
                if key in [
                    "endpoint",
                    "caching_groups",
                    "previous_models",
                ]:
                    continue
                else:
                    clean_metadata[key] = value

        # Build the initial payload
        payload = {
            "id": id,
            "call_type": call_type,
            "cache_hit": cache_hit,
            "start_time": start_time,
            "end_time": end_time,
            "response_time": response_time,
            "model": kwargs.get("model", ""),
            "user": kwargs.get("user", ""),
            "model_parameters": optional_params,
            "spend": kwargs.get("response_cost", 0),
            "messages": messages,
            "response": response_obj,
            "usage": usage,
            "metadata": clean_metadata,
        }

        make_json_serializable(payload)
        import json

        payload = json.dumps(payload)

        verbose_logger.debug("Datadog: Logger - Logging payload = %s", payload)

        dd_payload = DatadogPayload(
            ddsource=os.getenv("DD_SOURCE", "litellm"),
            ddtags="",
            hostname="",
            message=payload,
            service="litellm-server",
        )

        return dd_payload
