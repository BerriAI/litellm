"""
Splunk Integration - sends logs to /services/collector/event

Splunk HTTP Event Collector (HEC) Reference: https://docs.splunk.com/Documentation/HEC/latest/HTTPHEC/HEC
`async_log_event` - used by litellm proxy to send logs to Splunk
`log_event` - sync version of logging to Splunk, only used on litellm Python SDK, if user opts in to using sync functions

async_log_event:  will store batch of SPLUNK_MAX_BATCH_SIZE in memory and flush to Splunk once it reaches SPLUNK_MAX_BATCH_SIZE or every 5 seconds

For batching specific details see CustomBatchLogger class
"""

import asyncio
import json
import os
import traceback
from typing import Any, List, Optional, Union

from httpx import Response

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.splunk import *
from litellm.types.utils import StandardLoggingPayload


class SplunkLogger(
    CustomBatchLogger,
):
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initializes the Splunk logger, checks if the correct env variables are set

        Required environment variables:
        `SPLUNK_HEC_TOKEN` - your Splunk HEC token
        `SPLUNK_HEC_URL` - your Splunk HEC URL, example = `"https://localhost:8088"`
        """
        try:
            verbose_logger.debug("Splunk: in init Splunk logger")
            # Check if the correct env variables are set
            if os.getenv("SPLUNK_HEC_TOKEN", None) is None:
                raise Exception("SPLUNK_HEC_TOKEN is not set, set 'SPLUNK_HEC_TOKEN=<>")
            if os.getenv("SPLUNK_HEC_URL", None) is None:
                raise Exception("SPLUNK_HEC_URL is not set, set 'SPLUNK_HEC_URL=<>")

            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            self.SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN")
            self.hec_url = os.getenv("SPLUNK_HEC_URL")

            ###################################
            # OPTIONAL - only used for testing
            splunk_base_url: Optional[str] = (
                os.getenv("_SPLUNK_BASE_URL")
                or os.getenv("SPLUNK_BASE_URL")
                or os.getenv("SPLUNK_HEC_URL")
            )
            if splunk_base_url is not None:
                self.hec_url = f"{splunk_base_url}/services/collector/event"
            ###################################
            self.sync_client = _get_httpx_client()
            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            super().__init__(
                **kwargs,
                flush_lock=self.flush_lock,
            )
        except Exception as e:
            verbose_logger.exception(
                f"Splunk: Got exception on init Splunk client {str(e)}"
            )
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log events to Splunk

        - Creates a Splunk payload
        - Adds the Payload to the in-memory logs queue
        - Payload is flushed every 5 seconds or when batch size is greater than max size defined in CustomBatchLogger

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            verbose_logger.debug(
                "Splunk: Logging - Enters logging function for model %s", kwargs
            )
            await self._log_async_event(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )

        except Exception as e:
            verbose_logger.exception(
                f"Splunk Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log events to Splunk

        - Creates a Splunk payload
        - Adds the Payload to the in-memory logs queue
        - Payload is flushed every 5 seconds or when batch size is greater than max size defined in CustomBatchLogger

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            verbose_logger.debug(
                "Splunk: Logging - Enters logging function for model %s", kwargs
            )
            await self._log_async_event(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            verbose_logger.exception(
                f"Splunk Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    async def async_send_batch(self):
        """
        Sends the in-memory logs queue to Splunk HEC

        Logs sent to /services/collector/event

        Splunk HEC Ref: https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector
        """
        try:
            if not self.log_queue:
                verbose_logger.exception("Splunk: log_queue does not exist")
                return

            verbose_logger.debug(
                "Splunk - about to flush %s events to %s",
                len(self.log_queue),
                self.hec_url,
            )

            response = await self.async_send_data(self.log_queue)
            if response.status_code >= 400:
                verbose_logger.exception(
                    f"Splunk: Error sending batch API - status_code: {response.status_code}, text: {response.text}"
                )
                return

            verbose_logger.debug(
                "Splunk: Response from Splunk HEC status_code: %s, text: %s",
                response.status_code,
                response.text,
            )
        except Exception as e:
            verbose_logger.exception(
                f"Splunk Error sending batch API - {str(e)}\n{traceback.format_exc()}"
            )

    async def _log_async_event(self, kwargs, response_obj, start_time, end_time):

        splunk_payload = self.create_splunk_logging_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

        self.log_queue.append(splunk_payload)
        verbose_logger.debug(
            f"Splunk, event added to queue. Will flush in {self.flush_interval} seconds..."
        )

        if len(self.log_queue) >= self.batch_size:
            await self.async_send_batch()

    def create_splunk_logging_payload(
        self,
        kwargs: Union[dict, Any],
        response_obj: Any,
        start_time: Any,  # datetime.datetime or float
        end_time: Any,  # datetime.datetime or float
    ) -> SplunkPayload:
        """
        Helper function to create a Splunk payload for logging

        Args:
            kwargs (Union[dict, Any]): request kwargs
            response_obj (Any): llm api response
            start_time (Any): start time of request
            end_time (Any): end time of request

        Returns:
            SplunkPayload: defined in types.py
        """

        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_object is None:
            raise ValueError("standard_logging_object not found in kwargs")

        status = "INFO"
        if standard_logging_object.get("status") == "failure":
            status = "ERROR"

        # Build the initial payload
        self.truncate_standard_logging_payload_content(standard_logging_object)

        splunk_payload = SplunkPayload(
            event=SplunkEvent(
                **self._create_splunk_event(
                    standard_logging_object=standard_logging_object,
                    status=status,
                )
            )
        )
        return splunk_payload

    def _create_splunk_event(
        self,
        standard_logging_object: StandardLoggingPayload,
        status: str,
    ) -> dict:
        """
        Create the Splunk event dictionary

        Args:
            standard_logging_object (StandardLoggingPayload): standard logging payload
            status (str): log status

        Returns:
            dict: Splunk event
        """
        event = {
            "source": self._get_splunk_source(),
            "sourcetype": self._get_splunk_sourcetype(),
            "host": self._get_splunk_host(),
            "event": {
                "status": status,
                **standard_logging_object,
            },
            "index": self._get_splunk_index(),
            "tags": self._get_splunk_tags(
                standard_logging_object=standard_logging_object
            ),
        }
        verbose_logger.debug("Splunk: Logger - Logging event = %s", json.dumps(event))
        return event

    @staticmethod
    def _get_splunk_tags(
        standard_logging_object: Optional[StandardLoggingPayload] = None,
    ) -> List[str]:
        """
        Get the Splunk tags for the request

        Splunk tags can be added as a list of strings
        """
        base_tags = {
            "env": os.getenv("SPLUNK_ENV", "unknown"),
            "service": os.getenv("SPLUNK_SERVICE", "litellm"),
            "version": os.getenv("SPLUNK_VERSION", "unknown"),
            "host": SplunkLogger._get_splunk_host(),
            "pod_name": os.getenv("POD_NAME", "unknown"),
        }

        tags = [f"{k}:{v}" for k, v in base_tags.items()]

        if standard_logging_object:
            _request_tags: List[str] = (
                standard_logging_object.get("request_tags", []) or []
            )
            request_tags = [f"request_tag:{tag}" for tag in _request_tags]
            tags.extend(request_tags)

        return tags

    @staticmethod
    def _get_splunk_source():
        return os.getenv("SPLUNK_SOURCE", "litellm")

    @staticmethod
    def _get_splunk_sourcetype():
        return os.getenv("SPLUNK_SOURCETYPE", "_json")

    @staticmethod
    def _get_splunk_host():
        return os.getenv("HOSTNAME", "")

    @staticmethod
    def _get_splunk_index():
        return os.getenv("SPLUNK_INDEX", "main")

    async def async_send_data(self, data: List) -> Response:
        """
        Async helper to send data to Splunk HEC

        Splunk recommends using JSON format for events
        https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector
        """
        if self.hec_url is None:
            raise ValueError("Splunk HEC URL is not set")
        response = await self.async_client.post(
            url=self.hec_url,
            json={"event": data},  # type: ignore
            headers={
                "Authorization": f"Splunk {self.SPLUNK_HEC_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        return response
