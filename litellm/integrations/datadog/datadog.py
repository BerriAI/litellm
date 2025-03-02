"""
DataDog Integration - sends logs to /api/v2/log

DD Reference API: https://docs.datadoghq.com/api/latest/logs

`async_log_success_event` - used by litellm proxy to send logs to datadog
`log_success_event` - sync version of logging to DataDog, only used on litellm Python SDK, if user opts in to using sync functions

async_log_success_event:  will store batch of DD_MAX_BATCH_SIZE in memory and flush to Datadog once it reaches DD_MAX_BATCH_SIZE or every 5 seconds

async_service_failure_hook: Logs failures from Redis, Postgres (Adjacent systems), as 'WARNING' on DataDog

For batching specific details see CustomBatchLogger class
"""

import asyncio
import datetime
import json
import os
import traceback
import uuid
from datetime import datetime as datetimeObj
from typing import Any, List, Optional, Union

import httpx
from httpx import Response

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus
from litellm.types.integrations.datadog import *
from litellm.types.services import ServiceLoggerPayload, ServiceTypes
from litellm.types.utils import StandardLoggingPayload

from ..additional_logging_utils import AdditionalLoggingUtils

# max number of logs DD API can accept
DD_MAX_BATCH_SIZE = 1000

# specify what ServiceTypes are logged as success events to DD. (We don't want to spam DD traces with large number of service types)
DD_LOGGED_SUCCESS_SERVICE_TYPES = [
    ServiceTypes.RESET_BUDGET_JOB,
]


class DataDogLogger(
    CustomBatchLogger,
    AdditionalLoggingUtils,
):
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
            verbose_logger.debug("Datadog: in init datadog logger")
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

            ###################################
            # OPTIONAL -only used for testing
            dd_base_url: Optional[str] = (
                os.getenv("_DATADOG_BASE_URL")
                or os.getenv("DATADOG_BASE_URL")
                or os.getenv("DD_BASE_URL")
            )
            if dd_base_url is not None:
                self.intake_url = f"{dd_base_url}/api/v2/logs"
            ###################################
            self.sync_client = _get_httpx_client()
            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            super().__init__(
                **kwargs, flush_lock=self.flush_lock, batch_size=DD_MAX_BATCH_SIZE
            )
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
            await self._log_async_event(kwargs, response_obj, start_time, end_time)

        except Exception as e:
            verbose_logger.exception(
                f"Datadog Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "Datadog: Logging - Enters logging function for model %s", kwargs
            )
            await self._log_async_event(kwargs, response_obj, start_time, end_time)

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

            verbose_logger.debug(
                "Datadog - about to flush %s events on %s",
                len(self.log_queue),
                self.intake_url,
            )

            response = await self.async_send_compressed_data(self.log_queue)
            if response.status_code == 413:
                verbose_logger.exception(DD_ERRORS.DATADOG_413_ERROR.value)
                return

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
            if litellm.datadog_use_v1 is True:
                dd_payload = self._create_v0_logging_payload(
                    kwargs=kwargs,
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                dd_payload = self.create_datadog_logging_payload(
                    kwargs=kwargs,
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                )

            response = self.sync_client.post(
                url=self.intake_url,
                json=dd_payload,  # type: ignore
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

    async def _log_async_event(self, kwargs, response_obj, start_time, end_time):

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

    def _create_datadog_logging_payload_helper(
        self,
        standard_logging_object: StandardLoggingPayload,
        status: DataDogStatus,
    ) -> DatadogPayload:
        json_payload = json.dumps(standard_logging_object, default=str)
        verbose_logger.debug("Datadog: Logger - Logging payload = %s", json_payload)
        dd_payload = DatadogPayload(
            ddsource=self._get_datadog_source(),
            ddtags=self._get_datadog_tags(
                standard_logging_object=standard_logging_object
            ),
            hostname=self._get_datadog_hostname(),
            message=json_payload,
            service=self._get_datadog_service(),
            status=status,
        )
        return dd_payload

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

        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_object is None:
            raise ValueError("standard_logging_object not found in kwargs")

        status = DataDogStatus.INFO
        if standard_logging_object.get("status") == "failure":
            status = DataDogStatus.ERROR

        # Build the initial payload
        self.truncate_standard_logging_payload_content(standard_logging_object)

        dd_payload = self._create_datadog_logging_payload_helper(
            standard_logging_object=standard_logging_object,
            status=status,
        )
        return dd_payload

    async def async_send_compressed_data(self, data: List) -> Response:
        """
        Async helper to send compressed data to datadog self.intake_url

        Datadog recommends using gzip to compress data
        https://docs.datadoghq.com/api/latest/logs/

        "Datadog recommends sending your logs compressed. Add the Content-Encoding: gzip header to the request when sending"
        """

        import gzip
        import json

        compressed_data = gzip.compress(json.dumps(data, default=str).encode("utf-8"))
        response = await self.async_client.post(
            url=self.intake_url,
            data=compressed_data,  # type: ignore
            headers={
                "DD-API-KEY": self.DD_API_KEY,
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
        )
        return response

    async def async_service_failure_hook(
        self,
        payload: ServiceLoggerPayload,
        error: Optional[str] = "",
        parent_otel_span: Optional[Any] = None,
        start_time: Optional[Union[datetimeObj, float]] = None,
        end_time: Optional[Union[float, datetimeObj]] = None,
        event_metadata: Optional[dict] = None,
    ):
        """
        Logs failures from Redis, Postgres (Adjacent systems), as 'WARNING' on DataDog

        - example - Redis is failing / erroring, will be logged on DataDog
        """
        try:
            _payload_dict = payload.model_dump()
            _payload_dict.update(event_metadata or {})
            _dd_message_str = json.dumps(_payload_dict, default=str)
            _dd_payload = DatadogPayload(
                ddsource=self._get_datadog_source(),
                ddtags=self._get_datadog_tags(),
                hostname=self._get_datadog_hostname(),
                message=_dd_message_str,
                service=self._get_datadog_service(),
                status=DataDogStatus.WARN,
            )

            self.log_queue.append(_dd_payload)

        except Exception as e:
            verbose_logger.exception(
                f"Datadog: Logger - Exception in async_service_failure_hook: {e}"
            )
        pass

    async def async_service_success_hook(
        self,
        payload: ServiceLoggerPayload,
        error: Optional[str] = "",
        parent_otel_span: Optional[Any] = None,
        start_time: Optional[Union[datetimeObj, float]] = None,
        end_time: Optional[Union[float, datetimeObj]] = None,
        event_metadata: Optional[dict] = None,
    ):
        """
        Logs success from Redis, Postgres (Adjacent systems), as 'INFO' on DataDog

        No user has asked for this so far, this might be spammy on datatdog. If need arises we can implement this
        """
        try:
            # intentionally done. Don't want to log all service types to DD
            if payload.service not in DD_LOGGED_SUCCESS_SERVICE_TYPES:
                return

            _payload_dict = payload.model_dump()
            _payload_dict.update(event_metadata or {})

            _dd_message_str = json.dumps(_payload_dict, default=str)
            _dd_payload = DatadogPayload(
                ddsource=self._get_datadog_source(),
                ddtags=self._get_datadog_tags(),
                hostname=self._get_datadog_hostname(),
                message=_dd_message_str,
                service=self._get_datadog_service(),
                status=DataDogStatus.INFO,
            )

            self.log_queue.append(_dd_payload)

        except Exception as e:
            verbose_logger.exception(
                f"Datadog: Logger - Exception in async_service_failure_hook: {e}"
            )

    def _create_v0_logging_payload(
        self,
        kwargs: Union[dict, Any],
        response_obj: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> DatadogPayload:
        """
        Note: This is our V1 Version of DataDog Logging Payload


        (Not Recommended) If you want this to get logged set `litellm.datadog_use_v1 = True`
        """
        import json

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
        except Exception:
            response_time = None

        try:
            response_obj = dict(response_obj)
        except Exception:
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

        json_payload = json.dumps(payload, default=str)

        verbose_logger.debug("Datadog: Logger - Logging payload = %s", json_payload)

        dd_payload = DatadogPayload(
            ddsource=self._get_datadog_source(),
            ddtags=self._get_datadog_tags(),
            hostname=self._get_datadog_hostname(),
            message=json_payload,
            service=self._get_datadog_service(),
            status=DataDogStatus.INFO,
        )
        return dd_payload

    @staticmethod
    def _get_datadog_tags(
        standard_logging_object: Optional[StandardLoggingPayload] = None,
    ) -> str:
        """
        Get the datadog tags for the request

        DD tags need to be as follows:
            - tags: ["user_handle:dog@gmail.com", "app_version:1.0.0"]
        """
        base_tags = {
            "env": os.getenv("DD_ENV", "unknown"),
            "service": os.getenv("DD_SERVICE", "litellm"),
            "version": os.getenv("DD_VERSION", "unknown"),
            "HOSTNAME": DataDogLogger._get_datadog_hostname(),
            "POD_NAME": os.getenv("POD_NAME", "unknown"),
        }

        tags = [f"{k}:{v}" for k, v in base_tags.items()]

        if standard_logging_object:
            _request_tags: List[str] = (
                standard_logging_object.get("request_tags", []) or []
            )
            request_tags = [f"request_tag:{tag}" for tag in _request_tags]
            tags.extend(request_tags)

        return ",".join(tags)

    @staticmethod
    def _get_datadog_source():
        return os.getenv("DD_SOURCE", "litellm")

    @staticmethod
    def _get_datadog_service():
        return os.getenv("DD_SERVICE", "litellm-server")

    @staticmethod
    def _get_datadog_hostname():
        return os.getenv("HOSTNAME", "")

    @staticmethod
    def _get_datadog_env():
        return os.getenv("DD_ENV", "unknown")

    @staticmethod
    def _get_datadog_pod_name():
        return os.getenv("POD_NAME", "unknown")

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """
        Check if the service is healthy
        """
        from litellm.litellm_core_utils.litellm_logging import (
            create_dummy_standard_logging_payload,
        )

        standard_logging_object = create_dummy_standard_logging_payload()
        dd_payload = self._create_datadog_logging_payload_helper(
            standard_logging_object=standard_logging_object,
            status=DataDogStatus.INFO,
        )
        log_queue = [dd_payload]
        response = await self.async_send_compressed_data(log_queue)
        try:
            response.raise_for_status()
            return IntegrationHealthCheckStatus(
                status="healthy",
                error_message=None,
            )
        except httpx.HTTPStatusError as e:
            return IntegrationHealthCheckStatus(
                status="unhealthy",
                error_message=e.response.text,
            )
        except Exception as e:
            return IntegrationHealthCheckStatus(
                status="unhealthy",
                error_message=str(e),
            )

    async def get_request_response_payload(
        self,
        request_id: str,
        start_time_utc: Optional[datetimeObj],
        end_time_utc: Optional[datetimeObj],
    ) -> Optional[dict]:
        pass
