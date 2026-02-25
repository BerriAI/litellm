import asyncio
import os
import time
from datetime import datetime
from typing import List, Optional, Union

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus
from litellm.types.integrations.datadog_metrics import (
    DatadogMetricPoint,
    DatadogMetricSeries,
    DatadogMetricsPayload,
)
from litellm.types.utils import StandardLoggingPayload


class DatadogMetricsLogger(CustomBatchLogger):
    def __init__(self, **kwargs):
        self.dd_api_key = os.getenv("DD_API_KEY")
        self.dd_app_key = os.getenv("DD_APP_KEY")
        self.dd_site = os.getenv("DD_SITE", "datadoghq.com")

        if not self.dd_api_key:
            verbose_logger.warning(
                "Datadog Metrics: DD_API_KEY is required. Integration will not work."
            )

        self.upload_url = f"https://api.{self.dd_site}/api/v2/series"

        self.async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        # Initialize lock
        self.flush_lock = asyncio.Lock()

        # Check if flush_lock is already in kwargs to avoid double passing
        kwargs["flush_lock"] = self.flush_lock

        # Send metrics more quickly to datadog (every 5 seconds)
        if "flush_interval" not in kwargs:
            kwargs["flush_interval"] = 5

        super().__init__(**kwargs)

        # Start periodic flush task
        asyncio.create_task(self.periodic_flush())

    def _extract_tags(
        self,
        log: StandardLoggingPayload,
        status_code: Optional[Union[str, int]] = None,
    ) -> List[str]:
        """
        Builds the list of tags for a Datadog metric point
        """
        from litellm.integrations.datadog.datadog_handler import (
            get_datadog_env,
            get_datadog_hostname,
            get_datadog_pod_name,
            get_datadog_service,
        )

        # Base tags
        tags = [
            f"env:{get_datadog_env()}",
            f"service:{get_datadog_service()}",
            f"version:{os.getenv('DD_VERSION', 'unknown')}",
            f"HOSTNAME:{get_datadog_hostname()}",
            f"POD_NAME:{get_datadog_pod_name()}",
        ]

        # Add metric-specific tags
        if provider := log.get("custom_llm_provider"):
            tags.append(f"provider:{provider}")

        if model := log.get("model"):
            tags.append(f"model_name:{model}")

        if model_group := log.get("model_group"):
            tags.append(f"model_group:{model_group}")

        if status_code is not None:
            tags.append(f"status_code:{status_code}")

        # Extract team tag
        metadata = log.get("metadata", {}) or {}
        team_tag = (
            metadata.get("user_api_key_team_alias")
            or metadata.get("team_alias")  # type: ignore
            or metadata.get("user_api_key_team_id")
            or metadata.get("team_id")  # type: ignore
        )

        if team_tag:
            tags.append(f"team:{team_tag}")

        return tags

    def _add_metrics_from_log(
        self,
        log: StandardLoggingPayload,
        kwargs: dict,
        status_code: Union[str, int] = "200",
    ):
        """
        Extracts latencies and appends Datadog metric series to the queue
        """
        tags = self._extract_tags(log, status_code=status_code)

        # We record metrics with the end_time as the timestamp for the point
        end_time_dt = kwargs.get("end_time") or datetime.now()
        timestamp = int(end_time_dt.timestamp())

        # 1. Total Request Latency Metric (End to End)
        start_time_dt = kwargs.get("start_time")
        if start_time_dt and end_time_dt:
            total_duration = (end_time_dt - start_time_dt).total_seconds()
            series_total_latency: DatadogMetricSeries = {
                "metric": "litellm.request.total_latency",
                "type": 3,  # gauge
                "points": [{"timestamp": timestamp, "value": total_duration}],
                "tags": tags,
            }
            self.log_queue.append(series_total_latency)

        # 2. LLM API Latency Metric (Provider alone)
        api_call_start_time = kwargs.get("api_call_start_time")
        if api_call_start_time and end_time_dt:
            llm_api_duration = (end_time_dt - api_call_start_time).total_seconds()
            series_llm_latency: DatadogMetricSeries = {
                "metric": "litellm.llm_api.latency",
                "type": 3,  # gauge
                "points": [{"timestamp": timestamp, "value": llm_api_duration}],
                "tags": tags,
            }
            self.log_queue.append(series_llm_latency)

        # 3. Request Count / Status Code
        series_count: DatadogMetricSeries = {
            "metric": "litellm.llm_api.request_count",
            "type": 1,  # count
            "points": [{"timestamp": timestamp, "value": 1.0}],
            "tags": tags,
        }
        self.log_queue.append(series_count)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if standard_logging_object is None:
                return

            self._add_metrics_from_log(
                log=standard_logging_object, kwargs=kwargs, status_code="200"
            )

            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()

        except Exception as e:
            verbose_logger.exception(
                f"Datadog Metrics: Error in async_log_success_event: {str(e)}"
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if standard_logging_object is None:
                return

            # Extract status code from error information
            status_code = "500"  # default
            error_information = (
                standard_logging_object.get("error_information", {}) or {}
            )
            if "error_code" in error_information and error_information["error_code"] is not None:  # type: ignore
                status_code = str(error_information["error_code"])  # type: ignore

            self._add_metrics_from_log(
                log=standard_logging_object, kwargs=kwargs, status_code=status_code
            )

            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()

        except Exception as e:
            verbose_logger.exception(
                f"Datadog Metrics: Error in async_log_failure_event: {str(e)}"
            )

    async def async_send_batch(self):
        if not self.log_queue:
            return

        try:
            # We must only send the current batch, so copy and clear log queue
            batch = self.log_queue.copy()
            # Note: CustomBatchLogger clears queue in flush_queue, but we'll manually copy what we need

            payload_data: DatadogMetricsPayload = {"series": batch}

            await self._upload_to_datadog(payload_data)

        except Exception as e:
            verbose_logger.exception(
                f"Datadog Metrics: Error in async_send_batch: {str(e)}"
            )

    async def _upload_to_datadog(self, payload: DatadogMetricsPayload):
        if not self.dd_api_key:
            return

        headers = {
            "Content-Type": "application/json",
            "DD-API-KEY": self.dd_api_key,
        }

        if self.dd_app_key:
            headers["DD-APPLICATION-KEY"] = self.dd_app_key

        import gzip

        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        json_data = safe_dumps(payload)
        compressed_data = gzip.compress(json_data.encode("utf-8"))
        headers["Content-Encoding"] = "gzip"

        response = await self.async_client.post(
            self.upload_url, content=compressed_data, headers=headers  # type: ignore
        )

        response.raise_for_status()

        verbose_logger.debug(
            f"Datadog Metrics: Uploaded {len(payload['series'])} metric points. Status: {response.status_code}"
        )

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """
        Check if the service is healthy
        """
        try:
            # Send a test metric point to Datadog
            test_metric_point: DatadogMetricPoint = {
                "timestamp": int(time.time()),
                "value": 1.0,
            }
            test_metric_series: DatadogMetricSeries = {
                "metric": "litellm.health_check",
                "type": 3,  # Gauge
                "points": [test_metric_point],
                "tags": ["env:health_check"],
            }

            payload_data: DatadogMetricsPayload = {"series": [test_metric_series]}

            await self._upload_to_datadog(payload_data)

            return IntegrationHealthCheckStatus(
                status="healthy",
                error_message=None,
            )
        except Exception as e:
            return IntegrationHealthCheckStatus(
                status="unhealthy",
                error_message=str(e),
            )

    async def get_request_response_payload(
        self,
        request_id: str,
        start_time_utc: Optional[datetime],
        end_time_utc: Optional[datetime],
    ) -> Optional[dict]:
        pass
