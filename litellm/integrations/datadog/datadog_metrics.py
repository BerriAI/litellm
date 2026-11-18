import asyncio
import gzip
import os
import time
from datetime import datetime
from typing import Final

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.datadog.datadog_handler import (
    get_datadog_env,
    get_datadog_hostname,
    get_datadog_pod_name,
    get_datadog_service,
    normalize_datadog_tag_value,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
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
    def __init__(self, start_periodic_flush: bool = True, **kwargs):
        self.dd_api_key = os.getenv("DD_API_KEY")
        self.dd_app_key = os.getenv("DD_APP_KEY")
        self.dd_site = os.getenv("DD_SITE", "datadoghq.com")

        if not self.dd_api_key:
            verbose_logger.warning("Datadog Metrics: DD_API_KEY is required. Integration will not work.")

        self.upload_url = f"https://api.{self.dd_site}/api/v2/series"

        self.async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

        # Initialize lock
        self.flush_lock = asyncio.Lock()

        # Only set flush_lock if not already provided by caller
        if "flush_lock" not in kwargs:
            kwargs["flush_lock"] = self.flush_lock

        # Send metrics more quickly to datadog (every 5 seconds)
        if "flush_interval" not in kwargs:
            kwargs["flush_interval"] = 5

        super().__init__(**kwargs)

        # Start periodic flush task only if instructed
        if start_periodic_flush:
            asyncio.create_task(self.periodic_flush())

    def _extract_tags(
        self,
        log: StandardLoggingPayload,
        status_code: str | int | None = None,
    ) -> list[str]:
        """
        Builds the list of tags for a Datadog metric point
        """
        # Base tags
        tags: Final = [
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
        metadata: Final = log.get("metadata", {}) or {}
        team_tag: Final = (
            metadata.get("user_api_key_team_alias")
            or metadata.get("team_alias")
            or metadata.get("user_api_key_team_id")
            or metadata.get("team_id")
        )

        if team_tag:
            tags.append(f"team:{normalize_datadog_tag_value(team_tag)}")

        return tags

    def _add_metrics_from_log(
        self,
        log: StandardLoggingPayload,
        kwargs: dict,
        status_code: str | int = "200",
    ):
        """
        Extracts latencies and appends Datadog metric series to the queue
        """
        tags: Final = self._extract_tags(log, status_code=status_code)

        # We record metrics with the end_time as the timestamp for the point
        end_time_dt: Final = kwargs.get("end_time") or datetime.now()
        timestamp: Final = int(end_time_dt.timestamp())

        # 1. Total Request Latency Metric (End to End)
        start_time_dt: Final = kwargs.get("start_time")
        if start_time_dt and end_time_dt:
            total_duration: Final = (end_time_dt - start_time_dt).total_seconds()
            series_total_latency: Final[DatadogMetricSeries] = {
                "metric": "litellm.request.total_latency",
                "type": 3,  # gauge
                "points": [{"timestamp": timestamp, "value": total_duration}],
                "tags": tags,
            }
            self.log_queue.append(series_total_latency)

        # 2. LLM API Latency Metric (Provider alone)
        api_call_start_time: Final = kwargs.get("api_call_start_time")
        if api_call_start_time and end_time_dt:
            llm_api_duration: Final = (end_time_dt - api_call_start_time).total_seconds()
            series_llm_latency: Final[DatadogMetricSeries] = {
                "metric": "litellm.llm_api.latency",
                "type": 3,  # gauge
                "points": [{"timestamp": timestamp, "value": llm_api_duration}],
                "tags": tags,
            }
            self.log_queue.append(series_llm_latency)

        # 3. LiteLLM Overhead Latency Metric (total - llm_api time)
        hidden_params: Final = log.get("hidden_params", {}) or {}
        litellm_overhead_time_ms: Final = hidden_params.get("litellm_overhead_time_ms")
        if litellm_overhead_time_ms is not None:
            overhead_tags: Final = self._extract_tags(log)  # no status_code on latency metric
            series_overhead: Final[DatadogMetricSeries] = {
                "metric": "litellm.overhead.latency",
                "type": 3,  # gauge
                "points": [
                    {
                        "timestamp": timestamp,
                        "value": litellm_overhead_time_ms / 1000,  # convert ms → seconds
                    }
                ],
                "tags": overhead_tags,
            }
            self.log_queue.append(series_overhead)

        # 4. Request Count / Status Code
        series_count: Final[DatadogMetricSeries] = {
            "metric": "litellm.llm_api.request_count",
            "type": 1,  # count
            "points": [{"timestamp": timestamp, "value": 1.0}],
            "tags": tags,
            "interval": self.flush_interval,
        }
        self.log_queue.append(series_count)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)

            if standard_logging_object is None:
                return

            self._add_metrics_from_log(log=standard_logging_object, kwargs=kwargs, status_code="200")

            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()

        except Exception as e:
            verbose_logger.exception("Datadog Metrics: Error in async_log_success_event: %s", e)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)

            if standard_logging_object is None:
                return

            # Extract status code from error information
            status_code = "500"  # default
            error_information: Final = standard_logging_object.get("error_information", {}) or {}
            error_code: Final = error_information.get("error_code")
            if error_code is not None:
                status_code = str(error_code)

            self._add_metrics_from_log(log=standard_logging_object, kwargs=kwargs, status_code=status_code)

            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()

        except Exception as e:
            verbose_logger.exception("Datadog Metrics: Error in async_log_failure_event: %s", e)

    async def async_send_batch(self):
        if not self.log_queue:
            return

        batch: Final = self.log_queue.copy()
        payload_data: Final[DatadogMetricsPayload] = {"series": batch}

        try:
            await self._upload_to_datadog(payload_data)
        except Exception as e:
            verbose_logger.exception("Datadog Metrics: Error in async_send_batch: %s", e)
            raise

    async def _upload_to_datadog(self, payload: DatadogMetricsPayload):
        if not self.dd_api_key:
            return

        headers: Final = {
            "Content-Type": "application/json",
            "DD-API-KEY": self.dd_api_key,
        }

        if self.dd_app_key:
            headers["DD-APPLICATION-KEY"] = self.dd_app_key

        json_data: Final = safe_dumps(payload)
        compressed_data: Final = gzip.compress(json_data.encode("utf-8"))
        headers["Content-Encoding"] = "gzip"

        response: Final = await self.async_client.post(
            self.upload_url,
            content=compressed_data,
            headers=headers,
        )

        response.raise_for_status()

        verbose_logger.debug(
            "Datadog Metrics: Uploaded %s metric points. Status: %s", len(payload["series"]), response.status_code
        )

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """
        Check if the service is healthy
        """
        try:
            # Send a test metric point to Datadog
            test_metric_point: Final[DatadogMetricPoint] = {
                "timestamp": int(time.time()),
                "value": 1.0,
            }
            test_metric_series: Final[DatadogMetricSeries] = {
                "metric": "litellm.health_check",
                "type": 3,  # Gauge
                "points": [test_metric_point],
                "tags": ["env:health_check"],
            }

            payload_data: Final[DatadogMetricsPayload] = {"series": [test_metric_series]}

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
        start_time_utc: datetime | None,
        end_time_utc: datetime | None,
    ) -> dict | None:
        pass
