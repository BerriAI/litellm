import asyncio
import hashlib
import json
import os
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.constants import LITELLM_ASYNCIO_QUEUE_MAXSIZE
from litellm.integrations.additional_logging_utils import AdditionalLoggingUtils
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm.litellm_core_utils.cloud_storage_security import (
    sanitize_cloud_object_component,
)
from litellm.proxy._types import CommonProxyErrors
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus
from litellm.types.integrations.gcs_bucket import *
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
else:
    VertexBase = Any


class GCSBucketLogger(GCSBucketBase, AdditionalLoggingUtils):
    def __init__(self, bucket_name: str | None = None) -> None:
        from litellm.proxy.proxy_server import premium_user

        self.batch_size = int(os.getenv("GCS_BATCH_SIZE", GCS_DEFAULT_BATCH_SIZE))
        self.flush_interval = int(os.getenv("GCS_FLUSH_INTERVAL", GCS_DEFAULT_FLUSH_INTERVAL_SECONDS))
        self.use_batched_logging = (
            os.getenv("GCS_USE_BATCHED_LOGGING", str(GCS_DEFAULT_USE_BATCHED_LOGGING).lower()).lower() == "true"
        )
        self.flush_lock = asyncio.Lock()
        super().__init__(
            bucket_name=bucket_name,
            flush_lock=self.flush_lock,
            batch_size=self.batch_size,
            flush_interval=self.flush_interval,
        )
        self.log_queue: asyncio.Queue[GCSLogQueueItem] = asyncio.Queue(maxsize=LITELLM_ASYNCIO_QUEUE_MAXSIZE)
        asyncio.create_task(self.periodic_flush())
        AdditionalLoggingUtils.__init__(self)

        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )

    #### ASYNC ####
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )
        try:
            verbose_logger.debug(
                "GCS Logger: async_log_success_event logging kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )
            logging_payload: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)
            if logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")
            await self._enqueue(GCSLogQueueItem(payload=logging_payload, kwargs=kwargs, response_obj=response_obj))

        except Exception as e:
            verbose_logger.exception("GCS Bucket logging error: %s", e)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "GCS Logger: async_log_failure_event logging kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )

            logging_payload: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)
            if logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")
            await self._enqueue(GCSLogQueueItem(payload=logging_payload, kwargs=kwargs, response_obj=response_obj))

        except Exception as e:
            verbose_logger.exception("GCS Bucket logging error: %s", e)

    async def _enqueue(self, item: GCSLogQueueItem) -> None:
        if self.log_queue.full():
            await self.flush_queue()
        if self.log_queue.full():
            self.log_queue.get_nowait()
            verbose_logger.error("GCS Bucket log queue still full after flush, dropped the oldest queued event")
        self.log_queue.put_nowait(item)

    def _requeue(self, items: Sequence[GCSLogQueueItem]) -> None:
        dropped: Final = sum(1 for item in items if not self._put_nowait_or_drop(item))
        verbose_logger.error(
            "GCS Bucket upload failed for %s events, %s kept in queue for the next flush, %s dropped (queue full)",
            len(items),
            len(items) - dropped,
            dropped,
        )

    def _put_nowait_or_drop(self, item: GCSLogQueueItem) -> bool:
        try:
            self.log_queue.put_nowait(item)
        except asyncio.QueueFull:
            return False
        return True

    def _drain_queue_batch(self) -> list[GCSLogQueueItem]:
        """
        Drain items from the queue (non-blocking), respecting batch_size limit.

        This prevents unbounded queue growth when processing is slower than log accumulation.

        Returns:
            List of items to process, up to batch_size items
        """
        items_to_process: Final[list[GCSLogQueueItem]] = []
        while len(items_to_process) < self.batch_size:
            try:
                items_to_process.append(self.log_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items_to_process

    def _generate_batch_object_name(self, date_str: str, batch_id: str) -> str:
        """
        Generate object name for a batched log file.
        Format: {date}/batch-{batch_id}.ndjson
        """
        return f"{date_str}/batch-{batch_id}.ndjson"

    def _get_config_key(self, kwargs: dict[str, Any]) -> str:
        """
        Extract a synchronous grouping key from kwargs to group items by GCS config.
        This allows us to batch items with the same bucket/credentials together.

        Returns a string key that uniquely identifies the GCS config combination.
        This key may contain sensitive information (bucket names, paths) - use _sanitize_config_key()
        for logging purposes.
        """
        standard_callback_dynamic_params: Final = kwargs.get("standard_callback_dynamic_params", None) or {}

        bucket_name = standard_callback_dynamic_params.get("gcs_bucket_name", None) or self.BUCKET_NAME or "default"
        path_service_account: Final = (
            standard_callback_dynamic_params.get("gcs_path_service_account", None)
            or self.path_service_account_json
            or "default"
        )

        return f"{bucket_name}|{path_service_account}"

    def _sanitize_config_key(self, config_key: str) -> str:
        """
        Create a sanitized version of the config key for logging.
        Uses a hash to avoid exposing sensitive bucket names or service account paths.

        Returns a short hash prefix for safe logging.
        """
        hash_obj: Final = hashlib.sha256(config_key.encode("utf-8"))
        return f"config-{hash_obj.hexdigest()[:8]}"

    def _group_items_by_config(self, items: list[GCSLogQueueItem]) -> dict[str, list[GCSLogQueueItem]]:
        """
        Group items by their GCS config (bucket + credentials).
        This ensures items with different configs are processed separately.

        Returns a dict mapping config_key -> list of items with that config.
        """
        grouped: Final[dict[str, list[GCSLogQueueItem]]] = {}
        for item in items:
            config_key = self._get_config_key(item["kwargs"])
            if config_key not in grouped:
                grouped[config_key] = []
            grouped[config_key].append(item)
        return grouped

    def _combine_payloads_to_ndjson(self, items: list[GCSLogQueueItem]) -> str:
        """
        Combine multiple log payloads into newline-delimited JSON (NDJSON) format.
        Each line is a valid JSON object representing one log entry.
        """
        lines: Final = []
        for item in items:
            logging_payload = item["payload"]
            json_line = json.dumps(logging_payload, default=str, ensure_ascii=False)
            lines.append(json_line)
        return "\n".join(lines)

    async def _send_grouped_batch(self, items: list[GCSLogQueueItem], config_key: str) -> tuple[int, int]:
        """
        Send a batch of items that share the same GCS config.

        Returns:
            (success_count, error_count)
        """
        if not items:
            return (0, 0)

        first_kwargs: Final = items[0]["kwargs"]

        try:
            gcs_logging_config: Final[GCSLoggingConfig] = await self.get_gcs_logging_config(first_kwargs)

            headers: Final = await self.construct_request_headers(
                vertex_instance=gcs_logging_config["vertex_instance"],
                service_account_json=gcs_logging_config["path_service_account"],
            )
            bucket_name: Final = gcs_logging_config["bucket_name"]

            current_date: Final = self._get_object_date_from_datetime(datetime.now(timezone.utc))
            batch_id: Final = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            object_name: Final = self._generate_batch_object_name(current_date, batch_id)
            combined_payload: Final = self._combine_payloads_to_ndjson(items)

            await self._log_json_data_on_gcs(
                headers=headers,
                bucket_name=bucket_name,
                object_name=object_name,
                logging_payload=combined_payload,
            )

            success_count = len(items)
            error_count = 0
            return (success_count, error_count)

        except Exception as e:
            success_count = 0
            error_count = len(items)
            verbose_logger.exception("GCS Bucket error logging batch payload to GCS bucket: %s", e)
            return (success_count, error_count)

    async def _send_individual_logs(self, items: list[GCSLogQueueItem]) -> GCSFlushResult:
        """
        Send each log individually as separate GCS objects (legacy behavior).
        This is used when GCS_USE_BATCHED_LOGGING is disabled.
        """
        failed_items: Final = tuple([item for item in items if not await self._send_single_log_item(item)])
        if failed_items:
            self._requeue(failed_items)
        return GCSFlushResult(sent=len(items) - len(failed_items), failed=len(failed_items))

    async def _send_single_log_item(self, item: GCSLogQueueItem) -> bool:
        """
        Send a single log item to GCS as an individual object. Returns whether the upload succeeded.
        """
        try:
            gcs_logging_config: Final[GCSLoggingConfig] = await self.get_gcs_logging_config(item["kwargs"])

            headers: Final = await self.construct_request_headers(
                vertex_instance=gcs_logging_config["vertex_instance"],
                service_account_json=gcs_logging_config["path_service_account"],
            )
            bucket_name: Final = gcs_logging_config["bucket_name"]

            object_name: Final = self._get_object_name(
                kwargs=item["kwargs"],
                logging_payload=item["payload"],
                response_obj=item["response_obj"],
            )

            await self._log_json_data_on_gcs(
                headers=headers,
                bucket_name=bucket_name,
                object_name=object_name,
                logging_payload=item["payload"],
            )
        except Exception as e:
            verbose_logger.exception("GCS Bucket error logging individual payload to GCS bucket: %s", e)
            return False
        return True

    async def _send_grouped_batches(self, items: list[GCSLogQueueItem]) -> GCSFlushResult:
        results: Final = tuple(
            [
                (group_items, await self._send_grouped_batch(group_items, config_key))
                for config_key, group_items in self._group_items_by_config(items).items()
            ]
        )
        for group_items, (_, group_failed) in results:
            if group_failed:
                self._requeue(group_items)
        return GCSFlushResult(
            sent=sum(group_sent for _, (group_sent, _) in results),
            failed=sum(group_failed for _, (_, group_failed) in results),
        )

    async def async_send_batch(self) -> None:
        """
        Process queued logs - sends logs to GCS Bucket.

        If `GCS_USE_BATCHED_LOGGING` is enabled (default), batches multiple log payloads
        into single GCS object uploads (NDJSON format), dramatically reducing API calls.

        If disabled, sends each log individually as separate GCS objects (legacy behavior).
        """
        await self._send_queued_events()

    async def _send_queued_events(self) -> GCSFlushResult:
        items_to_process: Final = self._drain_queue_batch()

        if not items_to_process:
            return GCSFlushResult(sent=0, failed=0)

        if self.use_batched_logging:
            return await self._send_grouped_batches(items_to_process)
        return await self._send_individual_logs(items_to_process)

    def _get_object_name(self, kwargs: dict, logging_payload: StandardLoggingPayload, response_obj: Any) -> str:
        """
        Get the object name to use for the current payload
        """
        current_date: Final = self._get_object_date_from_datetime(datetime.now(timezone.utc))
        if logging_payload.get("error_str", None) is not None:
            object_name = self._generate_failure_object_name(
                request_date_str=current_date,
            )
        else:
            object_name = self._generate_success_object_name(
                request_date_str=current_date,
                response_id=response_obj.get("id", ""),
            )

        # used for testing
        _litellm_params: Final = kwargs.get("litellm_params", None) or {}
        _metadata: Final = _litellm_params.get("metadata", None) or {}
        if "gcs_log_id" in _metadata:
            safe_log_id: Final = sanitize_cloud_object_component(_metadata.get("gcs_log_id"), fallback="")
            if safe_log_id:
                object_name = f"{current_date}/custom-{uuid.uuid4().hex}-{safe_log_id}"

        return object_name

    async def get_request_response_payload(
        self,
        request_id: str,
        start_time_utc: datetime | None,
        end_time_utc: datetime | None,
    ) -> dict | None:
        """
        Get the request and response payload for a given `request_id`
        Tries current day, next day, and previous day until it finds the payload
        """
        if start_time_utc is None:
            raise ValueError("start_time_utc is required for getting a payload from GCS Bucket")

        dates_to_try: Final = [
            start_time_utc,
            start_time_utc + timedelta(days=1),
            start_time_utc - timedelta(days=1),
        ]
        date_str = None
        for date in dates_to_try:
            try:
                date_str = self._get_object_date_from_datetime(datetime_obj=date)
                object_name = self._generate_success_object_name(
                    request_date_str=date_str,
                    response_id=request_id,
                )
                response = await self.download_gcs_object(object_name)

                if response is not None:
                    loaded_response = json.loads(response)
                    return loaded_response
            except Exception as e:
                verbose_logger.debug("Failed to fetch payload for date %s: %s", date_str, e)
                continue

        return None

    def _generate_success_object_name(
        self,
        request_date_str: str,
        response_id: str,
    ) -> str:
        return f"{request_date_str}/{response_id}"

    def _generate_failure_object_name(
        self,
        request_date_str: str,
    ) -> str:
        return f"{request_date_str}/failure-{uuid.uuid4().hex}"

    def _get_object_date_from_datetime(self, datetime_obj: datetime) -> str:
        return datetime_obj.strftime("%Y-%m-%d")

    async def flush_queue(self) -> None:
        """
        Override flush_queue to work with asyncio.Queue.
        """
        await self.flush_queue_and_report()

    async def flush_queue_and_report(self) -> GCSFlushResult:
        result: Final = await self._send_queued_events()
        self.last_flush_time = time.time()
        return result

    async def periodic_flush(self):
        """
        Override periodic_flush to work with asyncio.Queue.
        """
        while True:
            await asyncio.sleep(self.flush_interval)
            verbose_logger.debug("GCS Bucket periodic flush after %s seconds", self.flush_interval)
            await self.flush_queue()

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        raise NotImplementedError("GCS Bucket does not support health check")
