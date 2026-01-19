import asyncio
import json
import os
import time
from litellm._uuid import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import quote

from litellm._logging import verbose_logger
from litellm.integrations.additional_logging_utils import AdditionalLoggingUtils
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm.proxy._types import CommonProxyErrors
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus
from litellm.types.integrations.gcs_bucket import *
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
else:
    VertexBase = Any


class GCSBucketLogger(GCSBucketBase, AdditionalLoggingUtils):
    def __init__(self, bucket_name: Optional[str] = None) -> None:
        from litellm.proxy.proxy_server import premium_user

        super().__init__(bucket_name=bucket_name)

        # Init Batch logging settings
        self.batch_size = int(os.getenv("GCS_BATCH_SIZE", GCS_DEFAULT_BATCH_SIZE))
        self.flush_interval = int(
            os.getenv("GCS_FLUSH_INTERVAL", GCS_DEFAULT_FLUSH_INTERVAL_SECONDS)
        )
        self.flush_lock = asyncio.Lock()
        super().__init__(
            flush_lock=self.flush_lock,
            batch_size=self.batch_size,
            flush_interval=self.flush_interval,
        )
        # Override log_queue with bounded asyncio.Queue for thread-safe concurrent access
        # Bounded queue prevents OOM when GCS is slower than log accumulation
        # Default maxsize is 10x batch_size to allow buffering during slow periods
        queue_maxsize = int(os.getenv("GCS_QUEUE_MAXSIZE", str(self.batch_size * 10)))
        self.log_queue: asyncio.Queue[GCSLogQueueItem] = asyncio.Queue(maxsize=queue_maxsize)  # type: ignore[assignment]
        asyncio.create_task(self.periodic_flush())
        AdditionalLoggingUtils.__init__(self)

        print(f"GCS Bucket Logger initialized: bucket_name={bucket_name or 'from env'}, batch_size={self.batch_size}, flush_interval={self.flush_interval}s, queue_maxsize={queue_maxsize}")

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
            logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )
            if logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")
            # Add to logging queue - this will be flushed periodically
            # Use asyncio.Queue.put() for thread-safe concurrent access
            # If queue is full, this will block until space is available (backpressure)
            await self.log_queue.put(
                GCSLogQueueItem(
                    payload=logging_payload, kwargs=kwargs, response_obj=response_obj
                )
            )
            queue_size = self.log_queue.qsize()
            queue_maxsize = self.log_queue.maxsize if hasattr(self.log_queue, 'maxsize') else None
            # Warn if queue is getting full (>80% capacity)
            if queue_maxsize and queue_size > queue_maxsize * 0.8:
                print(f"GCS Bucket: Success event queued. Queue size: {queue_size}/{queue_maxsize} (WARNING: queue nearly full)")
                verbose_logger.warning(f"GCS Bucket queue is {queue_size}/{queue_maxsize} full, processing may be slow")
            else:
                print(f"GCS Bucket: Success event queued. Queue size: {queue_size}")

        except Exception as e:
            print(f"GCS Bucket: Error queueing success event: {str(e)}")
            verbose_logger.exception(f"GCS Bucket logging error: {str(e)}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "GCS Logger: async_log_failure_event logging kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )

            logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )
            if logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")
            # Add to logging queue - this will be flushed periodically
            # Use asyncio.Queue.put() for thread-safe concurrent access
            # If queue is full, this will block until space is available (backpressure)
            await self.log_queue.put(
                GCSLogQueueItem(
                    payload=logging_payload, kwargs=kwargs, response_obj=response_obj
                )
            )
            queue_size = self.log_queue.qsize()
            queue_maxsize = self.log_queue.maxsize if hasattr(self.log_queue, 'maxsize') else None
            # Warn if queue is getting full (>80% capacity)
            if queue_maxsize and queue_size > queue_maxsize * 0.8:
                print(f"GCS Bucket: Failure event queued. Queue size: {queue_size}/{queue_maxsize} (WARNING: queue nearly full)")
                verbose_logger.warning(f"GCS Bucket queue is {queue_size}/{queue_maxsize} full, processing may be slow")
            else:
                print(f"GCS Bucket: Failure event queued. Queue size: {queue_size}")

        except Exception as e:
            print(f"GCS Bucket: Error queueing failure event: {str(e)}")
            verbose_logger.exception(f"GCS Bucket logging error: {str(e)}")

    def _drain_queue_batch(self) -> List[GCSLogQueueItem]:
        """
        Drain items from the queue (non-blocking), respecting batch_size limit.
        
        This prevents unbounded queue growth when processing is slower than log accumulation.
        
        Returns:
            List of items to process, up to batch_size items
        """
        items_to_process = []
        while len(items_to_process) < self.batch_size:
            try:
                item = self.log_queue.get_nowait()
                items_to_process.append(item)
            except asyncio.QueueEmpty:
                break
        return items_to_process

    def _generate_batch_object_name(self, date_str: str, batch_id: str) -> str:
        """
        Generate object name for a batched log file.
        Format: {date}/batch-{batch_id}.ndjson
        """
        return f"{date_str}/batch-{batch_id}.ndjson"

    def _combine_payloads_to_ndjson(self, items: List[GCSLogQueueItem]) -> str:
        """
        Combine multiple log payloads into newline-delimited JSON (NDJSON) format.
        Each line is a valid JSON object representing one log entry.
        """
        lines = []
        for item in items:
            logging_payload = item["payload"]
            # Serialize each payload as a JSON line
            json_line = json.dumps(logging_payload, default=str, ensure_ascii=False)
            lines.append(json_line)
        return "\n".join(lines)

    async def async_send_batch(self):
        """
        Process queued logs in batch - sends logs to GCS Bucket as a single batched object

        This implementation batches multiple log payloads into a single GCS object upload,
        dramatically reducing API calls and improving throughput.

        Strategy:
            - collect the logs to flush every `GCS_FLUSH_INTERVAL` seconds
            - combine all payloads in a batch into newline-delimited JSON (NDJSON)
            - make 1 POST request per batch (instead of 1 per log)
            - process up to `batch_size` items per flush to prevent unbounded queue growth

        Uses asyncio.Queue for thread-safe concurrent access. Processes up to `batch_size`
        items per flush, leaving remaining items for the next flush cycle.
        """
        items_to_process = self._drain_queue_batch()

        if not items_to_process:
            return

        print(f"GCS Bucket: Starting batch send. Processing {len(items_to_process)} items as a single batch")
        
        # Group items by GCS config (bucket, credentials) since they may differ
        # For now, we'll use the first item's config for the entire batch
        # (most common case is all items use the same config)
        first_item = items_to_process[0]
        first_kwargs = first_item["kwargs"]
        
        try:
            gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
                first_kwargs
            )

            headers = await self.construct_request_headers(
                vertex_instance=gcs_logging_config["vertex_instance"],
                service_account_json=gcs_logging_config["path_service_account"],
            )
            bucket_name = gcs_logging_config["bucket_name"]
            
            # Generate batch object name with timestamp and unique ID
            current_date = self._get_object_date_from_datetime(datetime.now(timezone.utc))
            batch_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            object_name = self._generate_batch_object_name(current_date, batch_id)
            
            # Combine all payloads into NDJSON format
            combined_payload = self._combine_payloads_to_ndjson(items_to_process)
            
            # Upload single batched object
            await self._log_json_data_on_gcs(
                headers=headers,
                bucket_name=bucket_name,
                object_name=object_name,
                logging_payload=combined_payload,  # Pass as string (NDJSON)
            )
            
            success_count = len(items_to_process)
            error_count = 0
            print(f"GCS Bucket: Successfully sent batch of {success_count} logs to bucket '{bucket_name}', object '{object_name}'")
            
        except Exception as e:
            # If batch upload fails, count all items as errors
            # In the future, we could implement retry or fallback to individual uploads
            success_count = 0
            error_count = len(items_to_process)
            print(f"GCS Bucket: Error sending batch to GCS bucket: {str(e)}")
            verbose_logger.exception(
                f"GCS Bucket error logging batch payload to GCS bucket: {str(e)}"
            )

        print(f"GCS Bucket: Batch send completed. Success: {success_count}, Errors: {error_count}, Remaining queue size: {self.log_queue.qsize()}")

    def _get_object_name(
        self, kwargs: Dict, logging_payload: StandardLoggingPayload, response_obj: Any
    ) -> str:
        """
        Get the object name to use for the current payload
        """
        current_date = self._get_object_date_from_datetime(datetime.now(timezone.utc))
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
        _litellm_params = kwargs.get("litellm_params", None) or {}
        _metadata = _litellm_params.get("metadata", None) or {}
        if "gcs_log_id" in _metadata:
            object_name = _metadata["gcs_log_id"]

        return object_name

    async def get_request_response_payload(
        self,
        request_id: str,
        start_time_utc: Optional[datetime],
        end_time_utc: Optional[datetime],
    ) -> Optional[dict]:
        """
        Get the request and response payload for a given `request_id`
        Tries current day, next day, and previous day until it finds the payload
        """
        if start_time_utc is None:
            raise ValueError(
                "start_time_utc is required for getting a payload from GCS Bucket"
            )

        # Try current day, next day, and previous day
        dates_to_try = [
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
                encoded_object_name = quote(object_name, safe="")
                response = await self.download_gcs_object(encoded_object_name)

                if response is not None:
                    loaded_response = json.loads(response)
                    return loaded_response
            except Exception as e:
                verbose_logger.debug(
                    f"Failed to fetch payload for date {date_str}: {str(e)}"
                )
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

    async def flush_queue(self):
        """
        Override flush_queue to work with asyncio.Queue.
        
        No lock needed: asyncio.Queue.get_nowait() is atomic, and async_send_batch()
        drains the queue completely, so concurrent flushes just compete for items safely.
        No qsize() check needed: async_send_batch() handles empty queues gracefully.
        """
        await self.async_send_batch()
        # Note: async_send_batch() already drains the queue and handles empty case
        self.last_flush_time = time.time()

    async def periodic_flush(self):
        """
        Override periodic_flush to add queue size observability.
        Logs the GCS queue size before each flush operation.
        """
        while True:
            await asyncio.sleep(self.flush_interval)
            queue_size = self.log_queue.qsize()
            print(
                f"GCS Bucket queue status: {queue_size} logs queued, batch_size={self.batch_size}, flush_interval={self.flush_interval}s"
            )
            verbose_logger.debug(
                f"GCS Bucket periodic flush after {self.flush_interval} seconds"
            )
            await self.flush_queue()

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        raise NotImplementedError("GCS Bucket does not support health check")
