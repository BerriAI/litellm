"""@private"""

import atexit
import logging
import queue
from queue import Queue
from typing import List, Optional

from langfuse.api.client import FernLangfuse
from langfuse.request import LangfuseClient
from langfuse.types import MaskFunction
from langfuse.utils import _get_timestamp

from .ingestion_consumer import IngestionConsumer
from .media_manager import MediaManager
from .media_upload_consumer import MediaUploadConsumer


class TaskManager(object):
    _log = logging.getLogger(__name__)
    _ingestion_consumers: List[IngestionConsumer]
    _enabled: bool
    _threads: int
    _max_task_queue_size: int
    _ingestion_queue: Queue
    _media_upload_queue: Queue
    _client: LangfuseClient
    _api_client: FernLangfuse
    _flush_at: int
    _flush_interval: float
    _max_retries: int
    _public_key: str
    _sdk_name: str
    _sdk_version: str
    _sdk_integration: str
    _sample_rate: float
    _mask: Optional[MaskFunction]

    def __init__(
        self,
        *,
        client: LangfuseClient,
        api_client: FernLangfuse,
        flush_at: int,
        flush_interval: float,
        max_retries: int,
        threads: int,
        public_key: str,
        sdk_name: str,
        sdk_version: str,
        sdk_integration: str,
        enabled: bool = True,
        max_task_queue_size: int = 100_000,
        sample_rate: float = 1,
        mask: Optional[MaskFunction] = None,
    ):
        self._max_task_queue_size = max_task_queue_size
        self._threads = threads
        self._ingestion_queue = queue.Queue(self._max_task_queue_size)
        self._media_upload_queue = Queue(self._max_task_queue_size)
        self._media_manager = MediaManager(
            api_client=api_client,
            media_upload_queue=self._media_upload_queue,
            max_retries=max_retries,
        )
        self._ingestion_consumers = []
        self._media_upload_consumers = []
        self._client = client
        self._api_client = api_client
        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._public_key = public_key
        self._sdk_name = sdk_name
        self._sdk_version = sdk_version
        self._sdk_integration = sdk_integration
        self._enabled = enabled
        self._sample_rate = sample_rate
        self._mask = mask

        self.init_resources()

        # cleans up when the python interpreter closes
        atexit.register(self.shutdown)

    def init_resources(self):
        for i in range(self._threads):
            ingestion_consumer = IngestionConsumer(
                ingestion_queue=self._ingestion_queue,
                identifier=i,
                client=self._client,
                media_manager=self._media_manager,
                flush_at=self._flush_at,
                flush_interval=self._flush_interval,
                max_retries=self._max_retries,
                public_key=self._public_key,
                sdk_name=self._sdk_name,
                sdk_version=self._sdk_version,
                sdk_integration=self._sdk_integration,
                sample_rate=self._sample_rate,
                mask=self._mask,
            )
            ingestion_consumer.start()
            self._ingestion_consumers.append(ingestion_consumer)

        for i in range(self._threads):
            media_upload_consumer = MediaUploadConsumer(
                identifier=i,
                media_manager=self._media_manager,
            )
            media_upload_consumer.start()
            self._media_upload_consumers.append(media_upload_consumer)

    def add_task(self, event: dict):
        if not self._enabled:
            return

        try:
            event["timestamp"] = _get_timestamp()

            self._ingestion_queue.put(event, block=False)
        except queue.Full:
            self._log.warning("analytics-python queue is full")
            return False
        except Exception as e:
            self._log.exception(f"Exception in adding task {e}")

            return False

    def flush(self):
        """Force a flush from the internal queue to the server."""
        self._log.debug("flushing ingestion and media upload queues")

        # Ingestion queue
        ingestion_queue_size = self._ingestion_queue.qsize()
        self._ingestion_queue.join()
        self._log.debug(
            f"Successfully flushed ~{ingestion_queue_size} items from ingestion queue"
        )

        # Media upload queue
        media_upload_queue_size = self._media_upload_queue.qsize()
        self._media_upload_queue.join()
        self._log.debug(
            f"Successfully flushed ~{media_upload_queue_size} items from media upload queue"
        )

    def join(self):
        """End the consumer threads once the queue is empty.

        Blocks execution until finished
        """
        self._log.debug(
            f"joining {len(self._ingestion_consumers)} ingestion consumer threads"
        )

        # pause all consumers before joining them so we don't have to wait for multiple
        # flush intervals to join them all.
        for ingestion_consumer in self._ingestion_consumers:
            ingestion_consumer.pause()

        for ingestion_consumer in self._ingestion_consumers:
            try:
                ingestion_consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            self._log.debug(
                f"IngestionConsumer thread {ingestion_consumer._identifier} joined"
            )

        self._log.debug(
            f"joining {len(self._media_upload_consumers)} media upload consumer threads"
        )
        for media_upload_consumer in self._media_upload_consumers:
            media_upload_consumer.pause()

        for media_upload_consumer in self._media_upload_consumers:
            try:
                media_upload_consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            self._log.debug(
                f"MediaUploadConsumer thread {media_upload_consumer._identifier} joined"
            )

    def shutdown(self):
        """Flush all messages and cleanly shutdown the client."""
        self._log.debug("shutdown initiated")

        # Unregister the atexit handler first
        atexit.unregister(self.shutdown)

        self.flush()
        self.join()

        self._log.debug("shutdown completed")
