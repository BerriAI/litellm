"""
Custom Logger that handles batching logic

Use this if you want your logs to be stored in memory and flushed periodically.
"""

import asyncio
import time
from typing import List, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger


class CustomBatchLogger(CustomLogger):
    preserve_events_added_during_flush = False

    # Default cap on the in-memory log queue. Prevents unbounded memory growth
    # if ``async_send_batch`` consistently fails (e.g. the destination is
    # unreachable) and events are preserved across flush attempts. Subclasses
    # may override by passing ``max_queue_size`` or by setting the attribute
    # directly (see ``RubrikLogger`` for an example).
    DEFAULT_MAX_QUEUE_SIZE = 50_000

    def __init__(
        self,
        flush_lock: Optional[asyncio.Lock] = None,
        batch_size: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            flush_lock (Optional[asyncio.Lock], optional): Lock to use when flushing the queue. Defaults to None. Only used for custom loggers that do batching
            max_queue_size (Optional[int], optional): Maximum number of events to retain in ``log_queue``. When the limit is exceeded (e.g. because the send destination is unreachable and events are preserved for retry), the oldest events are dropped. Defaults to ``DEFAULT_MAX_QUEUE_SIZE``.
        """
        self.log_queue: List = []
        self.flush_interval = flush_interval or litellm.DEFAULT_FLUSH_INTERVAL_SECONDS
        self.batch_size: int = batch_size or litellm.DEFAULT_BATCH_SIZE
        self.last_flush_time = time.time()
        self.flush_lock = flush_lock
        self.max_queue_size: int = (
            max_queue_size
            if max_queue_size is not None
            else self.DEFAULT_MAX_QUEUE_SIZE
        )

        super().__init__(**kwargs)

    async def periodic_flush(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            verbose_logger.debug(
                f"CustomLogger periodic flush after {self.flush_interval} seconds"
            )
            await self.flush_queue()

    async def flush_queue(self):
        if self.flush_lock is None:
            return

        async with self.flush_lock:
            if self.log_queue:
                log_queue_length = len(self.log_queue)
                verbose_logger.debug(
                    "CustomLogger: Flushing batch of %s events", len(self.log_queue)
                )
                try:
                    await self.async_send_batch()
                except Exception:
                    # If the underlying batch send raised, do NOT drop the
                    # in-flight events. They will be retried on the next flush.
                    # Most existing async_send_batch implementations swallow
                    # their own errors, so this only affects loggers that opt
                    # in to surfacing failures (e.g. Rubrik).
                    verbose_logger.exception(
                        "CustomLogger: async_send_batch raised; preserving "
                        "%s events in queue for retry",
                        log_queue_length,
                    )
                    # Guard against unbounded queue growth if the destination
                    # is persistently unreachable. Drop the oldest events
                    # beyond ``max_queue_size``.
                    overflow = len(self.log_queue) - self.max_queue_size
                    if overflow > 0:
                        del self.log_queue[:overflow]
                        verbose_logger.warning(
                            "CustomLogger: log queue exceeded max_queue_size=%s; "
                            "dropped %s oldest events.",
                            self.max_queue_size,
                            overflow,
                        )
                    return
                if self.preserve_events_added_during_flush:
                    del self.log_queue[:log_queue_length]
                else:
                    self.log_queue.clear()
                self.last_flush_time = time.time()

    async def async_send_batch(self, *args, **kwargs):
        pass
