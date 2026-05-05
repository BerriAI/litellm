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

    def __init__(
        self,
        flush_lock: Optional[asyncio.Lock] = None,
        batch_size: Optional[int] = None,
        flush_interval: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            flush_lock (Optional[asyncio.Lock], optional): Lock to use when flushing the queue. Defaults to None. Only used for custom loggers that do batching
        """
        self.log_queue: List = []
        self.flush_interval = flush_interval or litellm.DEFAULT_FLUSH_INTERVAL_SECONDS
        self.batch_size: int = batch_size or litellm.DEFAULT_BATCH_SIZE
        self.last_flush_time = time.time()
        self.flush_lock = flush_lock

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
                    return
                if self.preserve_events_added_during_flush:
                    del self.log_queue[:log_queue_length]
                else:
                    self.log_queue.clear()
                self.last_flush_time = time.time()

    async def async_send_batch(self, *args, **kwargs):
        pass
