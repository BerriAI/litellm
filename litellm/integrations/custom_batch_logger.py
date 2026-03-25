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
        # Track health transition so we emit one metric when callback goes unhealthy.
        self._is_in_failure_state: bool = False

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
                verbose_logger.debug(
                    "CustomLogger: Flushing batch of %s events", len(self.log_queue)
                )
                try:
                    await self.async_send_batch()
                except Exception as e:
                    callback_name = self._get_callback_failure_name()
                    verbose_logger.debug(
                        "CustomBatchLogger: batch flush failed for %s: %s",
                        callback_name,
                        str(e),
                    )
                    if not self._is_in_failure_state:
                        self._report_callback_failure(callback_name=callback_name)
                        self._is_in_failure_state = True
                    else:
                        verbose_logger.debug(
                            "CustomBatchLogger: callback %s already in failure state, skipping duplicate metric",
                            callback_name,
                        )
                    return
                self.log_queue.clear()
                self.last_flush_time = time.time()
                self._is_in_failure_state = False

    def _get_callback_failure_name(self) -> str:
        """
        Default callback name for batch failures.
        """
        return self.__class__.__name__

    async def async_send_batch(self, *args, **kwargs):
        pass
