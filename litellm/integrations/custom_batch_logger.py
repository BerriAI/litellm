"""
Custom Logger that handles batching logic 

Use this if you want your logs to be stored in memory and flushed periodically
"""

import asyncio
import time
from typing import List, Literal, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

DEFAULT_BATCH_SIZE = 512
DEFAULT_FLUSH_INTERVAL_SECONDS = 5


class CustomBatchLogger(CustomLogger):

    def __init__(
        self,
        flush_lock: Optional[asyncio.Lock] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        **kwargs,
    ) -> None:
        """
        Args:
            flush_lock (Optional[asyncio.Lock], optional): Lock to use when flushing the queue. Defaults to None. Only used for custom loggers that do batching
        """
        self.log_queue: List = []
        self.flush_interval = DEFAULT_FLUSH_INTERVAL_SECONDS  # 10 seconds
        self.batch_size: int = batch_size or DEFAULT_BATCH_SIZE
        self.last_flush_time = time.time()
        self.flush_lock = flush_lock

        super().__init__(**kwargs)
        pass

    async def periodic_flush(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            verbose_logger.debug(
                f"CustomLogger periodic flush after {self.flush_interval} seconds"
            )
            await self.flush_queue()

    async def flush_queue(self):
        async with self.flush_lock:
            if self.log_queue:
                verbose_logger.debug(
                    "CustomLogger: Flushing batch of %s events", len(self.log_queue)
                )
                await self.async_send_batch()
                self.log_queue.clear()
                self.last_flush_time = time.time()

    async def async_send_batch(self):
        pass
