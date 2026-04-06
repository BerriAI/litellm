"""
Base class for in memory buffer for database transactions
"""
import asyncio
from typing import Optional

from litellm._logging import verbose_proxy_logger
from litellm._service_logger import ServiceLogging

service_logger_obj = (
    ServiceLogging()
)  # used for tracking metrics for In memory buffer, redis buffer, pod lock manager
from litellm.constants import (
    LITELLM_ASYNCIO_QUEUE_MAXSIZE,
    MAX_IN_MEMORY_QUEUE_FLUSH_COUNT,
    MAX_SIZE_IN_MEMORY_QUEUE,
)


class BaseUpdateQueue:
    """Base class for in memory buffer for database transactions"""

    def __init__(self):
        self.update_queue = asyncio.Queue(maxsize=LITELLM_ASYNCIO_QUEUE_MAXSIZE)
        self.MAX_SIZE_IN_MEMORY_QUEUE = MAX_SIZE_IN_MEMORY_QUEUE
        if MAX_SIZE_IN_MEMORY_QUEUE >= LITELLM_ASYNCIO_QUEUE_MAXSIZE:
            verbose_proxy_logger.warning(
                "Misconfigured queue thresholds: MAX_SIZE_IN_MEMORY_QUEUE (%d) >= LITELLM_ASYNCIO_QUEUE_MAXSIZE (%d). "
                "The spend aggregation check will never trigger because the asyncio.Queue blocks at %d items. "
                "Set MAX_SIZE_IN_MEMORY_QUEUE to a value less than LITELLM_ASYNCIO_QUEUE_MAXSIZE (recommended: 80%% of it).",
                MAX_SIZE_IN_MEMORY_QUEUE,
                LITELLM_ASYNCIO_QUEUE_MAXSIZE,
                LITELLM_ASYNCIO_QUEUE_MAXSIZE,
            )

    async def add_update(self, update):
        """Enqueue an update."""
        verbose_proxy_logger.debug("Adding update to queue: %s", update)
        await self.update_queue.put(update)
        await self._emit_new_item_added_to_queue_event(
            queue_size=self.update_queue.qsize()
        )

    async def flush_all_updates_from_in_memory_queue(self):
        """Get all updates from the queue."""
        updates = []
        while not self.update_queue.empty():
            # Circuit breaker to ensure we're not stuck dequeuing updates. Protect CPU utilization
            if len(updates) >= MAX_IN_MEMORY_QUEUE_FLUSH_COUNT:
                verbose_proxy_logger.debug(
                    "Max in memory queue flush count reached, stopping flush"
                )
                break
            updates.append(await self.update_queue.get())
        return updates

    async def _emit_new_item_added_to_queue_event(
        self,
        queue_size: Optional[int] = None,
    ):
        """placeholder, emit event when a new item is added to the queue"""
        pass
