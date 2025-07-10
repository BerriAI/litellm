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
from litellm.constants import MAX_IN_MEMORY_QUEUE_FLUSH_COUNT, MAX_SIZE_IN_MEMORY_QUEUE


class BaseUpdateQueue:
    """Base class for in memory buffer for database transactions"""

    def __init__(self):
        self.update_queue = asyncio.Queue()
        self.MAX_SIZE_IN_MEMORY_QUEUE = MAX_SIZE_IN_MEMORY_QUEUE

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
