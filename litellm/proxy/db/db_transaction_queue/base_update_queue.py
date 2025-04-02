"""
Base class for in memory buffer for database transactions
"""
import asyncio

from litellm._logging import verbose_proxy_logger


class BaseUpdateQueue:
    """Base class for in memory buffer for database transactions"""

    def __init__(self):
        self.update_queue = asyncio.Queue()

    async def add_update(self, update):
        """Enqueue an update."""
        verbose_proxy_logger.debug("Adding update to queue: %s", update)
        await self.update_queue.put(update)

    async def flush_all_updates_from_in_memory_queue(self):
        """Get all updates from the queue."""
        updates = []
        while not self.update_queue.empty():
            updates.append(await self.update_queue.get())
        return updates
