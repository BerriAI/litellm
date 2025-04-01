import asyncio
from typing import TYPE_CHECKING, Any, Dict, List

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
else:
    PrismaClient = Any


class SpendUpdateQueue:
    """
    In memory buffer for spend updates that should be committed to the database
    """

    def __init__(
        self,
    ):
        self.update_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def add_update(self, update: Dict[str, Any]) -> None:
        """Enqueue an update. Each update might be a dict like {'entity_type': 'user', 'entity_id': '123', 'amount': 1.2}."""
        verbose_proxy_logger.debug("Adding update to queue: %s", update)
        await self.update_queue.put(update)

    async def flush_all_updates_from_in_memory_queue(self) -> List[Dict[str, Any]]:
        """Get all updates from the queue."""
        updates: List[Dict[str, Any]] = []
        while not self.update_queue.empty():
            updates.append(await self.update_queue.get())
        return updates

    async def flush_and_get_all_aggregated_updates_by_entity_type(
        self,
    ) -> Dict[str, Any]:
        """Flush all updates from the queue and return all updates aggregated by entity type."""
        updates = await self.flush_all_updates_from_in_memory_queue()
        verbose_proxy_logger.debug("Aggregating updates by entity type: %s", updates)
        return self.aggregate_updates_by_entity_type(updates)

    def aggregate_updates_by_entity_type(
        self, updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate updates by entity type."""
        aggregated_updates = {}
        for update in updates:
            entity_type = update["entity_type"]
            entity_id = update["entity_id"]
            amount = update["amount"]
            if entity_type not in aggregated_updates:
                aggregated_updates[entity_type] = {}
            if entity_id not in aggregated_updates[entity_type]:
                aggregated_updates[entity_type][entity_id] = 0
            aggregated_updates[entity_type][entity_id] += amount
        return aggregated_updates
