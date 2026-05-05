import asyncio
from copy import deepcopy
from typing import Dict, List

from litellm.constants import LITELLM_ASYNCIO_QUEUE_MAXSIZE
from litellm.proxy._types import DailyUserRequestDurationTransaction
from litellm.proxy.db.db_transaction_queue.base_update_queue import BaseUpdateQueue


class DailyUserRequestDurationQueue(BaseUpdateQueue):
    """
    In-memory buffer for daily user request duration updates.

    Keyed by "{user_id}_{date}". Aggregates total_request_duration_ms and
    api_requests before flushing to LiteLLM_DailyUserRequestDuration.
    """

    def __init__(self):
        super().__init__()
        self.update_queue: asyncio.Queue[
            Dict[str, DailyUserRequestDurationTransaction]
        ] = asyncio.Queue(maxsize=LITELLM_ASYNCIO_QUEUE_MAXSIZE)

    async def add_update(self, update: Dict[str, DailyUserRequestDurationTransaction]):
        await self.update_queue.put(update)
        if self.update_queue.qsize() >= self.MAX_SIZE_IN_MEMORY_QUEUE:
            await self.aggregate_queue_updates()

    async def aggregate_queue_updates(self):
        updates = await self.flush_all_updates_from_in_memory_queue()
        aggregated = self._aggregate(updates)
        await self.update_queue.put(aggregated)

    async def flush_and_get_aggregated(
        self,
    ) -> Dict[str, DailyUserRequestDurationTransaction]:
        updates = await self.flush_all_updates_from_in_memory_queue()
        return self._aggregate(updates)

    @staticmethod
    def _aggregate(
        updates: List[Dict[str, DailyUserRequestDurationTransaction]],
    ) -> Dict[str, DailyUserRequestDurationTransaction]:
        aggregated: Dict[str, DailyUserRequestDurationTransaction] = {}
        for _update in updates:
            for _key, payload in _update.items():
                if _key in aggregated:
                    aggregated[_key]["total_request_duration_ms"] += payload[
                        "total_request_duration_ms"
                    ]
                    aggregated[_key]["api_requests"] += payload["api_requests"]
                else:
                    aggregated[_key] = deepcopy(payload)
        return aggregated
