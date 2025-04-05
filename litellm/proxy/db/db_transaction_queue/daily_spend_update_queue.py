import asyncio
from copy import deepcopy
from typing import Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import DailyUserSpendTransaction
from litellm.proxy.db.db_transaction_queue.base_update_queue import (
    BaseUpdateQueue,
    service_logger_obj,
)
from litellm.types.services import ServiceTypes


class DailySpendUpdateQueue(BaseUpdateQueue):
    """
    In memory buffer for daily spend updates that should be committed to the database

    To add a new daily spend update transaction, use the following format:
        daily_spend_update_queue.add_update({
            "user1_date_api_key_model_custom_llm_provider": {
                "spend": 10,
                "prompt_tokens": 100,
                "completion_tokens": 100,
            }
        })

    Queue contains a list of daily spend update transactions

    eg
        queue = [
            {
                "user1_date_api_key_model_custom_llm_provider": {
                    "spend": 10,
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                    "api_requests": 100,
                    "successful_requests": 100,
                    "failed_requests": 100,
                }
            },
            {
                "user2_date_api_key_model_custom_llm_provider": {
                    "spend": 10,
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                    "api_requests": 100,
                    "successful_requests": 100,
                    "failed_requests": 100,
                }
            }
        ]
    """

    def __init__(self):
        super().__init__()
        self.update_queue: asyncio.Queue[
            Dict[str, DailyUserSpendTransaction]
        ] = asyncio.Queue()

    async def add_update(self, update: Dict[str, DailyUserSpendTransaction]):
        """Enqueue an update."""
        verbose_proxy_logger.debug("Adding update to queue: %s", update)
        await self.update_queue.put(update)
        if self.update_queue.qsize() >= self.MAX_SIZE_IN_MEMORY_QUEUE:
            verbose_proxy_logger.warning(
                "Spend update queue is full. Aggregating all entries in queue to concatenate entries."
            )
            await self.aggregate_queue_updates()

    async def aggregate_queue_updates(self):
        """
        Combine all updates in the queue into a single update.
        This is used to reduce the size of the in-memory queue.
        """
        updates: List[
            Dict[str, DailyUserSpendTransaction]
        ] = await self.flush_all_updates_from_in_memory_queue()
        aggregated_updates = self.get_aggregated_daily_spend_update_transactions(
            updates
        )
        await self.update_queue.put(aggregated_updates)

    async def flush_and_get_aggregated_daily_spend_update_transactions(
        self,
    ) -> Dict[str, DailyUserSpendTransaction]:
        """Get all updates from the queue and return all updates aggregated by daily_transaction_key."""
        updates = await self.flush_all_updates_from_in_memory_queue()
        aggregated_daily_spend_update_transactions = (
            DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
                updates
            )
        )
        verbose_proxy_logger.debug(
            "Aggregated daily spend update transactions: %s",
            aggregated_daily_spend_update_transactions,
        )
        return aggregated_daily_spend_update_transactions

    @staticmethod
    def get_aggregated_daily_spend_update_transactions(
        updates: List[Dict[str, DailyUserSpendTransaction]]
    ) -> Dict[str, DailyUserSpendTransaction]:
        """Aggregate updates by daily_transaction_key."""
        aggregated_daily_spend_update_transactions: Dict[
            str, DailyUserSpendTransaction
        ] = {}
        for _update in updates:
            for _key, payload in _update.items():
                if _key in aggregated_daily_spend_update_transactions:
                    daily_transaction = aggregated_daily_spend_update_transactions[_key]
                    daily_transaction["spend"] += payload["spend"]
                    daily_transaction["prompt_tokens"] += payload["prompt_tokens"]
                    daily_transaction["completion_tokens"] += payload[
                        "completion_tokens"
                    ]
                    daily_transaction["api_requests"] += payload["api_requests"]
                    daily_transaction["successful_requests"] += payload[
                        "successful_requests"
                    ]
                    daily_transaction["failed_requests"] += payload["failed_requests"]
                else:
                    aggregated_daily_spend_update_transactions[_key] = deepcopy(payload)
        return aggregated_daily_spend_update_transactions

    async def _emit_new_item_added_to_queue_event(
        self,
        queue_size: Optional[int] = None,
    ):
        asyncio.create_task(
            service_logger_obj.async_service_success_hook(
                service=ServiceTypes.IN_MEMORY_DAILY_SPEND_UPDATE_QUEUE,
                duration=0,
                call_type="_emit_new_item_added_to_queue_event",
                event_metadata={
                    "gauge_labels": ServiceTypes.IN_MEMORY_DAILY_SPEND_UPDATE_QUEUE,
                    "gauge_value": queue_size,
                },
            )
        )
