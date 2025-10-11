import asyncio
from typing import Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    DBSpendUpdateTransactions,
    Litellm_EntityType,
    SpendUpdateQueueItem,
)
from litellm.proxy.db.db_transaction_queue.base_update_queue import (
    BaseUpdateQueue,
    service_logger_obj,
)
from litellm.types.services import ServiceTypes


class SpendUpdateQueue(BaseUpdateQueue):
    """
    In memory buffer for spend updates that should be committed to the database
    """

    def __init__(self):
        super().__init__()
        self.update_queue: asyncio.Queue[SpendUpdateQueueItem] = asyncio.Queue()

    async def flush_and_get_aggregated_db_spend_update_transactions(
        self,
    ) -> DBSpendUpdateTransactions:
        """Flush all updates from the queue and return all updates aggregated by entity type."""
        updates = await self.flush_all_updates_from_in_memory_queue()
        verbose_proxy_logger.debug("Aggregating updates by entity type: %s", updates)
        return self.get_aggregated_db_spend_update_transactions(updates)

    async def add_update(self, update: SpendUpdateQueueItem):
        """Enqueue an update to the spend update queue"""
        verbose_proxy_logger.debug("Adding update to queue: %s", update)
        await self.update_queue.put(update)

        # if the queue is full, aggregate the updates
        if self.update_queue.qsize() >= self.MAX_SIZE_IN_MEMORY_QUEUE:
            verbose_proxy_logger.warning(
                "Spend update queue is full. Aggregating all entries in queue to concatenate entries."
            )
            await self.aggregate_queue_updates()

    async def aggregate_queue_updates(self):
        """Concatenate all updates in the queue to reduce the size of in-memory queue"""
        updates: List[
            SpendUpdateQueueItem
        ] = await self.flush_all_updates_from_in_memory_queue()
        aggregated_updates = self._get_aggregated_spend_update_queue_item(updates)
        for update in aggregated_updates:
            await self.update_queue.put(update)
        return

    def _get_aggregated_spend_update_queue_item(
        self, updates: List[SpendUpdateQueueItem]
    ) -> List[SpendUpdateQueueItem]:
        """
        This is used to reduce the size of the in-memory queue by aggregating updates by entity type + id


        Aggregate updates by entity type + id

        eg.

        ```
        [
            {
                "entity_type": "user",
                "entity_id": "123",
                "response_cost": 100
            },
            {
                "entity_type": "user",
                "entity_id": "123",
                "response_cost": 200
            }
        ]

        ```

        becomes

        ```

        [
            {
                "entity_type": "user",
                "entity_id": "123",
                "response_cost": 300
            }
        ]

        ```
        """
        verbose_proxy_logger.debug(
            "Aggregating spend updates, current queue size: %s",
            self.update_queue.qsize(),
        )
        aggregated_spend_updates: List[SpendUpdateQueueItem] = []

        _in_memory_map: Dict[str, SpendUpdateQueueItem] = {}
        """
        Used for combining several updates into a single update
        Key=entity_type:entity_id
        Value=SpendUpdateQueueItem
        """
        for update in updates:
            _key = f"{update.get('entity_type')}:{update.get('entity_id')}"
            if _key not in _in_memory_map:
                _in_memory_map[_key] = update
            else:
                current_cost = _in_memory_map[_key].get("response_cost", 0) or 0
                update_cost = update.get("response_cost", 0) or 0
                _in_memory_map[_key]["response_cost"] = current_cost + update_cost

        for _key, update in _in_memory_map.items():
            aggregated_spend_updates.append(update)

        verbose_proxy_logger.debug(
            "Aggregated spend updates: %s", aggregated_spend_updates
        )
        return aggregated_spend_updates

    def get_aggregated_db_spend_update_transactions(
        self, updates: List[SpendUpdateQueueItem]
    ) -> DBSpendUpdateTransactions:
        """Aggregate updates by entity type."""
        # Initialize all transaction lists as empty dicts
        db_spend_update_transactions = DBSpendUpdateTransactions(
            user_list_transactions={},
            end_user_list_transactions={},
            key_list_transactions={},
            team_list_transactions={},
            team_member_list_transactions={},
            org_list_transactions={},
            tag_list_transactions={},
        )

        # Map entity types to their corresponding transaction dictionary keys
        entity_type_to_dict_key = {
            Litellm_EntityType.USER: "user_list_transactions",
            Litellm_EntityType.END_USER: "end_user_list_transactions",
            Litellm_EntityType.KEY: "key_list_transactions",
            Litellm_EntityType.TEAM: "team_list_transactions",
            Litellm_EntityType.TEAM_MEMBER: "team_member_list_transactions",
            Litellm_EntityType.ORGANIZATION: "org_list_transactions",
            Litellm_EntityType.TAG: "tag_list_transactions",
        }

        for update in updates:
            entity_type = update.get("entity_type")
            entity_id = update.get("entity_id") or ""
            response_cost = update.get("response_cost") or 0

            if entity_type is None:
                verbose_proxy_logger.debug(
                    "Skipping update spend for update: %s, because entity_type is None",
                    update,
                )
                continue

            dict_key = entity_type_to_dict_key.get(entity_type)
            if dict_key is None:
                verbose_proxy_logger.debug(
                    "Skipping update spend for update: %s, because entity_type is not in entity_type_to_dict_key",
                    update,
                )
                continue  # Skip unknown entity types

            # Type-safe access using if/elif statements
            if dict_key == "user_list_transactions":
                transactions_dict = db_spend_update_transactions[
                    "user_list_transactions"
                ]
            elif dict_key == "end_user_list_transactions":
                transactions_dict = db_spend_update_transactions[
                    "end_user_list_transactions"
                ]
            elif dict_key == "key_list_transactions":
                transactions_dict = db_spend_update_transactions[
                    "key_list_transactions"
                ]
            elif dict_key == "team_list_transactions":
                transactions_dict = db_spend_update_transactions[
                    "team_list_transactions"
                ]
            elif dict_key == "team_member_list_transactions":
                transactions_dict = db_spend_update_transactions[
                    "team_member_list_transactions"
                ]
            elif dict_key == "org_list_transactions":
                transactions_dict = db_spend_update_transactions[
                    "org_list_transactions"
                ]
            elif dict_key == "tag_list_transactions":
                transactions_dict = db_spend_update_transactions[
                    "tag_list_transactions"
                ]
            else:
                continue

            if transactions_dict is None:
                transactions_dict = {}

                # type ignore: dict_key is guaranteed to be one of "one of ("user_list_transactions", "end_user_list_transactions", "key_list_transactions", "team_list_transactions", "team_member_list_transactions", "org_list_transactions")"
                db_spend_update_transactions[dict_key] = transactions_dict  # type: ignore

            if entity_id not in transactions_dict:
                transactions_dict[entity_id] = 0

            transactions_dict[entity_id] += response_cost or 0

        return db_spend_update_transactions

    async def _emit_new_item_added_to_queue_event(
        self,
        queue_size: Optional[int] = None,
    ):
        asyncio.create_task(
            service_logger_obj.async_service_success_hook(
                service=ServiceTypes.IN_MEMORY_SPEND_UPDATE_QUEUE,
                duration=0,
                call_type="_emit_new_item_added_to_queue_event",
                event_metadata={
                    "gauge_labels": ServiceTypes.IN_MEMORY_SPEND_UPDATE_QUEUE,
                    "gauge_value": queue_size,
                },
            )
        )
