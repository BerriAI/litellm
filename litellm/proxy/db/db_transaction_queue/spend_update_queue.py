import asyncio
from typing import List

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    DBSpendUpdateTransactions,
    Litellm_EntityType,
    SpendUpdateQueueItem,
)
from litellm.proxy.db.db_transaction_queue.base_update_queue import BaseUpdateQueue


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
        )

        # Map entity types to their corresponding transaction dictionary keys
        entity_type_to_dict_key = {
            Litellm_EntityType.USER: "user_list_transactions",
            Litellm_EntityType.END_USER: "end_user_list_transactions",
            Litellm_EntityType.KEY: "key_list_transactions",
            Litellm_EntityType.TEAM: "team_list_transactions",
            Litellm_EntityType.TEAM_MEMBER: "team_member_list_transactions",
            Litellm_EntityType.ORGANIZATION: "org_list_transactions",
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
