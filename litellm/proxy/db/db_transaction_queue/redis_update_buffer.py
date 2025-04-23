"""
Handles buffering database `UPDATE` transactions in Redis before committing them to the database

This is to prevent deadlocks and improve reliability
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    MAX_REDIS_BUFFER_DEQUEUE_COUNT,
    REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY,
    REDIS_UPDATE_BUFFER_KEY,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import (
    DailyTeamSpendTransaction,
    DailyUserSpendTransaction,
    DBSpendUpdateTransactions,
)
from litellm.proxy.db.db_transaction_queue.base_update_queue import service_logger_obj
from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import (
    DailySpendUpdateQueue,
)
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue
from litellm.secret_managers.main import str_to_bool
from litellm.types.services import ServiceTypes

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
else:
    PrismaClient = Any


class RedisUpdateBuffer:
    """
    Handles buffering database `UPDATE` transactions in Redis before committing them to the database

    This is to prevent deadlocks and improve reliability
    """

    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
    ):
        self.redis_cache = redis_cache

    @staticmethod
    def _should_commit_spend_updates_to_redis() -> bool:
        """
        Checks if the Pod should commit spend updates to Redis

        This setting enables buffering database transactions in Redis
        to improve reliability and reduce database contention
        """
        from litellm.proxy.proxy_server import general_settings

        _use_redis_transaction_buffer: Optional[
            Union[bool, str]
        ] = general_settings.get("use_redis_transaction_buffer", False)
        if isinstance(_use_redis_transaction_buffer, str):
            _use_redis_transaction_buffer = str_to_bool(_use_redis_transaction_buffer)
        if _use_redis_transaction_buffer is None:
            return False
        return _use_redis_transaction_buffer

    async def _store_transactions_in_redis(
        self,
        transactions: Any,
        redis_key: str,
        service_type: ServiceTypes,
    ) -> None:
        """
        Helper method to store transactions in Redis and emit an event

        Args:
            transactions: The transactions to store
            redis_key: The Redis key to store under
            service_type: The service type for event emission
        """
        if transactions is None or len(transactions) == 0:
            return

        list_of_transactions = [safe_dumps(transactions)]
        if self.redis_cache is None:
            return
        current_redis_buffer_size = await self.redis_cache.async_rpush(
            key=redis_key,
            values=list_of_transactions,
        )
        await self._emit_new_item_added_to_redis_buffer_event(
            queue_size=current_redis_buffer_size,
            service=service_type,
        )

    async def store_in_memory_spend_updates_in_redis(
        self,
        spend_update_queue: SpendUpdateQueue,
        daily_spend_update_queue: DailySpendUpdateQueue,
        daily_team_spend_update_queue: DailySpendUpdateQueue,
        daily_tag_spend_update_queue: DailySpendUpdateQueue,
    ):
        """
        Stores the in-memory spend updates to Redis

        Stores the following in memory data structures in Redis:
            - SpendUpdateQueue - Key, User, Team, TeamMember, Org, EndUser Spend updates
            - DailySpendUpdateQueue - Daily Spend updates Aggregate view

        For SpendUpdateQueue:
            Each transaction is a dict stored as following:
            - key is the entity id
            - value is the spend amount

                ```
                Redis List:
                key_list_transactions:
                [
                    "0929880201": 1.2,
                    "0929880202": 0.01,
                    "0929880203": 0.001,
                ]
                ```

        For DailySpendUpdateQueue:
            Each transaction is a Dict[str, DailyUserSpendTransaction] stored as following:
            - key is the daily_transaction_key
            - value is the DailyUserSpendTransaction

                ```
                Redis List:
                daily_spend_update_transactions:
                [
                    {
                        "user_keyhash_1_model_1": {
                            "spend": 1.2,
                            "prompt_tokens": 1000,
                            "completion_tokens": 1000,
                            "api_requests": 1000,
                            "successful_requests": 1000,
                        },

                    }
                ]
                ```
        """
        if self.redis_cache is None:
            verbose_proxy_logger.debug(
                "redis_cache is None, skipping store_in_memory_spend_updates_in_redis"
            )
            return

        # Get all transactions
        db_spend_update_transactions = (
            await spend_update_queue.flush_and_get_aggregated_db_spend_update_transactions()
        )
        daily_spend_update_transactions = (
            await daily_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
        )
        daily_team_spend_update_transactions = (
            await daily_team_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
        )
        daily_tag_spend_update_transactions = (
            await daily_tag_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
        )

        verbose_proxy_logger.debug(
            "ALL DB SPEND UPDATE TRANSACTIONS: %s", db_spend_update_transactions
        )
        verbose_proxy_logger.debug(
            "ALL DAILY SPEND UPDATE TRANSACTIONS: %s", daily_spend_update_transactions
        )

        # only store in redis if there are any updates to commit
        if (
            self._number_of_transactions_to_store_in_redis(db_spend_update_transactions)
            == 0
        ):
            return

        # Store all transaction types using the helper method
        await self._store_transactions_in_redis(
            transactions=db_spend_update_transactions,
            redis_key=REDIS_UPDATE_BUFFER_KEY,
            service_type=ServiceTypes.REDIS_SPEND_UPDATE_QUEUE,
        )

        await self._store_transactions_in_redis(
            transactions=daily_spend_update_transactions,
            redis_key=REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
            service_type=ServiceTypes.REDIS_DAILY_SPEND_UPDATE_QUEUE,
        )

        await self._store_transactions_in_redis(
            transactions=daily_team_spend_update_transactions,
            redis_key=REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY,
            service_type=ServiceTypes.REDIS_DAILY_TEAM_SPEND_UPDATE_QUEUE,
        )

        await self._store_transactions_in_redis(
            transactions=daily_tag_spend_update_transactions,
            redis_key=REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY,
            service_type=ServiceTypes.REDIS_DAILY_TAG_SPEND_UPDATE_QUEUE,
        )

    @staticmethod
    def _number_of_transactions_to_store_in_redis(
        db_spend_update_transactions: DBSpendUpdateTransactions,
    ) -> int:
        """
        Gets the number of transactions to store in Redis
        """
        num_transactions = 0
        for v in db_spend_update_transactions.values():
            if isinstance(v, dict):
                num_transactions += len(v)
        return num_transactions

    @staticmethod
    def _remove_prefix_from_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """
        Removes the specified prefix from the keys of a dictionary.
        """
        return {key.replace(prefix, "", 1): value for key, value in data.items()}

    async def get_all_update_transactions_from_redis_buffer(
        self,
    ) -> Optional[DBSpendUpdateTransactions]:
        """
        Gets all the update transactions from Redis

        On Redis we store a list of transactions as a JSON string

        eg.
            [
                DBSpendUpdateTransactions(
                    user_list_transactions={
                        "user_id_1": 1.2,
                        "user_id_2": 0.01,
                    },
                    end_user_list_transactions={},
                    key_list_transactions={
                        "0929880201": 1.2,
                        "0929880202": 0.01,
                    },
                    team_list_transactions={},
                    team_member_list_transactions={},
                    org_list_transactions={},
                ),
                DBSpendUpdateTransactions(
                    user_list_transactions={
                        "user_id_3": 1.2,
                        "user_id_4": 0.01,
                    },
                    end_user_list_transactions={},
                    key_list_transactions={
                        "key_id_1": 1.2,
                        "key_id_2": 0.01,
                    },
                    team_list_transactions={},
                    team_member_list_transactions={},
                    org_list_transactions={},
            ]
        """
        if self.redis_cache is None:
            return None
        list_of_transactions = await self.redis_cache.async_lpop(
            key=REDIS_UPDATE_BUFFER_KEY,
            count=MAX_REDIS_BUFFER_DEQUEUE_COUNT,
        )
        if list_of_transactions is None:
            return None

        # Parse the list of transactions from JSON strings
        parsed_transactions = self._parse_list_of_transactions(list_of_transactions)

        # If there are no transactions, return None
        if len(parsed_transactions) == 0:
            return None

        # Combine all transactions into a single transaction
        combined_transaction = self._combine_list_of_transactions(parsed_transactions)

        return combined_transaction

    async def get_all_daily_spend_update_transactions_from_redis_buffer(
        self,
    ) -> Optional[Dict[str, DailyUserSpendTransaction]]:
        """
        Gets all the daily spend update transactions from Redis
        """
        if self.redis_cache is None:
            return None
        list_of_transactions = await self.redis_cache.async_lpop(
            key=REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
            count=MAX_REDIS_BUFFER_DEQUEUE_COUNT,
        )
        if list_of_transactions is None:
            return None
        list_of_daily_spend_update_transactions = [
            json.loads(transaction) for transaction in list_of_transactions
        ]
        return cast(
            Dict[str, DailyUserSpendTransaction],
            DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
                list_of_daily_spend_update_transactions
            ),
        )

    async def get_all_daily_team_spend_update_transactions_from_redis_buffer(
        self,
    ) -> Optional[Dict[str, DailyTeamSpendTransaction]]:
        """
        Gets all the daily team spend update transactions from Redis
        """
        if self.redis_cache is None:
            return None
        list_of_transactions = await self.redis_cache.async_lpop(
            key=REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY,
            count=MAX_REDIS_BUFFER_DEQUEUE_COUNT,
        )
        if list_of_transactions is None:
            return None
        list_of_daily_spend_update_transactions = [
            json.loads(transaction) for transaction in list_of_transactions
        ]
        return cast(
            Dict[str, DailyTeamSpendTransaction],
            DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
                list_of_daily_spend_update_transactions
            ),
        )

    @staticmethod
    def _parse_list_of_transactions(
        list_of_transactions: Union[Any, List[Any]],
    ) -> List[DBSpendUpdateTransactions]:
        """
        Parses the list of transactions from Redis
        """
        if isinstance(list_of_transactions, list):
            return [json.loads(transaction) for transaction in list_of_transactions]
        else:
            return [json.loads(list_of_transactions)]

    @staticmethod
    def _combine_list_of_transactions(
        list_of_transactions: List[DBSpendUpdateTransactions],
    ) -> DBSpendUpdateTransactions:
        """
        Combines the list of transactions into a single DBSpendUpdateTransactions object
        """
        # Initialize a new combined transaction object with empty dictionaries
        combined_transaction = DBSpendUpdateTransactions(
            user_list_transactions={},
            end_user_list_transactions={},
            key_list_transactions={},
            team_list_transactions={},
            team_member_list_transactions={},
            org_list_transactions={},
        )

        # Define the transaction fields to process
        transaction_fields = [
            "user_list_transactions",
            "end_user_list_transactions",
            "key_list_transactions",
            "team_list_transactions",
            "team_member_list_transactions",
            "org_list_transactions",
        ]

        # Loop through each transaction and combine the values
        for transaction in list_of_transactions:
            # Process each field type
            for field in transaction_fields:
                if transaction.get(field):
                    for entity_id, amount in transaction[field].items():  # type: ignore
                        combined_transaction[field][entity_id] = (  # type: ignore
                            combined_transaction[field].get(entity_id, 0) + amount  # type: ignore
                        )

        return combined_transaction

    async def _emit_new_item_added_to_redis_buffer_event(
        self,
        service: ServiceTypes,
        queue_size: int,
    ):
        asyncio.create_task(
            service_logger_obj.async_service_success_hook(
                service=service,
                duration=0,
                call_type="_emit_new_item_added_to_queue_event",
                event_metadata={
                    "gauge_labels": service,
                    "gauge_value": queue_size,
                },
            )
        )
