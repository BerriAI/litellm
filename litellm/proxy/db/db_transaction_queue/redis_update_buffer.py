"""
Handles buffering database `UPDATE` transactions in Redis before committing them to the database

This is to prevent deadlocks and improve reliability
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    MAX_REDIS_BUFFER_DEQUEUE_COUNT,
    REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY,
    REDIS_UPDATE_BUFFER_KEY,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import (
    DailyTagSpendTransaction,
    DailyTeamSpendTransaction,
    DailyUserSpendTransaction,
    DailyOrganizationSpendTransaction,
    DailyEndUserSpendTransaction,
    DBSpendUpdateTransactions,
    DailyAgentSpendTransaction,
)
from litellm.proxy.db.db_transaction_queue.base_update_queue import service_logger_obj
from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import (
    DailySpendUpdateQueue,
)
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue
from litellm.secret_managers.main import str_to_bool
from litellm.types.caching import RedisPipelineLpopOperation, RedisPipelineRpushOperation
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

        _use_redis_transaction_buffer: Optional[Union[bool, str]] = (
            general_settings.get("use_redis_transaction_buffer", False)
        )
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
        try:
            current_redis_buffer_size = await self.redis_cache.async_rpush(
                key=redis_key,
                values=list_of_transactions,
            )
            verbose_proxy_logger.debug(
                "Spend tracking - pushed spend updates to Redis buffer. "
                "redis_key=%s, buffer_size=%s",
                redis_key,
                current_redis_buffer_size,
            )
            await self._emit_new_item_added_to_redis_buffer_event(
                queue_size=current_redis_buffer_size,
                service=service_type,
            )
        except Exception as e:
            verbose_proxy_logger.error(
                "Spend tracking - failed to push spend updates to Redis (redis_key=%s). "
                "Error: %s",
                redis_key,
                str(e),
            )

    async def store_in_memory_spend_updates_in_redis(
        self,
        spend_update_queue: SpendUpdateQueue,
        daily_spend_update_queue: DailySpendUpdateQueue,
        daily_team_spend_update_queue: DailySpendUpdateQueue,
        daily_org_spend_update_queue: DailySpendUpdateQueue,
        daily_end_user_spend_update_queue: DailySpendUpdateQueue,
        daily_agent_spend_update_queue: DailySpendUpdateQueue,
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
        daily_org_spend_update_transactions = (
            await daily_org_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
        )
        daily_end_user_spend_update_transactions = (
            await daily_end_user_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
        )
        daily_agent_spend_update_transactions = (
            await daily_agent_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
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

        # Build a list of rpush operations, skipping empty/None transaction sets
        _queue_configs: List[Tuple[Any, str, ServiceTypes]] = [
            (db_spend_update_transactions, REDIS_UPDATE_BUFFER_KEY, ServiceTypes.REDIS_SPEND_UPDATE_QUEUE),
            (daily_spend_update_transactions, REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY, ServiceTypes.REDIS_DAILY_SPEND_UPDATE_QUEUE),
            (daily_team_spend_update_transactions, REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY, ServiceTypes.REDIS_DAILY_TEAM_SPEND_UPDATE_QUEUE),
            (daily_org_spend_update_transactions, REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY, ServiceTypes.REDIS_DAILY_ORG_SPEND_UPDATE_QUEUE),
            (daily_end_user_spend_update_transactions, REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY, ServiceTypes.REDIS_DAILY_END_USER_SPEND_UPDATE_QUEUE),
            (daily_agent_spend_update_transactions, REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY, ServiceTypes.REDIS_DAILY_AGENT_SPEND_UPDATE_QUEUE),
            (daily_tag_spend_update_transactions, REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY, ServiceTypes.REDIS_DAILY_TAG_SPEND_UPDATE_QUEUE),
        ]

        rpush_list: List[RedisPipelineRpushOperation] = []
        service_types: List[ServiceTypes] = []
        for transactions, redis_key, service_type in _queue_configs:
            if transactions is None or len(transactions) == 0:
                continue
            rpush_list.append(
                RedisPipelineRpushOperation(
                    key=redis_key,
                    values=[safe_dumps(transactions)],
                )
            )
            service_types.append(service_type)

        if len(rpush_list) == 0:
            return

        result_lengths = await self.redis_cache.async_rpush_pipeline(
            rpush_list=rpush_list,
        )

        # Emit gauge events for each queue
        for i, queue_size in enumerate(result_lengths):
            if i < len(service_types):
                await self._emit_new_item_added_to_redis_buffer_event(
                    queue_size=queue_size,
                    service=service_types[i],
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

        verbose_proxy_logger.info(
            "Spend tracking - popped %d spend update batches from Redis buffer (key=%s). "
            "These items are now removed from Redis and must be committed to DB.",
            len(list_of_transactions) if isinstance(list_of_transactions, list) else 1,
            REDIS_UPDATE_BUFFER_KEY,
        )

        # Parse the list of transactions from JSON strings
        parsed_transactions = self._parse_list_of_transactions(list_of_transactions)

        # If there are no transactions, return None
        if len(parsed_transactions) == 0:
            return None

        # Combine all transactions into a single transaction
        combined_transaction = self._combine_list_of_transactions(parsed_transactions)

        return combined_transaction

    async def get_all_transactions_from_redis_buffer_pipeline(
        self,
    ) -> Tuple[
        Optional[DBSpendUpdateTransactions],
        Optional[Dict[str, DailyUserSpendTransaction]],
        Optional[Dict[str, DailyTeamSpendTransaction]],
        Optional[Dict[str, DailyOrganizationSpendTransaction]],
        Optional[Dict[str, DailyEndUserSpendTransaction]],
        Optional[Dict[str, DailyAgentSpendTransaction]],
        Optional[Dict[str, DailyTagSpendTransaction]],
    ]:
        """
        Drains all 7 Redis buffer queues in a single pipeline round-trip.

        Returns a 7-tuple of parsed results in this order:
            0: DBSpendUpdateTransactions
            1: daily user spend
            2: daily team spend
            3: daily org spend
            4: daily end-user spend
            5: daily agent spend
            6: daily tag spend
        """
        if self.redis_cache is None:
            return None, None, None, None, None, None, None

        lpop_list: List[RedisPipelineLpopOperation] = [
            RedisPipelineLpopOperation(key=REDIS_UPDATE_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT),
            RedisPipelineLpopOperation(key=REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT),
            RedisPipelineLpopOperation(key=REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT),
            RedisPipelineLpopOperation(key=REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT),
            RedisPipelineLpopOperation(key=REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT),
            RedisPipelineLpopOperation(key=REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT),
            RedisPipelineLpopOperation(key=REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT),
        ]

        raw_results = await self.redis_cache.async_lpop_pipeline(lpop_list=lpop_list)

        # Pad with None if pipeline returned fewer results than expected
        while len(raw_results) < 7:
            raw_results.append(None)

        # Slot 0: DBSpendUpdateTransactions
        db_spend: Optional[DBSpendUpdateTransactions] = None
        if raw_results[0] is not None:
            parsed = self._parse_list_of_transactions(raw_results[0])
            if len(parsed) > 0:
                db_spend = self._combine_list_of_transactions(parsed)

        # Slots 1-6: daily spend categories
        daily_results: List[Optional[Dict[str, Any]]] = []
        for slot in range(1, 7):
            if raw_results[slot] is None:
                daily_results.append(None)
            else:
                list_of_daily = [json.loads(t) for t in raw_results[slot]]  # type: ignore
                aggregated = DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
                    list_of_daily
                )
                daily_results.append(aggregated)

        return (
            db_spend,
            cast(Optional[Dict[str, DailyUserSpendTransaction]], daily_results[0]),
            cast(Optional[Dict[str, DailyTeamSpendTransaction]], daily_results[1]),
            cast(Optional[Dict[str, DailyOrganizationSpendTransaction]], daily_results[2]),
            cast(Optional[Dict[str, DailyEndUserSpendTransaction]], daily_results[3]),
            cast(Optional[Dict[str, DailyAgentSpendTransaction]], daily_results[4]),
            cast(Optional[Dict[str, DailyTagSpendTransaction]], daily_results[5]),
        )

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

    async def get_all_daily_org_spend_update_transactions_from_redis_buffer(
        self,
    ) -> Optional[Dict[str, DailyOrganizationSpendTransaction]]: 
        """
        Gets all the daily organization spend update transactions from Redis
        """
        if self.redis_cache is None:
            return None
        list_of_transactions = await self.redis_cache.async_lpop(
            key=REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY,
            count=MAX_REDIS_BUFFER_DEQUEUE_COUNT,
        )
        if list_of_transactions is None:
            return None
        list_of_daily_spend_update_transactions = [
            json.loads(transaction) for transaction in list_of_transactions
        ]
        return cast(
            Dict[str, DailyOrganizationSpendTransaction], 
            DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
                list_of_daily_spend_update_transactions
            ),
        )

    async def get_all_daily_end_user_spend_update_transactions_from_redis_buffer(
        self,
    ) -> Optional[Dict[str, DailyEndUserSpendTransaction]]:
        """
        Gets all the daily end-user spend update transactions from Redis
        """
        if self.redis_cache is None:
            return None
        list_of_transactions = await self.redis_cache.async_lpop(
            key=REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY,
            count=MAX_REDIS_BUFFER_DEQUEUE_COUNT,
        )
        if list_of_transactions is None:
            return None
        list_of_daily_spend_update_transactions = [
            json.loads(transaction) for transaction in list_of_transactions
        ]
        return cast(
            Dict[str, DailyEndUserSpendTransaction],
            DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
                list_of_daily_spend_update_transactions
            ),
        )

    async def get_all_daily_agent_spend_update_transactions_from_redis_buffer(
        self,
    ) -> Optional[Dict[str, DailyAgentSpendTransaction]]:
        """
        Gets all the daily agent spend update transactions from Redis
        """
        if self.redis_cache is None:
            return None
        list_of_transactions = await self.redis_cache.async_lpop(
            key=REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY,
            count=MAX_REDIS_BUFFER_DEQUEUE_COUNT,
        )
        if list_of_transactions is None:
            return None
        list_of_daily_spend_update_transactions = [
            json.loads(transaction) for transaction in list_of_transactions
        ]
        return cast(
            Dict[str, DailyAgentSpendTransaction],
            DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
                list_of_daily_spend_update_transactions
            ),
        )

    async def get_all_daily_tag_spend_update_transactions_from_redis_buffer(
        self,
    ) -> Optional[Dict[str, DailyTagSpendTransaction]]:
        """
        Gets all the daily tag spend update transactions from Redis
        """
        if self.redis_cache is None:
            return None
        list_of_transactions = await self.redis_cache.async_lpop(
            key=REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY,
            count=MAX_REDIS_BUFFER_DEQUEUE_COUNT,
        )
        if list_of_transactions is None:
            return None
        list_of_daily_spend_update_transactions = [
            json.loads(transaction) for transaction in list_of_transactions
        ]
        return cast(
            Dict[str, DailyTagSpendTransaction],
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
            tag_list_transactions={},
        )

        # Define the transaction fields to process
        transaction_fields = [
            "user_list_transactions",
            "end_user_list_transactions",
            "key_list_transactions",
            "team_list_transactions",
            "team_member_list_transactions",
            "org_list_transactions",
            "tag_list_transactions",
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
