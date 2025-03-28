"""
Handles buffering database `UPDATE` transactions in Redis before committing them to the database

This is to prevent deadlocks and improve reliability
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.secret_managers.main import str_to_bool

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
else:
    PrismaClient = Any


class DBSpendUpdateTransactions(TypedDict):
    user_list_transactions: Optional[Dict[str, float]]
    end_user_list_transactions: Optional[Dict[str, float]]
    key_list_transactions: Optional[Dict[str, float]]
    team_list_transactions: Optional[Dict[str, float]]
    team_member_list_transactions: Optional[Dict[str, float]]
    org_list_transactions: Optional[Dict[str, float]]


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

    async def store_in_memory_spend_updates_in_redis(
        self,
        prisma_client: PrismaClient,
    ):
        """
        Stores the in-memory spend updates to Redis

        Each transaction is a dict stored as following:
        - key is the entity id
        - value is the spend amount

            ```
            {
                "0929880201": 10,
                "0929880202": 20,
                "0929880203": 30,
            }
            ```
        """
        IN_MEMORY_UPDATE_TRANSACTIONS: DBSpendUpdateTransactions = (
            DBSpendUpdateTransactions(
                user_list_transactions=prisma_client.user_list_transactions,
                end_user_list_transactions=prisma_client.end_user_list_transactions,
                key_list_transactions=prisma_client.key_list_transactions,
                team_list_transactions=prisma_client.team_list_transactions,
                team_member_list_transactions=prisma_client.team_member_list_transactions,
                org_list_transactions=prisma_client.org_list_transactions,
            )
        )
        for key, _transactions in IN_MEMORY_UPDATE_TRANSACTIONS.items():
            await self.increment_all_transaction_objects_in_redis(
                key=key,
                transactions=cast(Dict, _transactions),
            )

    async def increment_all_transaction_objects_in_redis(
        self,
        key: str,
        transactions: Dict,
    ):
        """
        Increments all transaction objects in Redis
        """
        if self.redis_cache is None:
            verbose_proxy_logger.debug(
                "redis_cache is None, skipping increment_all_transaction_objects_in_redis"
            )
            return
        for transaction_id, transaction_amount in transactions.items():
            await self.redis_cache.async_increment(
                key=f"{key}:{transaction_id}",
                value=transaction_amount,
            )

    async def get_all_update_transactions_from_redis(
        self,
    ) -> Optional[DBSpendUpdateTransactions]:
        """
        Gets all the update transactions from Redis
        """
        if self.redis_cache is None:
            return None
        expected_keys = [
            "user_list_transactions",
            "end_user_list_transactions",
            "key_list_transactions",
            "team_list_transactions",
            "team_member_list_transactions",
            "org_list_transactions",
        ]
        result = await self.redis_cache.async_batch_get_cache(expected_keys)
        if result is None:
            return None
        return DBSpendUpdateTransactions(
            user_list_transactions=result.get("user_list_transactions", {}),
            end_user_list_transactions=result.get("end_user_list_transactions", {}),
            key_list_transactions=result.get("key_list_transactions", {}),
            team_list_transactions=result.get("team_list_transactions", {}),
            team_member_list_transactions=result.get(
                "team_member_list_transactions", {}
            ),
            org_list_transactions=result.get("org_list_transactions", {}),
        )
