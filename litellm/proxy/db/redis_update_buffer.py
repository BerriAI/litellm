"""
Handles buffering database `UPDATE` transactions in Redis before committing them to the database

This is to prevent deadlocks and improve reliability
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.proxy._types import DBSpendUpdateTransactions
from litellm.secret_managers.main import str_to_bool

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

    @staticmethod
    def _remove_prefix_from_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """
        Removes the specified prefix from the keys of a dictionary.
        """
        return {key.replace(prefix, "", 1): value for key, value in data.items()}

    async def get_all_update_transactions_from_redis(
        self,
    ) -> Optional[DBSpendUpdateTransactions]:
        """
        Gets all the update transactions from Redis
        """
        if self.redis_cache is None:
            return None
        user_transaction_keys = await self.redis_cache.async_scan_iter(
            "user_list_transactions:*"
        )
        end_user_transaction_keys = await self.redis_cache.async_scan_iter(
            "end_user_list_transactions:*"
        )
        key_transaction_keys = await self.redis_cache.async_scan_iter(
            "key_list_transactions:*"
        )
        team_transaction_keys = await self.redis_cache.async_scan_iter(
            "team_list_transactions:*"
        )
        team_member_transaction_keys = await self.redis_cache.async_scan_iter(
            "team_member_list_transactions:*"
        )
        org_transaction_keys = await self.redis_cache.async_scan_iter(
            "org_list_transactions:*"
        )

        user_list_transactions = await self.redis_cache.async_batch_get_cache(
            user_transaction_keys
        )
        end_user_list_transactions = await self.redis_cache.async_batch_get_cache(
            end_user_transaction_keys
        )
        key_list_transactions = await self.redis_cache.async_batch_get_cache(
            key_transaction_keys
        )
        team_list_transactions = await self.redis_cache.async_batch_get_cache(
            team_transaction_keys
        )
        team_member_list_transactions = await self.redis_cache.async_batch_get_cache(
            team_member_transaction_keys
        )
        org_list_transactions = await self.redis_cache.async_batch_get_cache(
            org_transaction_keys
        )

        # filter out the "prefix" from the keys using the helper method
        user_list_transactions = self._remove_prefix_from_keys(
            user_list_transactions, "user_list_transactions:"
        )
        end_user_list_transactions = self._remove_prefix_from_keys(
            end_user_list_transactions, "end_user_list_transactions:"
        )
        key_list_transactions = self._remove_prefix_from_keys(
            key_list_transactions, "key_list_transactions:"
        )
        team_list_transactions = self._remove_prefix_from_keys(
            team_list_transactions, "team_list_transactions:"
        )
        team_member_list_transactions = self._remove_prefix_from_keys(
            team_member_list_transactions, "team_member_list_transactions:"
        )
        org_list_transactions = self._remove_prefix_from_keys(
            org_list_transactions, "org_list_transactions:"
        )

        return DBSpendUpdateTransactions(
            user_list_transactions=user_list_transactions,
            end_user_list_transactions=end_user_list_transactions,
            key_list_transactions=key_list_transactions,
            team_list_transactions=team_list_transactions,
            team_member_list_transactions=team_member_list_transactions,
            org_list_transactions=org_list_transactions,
        )
