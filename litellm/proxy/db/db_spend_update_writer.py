"""
Module responsible for

1. Writing spend increments to either in memory list of transactions or to redis
2. Reading increments from redis or in memory list of transactions and committing them to db
"""

import asyncio
import json
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache, RedisCache
from litellm.constants import DB_SPEND_UPDATE_JOB_NAME
from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    BaseDailySpendTransaction,
    DailyTeamSpendTransaction,
    DailyUserSpendTransaction,
    DBSpendUpdateTransactions,
    Litellm_EntityType,
    LiteLLM_UserTable,
    SpendLogsMetadata,
    SpendLogsPayload,
    SpendUpdateQueueItem,
)
from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import (
    DailySpendUpdateQueue,
)
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging
else:
    PrismaClient = Any
    ProxyLogging = Any


class DBSpendUpdateWriter:
    """
    Module responsible for

    1. Writing spend increments to either in memory list of transactions or to redis
    2. Reading increments from redis or in memory list of transactions and committing them to db
    """

    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
    ):
        self.redis_cache = redis_cache
        self.redis_update_buffer = RedisUpdateBuffer(redis_cache=self.redis_cache)
        self.pod_lock_manager = PodLockManager()
        self.spend_update_queue = SpendUpdateQueue()
        self.daily_spend_update_queue = DailySpendUpdateQueue()
        self.daily_team_spend_update_queue = DailySpendUpdateQueue()

    async def update_database(
        # LiteLLM management object fields
        self,
        token: Optional[str],
        user_id: Optional[str],
        end_user_id: Optional[str],
        team_id: Optional[str],
        org_id: Optional[str],
        # Completion object fields
        kwargs: Optional[dict],
        completion_response: Optional[Union[litellm.ModelResponse, Any, Exception]],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        response_cost: Optional[float],
    ):
        from litellm.proxy.proxy_server import (
            disable_spend_logs,
            litellm_proxy_budget_name,
            prisma_client,
            user_api_key_cache,
        )
        from litellm.proxy.utils import ProxyUpdateSpend, hash_token

        try:
            verbose_proxy_logger.debug(
                f"Enters prisma db call, response_cost: {response_cost}, token: {token}; user_id: {user_id}; team_id: {team_id}"
            )
            if ProxyUpdateSpend.disable_spend_updates() is True:
                return
            if token is not None and isinstance(token, str) and token.startswith("sk-"):
                hashed_token = hash_token(token=token)
            else:
                hashed_token = token

            ## CREATE SPEND LOG PAYLOAD ##
            from litellm.proxy.spend_tracking.spend_tracking_utils import (
                get_logging_payload,
            )

            payload = get_logging_payload(
                kwargs=kwargs,
                response_obj=completion_response,
                start_time=start_time,
                end_time=end_time,
            )
            payload["spend"] = response_cost or 0.0
            if isinstance(payload["startTime"], datetime):
                payload["startTime"] = payload["startTime"].isoformat()
            if isinstance(payload["endTime"], datetime):
                payload["endTime"] = payload["endTime"].isoformat()

            asyncio.create_task(
                self._update_user_db(
                    response_cost=response_cost,
                    user_id=user_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    litellm_proxy_budget_name=litellm_proxy_budget_name,
                    end_user_id=end_user_id,
                )
            )
            asyncio.create_task(
                self._update_key_db(
                    response_cost=response_cost,
                    hashed_token=hashed_token,
                    prisma_client=prisma_client,
                )
            )
            asyncio.create_task(
                self._update_team_db(
                    response_cost=response_cost,
                    team_id=team_id,
                    user_id=user_id,
                    prisma_client=prisma_client,
                )
            )
            asyncio.create_task(
                self._update_org_db(
                    response_cost=response_cost,
                    org_id=org_id,
                    prisma_client=prisma_client,
                )
            )

            if disable_spend_logs is False:
                await self._insert_spend_log_to_db(
                    payload=payload,
                    prisma_client=prisma_client,
                )
            else:
                verbose_proxy_logger.info(
                    "disable_spend_logs=True. Skipping writing spend logs to db. Other spend updates - Key/User/Team table will still occur."
                )

            asyncio.create_task(
                self.add_spend_log_transaction_to_daily_user_transaction(
                    payload=payload,
                    prisma_client=prisma_client,
                )
            )

            asyncio.create_task(
                self.add_spend_log_transaction_to_daily_team_transaction(
                    payload=payload,
                    prisma_client=prisma_client,
                )
            )

            verbose_proxy_logger.debug("Runs spend update on all tables")
        except Exception:
            verbose_proxy_logger.debug(
                f"Error updating Prisma database: {traceback.format_exc()}"
            )

    async def _update_key_db(
        self,
        response_cost: Optional[float],
        hashed_token: Optional[str],
        prisma_client: Optional[PrismaClient],
    ):
        try:
            if hashed_token is None or prisma_client is None:
                return

            await self.spend_update_queue.add_update(
                update=SpendUpdateQueueItem(
                    entity_type=Litellm_EntityType.KEY,
                    entity_id=hashed_token,
                    response_cost=response_cost,
                )
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Update Key DB Call failed to execute - {str(e)}"
            )
            raise e

    async def _update_user_db(
        self,
        response_cost: Optional[float],
        user_id: Optional[str],
        prisma_client: Optional[PrismaClient],
        user_api_key_cache: DualCache,
        litellm_proxy_budget_name: Optional[str],
        end_user_id: Optional[str] = None,
    ):
        """
        - Update that user's row
        - Update litellm-proxy-budget row (global proxy spend)
        """
        ## if an end-user is passed in, do an upsert - we can't guarantee they already exist in db
        existing_user_obj = await user_api_key_cache.async_get_cache(key=user_id)
        if existing_user_obj is not None and isinstance(existing_user_obj, dict):
            existing_user_obj = LiteLLM_UserTable(**existing_user_obj)
        try:
            if prisma_client is not None:  # update
                user_ids = [user_id]
                if (
                    litellm.max_budget > 0
                ):  # track global proxy budget, if user set max budget
                    user_ids.append(litellm_proxy_budget_name)

                for _id in user_ids:
                    if _id is not None:
                        await self.spend_update_queue.add_update(
                            update=SpendUpdateQueueItem(
                                entity_type=Litellm_EntityType.USER,
                                entity_id=_id,
                                response_cost=response_cost,
                            )
                        )

                if end_user_id is not None:
                    await self.spend_update_queue.add_update(
                        update=SpendUpdateQueueItem(
                            entity_type=Litellm_EntityType.END_USER,
                            entity_id=end_user_id,
                            response_cost=response_cost,
                        )
                    )
        except Exception as e:
            verbose_proxy_logger.info(
                "\033[91m"
                + f"Update User DB call failed to execute {str(e)}\n{traceback.format_exc()}"
            )

    async def _update_team_db(
        self,
        response_cost: Optional[float],
        team_id: Optional[str],
        user_id: Optional[str],
        prisma_client: Optional[PrismaClient],
    ):
        try:
            if team_id is None or prisma_client is None:
                verbose_proxy_logger.debug(
                    "track_cost_callback: team_id is None or prisma_client is None. Not tracking spend for team"
                )
                return

            await self.spend_update_queue.add_update(
                update=SpendUpdateQueueItem(
                    entity_type=Litellm_EntityType.TEAM,
                    entity_id=team_id,
                    response_cost=response_cost,
                )
            )

            try:
                # Track spend of the team member within this team
                if user_id is not None:
                    # key is "team_id::<value>::user_id::<value>"
                    team_member_key = f"team_id::{team_id}::user_id::{user_id}"
                    await self.spend_update_queue.add_update(
                        update=SpendUpdateQueueItem(
                            entity_type=Litellm_EntityType.TEAM_MEMBER,
                            entity_id=team_member_key,
                            response_cost=response_cost,
                        )
                    )
            except Exception:
                pass
        except Exception as e:
            verbose_proxy_logger.info(
                f"Update Team DB failed to execute - {str(e)}\n{traceback.format_exc()}"
            )
            raise e

    async def _update_org_db(
        self,
        response_cost: Optional[float],
        org_id: Optional[str],
        prisma_client: Optional[PrismaClient],
    ):
        try:
            if org_id is None or prisma_client is None:
                verbose_proxy_logger.debug(
                    "track_cost_callback: org_id is None or prisma_client is None. Not tracking spend for org"
                )
                return

            await self.spend_update_queue.add_update(
                update=SpendUpdateQueueItem(
                    entity_type=Litellm_EntityType.ORGANIZATION,
                    entity_id=org_id,
                    response_cost=response_cost,
                )
            )
        except Exception as e:
            verbose_proxy_logger.info(
                f"Update Org DB failed to execute - {str(e)}\n{traceback.format_exc()}"
            )
            raise e

    async def _insert_spend_log_to_db(
        self,
        payload: Union[dict, SpendLogsPayload],
        prisma_client: Optional[PrismaClient] = None,
        spend_logs_url: Optional[str] = os.getenv("SPEND_LOGS_URL"),
    ) -> Optional[PrismaClient]:
        verbose_proxy_logger.info(
            "Writing spend log to db - request_id: {}, spend: {}".format(
                payload.get("request_id"), payload.get("spend")
            )
        )
        if prisma_client is not None and spend_logs_url is not None:
            prisma_client.spend_log_transactions.append(payload)
        elif prisma_client is not None:
            prisma_client.spend_log_transactions.append(payload)
        else:
            verbose_proxy_logger.debug(
                "prisma_client is None. Skipping writing spend logs to db."
            )

        return prisma_client

    async def db_update_spend_transaction_handler(
        self,
        prisma_client: PrismaClient,
        n_retry_times: int,
        proxy_logging_obj: ProxyLogging,
    ):
        """
        Handles commiting update spend transactions to db

        `UPDATES` can lead to deadlocks, hence we handle them separately

        Args:
            prisma_client: PrismaClient object
            n_retry_times: int, number of retry times
            proxy_logging_obj: ProxyLogging object

        How this works:
        - Check `general_settings.use_redis_transaction_buffer`
            - If enabled, write in-memory transactions to Redis
            - Check if this Pod should read from the DB
        else:
            - Regular flow of this method
        """
        if RedisUpdateBuffer._should_commit_spend_updates_to_redis():
            await self._commit_spend_updates_to_db_with_redis(
                prisma_client=prisma_client,
                n_retry_times=n_retry_times,
                proxy_logging_obj=proxy_logging_obj,
            )

        else:
            await self._commit_spend_updates_to_db_without_redis_buffer(
                prisma_client=prisma_client,
                n_retry_times=n_retry_times,
                proxy_logging_obj=proxy_logging_obj,
            )

    async def _commit_spend_updates_to_db_with_redis(
        self,
        prisma_client: PrismaClient,
        n_retry_times: int,
        proxy_logging_obj: ProxyLogging,
    ):
        """
        Handler to commit spend updates to Redis and attempt to acquire lock to commit to db

        This is a v2 scalable approach to first commit spend updates to redis, then commit to db

        This minimizes DB Deadlocks since
            - All pods only need to write their spend updates to redis
            - Only 1 pod will commit to db at a time (based on if it can acquire the lock over writing to DB)
        """
        await self.redis_update_buffer.store_in_memory_spend_updates_in_redis(
            spend_update_queue=self.spend_update_queue,
            daily_spend_update_queue=self.daily_spend_update_queue,
            daily_team_spend_update_queue=self.daily_team_spend_update_queue,
        )

        # Only commit from redis to db if this pod is the leader
        if await self.pod_lock_manager.acquire_lock(
            cronjob_id=DB_SPEND_UPDATE_JOB_NAME,
        ):
            verbose_proxy_logger.debug("acquired lock for spend updates")

            try:
                db_spend_update_transactions = (
                    await self.redis_update_buffer.get_all_update_transactions_from_redis_buffer()
                )
                if db_spend_update_transactions is not None:
                    await self._commit_spend_updates_to_db(
                        prisma_client=prisma_client,
                        n_retry_times=n_retry_times,
                        proxy_logging_obj=proxy_logging_obj,
                        db_spend_update_transactions=db_spend_update_transactions,
                    )

                daily_spend_update_transactions = (
                    await self.redis_update_buffer.get_all_daily_spend_update_transactions_from_redis_buffer()
                )
                if daily_spend_update_transactions is not None:
                    await DBSpendUpdateWriter.update_daily_user_spend(
                        n_retry_times=n_retry_times,
                        prisma_client=prisma_client,
                        proxy_logging_obj=proxy_logging_obj,
                        daily_spend_transactions=daily_spend_update_transactions,
                    )
                daily_team_spend_update_transactions = (
                    await self.redis_update_buffer.get_all_daily_team_spend_update_transactions_from_redis_buffer()
                )
                if daily_team_spend_update_transactions is not None:
                    await DBSpendUpdateWriter.update_daily_team_spend(
                        n_retry_times=n_retry_times,
                        prisma_client=prisma_client,
                        proxy_logging_obj=proxy_logging_obj,
                        daily_spend_transactions=daily_team_spend_update_transactions,
                    )
            except Exception as e:
                verbose_proxy_logger.error(f"Error committing spend updates: {e}")
            finally:
                await self.pod_lock_manager.release_lock(
                    cronjob_id=DB_SPEND_UPDATE_JOB_NAME,
                )

    async def _commit_spend_updates_to_db_without_redis_buffer(
        self,
        prisma_client: PrismaClient,
        n_retry_times: int,
        proxy_logging_obj: ProxyLogging,
    ):
        """
        Commits all the spend `UPDATE` transactions to the Database

        This is the regular flow of committing to db without using a redis buffer

        Note: This flow causes Deadlocks in production (1K RPS+). Use self._commit_spend_updates_to_db_with_redis() instead if you expect 1K+ RPS.
        """

        # Aggregate all in memory spend updates (key, user, end_user, team, team_member, org) and commit to db
        ################## Spend Update Transactions ##################
        db_spend_update_transactions = (
            await self.spend_update_queue.flush_and_get_aggregated_db_spend_update_transactions()
        )
        await self._commit_spend_updates_to_db(
            prisma_client=prisma_client,
            n_retry_times=n_retry_times,
            proxy_logging_obj=proxy_logging_obj,
            db_spend_update_transactions=db_spend_update_transactions,
        )

        ################## Daily Spend Update Transactions ##################
        # Aggregate all in memory daily spend transactions and commit to db
        daily_spend_update_transactions = cast(
            Dict[str, DailyUserSpendTransaction],
            await self.daily_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions(),
        )

        await DBSpendUpdateWriter.update_daily_user_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_spend_update_transactions,
        )

        ################## Daily Team Spend Update Transactions ##################
        # Aggregate all in memory daily team spend transactions and commit to db
        daily_team_spend_update_transactions = cast(
            Dict[str, DailyTeamSpendTransaction],
            await self.daily_team_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions(),
        )

        await DBSpendUpdateWriter.update_daily_team_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_team_spend_update_transactions,
        )

    async def _commit_spend_updates_to_db(  # noqa: PLR0915
        self,
        prisma_client: PrismaClient,
        n_retry_times: int,
        proxy_logging_obj: ProxyLogging,
        db_spend_update_transactions: DBSpendUpdateTransactions,
    ):
        """
        Commits all the spend `UPDATE` transactions to the Database

        """
        from litellm.proxy.utils import (
            ProxyUpdateSpend,
            _raise_failed_update_spend_exception,
        )

        ### UPDATE USER TABLE ###
        user_list_transactions = db_spend_update_transactions["user_list_transactions"]
        verbose_proxy_logger.debug(
            "User Spend transactions: {}".format(user_list_transactions)
        )
        if (
            user_list_transactions is not None
            and len(user_list_transactions.keys()) > 0
        ):
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with prisma_client.db.tx(
                        timeout=timedelta(seconds=60)
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            for (
                                user_id,
                                response_cost,
                            ) in user_list_transactions.items():
                                batcher.litellm_usertable.update_many(
                                    where={"user_id": user_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        _raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    _raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE END-USER TABLE ###
        end_user_list_transactions = db_spend_update_transactions[
            "end_user_list_transactions"
        ]
        verbose_proxy_logger.debug(
            "End-User Spend transactions: {}".format(end_user_list_transactions)
        )
        if (
            end_user_list_transactions is not None
            and len(end_user_list_transactions.keys()) > 0
        ):
            await ProxyUpdateSpend.update_end_user_spend(
                n_retry_times=n_retry_times,
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
                end_user_list_transactions=end_user_list_transactions,
            )
        ### UPDATE KEY TABLE ###
        key_list_transactions = db_spend_update_transactions["key_list_transactions"]
        verbose_proxy_logger.debug(
            "KEY Spend transactions: {}".format(key_list_transactions)
        )
        if key_list_transactions is not None and len(key_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with prisma_client.db.tx(
                        timeout=timedelta(seconds=60)
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            for (
                                token,
                                response_cost,
                            ) in key_list_transactions.items():
                                batcher.litellm_verificationtoken.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"token": token},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        _raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    _raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE TEAM TABLE ###
        team_list_transactions = db_spend_update_transactions["team_list_transactions"]
        verbose_proxy_logger.debug(
            "Team Spend transactions: {}".format(team_list_transactions)
        )
        if (
            team_list_transactions is not None
            and len(team_list_transactions.keys()) > 0
        ):
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with prisma_client.db.tx(
                        timeout=timedelta(seconds=60)
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            for (
                                team_id,
                                response_cost,
                            ) in team_list_transactions.items():
                                verbose_proxy_logger.debug(
                                    "Updating spend for team id={} by {}".format(
                                        team_id, response_cost
                                    )
                                )
                                batcher.litellm_teamtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"team_id": team_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        _raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    _raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE TEAM Membership TABLE with spend ###
        team_member_list_transactions = db_spend_update_transactions[
            "team_member_list_transactions"
        ]
        verbose_proxy_logger.debug(
            "Team Membership Spend transactions: {}".format(
                team_member_list_transactions
            )
        )
        if (
            team_member_list_transactions is not None
            and len(team_member_list_transactions.keys()) > 0
        ):
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with prisma_client.db.tx(
                        timeout=timedelta(seconds=60)
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            for (
                                key,
                                response_cost,
                            ) in team_member_list_transactions.items():
                                # key is "team_id::<value>::user_id::<value>"
                                team_id = key.split("::")[1]
                                user_id = key.split("::")[3]

                                batcher.litellm_teammembership.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"team_id": team_id, "user_id": user_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        _raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    _raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE ORG TABLE ###
        org_list_transactions = db_spend_update_transactions["org_list_transactions"]
        verbose_proxy_logger.debug(
            "Org Spend transactions: {}".format(org_list_transactions)
        )
        if org_list_transactions is not None and len(org_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with prisma_client.db.tx(
                        timeout=timedelta(seconds=60)
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            for (
                                org_id,
                                response_cost,
                            ) in org_list_transactions.items():
                                batcher.litellm_organizationtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"organization_id": org_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        _raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    _raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

    @staticmethod
    async def update_daily_user_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyUserSpendTransaction],
    ):
        """
        Batch job to update LiteLLM_DailyUserSpend table using in-memory daily_spend_transactions
        """
        from litellm.proxy.utils import _raise_failed_update_spend_exception

        ### UPDATE DAILY USER SPEND ###
        verbose_proxy_logger.debug(
            "Daily User Spend transactions: {}".format(len(daily_spend_transactions))
        )
        BATCH_SIZE = (
            100  # Number of aggregated records to update in each database operation
        )
        start_time = time.time()

        try:
            for i in range(n_retry_times + 1):
                try:
                    # Get transactions to process
                    transactions_to_process = dict(
                        list(daily_spend_transactions.items())[:BATCH_SIZE]
                    )

                    if len(transactions_to_process) == 0:
                        verbose_proxy_logger.debug(
                            "No new transactions to process for daily spend update"
                        )
                        break

                    # Update DailyUserSpend table in batches
                    async with prisma_client.db.batch_() as batcher:
                        for _, transaction in transactions_to_process.items():
                            user_id = transaction.get("user_id")
                            if not user_id:  # Skip if no user_id
                                continue

                            batcher.litellm_dailyuserspend.upsert(
                                where={
                                    "user_id_date_api_key_model_custom_llm_provider": {
                                        "user_id": user_id,
                                        "date": transaction["date"],
                                        "api_key": transaction["api_key"],
                                        "model": transaction["model"],
                                        "custom_llm_provider": transaction.get(
                                            "custom_llm_provider"
                                        ),
                                    }
                                },
                                data={
                                    "create": {
                                        "user_id": user_id,
                                        "date": transaction["date"],
                                        "api_key": transaction["api_key"],
                                        "model": transaction["model"],
                                        "model_group": transaction.get("model_group"),
                                        "custom_llm_provider": transaction.get(
                                            "custom_llm_provider"
                                        ),
                                        "prompt_tokens": transaction["prompt_tokens"],
                                        "completion_tokens": transaction[
                                            "completion_tokens"
                                        ],
                                        "cache_read_input_tokens": transaction.get(
                                            "cache_read_input_tokens", 0
                                        ),
                                        "cache_creation_input_tokens": transaction.get(
                                            "cache_creation_input_tokens", 0
                                        ),
                                        "spend": transaction["spend"],
                                        "api_requests": transaction["api_requests"],
                                        "successful_requests": transaction[
                                            "successful_requests"
                                        ],
                                        "failed_requests": transaction[
                                            "failed_requests"
                                        ],
                                    },
                                    "update": {
                                        "prompt_tokens": {
                                            "increment": transaction["prompt_tokens"]
                                        },
                                        "completion_tokens": {
                                            "increment": transaction[
                                                "completion_tokens"
                                            ]
                                        },
                                        "cache_read_input_tokens": {
                                            "increment": transaction.get(
                                                "cache_read_input_tokens", 0
                                            )
                                        },
                                        "cache_creation_input_tokens": {
                                            "increment": transaction.get(
                                                "cache_creation_input_tokens", 0
                                            )
                                        },
                                        "spend": {"increment": transaction["spend"]},
                                        "api_requests": {
                                            "increment": transaction["api_requests"]
                                        },
                                        "successful_requests": {
                                            "increment": transaction[
                                                "successful_requests"
                                            ]
                                        },
                                        "failed_requests": {
                                            "increment": transaction["failed_requests"]
                                        },
                                    },
                                },
                            )

                    verbose_proxy_logger.info(
                        f"Processed {len(transactions_to_process)} daily spend transactions in {time.time() - start_time:.2f}s"
                    )

                    # Remove processed transactions
                    for key in transactions_to_process.keys():
                        daily_spend_transactions.pop(key, None)

                    verbose_proxy_logger.debug(
                        f"Processed {len(transactions_to_process)} daily spend transactions in {time.time() - start_time:.2f}s"
                    )
                    break

                except DB_CONNECTION_ERROR_TYPES as e:
                    if i >= n_retry_times:
                        _raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    await asyncio.sleep(2**i)  # Exponential backoff

        except Exception as e:
            # Remove processed transactions even if there was an error
            if "transactions_to_process" in locals():
                for key in transactions_to_process.keys():  # type: ignore
                    daily_spend_transactions.pop(key, None)
            _raise_failed_update_spend_exception(
                e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
            )

    @staticmethod
    async def update_daily_team_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyTeamSpendTransaction],
    ):
        """
        Batch job to update LiteLLM_DailyTeamSpend table using in-memory daily_spend_transactions
        """
        from litellm.proxy.utils import _raise_failed_update_spend_exception

        ### UPDATE DAILY USER SPEND ###
        verbose_proxy_logger.debug(
            "Daily Team Spend transactions: {}".format(len(daily_spend_transactions))
        )
        BATCH_SIZE = (
            100  # Number of aggregated records to update in each database operation
        )
        start_time = time.time()

        try:
            for i in range(n_retry_times + 1):
                try:
                    # Get transactions to process
                    transactions_to_process = dict(
                        list(daily_spend_transactions.items())[:BATCH_SIZE]
                    )

                    if len(transactions_to_process) == 0:
                        verbose_proxy_logger.debug(
                            "No new transactions to process for daily spend update"
                        )
                        break

                    # Update DailyUserSpend table in batches
                    async with prisma_client.db.batch_() as batcher:
                        for _, transaction in transactions_to_process.items():
                            team_id = transaction.get("team_id")
                            if not team_id:  # Skip if no team_id
                                continue

                            batcher.litellm_dailyteamspend.upsert(
                                where={
                                    "team_id_date_api_key_model_custom_llm_provider": {
                                        "team_id": team_id,
                                        "date": transaction["date"],
                                        "api_key": transaction["api_key"],
                                        "model": transaction["model"],
                                        "custom_llm_provider": transaction.get(
                                            "custom_llm_provider"
                                        ),
                                    }
                                },
                                data={
                                    "create": {
                                        "team_id": team_id,
                                        "date": transaction["date"],
                                        "api_key": transaction["api_key"],
                                        "model": transaction["model"],
                                        "model_group": transaction.get("model_group"),
                                        "custom_llm_provider": transaction.get(
                                            "custom_llm_provider"
                                        ),
                                        "prompt_tokens": transaction["prompt_tokens"],
                                        "completion_tokens": transaction[
                                            "completion_tokens"
                                        ],
                                        "spend": transaction["spend"],
                                        "api_requests": transaction["api_requests"],
                                        "successful_requests": transaction[
                                            "successful_requests"
                                        ],
                                        "failed_requests": transaction[
                                            "failed_requests"
                                        ],
                                    },
                                    "update": {
                                        "prompt_tokens": {
                                            "increment": transaction["prompt_tokens"]
                                        },
                                        "completion_tokens": {
                                            "increment": transaction[
                                                "completion_tokens"
                                            ]
                                        },
                                        "spend": {"increment": transaction["spend"]},
                                        "api_requests": {
                                            "increment": transaction["api_requests"]
                                        },
                                        "successful_requests": {
                                            "increment": transaction[
                                                "successful_requests"
                                            ]
                                        },
                                        "failed_requests": {
                                            "increment": transaction["failed_requests"]
                                        },
                                    },
                                },
                            )

                    verbose_proxy_logger.info(
                        f"Processed {len(transactions_to_process)} daily team transactions in {time.time() - start_time:.2f}s"
                    )

                    # Remove processed transactions
                    for key in transactions_to_process.keys():
                        daily_spend_transactions.pop(key, None)

                    verbose_proxy_logger.debug(
                        f"Processed {len(transactions_to_process)} daily spend transactions in {time.time() - start_time:.2f}s"
                    )
                    break

                except DB_CONNECTION_ERROR_TYPES as e:
                    if i >= n_retry_times:
                        _raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    await asyncio.sleep(2**i)  # Exponential backoff

        except Exception as e:
            # Remove processed transactions even if there was an error
            if "transactions_to_process" in locals():
                for key in transactions_to_process.keys():  # type: ignore
                    daily_spend_transactions.pop(key, None)
            _raise_failed_update_spend_exception(
                e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
            )

    async def _common_add_spend_log_transaction_to_daily_transaction(
        self,
        payload: Union[dict, SpendLogsPayload],
        prisma_client: PrismaClient,
        type: Literal["user", "team"] = "user",
    ) -> Optional[BaseDailySpendTransaction]:
        common_expected_keys = ["startTime", "api_key", "model", "custom_llm_provider"]
        if type == "user":
            expected_keys = ["user", *common_expected_keys]
        else:
            expected_keys = ["team_id", *common_expected_keys]

        if not all(key in payload for key in expected_keys):
            verbose_proxy_logger.debug(
                f"Missing expected keys: {expected_keys}, in payload, skipping from daily_user_spend_transactions"
            )
            return None

        request_status = prisma_client.get_request_status(payload)
        verbose_proxy_logger.info(f"Logged request status: {request_status}")
        _metadata: SpendLogsMetadata = json.loads(payload["metadata"])
        usage_obj = _metadata.get("usage_object", {}) or {}
        if isinstance(payload["startTime"], datetime):
            start_time = payload["startTime"].isoformat()
            date = start_time.split("T")[0]
        elif isinstance(payload["startTime"], str):
            date = payload["startTime"].split("T")[0]
        else:
            verbose_proxy_logger.debug(
                f"Invalid start time: {payload['startTime']}, skipping from daily_user_spend_transactions"
            )
            return None
        try:
            daily_transaction = BaseDailySpendTransaction(
                date=date,
                api_key=payload["api_key"],
                model=payload["model"],
                model_group=payload["model_group"],
                custom_llm_provider=payload["custom_llm_provider"],
                prompt_tokens=payload["prompt_tokens"],
                completion_tokens=payload["completion_tokens"],
                spend=payload["spend"],
                api_requests=1,
                successful_requests=1 if request_status == "success" else 0,
                failed_requests=1 if request_status != "success" else 0,
                cache_read_input_tokens=usage_obj.get("cache_read_input_tokens", 0)
                or 0,
                cache_creation_input_tokens=usage_obj.get(
                    "cache_creation_input_tokens", 0
                )
                or 0,
            )
            return daily_transaction
        except Exception as e:
            raise e

    async def add_spend_log_transaction_to_daily_user_transaction(
        self,
        payload: Union[dict, SpendLogsPayload],
        prisma_client: Optional[PrismaClient] = None,
    ):
        """
        Add a spend log transaction to the `daily_spend_update_queue`

        Key = @@unique([user_id, date, api_key, model, custom_llm_provider])    )

        If key exists, update the transaction with the new spend and usage
        """
        if prisma_client is None:
            verbose_proxy_logger.debug(
                "prisma_client is None. Skipping writing spend logs to db."
            )
            return

        base_daily_transaction = (
            await self._common_add_spend_log_transaction_to_daily_transaction(
                payload, prisma_client, "user"
            )
        )
        if base_daily_transaction is None:
            return

        daily_transaction_key = f"{payload['user']}_{base_daily_transaction['date']}_{payload['api_key']}_{payload['model']}_{payload['custom_llm_provider']}"
        daily_transaction = DailyUserSpendTransaction(
            user_id=payload["user"], **base_daily_transaction
        )
        await self.daily_spend_update_queue.add_update(
            update={daily_transaction_key: daily_transaction}
        )

    async def add_spend_log_transaction_to_daily_team_transaction(
        self,
        payload: SpendLogsPayload,
        prisma_client: Optional[PrismaClient] = None,
    ) -> None:
        if prisma_client is None:
            verbose_proxy_logger.debug(
                "prisma_client is None. Skipping writing spend logs to db."
            )
            return

        base_daily_transaction = (
            await self._common_add_spend_log_transaction_to_daily_transaction(
                payload, prisma_client, "team"
            )
        )
        if base_daily_transaction is None:
            return
        if payload["team_id"] is None:
            verbose_proxy_logger.debug(
                "team_id is None for request. Skipping incrementing team spend."
            )
            return

        daily_transaction_key = f"{payload['team_id']}_{base_daily_transaction['date']}_{payload['api_key']}_{payload['model']}_{payload['custom_llm_provider']}"
        daily_transaction = DailyTeamSpendTransaction(
            team_id=payload["team_id"], **base_daily_transaction
        )
        await self.daily_team_spend_update_queue.add_update(
            update={daily_transaction_key: daily_transaction}
        )
