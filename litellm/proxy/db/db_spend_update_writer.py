"""
Module responsible for

1. Writing spend increments to either in memory list of transactions or to redis
2. Reading increments from redis or in memory list of transactions and committing them to db
"""

import asyncio
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache, RedisCache
from litellm.constants import DB_SPEND_UPDATE_JOB_NAME
from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    DBSpendUpdateTransactions,
    Litellm_EntityType,
    LiteLLM_UserTable,
    SpendLogsPayload,
    SpendUpdateQueueItem,
)
from litellm.proxy.db.pod_lock_manager import PodLockManager
from litellm.proxy.db.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.spend_update_queue import SpendUpdateQueue

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
        self.pod_lock_manager = PodLockManager(cronjob_id=DB_SPEND_UPDATE_JOB_NAME)
        self.spend_update_queue = SpendUpdateQueue()

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
                await DBSpendUpdateWriter._insert_spend_log_to_db(
                    kwargs=kwargs,
                    completion_response=completion_response,
                    start_time=start_time,
                    end_time=end_time,
                    response_cost=response_cost,
                    prisma_client=prisma_client,
                )
            else:
                verbose_proxy_logger.info(
                    "disable_spend_logs=True. Skipping writing spend logs to db. Other spend updates - Key/User/Team table will still occur."
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

    @staticmethod
    async def _insert_spend_log_to_db(
        kwargs: Optional[dict],
        completion_response: Optional[Union[litellm.ModelResponse, Any, Exception]],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        response_cost: Optional[float],
        prisma_client: Optional[PrismaClient],
    ):
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            get_logging_payload,
        )

        try:
            if prisma_client:
                payload = get_logging_payload(
                    kwargs=kwargs,
                    response_obj=completion_response,
                    start_time=start_time,
                    end_time=end_time,
                )
                payload["spend"] = response_cost or 0.0
                DBSpendUpdateWriter._set_spend_logs_payload(
                    payload=payload,
                    spend_logs_url=os.getenv("SPEND_LOGS_URL"),
                    prisma_client=prisma_client,
                )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Update Spend Logs DB failed to execute - {str(e)}\n{traceback.format_exc()}"
            )
            raise e

    @staticmethod
    def _set_spend_logs_payload(
        payload: Union[dict, SpendLogsPayload],
        prisma_client: PrismaClient,
        spend_logs_url: Optional[str] = None,
    ) -> PrismaClient:
        verbose_proxy_logger.info(
            "Writing spend log to db - request_id: {}, spend: {}".format(
                payload.get("request_id"), payload.get("spend")
            )
        )
        if prisma_client is not None and spend_logs_url is not None:
            if isinstance(payload["startTime"], datetime):
                payload["startTime"] = payload["startTime"].isoformat()
            if isinstance(payload["endTime"], datetime):
                payload["endTime"] = payload["endTime"].isoformat()
            prisma_client.spend_log_transactions.append(payload)
        elif prisma_client is not None:
            prisma_client.spend_log_transactions.append(payload)

        prisma_client.add_spend_log_transaction_to_daily_user_transaction(
            payload.copy()
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
        )

        # Only commit from redis to db if this pod is the leader
        if await self.pod_lock_manager.acquire_lock():
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
            except Exception as e:
                verbose_proxy_logger.error(f"Error committing spend updates: {e}")
            finally:
                await self.pod_lock_manager.release_lock()

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
        db_spend_update_transactions = (
            await self.spend_update_queue.flush_and_get_aggregated_db_spend_update_transactions()
        )
        await self._commit_spend_updates_to_db(
            prisma_client=prisma_client,
            n_retry_times=n_retry_times,
            proxy_logging_obj=proxy_logging_obj,
            db_spend_update_transactions=db_spend_update_transactions,
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
