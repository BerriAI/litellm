"""
Module responsible for

1. Writing spend increments to either in memory list of transactions or to redis
2. Reading increments from redis or in memory list of transactions and committing them to db
"""

import asyncio
import copy
import json
import os
import random
import time
import traceback
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union, cast, overload

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache, RedisCache
from litellm.constants import DB_SPEND_UPDATE_JOB_NAME
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    BaseDailySpendTransaction,
    DailyTagSpendTransaction,
    DailyOrganizationSpendTransaction,
    DailyTeamSpendTransaction,
    DailyEndUserSpendTransaction,
    DailyUserSpendTransaction,
    DailyAgentSpendTransaction,
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
        self.daily_end_user_spend_update_queue = DailySpendUpdateQueue()
        self.daily_agent_spend_update_queue = DailySpendUpdateQueue()
        self.daily_org_spend_update_queue = DailySpendUpdateQueue()
        self.daily_tag_spend_update_queue = DailySpendUpdateQueue()

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
            
            if org_id is not None and org_id != "":
                payload["organization_id"] = org_id

            if team_id is not None and team_id != "":
                payload["team_id"] = team_id

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
            asyncio.create_task(
                self._update_tag_db(
                    response_cost=response_cost,
                    request_tags=copy.deepcopy(payload.get("request_tags")),
                    prisma_client=prisma_client,
                )
            )

            if disable_spend_logs is False:
                await self._insert_spend_log_to_db(
                    payload=copy.deepcopy(payload),
                    prisma_client=prisma_client,
                )
            else:
                verbose_proxy_logger.debug(
                    "disable_spend_logs=True. Skipping writing spend logs to db. Other spend updates - Key/User/Team table will still occur."
                )

            asyncio.create_task(
                self.add_spend_log_transaction_to_daily_user_transaction(
                    payload=copy.deepcopy(payload),
                    prisma_client=prisma_client,
                )
            )

            asyncio.create_task(
                self.add_spend_log_transaction_to_daily_end_user_transaction(
                    payload=copy.deepcopy(payload),
                    prisma_client=prisma_client,
                )
            )

            asyncio.create_task(
                self.add_spend_log_transaction_to_daily_agent_transaction(
                    payload=payload,
                    prisma_client=prisma_client,
                )
            )

            asyncio.create_task(
                self.add_spend_log_transaction_to_daily_team_transaction(
                    payload=copy.deepcopy(payload),
                    prisma_client=prisma_client,
                )
            )
            asyncio.create_task(
                self.add_spend_log_transaction_to_daily_org_transaction(
                    payload=copy.deepcopy(payload),
                    org_id=org_id,
                    prisma_client=prisma_client,
                )
            )
            asyncio.create_task(
                self.add_spend_log_transaction_to_daily_tag_transaction(
                    payload=copy.deepcopy(payload),
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
            verbose_proxy_logger.debug(
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
            verbose_proxy_logger.debug(
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
            verbose_proxy_logger.debug(
                f"Update Org DB failed to execute - {str(e)}\n{traceback.format_exc()}"
            )
            raise e

    async def _update_tag_db(
        self,
        response_cost: Optional[float],
        request_tags: Optional[str],
        prisma_client: Optional[PrismaClient],
    ):
        """
        Update spend for all tags in the request.

        Args:
            response_cost: Cost of the request
            request_tags: JSON string of tags list e.g. '["prod-tag", "test-tag"]'
            prisma_client: Prisma client instance
        """
        try:
            if request_tags is None or prisma_client is None:
                return

            # Parse tags from JSON string
            tags = []
            if isinstance(request_tags, str):
                tags = safe_json_loads(request_tags, default=[])
                if not tags:
                    verbose_proxy_logger.debug(
                        f"Failed to parse request_tags JSON: {request_tags}"
                    )
                    return
            elif isinstance(request_tags, list):
                tags = request_tags
            else:
                return

            # Update spend for each tag
            for tag_name in tags:
                if tag_name and isinstance(tag_name, str):
                    await self.spend_update_queue.add_update(
                        update=SpendUpdateQueueItem(
                            entity_type=Litellm_EntityType.TAG,
                            entity_id=tag_name,
                            response_cost=response_cost,
                        )
                    )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Update Tag DB failed to execute - {str(e)}\n{traceback.format_exc()}"
            )
            raise e

    async def _insert_spend_log_to_db(
        self,
        payload: Union[dict, SpendLogsPayload],
        prisma_client: Optional[PrismaClient] = None,
        spend_logs_url: Optional[str] = os.getenv("SPEND_LOGS_URL"),
    ) -> Optional[PrismaClient]:
        verbose_proxy_logger.debug(
            "Writing spend log to db - request_id: {}, spend: {}".format(
                payload.get("request_id"), payload.get("spend")
            )
        )
        if prisma_client is not None and spend_logs_url is not None:
            async with prisma_client._spend_log_transactions_lock:
                prisma_client.spend_log_transactions.append(payload)
        elif prisma_client is not None:
            async with prisma_client._spend_log_transactions_lock:
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
            daily_org_spend_update_queue=self.daily_org_spend_update_queue,
            daily_end_user_spend_update_queue=self.daily_end_user_spend_update_queue,
            daily_agent_spend_update_queue=self.daily_agent_spend_update_queue,
            daily_tag_spend_update_queue=self.daily_tag_spend_update_queue,
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

                daily_org_spend_update_transactions = (
                    await self.redis_update_buffer.get_all_daily_org_spend_update_transactions_from_redis_buffer()
                )
                if daily_org_spend_update_transactions is not None:
                    await DBSpendUpdateWriter.update_daily_org_spend(
                        n_retry_times=n_retry_times,
                        prisma_client=prisma_client,
                        proxy_logging_obj=proxy_logging_obj,
                        daily_spend_transactions=daily_org_spend_update_transactions,
                    )

                daily_tag_spend_update_transactions = (
                    await self.redis_update_buffer.get_all_daily_tag_spend_update_transactions_from_redis_buffer()
                )
                if daily_tag_spend_update_transactions is not None:
                    await DBSpendUpdateWriter.update_daily_tag_spend(
                        n_retry_times=n_retry_times,
                        prisma_client=prisma_client,
                        proxy_logging_obj=proxy_logging_obj,
                        daily_spend_transactions=daily_tag_spend_update_transactions,
                    )
                daily_end_user_spend_update_transactions = (
                    await self.redis_update_buffer.get_all_daily_end_user_spend_update_transactions_from_redis_buffer()
                )
                if daily_end_user_spend_update_transactions is not None:
                    await DBSpendUpdateWriter.update_daily_end_user_spend(
                        n_retry_times=n_retry_times,
                        prisma_client=prisma_client,
                        proxy_logging_obj=proxy_logging_obj,
                        daily_spend_transactions=daily_end_user_spend_update_transactions,
                    )
                daily_agent_spend_update_transactions = (
                    await self.redis_update_buffer.get_all_daily_agent_spend_update_transactions_from_redis_buffer()
                )
                if daily_agent_spend_update_transactions is not None:
                    await DBSpendUpdateWriter.update_daily_agent_spend(
                        n_retry_times=n_retry_times,
                        prisma_client=prisma_client,
                        proxy_logging_obj=proxy_logging_obj,
                        daily_spend_transactions=daily_agent_spend_update_transactions,
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

        ################## Daily Organization Spend Update Transactions ##################
        # Aggregate all in memory daily org spend transactions and commit to db
        daily_org_spend_update_transactions = cast(
            Dict[str, DailyOrganizationSpendTransaction],
            await self.daily_org_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions(),
        )

        await DBSpendUpdateWriter.update_daily_org_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_org_spend_update_transactions,
        )

        ################## Daily Tag Spend Update Transactions ##################
        # Aggregate all in memory daily tag spend transactions and commit to db
        daily_tag_spend_update_transactions = cast(
            Dict[str, DailyTagSpendTransaction],
            await self.daily_tag_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions(),
        )

        await DBSpendUpdateWriter.update_daily_tag_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_tag_spend_update_transactions,
        )

        ################## Daily End-User Spend Update Transactions ##################
        # Aggregate all in memory daily end-user spend transactions and commit to db
        daily_end_user_spend_update_transactions = cast(
            Dict[str, DailyEndUserSpendTransaction],
            await self.daily_end_user_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions(),
        )

        await DBSpendUpdateWriter.update_daily_end_user_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_end_user_spend_update_transactions,
        )

        ################## Daily Agent Spend Update Transactions ##################
        # Aggregate all in memory daily agent spend transactions and commit to db
        daily_agent_spend_update_transactions = cast(
            Dict[str, DailyAgentSpendTransaction],
            await self.daily_agent_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions(),
        )

        await DBSpendUpdateWriter.update_daily_agent_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_agent_spend_update_transactions,
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
                    await asyncio.sleep(
                        # Sleep a random amount to avoid retrying and deadlocking again: when two transactions deadlock they are
                        # cancelled basically at the same time, so if they wait the same time they will also retry at the same time
                        # and thus they are more likely to deadlock again.
                        # Instead, we sleep a random amount so that they retry at slightly different times, lowering the chance of
                        # repeated deadlocks, and therefore of exceeding the retry limit.
                        random.uniform(2**i, 2 ** (i + 1))
                    )
                except Exception as e:
                    _raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE TAG TABLE ###
        tag_list_transactions = db_spend_update_transactions["tag_list_transactions"]
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Tag",
            transactions=tag_list_transactions,
            table_accessor="litellm_tagtable",
            where_field="tag_name",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )

    @staticmethod
    async def _update_entity_spend_in_db(
        entity_name: str,
        transactions: Optional[Dict[str, float]],
        table_accessor: Any,
        where_field: str,
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
    ):
        """
        Helper function to update spend for any entity type (team, org, tag, etc).

        Args:
            entity_name: Name of entity for logging (e.g., "Team", "Org", "Tag")
            transactions: Dictionary of {entity_id: response_cost}
            table_accessor: Prisma table accessor (e.g., prisma_client.db.litellm_teamtable)
            where_field: Field name for where clause (e.g., "team_id", "organization_id", "tag_name")
            n_retry_times: Number of retries on failure
            prisma_client: Prisma client instance
            proxy_logging_obj: Proxy logging object
        """
        from litellm.proxy.utils import _raise_failed_update_spend_exception

        verbose_proxy_logger.debug(f"{entity_name} Spend transactions: {transactions}")
        if transactions is not None and len(transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with prisma_client.db.tx(
                        timeout=timedelta(seconds=60)
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            for entity_id, response_cost in transactions.items():
                                verbose_proxy_logger.debug(
                                    f"Updating spend for {entity_name} {where_field}={entity_id} by {response_cost}"
                                )
                                getattr(batcher, table_accessor).update_many(
                                    where={where_field: entity_id},
                                    data={"spend": {"increment": response_cost}},
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
                    _raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

    # fmt: off

    @overload
    @staticmethod
    async def _update_daily_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyUserSpendTransaction],
        entity_type: Literal["user"],
        entity_id_field: str,
        table_name: str,
        unique_constraint_name: str,
    ) -> None: 
        ...

    @overload
    @staticmethod
    async def _update_daily_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyTeamSpendTransaction],
        entity_type: Literal["team"],
        entity_id_field: str,
        table_name: str,
        unique_constraint_name: str,
    ) -> None:
        ...

    @overload
    @staticmethod
    async def _update_daily_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyOrganizationSpendTransaction],
        entity_type: Literal["org"],
        entity_id_field: str,
        table_name: str,
        unique_constraint_name: str,
    ) -> None:
        ...

    @overload
    @staticmethod
    async def _update_daily_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyEndUserSpendTransaction],
        entity_type: Literal["end_user"],
        entity_id_field: str,
        table_name: str,
        unique_constraint_name: str,
    ) -> None:
        ...

    @overload
    @staticmethod
    async def _update_daily_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyAgentSpendTransaction],
        entity_type: Literal["agent"],
        entity_id_field: str,
        table_name: str,
        unique_constraint_name: str,
    ) -> None:
        ...

    @overload
    @staticmethod
    async def _update_daily_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyTagSpendTransaction],
        entity_type: Literal["tag"],
        entity_id_field: str,
        table_name: str,
        unique_constraint_name: str,
    ) -> None: 
        ...
    # fmt: on

    @staticmethod
    async def _update_daily_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Union[
            Dict[str, DailyUserSpendTransaction],
            Dict[str, DailyTeamSpendTransaction],
            Dict[str, DailyTagSpendTransaction],
            Dict[str, DailyOrganizationSpendTransaction],
            Dict[str, DailyEndUserSpendTransaction],
            Dict[str, DailyAgentSpendTransaction],
        ],
        entity_type: Literal["user", "team", "org", "tag", "end_user", "agent"],
        entity_id_field: str,
        table_name: str,
        unique_constraint_name: str,
    ) -> None:
        """
        Generic function to update daily spend for any entity type (user, team, org, tag, end_user, agent)
        """
        from litellm.proxy.utils import _raise_failed_update_spend_exception

        verbose_proxy_logger.debug(
            f"Daily {entity_type.capitalize()} Spend transactions: {len(daily_spend_transactions)}"
        )
        BATCH_SIZE = 100
        start_time = time.time()

        try:
            for i in range(n_retry_times + 1):
                try:
                    # Sort the transactions to minimize the probability of deadlocks by reducing the chance of concurrent
                    # trasactions locking the same rows/ranges in different orders.
                    transactions_to_process = dict(
                        sorted(
                            daily_spend_transactions.items(),
                            # Normally to avoid deadlocks we would sort by the index, but since we have sprinkled indexes
                            # on our schema like we're discount Salt Bae, we just sort by all fields that have an index,
                            # in an ad-hoc (but hopefully sensible) order of indexes. The actual ordering matters less than
                            # ensuring that all concurrent transactions sort in the same order.
                            # We could in theory use the dict key, as it contains basically the same fields, but this is more
                            # robust to future changes in the key format.
                            # If _update_daily_spend ever gets the ability to write to multiple tables at once, the sorting
                            # should sort by the table first.
                            key=lambda x: (
                                x[1].get("date") or "",
                                x[1].get(entity_id_field) or "",
                                x[1].get("api_key") or "",
                                x[1].get("model") or "",
                                x[1].get("custom_llm_provider") or "",
                            ),
                        )[:BATCH_SIZE]
                    )

                    if len(transactions_to_process) == 0:
                        verbose_proxy_logger.debug(
                            f"No new transactions to process for daily {entity_type} spend update"
                        )
                        break

                    async with prisma_client.db.batch_() as batcher:
                        for _, transaction in transactions_to_process.items():
                            entity_id = transaction.get(entity_id_field)

                            # Construct the where clause dynamically
                            where_clause = {
                                unique_constraint_name: {
                                    entity_id_field: entity_id,
                                    "date": transaction["date"],
                                    "api_key": transaction["api_key"],
                                    "model": transaction["model"],
                                    "custom_llm_provider": transaction.get(
                                        "custom_llm_provider"
                                    )
                                    or "",
                                    "mcp_namespaced_tool_name": transaction.get(
                                        "mcp_namespaced_tool_name"
                                    )
                                    or "",
                                }
                            }

                            # Get the table dynamically
                            table = getattr(batcher, table_name)

                            # Common data structure for both create and update
                            common_data = {
                                entity_id_field: entity_id,
                                "date": transaction["date"],
                                "api_key": transaction["api_key"],
                                "model": transaction.get("model"),
                                "model_group": transaction.get("model_group"),
                                "mcp_namespaced_tool_name": transaction.get(
                                    "mcp_namespaced_tool_name"
                                )
                                or "",
                                "custom_llm_provider": transaction.get(
                                    "custom_llm_provider"
                                ),
                                "prompt_tokens": transaction["prompt_tokens"],
                                "completion_tokens": transaction["completion_tokens"],
                                "spend": transaction["spend"],
                                "api_requests": transaction["api_requests"],
                                "successful_requests": transaction[
                                    "successful_requests"
                                ],
                                "failed_requests": transaction["failed_requests"],
                            }

                            # Add cache-related fields if they exist
                            if "cache_read_input_tokens" in transaction:
                                common_data["cache_read_input_tokens"] = (
                                    transaction.get("cache_read_input_tokens", 0)
                                )
                            if "cache_creation_input_tokens" in transaction:
                                common_data["cache_creation_input_tokens"] = (
                                    transaction.get("cache_creation_input_tokens", 0)
                                )

                            if entity_type == "tag" and "request_id" in transaction:
                                common_data["request_id"] = transaction.get(
                                    "request_id"
                                )

                            # Create update data structure
                            update_data = {
                                "prompt_tokens": {
                                    "increment": transaction["prompt_tokens"]
                                },
                                "completion_tokens": {
                                    "increment": transaction["completion_tokens"]
                                },
                                "spend": {"increment": transaction["spend"]},
                                "api_requests": {
                                    "increment": transaction["api_requests"]
                                },
                                "successful_requests": {
                                    "increment": transaction["successful_requests"]
                                },
                                "failed_requests": {
                                    "increment": transaction["failed_requests"]
                                },
                            }

                            # Add cache-related fields to update if they exist
                            if "cache_read_input_tokens" in transaction:
                                update_data["cache_read_input_tokens"] = {
                                    "increment": transaction.get(
                                        "cache_read_input_tokens", 0
                                    )
                                }
                            if "cache_creation_input_tokens" in transaction:
                                update_data["cache_creation_input_tokens"] = {
                                    "increment": transaction.get(
                                        "cache_creation_input_tokens", 0
                                    )
                                }

                            if entity_type == "tag" and "request_id" in transaction:
                                update_data["request_id"] = transaction.get("request_id")

                            table.upsert(
                                where=where_clause,
                                data={
                                    "create": common_data,
                                    "update": update_data,
                                },
                            )

                    verbose_proxy_logger.debug(
                        f"Processed {len(transactions_to_process)} daily {entity_type} transactions in {time.time() - start_time:.2f}s"
                    )

                    # Remove processed transactions
                    for key in transactions_to_process.keys():
                        daily_spend_transactions.pop(key, None)

                    break

                except DB_CONNECTION_ERROR_TYPES as e:
                    if i >= n_retry_times:
                        _raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    await asyncio.sleep(
                        # Sleep a random amount to avoid retrying and deadlocking again: when two transactions deadlock they are
                        # cancelled basically at the same time, so if they wait the same time they will also retry at the same time
                        # and thus they are more likely to deadlock again.
                        # Instead, we sleep a random amount so that they retry at slightly different times, lowering the chance of
                        # repeated deadlocks, and therefore of exceeding the retry limit.
                        random.uniform(2**i, 2 ** (i + 1))
                    )

        except Exception as e:
            if "transactions_to_process" in locals():
                for key in transactions_to_process.keys():  # type: ignore
                    daily_spend_transactions.pop(key, None)
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
        await DBSpendUpdateWriter._update_daily_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_spend_transactions,
            entity_type="user",
            entity_id_field="user_id",
            table_name="litellm_dailyuserspend",
            unique_constraint_name="user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name",
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
        await DBSpendUpdateWriter._update_daily_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_spend_transactions,
            entity_type="team",
            entity_id_field="team_id",
            table_name="litellm_dailyteamspend",
            unique_constraint_name="team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name",
        )

    @staticmethod
    async def update_daily_org_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyOrganizationSpendTransaction],
    ):
        """
        Batch job to update LiteLLM_DailyOrganizationSpend table using in-memory daily_spend_transactions
        """
        await DBSpendUpdateWriter._update_daily_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_spend_transactions,
            entity_type="org",
            entity_id_field="organization_id",
            table_name="litellm_dailyorganizationspend",
            unique_constraint_name="organization_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name",
        )

    @staticmethod
    async def update_daily_end_user_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyEndUserSpendTransaction],
    ):
        """
        Batch job to update LiteLLM_DailyEndUserSpend table using in-memory daily_spend_transactions
        """
        await DBSpendUpdateWriter._update_daily_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_spend_transactions,
            entity_type="end_user",
            entity_id_field="end_user_id",
            table_name="litellm_dailyenduserspend",
            unique_constraint_name="end_user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name",
        )

    @staticmethod
    async def update_daily_agent_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyAgentSpendTransaction],
    ):
        """
        Batch job to update LiteLLM_DailyAgentSpend table using in-memory daily_spend_transactions
        """
        await DBSpendUpdateWriter._update_daily_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_spend_transactions,
            entity_type="agent",
            entity_id_field="agent_id",
            table_name="litellm_dailyagentspend",
            unique_constraint_name="agent_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name",
        )

    @staticmethod
    async def update_daily_tag_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        daily_spend_transactions: Dict[str, DailyTagSpendTransaction],
    ):
        """
        Batch job to update LiteLLM_DailyTagSpend table using in-memory daily_spend_transactions
        """
        await DBSpendUpdateWriter._update_daily_spend(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_spend_transactions,
            entity_type="tag",
            entity_id_field="tag",
            table_name="litellm_dailytagspend",
            unique_constraint_name="tag_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name",
        )

    async def _common_add_spend_log_transaction_to_daily_transaction(
        self,
        payload: Union[dict, SpendLogsPayload],
        prisma_client: PrismaClient,
        type: Literal["user", "team", "org", "request_tags", "end_user", "agent"] = "user",
    ) -> Optional[BaseDailySpendTransaction]:
        common_expected_keys = ["startTime", "api_key"]
        if type == "user":
            expected_keys = ["user", *common_expected_keys]
        elif type == "team":
            expected_keys = ["team_id", *common_expected_keys]
        elif type == "org":
            expected_keys = ["organization_id", *common_expected_keys]
        elif type == "request_tags":
            expected_keys = ["request_tags", *common_expected_keys]
        elif type == "end_user":
            expected_keys = ["end_user_id", *common_expected_keys]
        elif type == "agent":
            expected_keys = ["agent_id", *common_expected_keys]
        else:
            raise ValueError(f"Invalid type: {type}")
        if not all(key in payload for key in expected_keys):
            verbose_proxy_logger.debug(
                f"Missing expected keys: {expected_keys}, in payload, skipping from daily_user_spend_transactions"
            )
            return None

        any_expected_keys = ["model", "mcp_namespaced_tool_name"]
        if not any(key in payload for key in any_expected_keys):
            verbose_proxy_logger.debug(
                f"Missing any expected keys: {any_expected_keys}, in payload, skipping from daily_user_spend_transactions"
            )
            return None
        elif "mcp_namespaced_tool_name" in payload:
            pass
        elif "model" in payload and (
            "custom_llm_provider" not in payload or "model_group" not in payload
        ):
            verbose_proxy_logger.debug(
                "Missing custom_llm_provider or model_group in payload, skipping from daily_user_spend_transactions"
            )
            return None

        request_status = prisma_client.get_request_status(payload)
        verbose_proxy_logger.debug(f"Logged request status: {request_status}")
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
                model=payload.get("model", None),
                model_group=payload.get("model_group", None),
                mcp_namespaced_tool_name=payload.get("mcp_namespaced_tool_name", None),
                custom_llm_provider=payload.get("custom_llm_provider", None),
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

    async def add_spend_log_transaction_to_daily_org_transaction(
        self,
        payload: SpendLogsPayload,
        prisma_client: Optional[PrismaClient] = None,
        org_id: Optional[str] = None,
    ) -> None:
        if prisma_client is None:
            verbose_proxy_logger.debug(
                "prisma_client is None. Skipping writing spend logs to db."
            )
            return

        if org_id is None:
            verbose_proxy_logger.debug(
                "organization_id is None for request. Skipping incrementing organization spend."
            )
            return

        payload_with_org = cast(
            SpendLogsPayload,
            {
                **payload,
                "organization_id": org_id,
            },
        )

        base_daily_transaction = (
            await self._common_add_spend_log_transaction_to_daily_transaction(
                payload_with_org, prisma_client, "org"
            )
        )
        if base_daily_transaction is None:
            return

        daily_transaction_key = f"{org_id}_{base_daily_transaction['date']}_{payload_with_org['api_key']}_{payload_with_org['model']}_{payload_with_org['custom_llm_provider']}"
        daily_transaction = DailyOrganizationSpendTransaction(
            organization_id=org_id, **base_daily_transaction
        )
        await self.daily_org_spend_update_queue.add_update(
            update={daily_transaction_key: daily_transaction}
        )

    async def add_spend_log_transaction_to_daily_end_user_transaction(
        self,
        payload: SpendLogsPayload,
        prisma_client: Optional[PrismaClient] = None,
    ) -> None:
        if prisma_client is None:
            verbose_proxy_logger.debug(
                "prisma_client is None. Skipping writing spend logs to db."
            )
            return

        end_user_id = payload.get("end_user")
        if end_user_id is None or end_user_id == "":
            verbose_proxy_logger.debug(
                "end_user is None or empty for request. Skipping incrementing end user spend."
            )
            return

        payload_with_end_user_id = cast(
            SpendLogsPayload,
            {
                **payload,
                "end_user_id": end_user_id,
            },
        )

        base_daily_transaction = (
            await self._common_add_spend_log_transaction_to_daily_transaction(
                payload_with_end_user_id, prisma_client, "end_user"
            )
        )
        if base_daily_transaction is None:
            return

        daily_transaction_key = f"{end_user_id}_{base_daily_transaction['date']}_{payload_with_end_user_id['api_key']}_{payload_with_end_user_id['model']}_{payload_with_end_user_id['custom_llm_provider']}"
        daily_transaction = DailyEndUserSpendTransaction(
            end_user_id=end_user_id, **base_daily_transaction
        )
        await self.daily_end_user_spend_update_queue.add_update(
            update={daily_transaction_key: daily_transaction}
        )

    async def add_spend_log_transaction_to_daily_agent_transaction(
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
                payload, prisma_client, "agent"
            )
        )
        if base_daily_transaction is None:
            return
        if payload["agent_id"] is None:
            verbose_proxy_logger.debug(
                "agent_id is None for request. Skipping incrementing agent spend."
            )
            return
        payload_with_agent_id = cast(
            SpendLogsPayload,
            {
                **payload,
                "agent_id": payload["agent_id"],
            },
        )
        base_daily_transaction = (
            await self._common_add_spend_log_transaction_to_daily_transaction(
                payload_with_agent_id, prisma_client, "agent"
            )
        )
        if base_daily_transaction is None:
            return
        daily_transaction_key = f"{payload['agent_id']}_{base_daily_transaction['date']}_{payload_with_agent_id['api_key']}_{payload_with_agent_id['model']}_{payload_with_agent_id['custom_llm_provider']}"
        daily_transaction = DailyAgentSpendTransaction(
            agent_id=payload['agent_id'], **base_daily_transaction
        )
        await self.daily_agent_spend_update_queue.add_update(
            update={daily_transaction_key: daily_transaction}
        )

    async def add_spend_log_transaction_to_daily_tag_transaction(
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
                payload, prisma_client, "request_tags"
            )
        )
        if base_daily_transaction is None:
            return
        if payload["request_tags"] is None:
            verbose_proxy_logger.debug(
                "request_tags is None for request. Skipping incrementing tag spend."
            )
            return

        request_tags = []
        if isinstance(payload["request_tags"], str):
            request_tags = json.loads(payload["request_tags"])
        elif isinstance(payload["request_tags"], list):
            request_tags = payload["request_tags"]
        else:
            raise ValueError(f"Invalid request_tags: {payload['request_tags']}")
        for tag in request_tags:
            daily_transaction_key = f"{tag}_{base_daily_transaction['date']}_{payload['api_key']}_{payload['model']}_{payload['custom_llm_provider']}"
            daily_transaction = DailyTagSpendTransaction(
                tag=tag, **base_daily_transaction, request_id=payload["request_id"]
            )

            await self.daily_tag_spend_update_queue.add_update(
                update={daily_transaction_key: daily_transaction}
            )
