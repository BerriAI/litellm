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
from typing import Any, Optional, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    Litellm_EntityType,
    LiteLLM_UserTable,
    SpendLogsPayload,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token


class DBSpendUpdateWriter:
    """
    Module responsible for

    1. Writing spend increments to either in memory list of transactions or to redis
    2. Reading increments from redis or in memory list of transactions and committing them to db
    """

    @staticmethod
    async def update_database(
        # LiteLLM management object fields
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

        try:
            verbose_proxy_logger.debug(
                f"Enters prisma db call, response_cost: {response_cost}, token: {token}; user_id: {user_id}; team_id: {team_id}"
            )
            if DBSpendUpdateWriter.disable_spend_updates() is True:
                return
            if token is not None and isinstance(token, str) and token.startswith("sk-"):
                hashed_token = hash_token(token=token)
            else:
                hashed_token = token

            asyncio.create_task(
                DBSpendUpdateWriter._update_user_db(
                    response_cost=response_cost,
                    user_id=user_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    litellm_proxy_budget_name=litellm_proxy_budget_name,
                    end_user_id=end_user_id,
                )
            )
            asyncio.create_task(
                DBSpendUpdateWriter._update_key_db(
                    response_cost=response_cost,
                    hashed_token=hashed_token,
                    prisma_client=prisma_client,
                )
            )
            asyncio.create_task(
                DBSpendUpdateWriter._update_team_db(
                    response_cost=response_cost,
                    team_id=team_id,
                    user_id=user_id,
                    prisma_client=prisma_client,
                )
            )
            asyncio.create_task(
                DBSpendUpdateWriter._update_org_db(
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

    @staticmethod
    async def _update_transaction_list(
        response_cost: Optional[float],
        entity_id: Optional[str],
        transaction_list: dict,
        entity_type: Litellm_EntityType,
        debug_msg: Optional[str] = None,
    ) -> bool:
        """
        Common helper method to update a transaction list for an entity

        Args:
            response_cost: The cost to add
            entity_id: The ID of the entity to update
            transaction_list: The transaction list dictionary to update
            entity_type: The type of entity (from EntityType enum)
            debug_msg: Optional custom debug message

        Returns:
            bool: True if update happened, False otherwise
        """
        try:
            if debug_msg:
                verbose_proxy_logger.debug(debug_msg)
            else:
                verbose_proxy_logger.debug(
                    f"adding spend to {entity_type.value} db. Response cost: {response_cost}. {entity_type.value}_id: {entity_id}."
                )

            if entity_id is None:
                verbose_proxy_logger.debug(
                    f"track_cost_callback: {entity_type.value}_id is None. Not tracking spend for {entity_type.value}"
                )
                return False

            transaction_list[entity_id] = response_cost + transaction_list.get(
                entity_id, 0
            )
            return True

        except Exception as e:
            verbose_proxy_logger.info(
                f"Update {entity_type.value.capitalize()} DB failed to execute - {str(e)}\n{traceback.format_exc()}"
            )
            raise e

    @staticmethod
    async def _update_key_db(
        response_cost: Optional[float],
        hashed_token: Optional[str],
        prisma_client: Optional[PrismaClient],
    ):
        try:
            if hashed_token is None or prisma_client is None:
                return

            await DBSpendUpdateWriter._update_transaction_list(
                response_cost=response_cost,
                entity_id=hashed_token,
                transaction_list=prisma_client.key_list_transactons,
                entity_type=Litellm_EntityType.KEY,
                debug_msg=f"adding spend to key db. Response cost: {response_cost}. Token: {hashed_token}.",
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Update Key DB Call failed to execute - {str(e)}"
            )
            raise e

    @staticmethod
    async def _update_user_db(
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
                        await DBSpendUpdateWriter._update_transaction_list(
                            response_cost=response_cost,
                            entity_id=_id,
                            transaction_list=prisma_client.user_list_transactons,
                            entity_type=Litellm_EntityType.USER,
                        )

                if end_user_id is not None:
                    await DBSpendUpdateWriter._update_transaction_list(
                        response_cost=response_cost,
                        entity_id=end_user_id,
                        transaction_list=prisma_client.end_user_list_transactons,
                        entity_type=Litellm_EntityType.END_USER,
                    )
        except Exception as e:
            verbose_proxy_logger.info(
                "\033[91m"
                + f"Update User DB call failed to execute {str(e)}\n{traceback.format_exc()}"
            )

    @staticmethod
    async def _update_team_db(
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

            await DBSpendUpdateWriter._update_transaction_list(
                response_cost=response_cost,
                entity_id=team_id,
                transaction_list=prisma_client.team_list_transactons,
                entity_type=Litellm_EntityType.TEAM,
            )

            try:
                # Track spend of the team member within this team
                if user_id is not None:
                    # key is "team_id::<value>::user_id::<value>"
                    team_member_key = f"team_id::{team_id}::user_id::{user_id}"
                    await DBSpendUpdateWriter._update_transaction_list(
                        response_cost=response_cost,
                        entity_id=team_member_key,
                        transaction_list=prisma_client.team_member_list_transactons,
                        entity_type=Litellm_EntityType.TEAM_MEMBER,
                    )
            except Exception:
                pass
        except Exception as e:
            verbose_proxy_logger.info(
                f"Update Team DB failed to execute - {str(e)}\n{traceback.format_exc()}"
            )
            raise e

    @staticmethod
    async def _update_org_db(
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

            await DBSpendUpdateWriter._update_transaction_list(
                response_cost=response_cost,
                entity_id=org_id,
                transaction_list=prisma_client.org_list_transactons,
                entity_type=Litellm_EntityType.ORGANIZATION,
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

    @staticmethod
    async def update_end_user_spend(
        n_retry_times: int, prisma_client: PrismaClient, proxy_logging_obj: ProxyLogging
    ):
        for i in range(n_retry_times + 1):
            start_time = time.time()
            try:
                async with prisma_client.db.tx(
                    timeout=timedelta(seconds=60)
                ) as transaction:
                    async with transaction.batch_() as batcher:
                        for (
                            end_user_id,
                            response_cost,
                        ) in prisma_client.end_user_list_transactons.items():
                            if litellm.max_end_user_budget is not None:
                                pass
                            batcher.litellm_endusertable.upsert(
                                where={"user_id": end_user_id},
                                data={
                                    "create": {
                                        "user_id": end_user_id,
                                        "spend": response_cost,
                                        "blocked": False,
                                    },
                                    "update": {"spend": {"increment": response_cost}},
                                },
                            )

                break
            except DB_CONNECTION_ERROR_TYPES as e:
                if i >= n_retry_times:  # If we've reached the maximum number of retries
                    DBSpendUpdateWriter._raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )
                # Optionally, sleep for a bit before retrying
                await asyncio.sleep(2**i)  # Exponential backoff
            except Exception as e:
                DBSpendUpdateWriter._raise_failed_update_spend_exception(
                    e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                )
            finally:
                prisma_client.end_user_list_transactons = (
                    {}
                )  # reset the end user list transactions - prevent bad data from causing issues

    @staticmethod
    async def update_spend_logs(
        n_retry_times: int,
        prisma_client: PrismaClient,
        db_writer_client: Optional[HTTPHandler],
        proxy_logging_obj: ProxyLogging,
    ):
        BATCH_SIZE = 100  # Preferred size of each batch to write to the database
        MAX_LOGS_PER_INTERVAL = (
            1000  # Maximum number of logs to flush in a single interval
        )
        # Get initial logs to process
        logs_to_process = prisma_client.spend_log_transactions[:MAX_LOGS_PER_INTERVAL]
        start_time = time.time()
        try:
            for i in range(n_retry_times + 1):
                try:
                    base_url = os.getenv("SPEND_LOGS_URL", None)
                    if (
                        len(logs_to_process) > 0
                        and base_url is not None
                        and db_writer_client is not None
                    ):
                        if not base_url.endswith("/"):
                            base_url += "/"
                        verbose_proxy_logger.debug("base_url: {}".format(base_url))
                        response = await db_writer_client.post(
                            url=base_url + "spend/update",
                            data=json.dumps(logs_to_process),
                            headers={"Content-Type": "application/json"},
                        )
                        if response.status_code == 200:
                            prisma_client.spend_log_transactions = (
                                prisma_client.spend_log_transactions[
                                    len(logs_to_process) :
                                ]
                            )
                    else:
                        for j in range(0, len(logs_to_process), BATCH_SIZE):
                            batch = logs_to_process[j : j + BATCH_SIZE]
                            batch_with_dates = [
                                prisma_client.jsonify_object({**entry})
                                for entry in batch
                            ]
                            await prisma_client.db.litellm_spendlogs.create_many(
                                data=batch_with_dates, skip_duplicates=True
                            )
                            verbose_proxy_logger.debug(
                                f"Flushed {len(batch)} logs to the DB."
                            )

                        prisma_client.spend_log_transactions = (
                            prisma_client.spend_log_transactions[len(logs_to_process) :]
                        )
                        verbose_proxy_logger.debug(
                            f"{len(logs_to_process)} logs processed. Remaining in queue: {len(prisma_client.spend_log_transactions)}"
                        )
                    break
                except DB_CONNECTION_ERROR_TYPES:
                    if i is None:
                        i = 0
                    if i >= n_retry_times:
                        raise
                    await asyncio.sleep(2**i)
        except Exception as e:
            prisma_client.spend_log_transactions = prisma_client.spend_log_transactions[
                len(logs_to_process) :
            ]
            DBSpendUpdateWriter._raise_failed_update_spend_exception(
                e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
            )

    @staticmethod
    async def update_daily_user_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
    ):
        """
        Batch job to update LiteLLM_DailyUserSpend table using in-memory daily_spend_transactions
        """
        BATCH_SIZE = (
            100  # Number of aggregated records to update in each database operation
        )
        start_time = time.time()

        try:
            for i in range(n_retry_times + 1):
                try:
                    # Get transactions to process
                    transactions_to_process = dict(
                        list(prisma_client.daily_user_spend_transactions.items())[
                            :BATCH_SIZE
                        ]
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
                                        "spend": transaction["spend"],
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
                                    },
                                },
                            )

                    verbose_proxy_logger.info(
                        f"Processed {len(transactions_to_process)} daily spend transactions in {time.time() - start_time:.2f}s"
                    )

                    # Remove processed transactions
                    for key in transactions_to_process.keys():
                        prisma_client.daily_user_spend_transactions.pop(key, None)

                    verbose_proxy_logger.debug(
                        f"Processed {len(transactions_to_process)} daily spend transactions in {time.time() - start_time:.2f}s"
                    )
                    break

                except DB_CONNECTION_ERROR_TYPES as e:
                    if i >= n_retry_times:
                        DBSpendUpdateWriter._raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    await asyncio.sleep(2**i)  # Exponential backoff

        except Exception as e:
            # Remove processed transactions even if there was an error
            if "transactions_to_process" in locals():
                for key in transactions_to_process.keys():  # type: ignore
                    prisma_client.daily_user_spend_transactions.pop(key, None)
            DBSpendUpdateWriter._raise_failed_update_spend_exception(
                e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
            )

    @staticmethod
    def disable_spend_updates() -> bool:
        """
        returns True if should not update spend in db
        Skips writing spend logs and updates to key, team, user spend to DB
        """
        from litellm.proxy.proxy_server import general_settings

        if general_settings.get("disable_spend_updates") is True:
            return True
        return False

    @staticmethod
    def _raise_failed_update_spend_exception(
        e: Exception, start_time: float, proxy_logging_obj: ProxyLogging
    ):
        """
        Raise an exception for failed update spend logs

        - Calls proxy_logging_obj.failure_handler to log the error
        - Ensures error messages says "Non-Blocking"
        """
        import traceback

        error_msg = f"[Non-Blocking]LiteLLM Prisma Client Exception - update spend logs: {str(e)}"
        error_traceback = error_msg + "\n" + traceback.format_exc()
        end_time = time.time()
        _duration = end_time - start_time
        asyncio.create_task(
            proxy_logging_obj.failure_handler(
                original_exception=e,
                duration=_duration,
                call_type="update_spend",
                traceback_str=error_traceback,
            )
        )
        raise e

    @staticmethod
    async def update_spend(  # noqa: PLR0915
        prisma_client: PrismaClient,
        db_writer_client: Optional[HTTPHandler],
        proxy_logging_obj: ProxyLogging,
    ):
        """
        Batch write updates to db.

        Triggered every minute.

        Requires:
        user_id_list: dict,
        keys_list: list,
        team_list: list,
        spend_logs: list,
        """
        n_retry_times = 3
        i = None
        ### UPDATE USER TABLE ###
        if len(prisma_client.user_list_transactons.keys()) > 0:
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
                            ) in prisma_client.user_list_transactons.items():
                                batcher.litellm_usertable.update_many(
                                    where={"user_id": user_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    prisma_client.user_list_transactons = (
                        {}
                    )  # Clear the remaining transactions after processing all batches in the loop.
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        DBSpendUpdateWriter._raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    DBSpendUpdateWriter._raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE END-USER TABLE ###
        verbose_proxy_logger.debug(
            "End-User Spend transactions: {}".format(
                len(prisma_client.end_user_list_transactons.keys())
            )
        )
        if len(prisma_client.end_user_list_transactons.keys()) > 0:
            await DBSpendUpdateWriter.update_end_user_spend(
                n_retry_times=n_retry_times,
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
            )
        ### UPDATE KEY TABLE ###
        verbose_proxy_logger.debug(
            "KEY Spend transactions: {}".format(
                len(prisma_client.key_list_transactons.keys())
            )
        )
        if len(prisma_client.key_list_transactons.keys()) > 0:
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
                            ) in prisma_client.key_list_transactons.items():
                                batcher.litellm_verificationtoken.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"token": token},
                                    data={"spend": {"increment": response_cost}},
                                )
                    prisma_client.key_list_transactons = (
                        {}
                    )  # Clear the remaining transactions after processing all batches in the loop.
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        DBSpendUpdateWriter._raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    DBSpendUpdateWriter._raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE TEAM TABLE ###
        verbose_proxy_logger.debug(
            "Team Spend transactions: {}".format(
                len(prisma_client.team_list_transactons.keys())
            )
        )
        if len(prisma_client.team_list_transactons.keys()) > 0:
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
                            ) in prisma_client.team_list_transactons.items():
                                verbose_proxy_logger.debug(
                                    "Updating spend for team id={} by {}".format(
                                        team_id, response_cost
                                    )
                                )
                                batcher.litellm_teamtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"team_id": team_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    prisma_client.team_list_transactons = (
                        {}
                    )  # Clear the remaining transactions after processing all batches in the loop.
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        DBSpendUpdateWriter._raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    DBSpendUpdateWriter._raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE TEAM Membership TABLE with spend ###
        if len(prisma_client.team_member_list_transactons.keys()) > 0:
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
                            ) in prisma_client.team_member_list_transactons.items():
                                # key is "team_id::<value>::user_id::<value>"
                                team_id = key.split("::")[1]
                                user_id = key.split("::")[3]

                                batcher.litellm_teammembership.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"team_id": team_id, "user_id": user_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    prisma_client.team_member_list_transactons = (
                        {}
                    )  # Clear the remaining transactions after processing all batches in the loop.
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        DBSpendUpdateWriter._raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    DBSpendUpdateWriter._raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE ORG TABLE ###
        if len(prisma_client.org_list_transactons.keys()) > 0:
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
                            ) in prisma_client.org_list_transactons.items():
                                batcher.litellm_organizationtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"organization_id": org_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    prisma_client.org_list_transactons = (
                        {}
                    )  # Clear the remaining transactions after processing all batches in the loop.
                    break
                except DB_CONNECTION_ERROR_TYPES as e:
                    if (
                        i >= n_retry_times
                    ):  # If we've reached the maximum number of retries
                        DBSpendUpdateWriter._raise_failed_update_spend_exception(
                            e=e,
                            start_time=start_time,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    # Optionally, sleep for a bit before retrying
                    await asyncio.sleep(2**i)  # Exponential backoff
                except Exception as e:
                    DBSpendUpdateWriter._raise_failed_update_spend_exception(
                        e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj
                    )

        ### UPDATE SPEND LOGS ###
        verbose_proxy_logger.debug(
            "Spend Logs transactions: {}".format(
                len(prisma_client.spend_log_transactions)
            )
        )

        if len(prisma_client.spend_log_transactions) > 0:
            await DBSpendUpdateWriter.update_spend_logs(
                n_retry_times=n_retry_times,
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
                db_writer_client=db_writer_client,
            )

        ### UPDATE DAILY USER SPEND ###
        verbose_proxy_logger.debug(
            "Daily User Spend transactions: {}".format(
                len(prisma_client.daily_user_spend_transactions)
            )
        )

        if len(prisma_client.daily_user_spend_transactions) > 0:
            await DBSpendUpdateWriter.update_daily_user_spend(
                n_retry_times=n_retry_times,
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
            )
