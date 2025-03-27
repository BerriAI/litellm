"""
Module responsible for

1. Writing spend increments to either in memory list of transactions or to redis
2. Reading increments from redis or in memory list of transactions and committing them to db
"""

import asyncio
import os
import traceback
from datetime import datetime
from typing import Any, Optional, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.proxy._types import LiteLLM_UserTable, SpendLogsPayload
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import PrismaClient, ProxyUpdateSpend, hash_token


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
            if ProxyUpdateSpend.disable_spend_updates() is True:
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
    async def _update_key_db(
        response_cost: Optional[float],
        hashed_token: Optional[str],
        prisma_client: Optional[PrismaClient],
    ):
        try:
            verbose_proxy_logger.debug(
                f"adding spend to key db. Response cost: {response_cost}. Token: {hashed_token}."
            )
            if hashed_token is None:
                return
            if prisma_client is not None:
                prisma_client.key_list_transactons[hashed_token] = (
                    response_cost
                    + prisma_client.key_list_transactons.get(hashed_token, 0)
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
                ### KEY CHANGE ###
                for _id in user_ids:
                    if _id is not None:
                        prisma_client.user_list_transactons[_id] = (
                            response_cost
                            + prisma_client.user_list_transactons.get(_id, 0)
                        )
                if end_user_id is not None:
                    prisma_client.end_user_list_transactons[end_user_id] = (
                        response_cost
                        + prisma_client.end_user_list_transactons.get(end_user_id, 0)
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
            verbose_proxy_logger.debug(
                f"adding spend to team db. Response cost: {response_cost}. team_id: {team_id}."
            )
            if team_id is None:
                verbose_proxy_logger.debug(
                    "track_cost_callback: team_id is None. Not tracking spend for team"
                )
                return
            if prisma_client is not None:
                prisma_client.team_list_transactons[team_id] = (
                    response_cost + prisma_client.team_list_transactons.get(team_id, 0)
                )

                try:
                    # Track spend of the team member within this team
                    # key is "team_id::<value>::user_id::<value>"
                    team_member_key = f"team_id::{team_id}::user_id::{user_id}"
                    prisma_client.team_member_list_transactons[team_member_key] = (
                        response_cost
                        + prisma_client.team_member_list_transactons.get(
                            team_member_key, 0
                        )
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
            verbose_proxy_logger.debug(
                "adding spend to org db. Response cost: {}. org_id: {}.".format(
                    response_cost, org_id
                )
            )
            if org_id is None:
                verbose_proxy_logger.debug(
                    "track_cost_callback: org_id is None. Not tracking spend for org"
                )
                return
            if prisma_client is not None:
                prisma_client.org_list_transactons[org_id] = (
                    response_cost + prisma_client.org_list_transactons.get(org_id, 0)
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
