import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LiteLLM_BudgetTableFull,
    LiteLLM_EndUserTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LiteLLM_VerificationToken,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.types.services import ServiceTypes


class ResetBudgetJob:
    """
    Resets the budget for all the keys, users, and teams that need it
    """

    def __init__(self, proxy_logging_obj: ProxyLogging, prisma_client: PrismaClient):
        self.proxy_logging_obj: ProxyLogging = proxy_logging_obj
        self.prisma_client: PrismaClient = prisma_client

    async def reset_budget(
        self,
    ):
        """
        Gets all the non-expired keys for a db, which need spend to be reset

        Resets their spend

        Updates db
        """
        if self.prisma_client is not None:
            ### RESET KEY BUDGET ###
            await self.reset_budget_for_litellm_keys()

            ### RESET USER BUDGET ###
            await self.reset_budget_for_litellm_users()

            ## Reset Team Budget
            await self.reset_budget_for_litellm_teams()

            ### RESET ENDUSER (Customer) BUDGET and corresponding Budget duration ###
            await self.reset_budget_for_litellm_budget_table()

    async def reset_budget_for_litellm_team_members(
        self, budgets_to_reset: List[LiteLLM_BudgetTableFull]
    ):
        """
        Resets the budget for all LiteLLM Team Members if their budget has expired
        """
        return await self.prisma_client.db.litellm_teammembership.update_many(
            where={
                "budget_id": {
                    "in": [
                        budget.budget_id
                        for budget in budgets_to_reset
                        if budget.budget_id is not None
                    ]
                }
            },
            data={
                "spend": 0,
            },
        )

    async def reset_budget_for_litellm_budget_table(self):
        """
        Resets the budget for all LiteLLM End-Users (Customers), and Team Members if their budget has expired
        The corresponding Budget duration is also updated.
        """

        now = datetime.now(timezone.utc)
        start_time = time.time()
        endusers_to_reset: Optional[List[LiteLLM_EndUserTable]] = None
        budgets_to_reset: Optional[List[LiteLLM_BudgetTableFull]] = None
        updated_endusers: List[LiteLLM_EndUserTable] = []
        failed_endusers = []
        try:
            budgets_to_reset = await self.prisma_client.get_data(
                table_name="budget", query_type="find_all", reset_at=now
            )

            if budgets_to_reset is not None and len(budgets_to_reset) > 0:
                for budget in budgets_to_reset:
                    budget = await ResetBudgetJob._reset_budget_reset_at_date(
                        budget, now
                    )

                await self.prisma_client.update_data(
                    query_type="update_many",
                    data_list=budgets_to_reset,
                    table_name="budget",
                )

                endusers_to_reset = await self.prisma_client.get_data(
                    table_name="enduser",
                    query_type="find_all",
                    budget_id_list=[
                        budget.budget_id
                        for budget in budgets_to_reset
                        if budget.budget_id is not None
                    ],
                )

                await self.reset_budget_for_litellm_team_members(
                    budgets_to_reset=budgets_to_reset
                )

            if endusers_to_reset is not None and len(endusers_to_reset) > 0:
                for enduser in endusers_to_reset:
                    try:
                        updated_enduser = (
                            await ResetBudgetJob._reset_budget_for_enduser(
                                enduser=enduser
                            )
                        )
                        if updated_enduser is not None:
                            updated_endusers.append(updated_enduser)
                        else:
                            failed_endusers.append(
                                {
                                    "enduser": enduser,
                                    "error": "Returned None without exception",
                                }
                            )
                    except Exception as e:
                        failed_endusers.append({"enduser": enduser, "error": str(e)})
                        verbose_proxy_logger.exception(
                            "Failed to reset budget for enduser: %s", enduser
                        )

                verbose_proxy_logger.debug(
                    "Updated users %s",
                    json.dumps(updated_endusers, indent=4, default=str),
                )

                await self.prisma_client.update_data(
                    query_type="update_many",
                    data_list=updated_endusers,
                    table_name="enduser",
                )

            end_time = time.time()
            if len(failed_endusers) > 0:  # If any endusers failed to reset
                raise Exception(
                    f"Failed to reset {len(failed_endusers)} endusers: {json.dumps(failed_endusers, default=str)}"
                )

            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_success_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    call_type="reset_budget_budget_table",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_budgets_found": (
                            len(budgets_to_reset) if budgets_to_reset else 0
                        ),
                        "budgets_found": json.dumps(
                            budgets_to_reset, indent=4, default=str
                        ),
                        "num_endusers_found": (
                            len(endusers_to_reset) if endusers_to_reset else 0
                        ),
                        "endusers_found": json.dumps(
                            endusers_to_reset, indent=4, default=str
                        ),
                        "num_endusers_updated": len(updated_endusers),
                        "endusers_updated": json.dumps(
                            updated_endusers, indent=4, default=str
                        ),
                        "num_endusers_failed": len(failed_endusers),
                        "endusers_failed": json.dumps(
                            failed_endusers, indent=4, default=str
                        ),
                    },
                )
            )
        except Exception as e:
            end_time = time.time()
            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_failure_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    error=e,
                    call_type="reset_budget_endusers",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_budgets_found": (
                            len(budgets_to_reset) if budgets_to_reset else 0
                        ),
                        "budgets_found": json.dumps(
                            budgets_to_reset, indent=4, default=str
                        ),
                        "num_endusers_found": (
                            len(endusers_to_reset) if endusers_to_reset else 0
                        ),
                        "endusers_found": json.dumps(
                            endusers_to_reset, indent=4, default=str
                        ),
                    },
                )
            )
            verbose_proxy_logger.exception("Failed to reset budget for endusers: %s", e)

    async def reset_budget_for_litellm_keys(self):
        """
        Resets the budget for all the litellm keys

        Catches Exceptions and logs them
        """
        now = datetime.utcnow()
        start_time = time.time()
        keys_to_reset: Optional[List[LiteLLM_VerificationToken]] = None
        try:
            keys_to_reset = await self.prisma_client.get_data(
                table_name="key", query_type="find_all", expires=now, reset_at=now
            )
            verbose_proxy_logger.debug(
                "Keys to reset %s", json.dumps(keys_to_reset, indent=4, default=str)
            )
            updated_keys: List[LiteLLM_VerificationToken] = []
            failed_keys = []
            if keys_to_reset is not None and len(keys_to_reset) > 0:
                for key in keys_to_reset:
                    try:
                        updated_key = await ResetBudgetJob._reset_budget_for_key(
                            key=key, current_time=now
                        )
                        if updated_key is not None:
                            updated_keys.append(updated_key)
                        else:
                            failed_keys.append(
                                {"key": key, "error": "Returned None without exception"}
                            )
                    except Exception as e:
                        failed_keys.append({"key": key, "error": str(e)})
                        verbose_proxy_logger.exception(
                            "Failed to reset budget for key: %s", key
                        )

                verbose_proxy_logger.debug(
                    "Updated keys %s", json.dumps(updated_keys, indent=4, default=str)
                )

                if updated_keys:
                    await self.prisma_client.update_data(
                        query_type="update_many",
                        data_list=updated_keys,
                        table_name="key",
                    )

            end_time = time.time()
            if len(failed_keys) > 0:  # If any keys failed to reset
                raise Exception(
                    f"Failed to reset {len(failed_keys)} keys: {json.dumps(failed_keys, default=str)}"
                )

            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_success_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    call_type="reset_budget_keys",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_keys_found": len(keys_to_reset) if keys_to_reset else 0,
                        "keys_found": json.dumps(keys_to_reset, indent=4, default=str),
                        "num_keys_updated": len(updated_keys),
                        "keys_updated": json.dumps(updated_keys, indent=4, default=str),
                        "num_keys_failed": len(failed_keys),
                        "keys_failed": json.dumps(failed_keys, indent=4, default=str),
                    },
                )
            )
        except Exception as e:
            end_time = time.time()
            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_failure_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    error=e,
                    call_type="reset_budget_keys",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_keys_found": len(keys_to_reset) if keys_to_reset else 0,
                        "keys_found": json.dumps(keys_to_reset, indent=4, default=str),
                    },
                )
            )
            verbose_proxy_logger.exception("Failed to reset budget for keys: %s", e)

    async def reset_budget_for_litellm_users(self):
        """
        Resets the budget for all LiteLLM Internal Users if their budget has expired
        """
        now = datetime.utcnow()
        start_time = time.time()
        users_to_reset: Optional[List[LiteLLM_UserTable]] = None
        try:
            users_to_reset = await self.prisma_client.get_data(
                table_name="user", query_type="find_all", reset_at=now
            )
            updated_users: List[LiteLLM_UserTable] = []
            failed_users = []
            if users_to_reset is not None and len(users_to_reset) > 0:
                for user in users_to_reset:
                    try:
                        updated_user = await ResetBudgetJob._reset_budget_for_user(
                            user=user, current_time=now
                        )
                        if updated_user is not None:
                            updated_users.append(updated_user)
                        else:
                            failed_users.append(
                                {
                                    "user": user,
                                    "error": "Returned None without exception",
                                }
                            )
                    except Exception as e:
                        failed_users.append({"user": user, "error": str(e)})
                        verbose_proxy_logger.exception(
                            "Failed to reset budget for user: %s", user
                        )

                verbose_proxy_logger.debug(
                    "Updated users %s", json.dumps(updated_users, indent=4, default=str)
                )
                if updated_users:
                    await self.prisma_client.update_data(
                        query_type="update_many",
                        data_list=updated_users,
                        table_name="user",
                    )

            end_time = time.time()
            if len(failed_users) > 0:  # If any users failed to reset
                raise Exception(
                    f"Failed to reset {len(failed_users)} users: {json.dumps(failed_users, default=str)}"
                )

            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_success_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    call_type="reset_budget_users",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_users_found": len(users_to_reset) if users_to_reset else 0,
                        "users_found": json.dumps(
                            users_to_reset, indent=4, default=str
                        ),
                        "num_users_updated": len(updated_users),
                        "users_updated": json.dumps(
                            updated_users, indent=4, default=str
                        ),
                        "num_users_failed": len(failed_users),
                        "users_failed": json.dumps(failed_users, indent=4, default=str),
                    },
                )
            )
        except Exception as e:
            end_time = time.time()
            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_failure_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    error=e,
                    call_type="reset_budget_users",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_users_found": len(users_to_reset) if users_to_reset else 0,
                        "users_found": json.dumps(
                            users_to_reset, indent=4, default=str
                        ),
                    },
                )
            )
            verbose_proxy_logger.exception("Failed to reset budget for users: %s", e)

    async def reset_budget_for_litellm_teams(self):
        """
        Resets the budget for all LiteLLM Internal Teams if their budget has expired
        """
        now = datetime.utcnow()
        start_time = time.time()
        teams_to_reset: Optional[List[LiteLLM_TeamTable]] = None
        try:
            teams_to_reset = await self.prisma_client.get_data(
                table_name="team", query_type="find_all", reset_at=now
            )
            updated_teams: List[LiteLLM_TeamTable] = []
            failed_teams = []
            if teams_to_reset is not None and len(teams_to_reset) > 0:
                for team in teams_to_reset:
                    try:
                        updated_team = await ResetBudgetJob._reset_budget_for_team(
                            team=team, current_time=now
                        )
                        if updated_team is not None:
                            updated_teams.append(updated_team)
                        else:
                            failed_teams.append(
                                {
                                    "team": team,
                                    "error": "Returned None without exception",
                                }
                            )
                    except Exception as e:
                        failed_teams.append({"team": team, "error": str(e)})
                        verbose_proxy_logger.exception(
                            "Failed to reset budget for team: %s", team
                        )

                verbose_proxy_logger.debug(
                    "Updated teams %s", json.dumps(updated_teams, indent=4, default=str)
                )
                if updated_teams:
                    await self.prisma_client.update_data(
                        query_type="update_many",
                        data_list=updated_teams,
                        table_name="team",
                    )

            end_time = time.time()
            if len(failed_teams) > 0:  # If any teams failed to reset
                raise Exception(
                    f"Failed to reset {len(failed_teams)} teams: {json.dumps(failed_teams, default=str)}"
                )

            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_success_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    call_type="reset_budget_teams",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_teams_found": len(teams_to_reset) if teams_to_reset else 0,
                        "teams_found": json.dumps(
                            teams_to_reset, indent=4, default=str
                        ),
                        "num_teams_updated": len(updated_teams),
                        "teams_updated": json.dumps(
                            updated_teams, indent=4, default=str
                        ),
                        "num_teams_failed": len(failed_teams),
                        "teams_failed": json.dumps(failed_teams, indent=4, default=str),
                    },
                )
            )
        except Exception as e:
            end_time = time.time()
            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_failure_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    error=e,
                    call_type="reset_budget_teams",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_teams_found": len(teams_to_reset) if teams_to_reset else 0,
                        "teams_found": json.dumps(
                            teams_to_reset, indent=4, default=str
                        ),
                    },
                )
            )
            verbose_proxy_logger.exception("Failed to reset budget for teams: %s", e)

    @staticmethod
    async def _reset_budget_common(
        item: Union[LiteLLM_TeamTable, LiteLLM_UserTable, LiteLLM_VerificationToken],
        current_time: datetime,
        item_type: Literal["key", "team", "user"],
    ):
        """
        In-place, updates spend=0, and sets budget_reset_at to current_time + budget_duration

        Common logic for resetting budget for a team, user, or key
        """
        try:
            item.spend = 0.0
            if hasattr(item, "budget_duration") and item.budget_duration is not None:
                # Get standardized reset time based on budget duration
                from litellm.proxy.common_utils.timezone_utils import (
                    get_budget_reset_time,
                )

                item.budget_reset_at = get_budget_reset_time(
                    budget_duration=item.budget_duration
                )
            return item
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error resetting budget for %s: %s. Item: %s", item_type, e, item
            )
            raise e

    @staticmethod
    async def _reset_budget_for_team(
        team: LiteLLM_TeamTable, current_time: datetime
    ) -> Optional[LiteLLM_TeamTable]:
        await ResetBudgetJob._reset_budget_common(
            item=team, current_time=current_time, item_type="team"
        )
        return team

    @staticmethod
    async def _reset_budget_for_user(
        user: LiteLLM_UserTable, current_time: datetime
    ) -> Optional[LiteLLM_UserTable]:
        await ResetBudgetJob._reset_budget_common(
            item=user, current_time=current_time, item_type="user"
        )
        return user

    @staticmethod
    async def _reset_budget_for_enduser(
        enduser: LiteLLM_EndUserTable,
    ) -> Optional[LiteLLM_EndUserTable]:
        try:
            enduser.spend = 0.0
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error resetting budget for enduser: %s. Item: %s", e, enduser
            )
            raise e
        return enduser

    @staticmethod
    async def _reset_budget_reset_at_date(
        budget: LiteLLM_BudgetTableFull, current_time: datetime
    ) -> LiteLLM_BudgetTableFull:
        try:
            if budget.budget_duration is not None:
                from litellm.litellm_core_utils.duration_parser import (
                    duration_in_seconds,
                )

                duration_s = duration_in_seconds(duration=budget.budget_duration)

                # Fallback for existing budgets that do not have a budget_reset_at date set, ensuring the duration is taken into account
                if (
                    budget.budget_reset_at is None
                    and budget.created_at + timedelta(seconds=duration_s) > current_time
                ):
                    budget.budget_reset_at = budget.created_at + timedelta(
                        seconds=duration_s
                    )
                else:
                    budget.budget_reset_at = current_time + timedelta(
                        seconds=duration_s
                    )
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error resetting budget_reset_at for budget: %s. Item: %s", e, budget
            )
            raise e
        return budget

    @staticmethod
    async def _reset_budget_for_key(
        key: LiteLLM_VerificationToken, current_time: datetime
    ) -> Optional[LiteLLM_VerificationToken]:
        await ResetBudgetJob._reset_budget_common(
            item=key, current_time=current_time, item_type="key"
        )
        return key
