import json
from datetime import datetime, timedelta
from typing import List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_VerificationToken, ResetTeamBudgetRequest
from litellm.proxy.utils import PrismaClient, duration_in_seconds


class ResetBudgetJob:
    """
    Resets the budget for all the keys, users, and teams that need it
    """

    @staticmethod
    async def reset_budget(prisma_client: PrismaClient):
        """
        Gets all the non-expired keys for a db, which need spend to be reset

        Resets their spend

        Updates db
        """
        if prisma_client is not None:
            ### RESET KEY BUDGET ###
            await ResetBudgetJob.reset_budget_for_litellm_keys(
                prisma_client=prisma_client
            )

            ### RESET USER BUDGET ###
            now = datetime.utcnow()
            users_to_reset = await prisma_client.get_data(
                table_name="user", query_type="find_all", reset_at=now
            )

            if users_to_reset is not None and len(users_to_reset) > 0:
                for user in users_to_reset:
                    user.spend = 0.0
                    duration_s = duration_in_seconds(duration=user.budget_duration)
                    user.budget_reset_at = now + timedelta(seconds=duration_s)

                await prisma_client.update_data(
                    query_type="update_many",
                    data_list=users_to_reset,
                    table_name="user",
                )

            ## Reset Team Budget
            now = datetime.utcnow()
            teams_to_reset = await prisma_client.get_data(
                table_name="team",
                query_type="find_all",
                reset_at=now,
            )

            if teams_to_reset is not None and len(teams_to_reset) > 0:
                team_reset_requests = []
                for team in teams_to_reset:
                    duration_s = duration_in_seconds(duration=team.budget_duration)
                    reset_team_budget_request = ResetTeamBudgetRequest(
                        team_id=team.team_id,
                        spend=0.0,
                        budget_reset_at=now + timedelta(seconds=duration_s),
                        updated_at=now,
                    )
                    team_reset_requests.append(reset_team_budget_request)
                await prisma_client.update_data(
                    query_type="update_many",
                    data_list=team_reset_requests,
                    table_name="team",
                )

    @staticmethod
    async def reset_budget_for_litellm_keys(prisma_client: PrismaClient):
        """
        Resets the budget for all the litellm keys
        """
        now = datetime.utcnow()
        keys_to_reset = await prisma_client.get_data(
            table_name="key", query_type="find_all", expires=now, reset_at=now
        )
        verbose_proxy_logger.debug(
            "Keys to reset %s", json.dumps(keys_to_reset, indent=4, default=str)
        )
        updated_keys: List[LiteLLM_VerificationToken] = []
        if keys_to_reset is not None and len(keys_to_reset) > 0:
            for key in keys_to_reset:
                updated_key = await ResetBudgetJob._reset_budget_for_key(
                    key=key, current_time=now
                )
                if updated_key is not None:
                    updated_keys.append(updated_key)

            verbose_proxy_logger.debug(
                "Updated keys %s", json.dumps(updated_keys, indent=4, default=str)
            )

            await prisma_client.update_data(
                query_type="update_many", data_list=updated_keys, table_name="key"
            )
        pass

    @staticmethod
    async def _reset_budget_for_key(
        key: LiteLLM_VerificationToken, current_time: datetime
    ) -> Optional[LiteLLM_VerificationToken]:
        """
        Resets the budget for a single key
        """
        try:
            key.spend = 0.0
            if key.budget_duration is not None:
                duration_s = duration_in_seconds(duration=key.budget_duration)
                key.budget_reset_at = current_time + timedelta(seconds=duration_s)
            return key
        except Exception as e:
            verbose_proxy_logger.exception(f"Error resetting budget for key: {key} {e}")
            return None
