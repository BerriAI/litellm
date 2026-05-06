import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional, Union

import litellm
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

            ### RESET MULTI-WINDOW BUDGETS ###
            await self.reset_budget_windows()

    @staticmethod
    async def _invalidate_spend_counter(counter_key: str) -> None:
        """Zero a spend counter so a DB-row reset takes effect immediately.

        Call AFTER the DB write commits. Clearing Redis before the DB
        commit opens a window where get_current_spend reads 0 from Redis
        while the DB still holds the pre-reset value, allowing bypass.
        """
        try:
            from litellm.proxy.proxy_server import spend_counter_cache

            spend_counter_cache.in_memory_cache.set_cache(
                key=counter_key, value=0.0, ttl=60
            )
            if spend_counter_cache.redis_cache is not None:
                try:
                    await spend_counter_cache.redis_cache.async_set_cache(
                        key=counter_key, value=0.0, ttl=60
                    )
                except Exception as redis_err:
                    verbose_proxy_logger.warning(
                        "Failed to reset spend counter %s in Redis: %s. "
                        "Budget may be over-enforced until counter expires.",
                        counter_key,
                        redis_err,
                    )
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to reset spend counter %s: %s", counter_key, e
            )

    async def reset_budget_for_litellm_team_members(
        self, budgets_to_reset: List[LiteLLM_BudgetTableFull]
    ):
        """
        Resets the budget for all LiteLLM Team Members if their budget has expired
        """
        budget_ids = [
            budget.budget_id
            for budget in budgets_to_reset
            if budget.budget_id is not None
        ]

        try:
            memberships = await self.prisma_client.db.litellm_teammembership.find_many(
                where={"budget_id": {"in": budget_ids}}
            )
        except Exception as e:
            memberships = []
            verbose_proxy_logger.warning(
                "Failed to fetch team memberships for counter invalidation: %s", e
            )

        update_result = await self.prisma_client.db.litellm_teammembership.update_many(
            where={"budget_id": {"in": budget_ids}},
            data={
                "spend": 0,
            },
        )

        for m in memberships:
            await self._invalidate_spend_counter(
                f"spend:team_member:{m.user_id}:{m.team_id}"
            )

        return update_result

    async def reset_budget_for_keys_linked_to_budgets(
        self, budgets_to_reset: List[LiteLLM_BudgetTableFull]
    ):
        """
        Resets the spend for keys linked to budget tiers that are being reset.

        This handles keys that have budget_id but no budget_duration set on the key
        itself. Keys with budget_id rely on their linked budget tier's reset schedule
        rather than having their own budget_duration.

        Keys that have their own budget_duration are already handled by
        reset_budget_for_litellm_keys() and are excluded here to avoid
        double-resetting.
        """
        budget_ids = [
            budget.budget_id
            for budget in budgets_to_reset
            if budget.budget_id is not None
        ]
        if not budget_ids:
            return

        where_clause: dict = {
            "budget_id": {"in": budget_ids},
            "budget_duration": None,  # only keys without their own reset schedule
            "spend": {"gt": 0},  # only reset keys that have accumulated spend
        }

        try:
            keys = await self.prisma_client.db.litellm_verificationtoken.find_many(
                where=where_clause
            )
        except Exception as e:
            keys = []
            verbose_proxy_logger.warning(
                "Failed to fetch keys for counter invalidation: %s", e
            )

        update_result = (
            await self.prisma_client.db.litellm_verificationtoken.update_many(
                where=where_clause,
                data={
                    "spend": 0,
                },
            )
        )

        for k in keys:
            await self._invalidate_spend_counter(f"spend:key:{k.token}")

        return update_result

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

                budget_ids_to_reset = [
                    budget.budget_id
                    for budget in budgets_to_reset
                    if budget.budget_id is not None
                ]

                endusers_to_reset = await self.prisma_client.get_data(
                    table_name="enduser",
                    query_type="find_all",
                    budget_id_list=budget_ids_to_reset,
                )

                # Also reset end users with no budget_id (NULL) who use the
                # default budget via litellm.max_end_user_budget_id.  These
                # users are enforced in-memory but never had budget_id
                # persisted, so the query above misses them.
                if (
                    litellm.max_end_user_budget_id is not None
                    and litellm.max_end_user_budget_id in budget_ids_to_reset
                ):
                    default_budget_endusers = (
                        await self._get_endusers_with_no_budget_id()
                    )
                    if default_budget_endusers:
                        if endusers_to_reset is None:
                            endusers_to_reset = default_budget_endusers
                        else:
                            endusers_to_reset.extend(default_budget_endusers)

                await self.reset_budget_for_litellm_team_members(
                    budgets_to_reset=budgets_to_reset
                )

                await self.reset_budget_for_keys_linked_to_budgets(
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

    async def _get_endusers_with_no_budget_id(
        self,
    ) -> List[LiteLLM_EndUserTable]:
        """
        Fetch end users that have no explicit budget_id set (NULL) and have
        accumulated spend > 0.  These are implicitly-created end users that
        rely on the default budget (litellm.max_end_user_budget_id) applied
        in-memory during auth checks.
        """
        rows = await self.prisma_client.db.litellm_endusertable.find_many(
            where={
                "budget_id": None,
                "spend": {"gt": 0},
            },
        )
        return [LiteLLM_EndUserTable(**row.dict()) for row in rows]

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
                    for k in updated_keys:
                        token = getattr(k, "token", None)
                        if token:
                            await self._invalidate_spend_counter(f"spend:key:{token}")

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
                    for u in updated_users:
                        user_id = getattr(u, "user_id", None)
                        if user_id:
                            await self._invalidate_spend_counter(
                                f"spend:user:{user_id}"
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
                    for t in updated_teams:
                        team_id = getattr(t, "team_id", None)
                        if team_id:
                            await self._invalidate_spend_counter(
                                f"spend:team:{team_id}"
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
    async def _reset_expired_window(
        window: dict,
        counter_key: str,
        spend_counter_cache: Any,
        now: datetime,
    ) -> bool:
        """Reset a single budget window if expired. Returns True if the window was reset."""
        from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

        reset_at_str = window.get("reset_at")
        if not reset_at_str:
            return False
        reset_at = datetime.fromisoformat(reset_at_str.replace("Z", "+00:00")).replace(
            tzinfo=None
        )
        if reset_at > now:
            return False
        spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=0.0)
        if spend_counter_cache.redis_cache is not None:
            try:
                await spend_counter_cache.redis_cache.async_set_cache(
                    key=counter_key, value=0.0
                )
            except Exception as redis_err:
                verbose_proxy_logger.warning(
                    "Failed to reset Redis counter %s: %s", counter_key, redis_err
                )
        window["reset_at"] = get_budget_reset_time(
            budget_duration=window["budget_duration"]
        ).isoformat()
        return True

    async def reset_budget_windows(self) -> None:
        """
        For keys and teams with budget_limits, reset any individual windows where
        reset_at <= now. Only the expired windows are reset; other windows are untouched.
        """

        from litellm.proxy.proxy_server import spend_counter_cache

        now = datetime.utcnow()

        # Note on raw SQL: prisma-client-python does not support null-filtering
        # on `Json?` columns (no DbNull/JsonNull sentinel — see
        # RobertCraigie/prisma-client-py#714). We use `query_raw` with
        # `IS NOT NULL` so we don't materialize every key/team row on each
        # tick of the reset job. Writes still go through the ORM.

        # --- Keys ---
        try:
            key_rows = await self.prisma_client.db.query_raw(
                'SELECT token, budget_limits FROM "LiteLLM_VerificationToken" '
                "WHERE budget_limits IS NOT NULL"
            )
            for row in key_rows:
                raw = row["budget_limits"]
                if not raw:
                    continue
                windows: list = raw if isinstance(raw, list) else json.loads(raw)
                changed = False
                for window in windows:
                    counter_key = (
                        f"spend:key:{row['token']}:window:{window['budget_duration']}"
                    )
                    if await ResetBudgetJob._reset_expired_window(
                        window, counter_key, spend_counter_cache, now
                    ):
                        changed = True
                if changed:
                    await self.prisma_client.db.litellm_verificationtoken.update(
                        where={"token": row["token"]},
                        data={"budget_limits": json.dumps(windows)},  # type: ignore[arg-type]
                    )
        except Exception as e:
            verbose_proxy_logger.exception(
                "Failed to reset budget windows for keys: %s", e
            )

        # --- Teams ---
        try:
            team_rows = await self.prisma_client.db.query_raw(
                'SELECT team_id, budget_limits FROM "LiteLLM_TeamTable" '
                "WHERE budget_limits IS NOT NULL"
            )
            for row in team_rows:
                raw = row["budget_limits"]
                if not raw:
                    continue
                windows = raw if isinstance(raw, list) else json.loads(raw)
                changed = False
                for window in windows:
                    counter_key = f"spend:team:{row['team_id']}:window:{window['budget_duration']}"
                    if await ResetBudgetJob._reset_expired_window(
                        window, counter_key, spend_counter_cache, now
                    ):
                        changed = True
                if changed:
                    await self.prisma_client.db.litellm_teamtable.update(
                        where={"team_id": row["team_id"]},
                        data={"budget_limits": json.dumps(windows)},  # type: ignore[arg-type]
                    )
        except Exception as e:
            verbose_proxy_logger.exception(
                "Failed to reset budget windows for teams: %s", e
            )

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

            # Reset the cross-pod spend counter.
            # Reset Redis directly (not via DualCache) so a Redis failure
            # doesn't silently leave a stale counter that get_current_spend
            # would read as authoritative, permanently blocking the user.
            from litellm.proxy.proxy_server import spend_counter_cache

            counter_key = None
            if item_type == "key" and hasattr(item, "token") and item.token is not None:  # type: ignore[union-attr]
                counter_key = f"spend:key:{item.token}"  # type: ignore[union-attr]
            elif (
                item_type == "team"
                and hasattr(item, "team_id")
                and item.team_id is not None  # type: ignore[union-attr]
            ):
                counter_key = f"spend:team:{item.team_id}"  # type: ignore[union-attr]

            if counter_key is not None:
                # Always reset in-memory (local fallback)
                spend_counter_cache.in_memory_cache.set_cache(
                    key=counter_key, value=0.0
                )
                # Explicitly reset Redis with warning on failure
                if spend_counter_cache.redis_cache is not None:
                    try:
                        await spend_counter_cache.redis_cache.async_set_cache(
                            key=counter_key, value=0.0
                        )
                    except Exception as redis_err:
                        verbose_proxy_logger.warning(
                            "Failed to reset spend counter in Redis for %s key=%s: %s. "
                            "Budget may be over-enforced until counter expires.",
                            item_type,
                            counter_key,
                            redis_err,
                        )

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
                from litellm.proxy.common_utils.timezone_utils import (
                    get_budget_reset_time,
                )

                budget.budget_reset_at = get_budget_reset_time(
                    budget_duration=budget.budget_duration
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
