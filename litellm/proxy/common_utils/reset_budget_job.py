import asyncio
import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Final, Literal, Protocol, TypeVar

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import GLOBAL_PROXY_SPEND_CACHE_KEY, LITELLM_PROXY_BUDGET_NAME
from litellm.proxy._types import (
    LiteLLM_BudgetTableFull,
    LiteLLM_EndUserTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LiteLLM_VerificationToken,
)
from litellm.proxy.common_utils.timezone_utils import (
    BudgetResetSettings,
    compute_budget_reset_at,
    get_budget_reset_settings,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.organization_repository import OrganizationRepository
from litellm.repositories.prisma_protocols import ReadOnlyTable, SpendLinkedTable
from litellm.repositories.table_repositories import (
    EndUserRepository,
    TagRepository,
    TeamMembershipRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.unit_of_work import spend_reset_unit_of_work
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
from litellm.types.services import ServiceTypes

_RowT = TypeVar("_RowT")


class _TeamMembershipRow(Protocol):
    @property
    def user_id(self) -> str: ...

    @property
    def team_id(self) -> str: ...


class _KeyRow(Protocol):
    @property
    def token(self) -> str: ...


class _OrgRow(Protocol):
    @property
    def organization_id(self) -> str: ...


class _TagRow(Protocol):
    @property
    def tag_name(self) -> str: ...


def _team_membership_counter_key(row: _TeamMembershipRow) -> str:
    return f"spend:team_member:{row.user_id}:{row.team_id}"


def _team_membership_cache_key(row: _TeamMembershipRow) -> str:
    return f"{row.team_id}_{row.user_id}"


def _key_counter_key(row: _KeyRow) -> str:
    return f"spend:key:{row.token}"


def _key_cache_key(row: _KeyRow) -> str:
    return row.token


def _org_counter_key(row: _OrgRow) -> str:
    return f"spend:org:{row.organization_id}"


def _org_cache_keys(row: _OrgRow) -> Sequence[str]:
    return [
        f"org_id:{row.organization_id}",
        f"org_id:{row.organization_id}:with_budget",
    ]


def _tag_counter_key(row: _TagRow) -> str:
    return f"spend:tag:{row.tag_name}"


def _tag_cache_key(row: _TagRow) -> str:
    return f"tag:{row.tag_name}"


class ResetBudgetJob:
    """
    Resets the budget for all the keys, users, and teams that need it
    """

    def __init__(
        self,
        proxy_logging_obj: ProxyLogging,
        prisma_client: PrismaClient,
        reset_settings: BudgetResetSettings | None = None,
    ):
        self.proxy_logging_obj: ProxyLogging = proxy_logging_obj
        self.prisma_client: PrismaClient = prisma_client
        self.reset_settings: BudgetResetSettings = reset_settings or get_budget_reset_settings()

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

            spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=0.0, ttl=60)
            if spend_counter_cache.redis_cache is not None:
                try:
                    await spend_counter_cache.redis_cache.async_set_cache(key=counter_key, value=0.0, ttl=60)
                except Exception as redis_err:
                    verbose_proxy_logger.warning(
                        "Failed to reset spend counter %s in Redis: %s. "
                        "Budget may be over-enforced until counter expires.",
                        counter_key,
                        redis_err,
                    )
        except Exception as e:
            verbose_proxy_logger.warning("Failed to reset spend counter %s: %s", counter_key, e)

    @staticmethod
    async def _invalidate_global_proxy_spend_cache() -> None:
        """Drop the cached global-proxy spend accumulator after the proxy
        budget aggregate row is reset, so the next auth-time load reads the
        zeroed row instead of a stale (potentially never-expiring) counter.
        """
        await ResetBudgetJob._invalidate_user_api_key_cache_entry(GLOBAL_PROXY_SPEND_CACHE_KEY)

    @staticmethod
    async def _invalidate_user_api_key_cache_entry(cache_key: str) -> None:
        """Drop a stale management-cache entry so the next read fetches from DB.

        Tags and end-users are not reseeded by ``SpendCounterReseed.from_db``;
        for those, when the spend counter expires the budget check falls back
        to ``cached_obj.spend``. Keys, orgs, and team memberships are reseeded
        from the DB, but auth still may consult ``user_api_key_cache`` objects
        whose ``.spend`` field can lag a cross-pod DB reset. Deleting the cache
        entry forces the next auth-time fetch to reload the zeroed row from
        Postgres.
        """
        try:
            from litellm.proxy.proxy_server import user_api_key_cache

            await user_api_key_cache.async_delete_cache(key=cache_key)
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to invalidate user_api_key_cache entry %s: %s",
                cache_key,
                e,
            )

    async def _cascade_reset_spend_for_budget_link(
        self,
        budgets_to_reset: list[LiteLLM_BudgetTableFull],
        table: SpendLinkedTable[_RowT],
        counter_key_fn: Callable[[_RowT], str],
        log_subject: str,
        extra_where: dict[str, object] | None = None,
        cache_key_fn: Callable[[_RowT], str | Sequence[str]] | None = None,
    ):
        """
        Generic cascade: zero spend on rows whose budget_id is in the reset set.

        ``cache_key_fn`` is optional: when provided, after the DB update each
        matching row's entry or entries in ``user_api_key_cache`` are dropped so
        cached spend cannot stay pinned above the zeroed DB row after a reset.
        """
        budget_ids: Final = [b.budget_id for b in budgets_to_reset if b.budget_id is not None]
        if not budget_ids:
            return

        where: Final[dict[str, object]] = {"budget_id": {"in": budget_ids}}
        if extra_where:
            where.update(extra_where)

        try:
            rows: Sequence[_RowT] = await table.find_many(where=where)
        except Exception as e:
            rows = ()
            verbose_proxy_logger.warning("Failed to fetch %s for counter invalidation: %s", log_subject, e)

        update_result: Final = await table.update_many(where=where, data={"spend": 0})

        for row in rows:
            await self._invalidate_spend_counter(counter_key_fn(row))
            if cache_key_fn is not None:
                cache_keys = cache_key_fn(row)
                if isinstance(cache_keys, str):
                    cache_keys = [cache_keys]
                for cache_key in cache_keys:
                    await self._invalidate_user_api_key_cache_entry(cache_key)

        return update_result

    async def reset_budget_for_litellm_team_members(self, budgets_to_reset: list[LiteLLM_BudgetTableFull]):
        """
        Resets the budget for all LiteLLM Team Members if their budget has expired
        """
        return await self._cascade_reset_spend_for_budget_link(
            budgets_to_reset=budgets_to_reset,
            table=TeamMembershipRepository(self.prisma_client).table,
            counter_key_fn=_team_membership_counter_key,
            log_subject="team memberships",
            cache_key_fn=_team_membership_cache_key,
        )

    async def reset_budget_for_keys_linked_to_budgets(self, budgets_to_reset: list[LiteLLM_BudgetTableFull]):
        """
        Resets the spend for keys linked to budget tiers that are being reset.

        Excludes keys with their own budget_duration; those are reset by
        reset_budget_for_litellm_keys() to avoid double-resetting.
        """
        return await self._cascade_reset_spend_for_budget_link(
            budgets_to_reset=budgets_to_reset,
            table=VerificationTokenRepository(self.prisma_client).table,
            counter_key_fn=_key_counter_key,
            log_subject="keys",
            extra_where={"budget_duration": None, "spend": {"gt": 0}},
            cache_key_fn=_key_cache_key,
        )

    async def reset_budget_for_orgs_linked_to_budgets(self, budgets_to_reset: list[LiteLLM_BudgetTableFull]):
        """
        Resets the spend for orgs linked to budget tiers that are being reset.
        """
        return await self._cascade_reset_spend_for_budget_link(
            budgets_to_reset=budgets_to_reset,
            table=OrganizationRepository(self.prisma_client).table,
            counter_key_fn=_org_counter_key,
            log_subject="orgs",
            extra_where={"spend": {"gt": 0}},
            cache_key_fn=_org_cache_keys,
        )

    async def reset_budget_for_tags_linked_to_budgets(self, budgets_to_reset: list[LiteLLM_BudgetTableFull]):
        """
        Resets the spend for tags linked to budget tiers that are being reset.

        Also drops each tag's ``user_api_key_cache`` entry so the next
        ``_tag_max_budget_check`` reloads the zeroed row from the DB.
        ``SpendCounterReseed.from_db`` intentionally returns ``None`` for
        tags, so the budget check falls back to the cached
        ``LiteLLM_TagTable.spend`` once the spend counter expires; without
        this invalidation, that stale ``.spend`` keeps the tag over-budget
        indefinitely.
        """
        return await self._cascade_reset_spend_for_budget_link(
            budgets_to_reset=budgets_to_reset,
            table=TagRepository(self.prisma_client).table,
            counter_key_fn=_tag_counter_key,
            log_subject="tags",
            extra_where={"spend": {"gt": 0}},
            cache_key_fn=_tag_cache_key,
        )

    async def reset_budget_for_litellm_budget_table(self):
        """
        Resets the budget for all LiteLLM End-Users (Customers), and Team Members if their budget has expired
        The corresponding Budget duration is also updated.
        """

        now: Final = datetime.now(timezone.utc)
        start_time: Final = time.time()
        endusers_to_reset: list[LiteLLM_EndUserTable] | None = None
        budgets_to_reset: list[LiteLLM_BudgetTableFull] | None = None
        updated_endusers: Final[list[LiteLLM_EndUserTable]] = []
        failed_endusers: Final = []
        try:
            budgets_to_reset = await self.prisma_client.get_data(
                table_name="budget", query_type="find_all", reset_at=now
            )

            if budgets_to_reset is not None and len(budgets_to_reset) > 0:
                for budget in budgets_to_reset:
                    budget = await ResetBudgetJob._reset_budget_reset_at_date(budget, now, self.reset_settings)

                await self.prisma_client.update_data(
                    query_type="update_many",
                    data_list=budgets_to_reset,
                    table_name="budget",
                )

                budget_ids_to_reset = [budget.budget_id for budget in budgets_to_reset if budget.budget_id is not None]

                endusers_to_reset = await self.prisma_client.get_data(
                    table_name="enduser",
                    query_type="find_all",
                    budget_id_list=budget_ids_to_reset,
                )

                # Also reset end users with no budget_id (NULL) who use the
                # default budget via litellm.max_end_user_budget_id.  These
                # users are enforced in-memory but never had budget_id
                # persisted, so the query above misses them.
                if litellm.max_end_user_budget_id is not None and litellm.max_end_user_budget_id in budget_ids_to_reset:
                    default_budget_endusers: Final = await self._get_endusers_with_no_budget_id()
                    if default_budget_endusers:
                        if endusers_to_reset is None:
                            endusers_to_reset = default_budget_endusers
                        else:
                            endusers_to_reset.extend(default_budget_endusers)

                await self.reset_budget_for_litellm_team_members(budgets_to_reset=budgets_to_reset)

                await self.reset_budget_for_keys_linked_to_budgets(budgets_to_reset=budgets_to_reset)

                await self.reset_budget_for_orgs_linked_to_budgets(budgets_to_reset=budgets_to_reset)

                await self.reset_budget_for_tags_linked_to_budgets(budgets_to_reset=budgets_to_reset)

            if endusers_to_reset is not None and len(endusers_to_reset) > 0:
                for enduser in endusers_to_reset:
                    try:
                        updated_enduser = await ResetBudgetJob._reset_budget_for_enduser(enduser=enduser)
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
                        verbose_proxy_logger.exception("Failed to reset budget for enduser: %s", enduser)

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
                        "num_budgets_found": (len(budgets_to_reset) if budgets_to_reset else 0),
                        "budgets_found": json.dumps(budgets_to_reset, indent=4, default=str),
                        "num_endusers_found": (len(endusers_to_reset) if endusers_to_reset else 0),
                        "endusers_found": json.dumps(endusers_to_reset, indent=4, default=str),
                        "num_endusers_updated": len(updated_endusers),
                        "endusers_updated": json.dumps(updated_endusers, indent=4, default=str),
                        "num_endusers_failed": len(failed_endusers),
                        "endusers_failed": json.dumps(failed_endusers, indent=4, default=str),
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
                        "num_budgets_found": (len(budgets_to_reset) if budgets_to_reset else 0),
                        "budgets_found": json.dumps(budgets_to_reset, indent=4, default=str),
                        "num_endusers_found": (len(endusers_to_reset) if endusers_to_reset else 0),
                        "endusers_found": json.dumps(endusers_to_reset, indent=4, default=str),
                    },
                )
            )
            verbose_proxy_logger.exception("Failed to reset budget for endusers: %s", e)

    async def _get_endusers_with_no_budget_id(
        self,
    ) -> list[LiteLLM_EndUserTable]:
        """
        Fetch end users that have no explicit budget_id set (NULL) and have
        accumulated spend > 0.  These are implicitly-created end users that
        rely on the default budget (litellm.max_end_user_budget_id) applied
        in-memory during auth checks.
        """
        table: Final[ReadOnlyTable] = EndUserRepository(self.prisma_client).table
        rows: Final = await table.find_many(
            where={
                "budget_id": None,
                "spend": {"gt": 0},
            },
        )
        return [LiteLLM_EndUserTable.model_validate(row.dict()) for row in rows]

    async def _write_key_reset_updates(self, updated_keys: list[LiteLLM_VerificationToken]) -> None:
        """
        Write per-row {spend, budget_reset_at} updates for keys.

        Avoids the batched full-model update path, which trips
        prisma.errors.DataError on any row carrying object_permission_id or
        budget_limits (see #27730). Both fields are rejected by Prisma's
        update input type for LiteLLM_VerificationToken, and the failure
        aborts the entire batch — silently leaving spend over the cap and
        budget_reset_at unchanged forever.
        """
        async with spend_reset_unit_of_work(self.prisma_client.db.batch_) as uow:
            for k in updated_keys:
                if k.token is None:
                    continue
                uow.keys.queue_spend_reset(token=k.token, budget_reset_at=k.budget_reset_at)

    async def _write_user_reset_updates(self, updated_users: list[LiteLLM_UserTable]) -> None:
        """
        Write per-row {spend, budget_reset_at} updates for users.

        Mirrors _write_key_reset_updates — avoids the full-model update path
        that trips Prisma's DataError on rows carrying unrecognised fields
        (see #27730).
        """
        async with spend_reset_unit_of_work(self.prisma_client.db.batch_) as uow:
            for u in updated_users:
                uow.users.queue_spend_reset(user_id=u.user_id, budget_reset_at=u.budget_reset_at)

    async def _write_team_reset_updates(self, updated_teams: list[LiteLLM_TeamTable]) -> None:
        """
        Write per-row {spend, budget_reset_at} updates for teams.

        Mirrors _write_key_reset_updates — avoids the full-model update path
        that trips Prisma's DataError on rows carrying unrecognised fields
        (see #27730).
        """
        async with spend_reset_unit_of_work(self.prisma_client.db.batch_) as uow:
            for t in updated_teams:
                uow.teams.queue_spend_reset(team_id=t.team_id, budget_reset_at=t.budget_reset_at)

    async def reset_budget_for_litellm_keys(self):
        """
        Resets the budget for all the litellm keys

        Catches Exceptions and logs them
        """
        now: Final = datetime.utcnow()
        start_time: Final = time.time()
        keys_to_reset: list[LiteLLM_VerificationToken] | None = None
        try:
            keys_to_reset = await self.prisma_client.get_data(
                table_name="key", query_type="find_all", expires=now, reset_at=now
            )
            verbose_proxy_logger.debug("Keys to reset %s", json.dumps(keys_to_reset, indent=4, default=str))
            updated_keys: Final[list[LiteLLM_VerificationToken]] = []
            failed_keys: Final = []
            if keys_to_reset is not None and len(keys_to_reset) > 0:
                for key in keys_to_reset:
                    try:
                        updated_key = await ResetBudgetJob._reset_budget_for_key(
                            key=key,
                            current_time=now,
                            reset_settings=self.reset_settings,
                        )
                        if updated_key is not None:
                            updated_keys.append(updated_key)
                        else:
                            failed_keys.append({"key": key, "error": "Returned None without exception"})
                    except Exception as e:
                        failed_keys.append({"key": key, "error": str(e)})
                        verbose_proxy_logger.exception("Failed to reset budget for key: %s", key)

                verbose_proxy_logger.debug("Updated keys %s", json.dumps(updated_keys, indent=4, default=str))

                if updated_keys:
                    await self._write_key_reset_updates(updated_keys=updated_keys)
                    for k in updated_keys:
                        token = getattr(k, "token", None)
                        if token:
                            await self._invalidate_spend_counter(f"spend:key:{token}")

            end_time = time.time()
            if len(failed_keys) > 0:  # If any keys failed to reset
                raise Exception(f"Failed to reset {len(failed_keys)} keys: {json.dumps(failed_keys, default=str)}")

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
        now: Final = datetime.utcnow()
        start_time: Final = time.time()
        users_to_reset: list[LiteLLM_UserTable] | None = None
        try:
            users_to_reset = await self.prisma_client.get_data(table_name="user", query_type="find_all", reset_at=now)
            updated_users: Final[list[LiteLLM_UserTable]] = []
            failed_users: Final = []
            if users_to_reset is not None and len(users_to_reset) > 0:
                for user in users_to_reset:
                    try:
                        updated_user = await ResetBudgetJob._reset_budget_for_user(
                            user=user,
                            current_time=now,
                            reset_settings=self.reset_settings,
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
                        verbose_proxy_logger.exception("Failed to reset budget for user: %s", user)

                verbose_proxy_logger.debug("Updated users %s", json.dumps(updated_users, indent=4, default=str))
                if updated_users:
                    await self._write_user_reset_updates(updated_users=updated_users)
                    for u in updated_users:
                        user_id = getattr(u, "user_id", None)
                        if user_id:
                            await self._invalidate_spend_counter(f"spend:user:{user_id}")
                        if user_id == LITELLM_PROXY_BUDGET_NAME:
                            await self._invalidate_global_proxy_spend_cache()

            end_time = time.time()
            if len(failed_users) > 0:  # If any users failed to reset
                raise Exception(f"Failed to reset {len(failed_users)} users: {json.dumps(failed_users, default=str)}")

            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_success_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    call_type="reset_budget_users",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_users_found": len(users_to_reset) if users_to_reset else 0,
                        "users_found": json.dumps(users_to_reset, indent=4, default=str),
                        "num_users_updated": len(updated_users),
                        "users_updated": json.dumps(updated_users, indent=4, default=str),
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
                        "users_found": json.dumps(users_to_reset, indent=4, default=str),
                    },
                )
            )
            verbose_proxy_logger.exception("Failed to reset budget for users: %s", e)

    async def reset_budget_for_litellm_teams(self):
        """
        Resets the budget for all LiteLLM Internal Teams if their budget has expired
        """
        now: Final = datetime.utcnow()
        start_time: Final = time.time()
        teams_to_reset: list[LiteLLM_TeamTable] | None = None
        try:
            teams_to_reset = await self.prisma_client.get_data(table_name="team", query_type="find_all", reset_at=now)
            updated_teams: Final[list[LiteLLM_TeamTable]] = []
            failed_teams: Final = []
            if teams_to_reset is not None and len(teams_to_reset) > 0:
                for team in teams_to_reset:
                    try:
                        updated_team = await ResetBudgetJob._reset_budget_for_team(
                            team=team,
                            current_time=now,
                            reset_settings=self.reset_settings,
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
                        verbose_proxy_logger.exception("Failed to reset budget for team: %s", team)

                verbose_proxy_logger.debug("Updated teams %s", json.dumps(updated_teams, indent=4, default=str))
                if updated_teams:
                    await self._write_team_reset_updates(updated_teams=updated_teams)
                    for t in updated_teams:
                        team_id = getattr(t, "team_id", None)
                        if team_id:
                            await self._invalidate_spend_counter(f"spend:team:{team_id}")

            end_time = time.time()
            if len(failed_teams) > 0:  # If any teams failed to reset
                raise Exception(f"Failed to reset {len(failed_teams)} teams: {json.dumps(failed_teams, default=str)}")

            asyncio.create_task(
                self.proxy_logging_obj.service_logging_obj.async_service_success_hook(
                    service=ServiceTypes.RESET_BUDGET_JOB,
                    duration=end_time - start_time,
                    call_type="reset_budget_teams",
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_teams_found": len(teams_to_reset) if teams_to_reset else 0,
                        "teams_found": json.dumps(teams_to_reset, indent=4, default=str),
                        "num_teams_updated": len(updated_teams),
                        "teams_updated": json.dumps(updated_teams, indent=4, default=str),
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
                        "teams_found": json.dumps(teams_to_reset, indent=4, default=str),
                    },
                )
            )
            verbose_proxy_logger.exception("Failed to reset budget for teams: %s", e)

    @staticmethod
    async def _reset_expired_window(
        window: dict,
        counter_key: str,
        spend_counter_cache: DualCache,
        now: datetime,
        reset_settings: BudgetResetSettings,
    ) -> bool:
        """Reset a single budget window if expired. Returns True if the window was reset."""
        reset_at_str: Final = window.get("reset_at")
        if not reset_at_str:
            return False
        reset_at: Final = datetime.fromisoformat(reset_at_str.replace("Z", "+00:00")).replace(tzinfo=None)
        if reset_at > now:
            return False
        spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=0.0)
        if spend_counter_cache.redis_cache is not None:
            try:
                await spend_counter_cache.redis_cache.async_set_cache(key=counter_key, value=0.0)
            except Exception as redis_err:
                verbose_proxy_logger.warning("Failed to reset Redis counter %s: %s", counter_key, redis_err)
        window["reset_at"] = compute_budget_reset_at(
            budget_duration=window["budget_duration"], settings=reset_settings
        ).isoformat()
        return True

    async def reset_budget_windows(self) -> None:
        """
        For keys and teams with budget_limits, reset any individual windows where
        reset_at <= now. Only the expired windows are reset; other windows are untouched.
        """

        from litellm.proxy.proxy_server import spend_counter_cache

        now: Final = datetime.utcnow()

        # Note on raw SQL: prisma-client-python does not support null-filtering
        # on `Json?` columns (no DbNull/JsonNull sentinel — see
        # RobertCraigie/prisma-client-py#714). We use `query_raw` with
        # `IS NOT NULL` so we don't materialize every key/team row on each
        # tick of the reset job. Writes still go through the ORM.

        # --- Keys ---
        try:
            key_rows: Final = await self.prisma_client.db.query_raw(
                'SELECT token, budget_limits FROM "LiteLLM_VerificationToken" WHERE budget_limits IS NOT NULL'
            )
            for row in key_rows:
                raw = row["budget_limits"]
                if not raw:
                    continue
                windows: list = raw if isinstance(raw, list) else json.loads(raw)
                changed = False
                for window in windows:
                    counter_key = f"spend:key:{row['token']}:window:{window['budget_duration']}"
                    if await ResetBudgetJob._reset_expired_window(
                        window,
                        counter_key,
                        spend_counter_cache,
                        now,
                        self.reset_settings,
                    ):
                        changed = True
                if changed:
                    await VerificationTokenRepository(self.prisma_client).table.update(
                        where={"token": row["token"]},
                        data={"budget_limits": json.dumps(windows)},
                    )
        except Exception as e:
            verbose_proxy_logger.exception("Failed to reset budget windows for keys: %s", e)

        # --- Teams ---
        try:
            team_rows: Final = await self.prisma_client.db.query_raw(
                'SELECT team_id, budget_limits FROM "LiteLLM_TeamTable" WHERE budget_limits IS NOT NULL'
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
                        window,
                        counter_key,
                        spend_counter_cache,
                        now,
                        self.reset_settings,
                    ):
                        changed = True
                if changed:
                    await TeamRepository(self.prisma_client).table.update(
                        where={"team_id": row["team_id"]},
                        data={"budget_limits": json.dumps(windows)},
                    )
        except Exception as e:
            verbose_proxy_logger.exception("Failed to reset budget windows for teams: %s", e)

    @staticmethod
    async def _reset_budget_common(
        item: LiteLLM_TeamTable | LiteLLM_UserTable | LiteLLM_VerificationToken,
        current_time: datetime,
        item_type: Literal["key", "team", "user"],
        reset_settings: BudgetResetSettings,
    ):
        """
        In-place, updates spend=0, and sets budget_reset_at to current_time + budget_duration

        Common logic for resetting budget for a team, user, or key.

        Spend-counter invalidation happens in the caller, AFTER the DB write
        commits. Zeroing the counter here would open a bypass window when the
        DB write fails: get_current_spend reads 0 from Redis while the DB
        still holds the pre-reset value, admitting requests past the cap.
        """
        try:
            item.spend = 0.0
            if hasattr(item, "budget_duration") and item.budget_duration is not None:
                item.budget_reset_at = compute_budget_reset_at(
                    budget_duration=item.budget_duration, settings=reset_settings
                )
            return item
        except Exception as e:
            verbose_proxy_logger.exception("Error resetting budget for %s: %s. Item: %s", item_type, e, item)
            raise e

    @staticmethod
    async def _reset_budget_for_team(
        team: LiteLLM_TeamTable,
        current_time: datetime,
        reset_settings: BudgetResetSettings,
    ) -> LiteLLM_TeamTable | None:
        await ResetBudgetJob._reset_budget_common(
            item=team,
            current_time=current_time,
            item_type="team",
            reset_settings=reset_settings,
        )
        return team

    @staticmethod
    async def _reset_budget_for_user(
        user: LiteLLM_UserTable,
        current_time: datetime,
        reset_settings: BudgetResetSettings,
    ) -> LiteLLM_UserTable | None:
        await ResetBudgetJob._reset_budget_common(
            item=user,
            current_time=current_time,
            item_type="user",
            reset_settings=reset_settings,
        )
        return user

    @staticmethod
    async def _reset_budget_for_enduser(
        enduser: LiteLLM_EndUserTable,
    ) -> LiteLLM_EndUserTable | None:
        try:
            enduser.spend = 0.0
        except Exception as e:
            verbose_proxy_logger.exception("Error resetting budget for enduser: %s. Item: %s", e, enduser)
            raise e
        return enduser

    @staticmethod
    async def _reset_budget_reset_at_date(
        budget: LiteLLM_BudgetTableFull,
        current_time: datetime,
        reset_settings: BudgetResetSettings,
    ) -> LiteLLM_BudgetTableFull:
        try:
            if budget.budget_duration is not None:
                budget.budget_reset_at = compute_budget_reset_at(
                    budget_duration=budget.budget_duration, settings=reset_settings
                )
        except Exception as e:
            verbose_proxy_logger.exception("Error resetting budget_reset_at for budget: %s. Item: %s", e, budget)
            raise e
        return budget

    @staticmethod
    async def _reset_budget_for_key(
        key: LiteLLM_VerificationToken,
        current_time: datetime,
        reset_settings: BudgetResetSettings,
    ) -> LiteLLM_VerificationToken | None:
        await ResetBudgetJob._reset_budget_common(
            item=key,
            current_time=current_time,
            item_type="key",
            reset_settings=reset_settings,
        )
        return key
