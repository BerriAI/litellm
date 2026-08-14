import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeVar, assert_never

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import (
    GLOBAL_PROXY_SPEND_CACHE_KEY,
    LITELLM_PROXY_BUDGET_NAME,
    RESET_BUDGET_JOB_BATCH_SIZE,
    RESET_BUDGET_JOB_MAX_CHUNKS_PER_RUN,
)
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
from litellm.repositories.unit_of_work import (
    budget_cascade_unit_of_work,
    spend_reset_unit_of_work,
)
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
from litellm.types.services import ServiceTypes

_RowT = TypeVar("_RowT")

_LINKED_KEYS_WHERE: Final[Mapping[str, object]] = MappingProxyType({"budget_duration": None, "spend": {"gt": 0}})
_SPENT_ROWS_WHERE: Final[Mapping[str, object]] = MappingProxyType({"spend": {"gt": 0}})


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


class _EndUserRow(Protocol):
    @property
    def user_id(self) -> str: ...


def _team_membership_counter_key(row: _TeamMembershipRow) -> str:
    return f"spend:team_member:{row.user_id}:{row.team_id}"


def _team_membership_cache_keys(row: _TeamMembershipRow) -> tuple[str, ...]:
    return (f"{row.team_id}_{row.user_id}",)


def _key_counter_key(row: _KeyRow) -> str:
    return f"spend:key:{row.token}"


def _key_cache_keys(row: _KeyRow) -> tuple[str, ...]:
    return (row.token,)


def _org_counter_key(row: _OrgRow) -> str:
    return f"spend:org:{row.organization_id}"


def _org_cache_keys(row: _OrgRow) -> tuple[str, ...]:
    return (
        f"org_id:{row.organization_id}",
        f"org_id:{row.organization_id}:with_budget",
    )


def _tag_counter_key(row: _TagRow) -> str:
    return f"spend:tag:{row.tag_name}"


def _tag_cache_keys(row: _TagRow) -> tuple[str, ...]:
    return (f"tag:{row.tag_name}",)


def _budget_link_where(
    budget_ids: Sequence[str],
    extra: Mapping[str, object] = MappingProxyType({}),
) -> dict[str, object]:
    return {"budget_id": {"in": list(budget_ids)}, **extra}


@dataclass(frozen=True, slots=True)
class _BudgetCascade:
    """Everything one budget-tier reset touches, resolved before any write."""

    budgets: tuple[LiteLLM_BudgetTableFull, ...] = ()
    budget_ids: tuple[str, ...] = ()
    budget_resets: tuple[tuple[str, datetime], ...] = ()
    endusers: tuple[_EndUserRow, ...] = ()
    counter_keys: tuple[str, ...] = ()
    cache_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _BudgetCascadeCommitted:
    cascade: _BudgetCascade
    advanced: int


@dataclass(frozen=True, slots=True)
class _BudgetCascadeFailed:
    cascade: _BudgetCascade
    error: Exception


_EMPTY_CASCADE: Final = _BudgetCascade()


@dataclass(frozen=True, slots=True)
class _ChunkOutcome:
    """One chunk of a reset phase: rows read, and rows whose new budget_reset_at
    cleared the due cutoff. Anything else is still due and would come straight
    back on the next fetch, so it is not progress."""

    fetched: int
    advanced: int


_NO_PROGRESS: Final = _ChunkOutcome(fetched=0, advanced=0)


def _as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _count_advanced(reset_ats: Iterable[object], cutoff: datetime) -> int:
    """How many rows the write actually moved past the due cutoff.

    A budget_duration of "0s" (or one the parser cannot read) resolves to the
    current time, so the row is written and stays due. Counting it as progress
    would re-read the same chunk until the per-run cap on every tick.
    """
    utc_cutoff: Final = _as_utc(cutoff)
    return sum(1 for reset_at in reset_ats if isinstance(reset_at, datetime) and _as_utc(reset_at) > utc_cutoff)


def _phase_is_drained(outcome: _ChunkOutcome) -> bool:
    """A short chunk means the due rows ran out. A full chunk that advanced
    nothing would be re-read unchanged forever, so it ends the phase too and
    those rows wait for the next tick."""
    return outcome.fetched < RESET_BUDGET_JOB_BATCH_SIZE or outcome.advanced == 0


async def _run_phase_in_chunks(process_chunk: Callable[[], Awaitable[_ChunkOutcome]]) -> None:
    """Drive one reset phase a chunk at a time, capped so a single run cannot
    spin unbounded: leftovers are picked up by the next tick."""
    for _ in range(RESET_BUDGET_JOB_MAX_CHUNKS_PER_RUN):
        if _phase_is_drained(await process_chunk()):
            return


def _budget_cascade_event_metadata(cascade: _BudgetCascade) -> dict[str, object]:
    return {
        "num_budgets_found": len(cascade.budgets),
        "budgets_found": json.dumps(cascade.budgets, indent=4, default=str),
        "num_endusers_found": len(cascade.endusers),
        "endusers_found": json.dumps(cascade.endusers, indent=4, default=str),
    }


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
        if self.prisma_client is None:
            return

        await self.reset_budget_for_litellm_keys()
        await self.reset_budget_for_litellm_users()
        await self.reset_budget_for_litellm_teams()
        await self.reset_budget_for_litellm_budget_table()
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

    async def _fetch_linked_rows(
        self,
        table: SpendLinkedTable[_RowT],
        where: Mapping[str, object],
        log_subject: str,
    ) -> tuple[_RowT, ...]:
        """Read the rows the cascade will zero, so their counters can be
        invalidated once the transaction commits."""
        try:
            return tuple(await table.find_many(where=where))
        except Exception as e:
            verbose_proxy_logger.warning("Failed to fetch %s for counter invalidation: %s", log_subject, e)
            return ()

    async def _collect_endusers_to_reset(self, budget_ids: Sequence[str]) -> tuple[_EndUserRow, ...]:
        linked: Final[Sequence[_EndUserRow] | None] = await self.prisma_client.get_data(
            table_name="enduser",
            query_type="find_all",
            budget_id_list=list(budget_ids),
        )
        if litellm.max_end_user_budget_id is None or litellm.max_end_user_budget_id not in budget_ids:
            return tuple(linked or ())
        return (*(linked or ()), *await self._get_endusers_with_no_budget_id())

    async def _collect_budget_cascade(self, budgets_to_reset: Sequence[LiteLLM_BudgetTableFull]) -> _BudgetCascade:
        """Resolve every row the expiring budget tiers gate, before any write.

        Keys carrying their own budget_duration are left out: they run on their
        own schedule via reset_budget_for_litellm_keys(), so sweeping them here
        would reset them twice.
        """
        budget_ids: Final = tuple(b.budget_id for b in budgets_to_reset if b.budget_id is not None)
        if not budget_ids:
            return _EMPTY_CASCADE

        team_memberships: Final[tuple[_TeamMembershipRow, ...]] = await self._fetch_linked_rows(
            table=TeamMembershipRepository(self.prisma_client).table,
            where=_budget_link_where(budget_ids),
            log_subject="team memberships",
        )
        keys: Final[tuple[_KeyRow, ...]] = await self._fetch_linked_rows(
            table=VerificationTokenRepository(self.prisma_client).table,
            where=_budget_link_where(budget_ids, _LINKED_KEYS_WHERE),
            log_subject="keys",
        )
        orgs: Final[tuple[_OrgRow, ...]] = await self._fetch_linked_rows(
            table=OrganizationRepository(self.prisma_client).table,
            where=_budget_link_where(budget_ids, _SPENT_ROWS_WHERE),
            log_subject="orgs",
        )
        tags: Final[tuple[_TagRow, ...]] = await self._fetch_linked_rows(
            table=TagRepository(self.prisma_client).table,
            where=_budget_link_where(budget_ids, _SPENT_ROWS_WHERE),
            log_subject="tags",
        )
        return _BudgetCascade(
            budgets=tuple(budgets_to_reset),
            budget_ids=budget_ids,
            budget_resets=tuple(
                (
                    b.budget_id,
                    compute_budget_reset_at(budget_duration=b.budget_duration, settings=self.reset_settings),
                )
                for b in budgets_to_reset
                if b.budget_id is not None and b.budget_duration is not None
            ),
            endusers=await self._collect_endusers_to_reset(budget_ids),
            counter_keys=(
                *(_team_membership_counter_key(row) for row in team_memberships),
                *(_key_counter_key(row) for row in keys),
                *(_org_counter_key(row) for row in orgs),
                *(_tag_counter_key(row) for row in tags),
            ),
            cache_keys=(
                *(key for row in team_memberships for key in _team_membership_cache_keys(row)),
                *(key for row in keys for key in _key_cache_keys(row)),
                *(key for row in orgs for key in _org_cache_keys(row)),
                *(key for row in tags for key in _tag_cache_keys(row)),
            ),
        )

    async def _commit_budget_cascade(self, cascade: _BudgetCascade) -> None:
        """Zero the gated spend and advance ``budget_reset_at`` in one transaction.

        Advancing the window on its own hides the tier from every later tick
        while its dependents stay pinned at the cap for the whole window;
        batching both means a mid-cascade failure persists nothing and the rows
        stay due for the next run.
        """
        if not cascade.budget_ids:
            return

        enduser_ids: Final = tuple(row.user_id for row in cascade.endusers)
        async with budget_cascade_unit_of_work(self.prisma_client.db.batch_) as uow:
            uow.team_memberships.queue_spend_zero(where=_budget_link_where(cascade.budget_ids))
            uow.keys.queue_spend_zero(where=_budget_link_where(cascade.budget_ids, _LINKED_KEYS_WHERE))
            uow.organizations.queue_spend_zero(where=_budget_link_where(cascade.budget_ids, _SPENT_ROWS_WHERE))
            uow.tags.queue_spend_zero(where=_budget_link_where(cascade.budget_ids, _SPENT_ROWS_WHERE))
            if enduser_ids:
                uow.endusers.queue_spend_zero(where={"user_id": {"in": list(enduser_ids)}})
            for budget_id, budget_reset_at in cascade.budget_resets:
                uow.budgets.queue_window_advance(budget_id=budget_id, budget_reset_at=budget_reset_at)

    async def _invalidate_budget_cascade_caches(self, cascade: _BudgetCascade) -> None:
        for counter_key in cascade.counter_keys:
            await self._invalidate_spend_counter(counter_key)
        for cache_key in cascade.cache_keys:
            await self._invalidate_user_api_key_cache_entry(cache_key)

    async def _reset_expired_budget_cascade(self) -> _BudgetCascadeCommitted | _BudgetCascadeFailed:
        now: Final = datetime.now(timezone.utc)
        try:
            budgets_to_reset: Final[Sequence[LiteLLM_BudgetTableFull] | None] = await self.prisma_client.get_data(
                table_name="budget",
                query_type="find_all",
                reset_at=now,
                limit=RESET_BUDGET_JOB_BATCH_SIZE,
            )
            cascade: Final = await self._collect_budget_cascade(budgets_to_reset or ())
        except Exception as e:
            return _BudgetCascadeFailed(cascade=_EMPTY_CASCADE, error=e)

        try:
            await self._commit_budget_cascade(cascade)
        except Exception as e:
            return _BudgetCascadeFailed(cascade=cascade, error=e)

        await self._invalidate_budget_cascade_caches(cascade)
        return _BudgetCascadeCommitted(
            cascade=cascade,
            advanced=_count_advanced(
                (reset_at for _, reset_at in cascade.budget_resets),
                cutoff=datetime.now(timezone.utc),
            ),
        )

    async def reset_budget_for_litellm_budget_table(self) -> None:
        """
        Resets the spend a budget tier gates (end users, team members, keys,
        orgs, tags) and advances the tier's budget_reset_at, atomically.

        Caches are invalidated only after the transaction commits, so a failed
        run cannot leave a zeroed counter in front of an un-reset DB row.
        """
        await _run_phase_in_chunks(self._reset_budget_for_litellm_budget_table_chunk)

    async def _reset_budget_for_litellm_budget_table_chunk(self) -> _ChunkOutcome:
        start_time: Final = time.time()
        outcome: Final = await self._reset_expired_budget_cascade()
        end_time: Final = time.time()

        match outcome:
            case _BudgetCascadeCommitted(cascade=cascade, advanced=advanced):
                asyncio.create_task(
                    self.proxy_logging_obj.service_logging_obj.async_service_success_hook(
                        service=ServiceTypes.RESET_BUDGET_JOB,
                        duration=end_time - start_time,
                        call_type="reset_budget_budget_table",
                        start_time=start_time,
                        end_time=end_time,
                        event_metadata={
                            **_budget_cascade_event_metadata(cascade),
                            "num_endusers_updated": len(cascade.endusers),
                            "num_endusers_failed": 0,
                        },
                    )
                )
                return _ChunkOutcome(fetched=len(cascade.budgets), advanced=advanced)
            case _BudgetCascadeFailed(cascade=cascade, error=error):
                verbose_proxy_logger.exception(
                    "Failed to reset the budget table cascade (team member, enduser, org and tag spend, plus "
                    "budget_reset_at); nothing was committed and the budgets stay due for the next run: %s",
                    error,
                    exc_info=error,
                )
                asyncio.create_task(
                    self.proxy_logging_obj.service_logging_obj.async_service_failure_hook(
                        service=ServiceTypes.RESET_BUDGET_JOB,
                        duration=end_time - start_time,
                        error=error,
                        call_type="reset_budget_endusers",
                        start_time=start_time,
                        end_time=end_time,
                        event_metadata=_budget_cascade_event_metadata(cascade),
                    )
                )
                return _NO_PROGRESS
            case _:
                assert_never(outcome)

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

    def _emit_phase_failure(
        self,
        call_type: str,
        error: Exception,
        start_time: float,
        end_time: float,
        event_metadata: dict[str, object],
    ) -> None:
        """Report rows that could not be reset without failing the chunk: the
        rows that did reset are already committed, and raising here would cost
        the phase every remaining chunk this tick.
        """
        verbose_proxy_logger.error("%s: %s", call_type, error)
        asyncio.create_task(
            self.proxy_logging_obj.service_logging_obj.async_service_failure_hook(
                service=ServiceTypes.RESET_BUDGET_JOB,
                duration=end_time - start_time,
                error=error,
                call_type=call_type,
                start_time=start_time,
                end_time=end_time,
                event_metadata=event_metadata,
            )
        )

    async def reset_budget_for_litellm_keys(self) -> None:
        """
        Resets the budget for all the litellm keys

        Catches Exceptions and logs them
        """
        await _run_phase_in_chunks(self._reset_budget_for_litellm_keys_chunk)

    async def _reset_budget_for_litellm_keys_chunk(self) -> _ChunkOutcome:
        now: Final = datetime.utcnow()
        start_time: Final = time.time()
        keys_to_reset: list[LiteLLM_VerificationToken] | None = None
        try:
            keys_to_reset = await self.prisma_client.get_data(
                table_name="key",
                query_type="find_all",
                expires=now,
                reset_at=now,
                limit=RESET_BUDGET_JOB_BATCH_SIZE,
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
            outcome: Final = _ChunkOutcome(
                fetched=len(keys_to_reset) if keys_to_reset else 0,
                advanced=_count_advanced(
                    (k.budget_reset_at for k in updated_keys),
                    cutoff=datetime.now(timezone.utc),
                ),
            )
            if len(failed_keys) > 0:
                self._emit_phase_failure(
                    call_type="reset_budget_keys",
                    error=Exception(f"Failed to reset {len(failed_keys)} keys: {json.dumps(failed_keys, default=str)}"),
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_keys_found": len(keys_to_reset) if keys_to_reset else 0,
                        "keys_found": json.dumps(keys_to_reset, indent=4, default=str),
                    },
                )
                return outcome

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
            return _NO_PROGRESS
        else:
            return outcome

    async def reset_budget_for_litellm_users(self) -> None:
        """
        Resets the budget for all LiteLLM Internal Users if their budget has expired
        """
        await _run_phase_in_chunks(self._reset_budget_for_litellm_users_chunk)

    async def _reset_budget_for_litellm_users_chunk(self) -> _ChunkOutcome:
        now: Final = datetime.utcnow()
        start_time: Final = time.time()
        users_to_reset: list[LiteLLM_UserTable] | None = None
        try:
            users_to_reset = await self.prisma_client.get_data(
                table_name="user",
                query_type="find_all",
                reset_at=now,
                limit=RESET_BUDGET_JOB_BATCH_SIZE,
            )
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
            outcome: Final = _ChunkOutcome(
                fetched=len(users_to_reset) if users_to_reset else 0,
                advanced=_count_advanced(
                    (u.budget_reset_at for u in updated_users),
                    cutoff=datetime.now(timezone.utc),
                ),
            )
            if len(failed_users) > 0:
                self._emit_phase_failure(
                    call_type="reset_budget_users",
                    error=Exception(
                        f"Failed to reset {len(failed_users)} users: {json.dumps(failed_users, default=str)}"
                    ),
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_users_found": len(users_to_reset) if users_to_reset else 0,
                        "users_found": json.dumps(users_to_reset, indent=4, default=str),
                    },
                )
                return outcome

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
            return _NO_PROGRESS
        else:
            return outcome

    async def reset_budget_for_litellm_teams(self) -> None:
        """
        Resets the budget for all LiteLLM Internal Teams if their budget has expired
        """
        await _run_phase_in_chunks(self._reset_budget_for_litellm_teams_chunk)

    async def _reset_budget_for_litellm_teams_chunk(self) -> _ChunkOutcome:
        now: Final = datetime.utcnow()
        start_time: Final = time.time()
        teams_to_reset: list[LiteLLM_TeamTable] | None = None
        try:
            teams_to_reset = await self.prisma_client.get_data(
                table_name="team",
                query_type="find_all",
                reset_at=now,
                limit=RESET_BUDGET_JOB_BATCH_SIZE,
            )
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
            outcome: Final = _ChunkOutcome(
                fetched=len(teams_to_reset) if teams_to_reset else 0,
                advanced=_count_advanced(
                    (t.budget_reset_at for t in updated_teams),
                    cutoff=datetime.now(timezone.utc),
                ),
            )
            if len(failed_teams) > 0:
                self._emit_phase_failure(
                    call_type="reset_budget_teams",
                    error=Exception(
                        f"Failed to reset {len(failed_teams)} teams: {json.dumps(failed_teams, default=str)}"
                    ),
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata={
                        "num_teams_found": len(teams_to_reset) if teams_to_reset else 0,
                        "teams_found": json.dumps(teams_to_reset, indent=4, default=str),
                    },
                )
                return outcome

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
            return _NO_PROGRESS
        else:
            return outcome

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
