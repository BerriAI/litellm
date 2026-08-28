import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeVar, assert_never

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import (
    GLOBAL_PROXY_SPEND_CACHE_KEY,
    LITELLM_PROXY_BUDGET_NAME,
    RESET_BUDGET_JOB_BATCH_SIZE,
    RESET_BUDGET_JOB_LOCK_TTL_SECONDS,
    RESET_BUDGET_JOB_MAX_CHUNKS_PER_RUN,
    RESET_BUDGET_JOB_NAME,
)
from litellm.proxy._types import (
    DB_RETRY_SAFE_ERROR_TYPES,
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
from litellm.proxy.common_utils.user_api_key_cache import tag_cache_key
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.db.exception_handler import call_with_db_reconnect_retry
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.organization_repository import OrganizationRepository
from litellm.repositories.prisma_protocols import SpendLinkedTable
from litellm.repositories.table_repositories import (
    EndUserRepository,
    TagRepository,
    TeamMembershipRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.unit_of_work import (
    LinkedSpendResetWrites,
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


class _BudgetLinkedRow(Protocol):
    @property
    def spend(self) -> float | None: ...

    @property
    def budget_id(self) -> str | None: ...


class _TeamMembershipRow(_BudgetLinkedRow, Protocol):
    @property
    def user_id(self) -> str: ...

    @property
    def team_id(self) -> str: ...


class _KeyRow(_BudgetLinkedRow, Protocol):
    @property
    def token(self) -> str: ...


class _OrgRow(_BudgetLinkedRow, Protocol):
    @property
    def organization_id(self) -> str: ...


class _TagRow(_BudgetLinkedRow, Protocol):
    @property
    def tag_name(self) -> str: ...


class _EndUserRow(_BudgetLinkedRow, Protocol):
    @property
    def user_id(self) -> str: ...


def _rollover_enabled() -> bool:
    return litellm.budget_rollover is True


def _rollover_cap(max_budget: float | None) -> float | None:
    if max_budget is None or not math.isfinite(max_budget):
        return None
    return max_budget


def _carried_spend(spend: float | None, cap: float | None) -> float:
    if cap is None:
        return 0.0
    return max(0.0, (spend or 0.0) - cap)


def _row_carried_spend(row: _BudgetLinkedRow, caps: Mapping[str, float]) -> float:
    if not caps:
        return 0.0
    return _carried_spend(row.spend, caps.get(row.budget_id) if row.budget_id is not None else None)


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
    return (tag_cache_key(row.tag_name),)


def _budget_link_where(
    budget_ids: Sequence[str],
    extra: Mapping[str, object] = MappingProxyType({}),
) -> dict[str, object]:
    return {"budget_id": {"in": list(budget_ids)}, **extra}


def _queue_budget_linked_resets(
    writes: LinkedSpendResetWrites,
    cascade: "_BudgetCascade",
    extra: Mapping[str, object] = MappingProxyType({}),
) -> None:
    """Reset one linked table's spend for every expiring tier: tiers with a
    rollover cap keep spend beyond the cap (decrement preserves writes racing
    the reset), everything else is zeroed as before. Zero the under-cap rows
    BEFORE decrementing the over-cap ones: the statements run sequentially in
    one transaction, so the reverse order lets the zero re-match a row the
    decrement just moved into the (0, cap] range and erase its carried spend."""
    for budget_id, cap in cascade.rollover_caps.items():
        writes.queue_spend_zero(
            where={"budget_id": budget_id, **extra, "spend": {"gt": 0, "lte": cap}}
        )  # mutable-ok: prisma where filter must be a dict
        writes.queue_spend_decrement(
            where={"budget_id": budget_id, **extra, "spend": {"gt": cap}}, amount=cap
        )  # mutable-ok: prisma where filter must be a dict
    plain_ids: Final = tuple(bid for bid in cascade.budget_ids if bid not in cascade.rollover_caps)
    if plain_ids:
        writes.queue_spend_zero(where=_budget_link_where(plain_ids, extra))


def _queue_enduser_resets(writes: LinkedSpendResetWrites, cascade: "_BudgetCascade") -> None:
    """End users are matched by id rather than budget link: rows with no
    budget_id ride the default budget tier (litellm.max_end_user_budget_id).
    Zero-before-decrement ordering matters here too (see
    _queue_budget_linked_resets)."""
    if not cascade.rollover_caps:
        if cascade.endusers:
            writes.queue_spend_zero(
                where={"user_id": {"in": [row.user_id for row in cascade.endusers]}}
            )  # mutable-ok: prisma where filter must be a dict
        return
    tiered: Final = tuple((row.budget_id or litellm.max_end_user_budget_id, row.user_id) for row in cascade.endusers)
    for budget_id, cap in cascade.rollover_caps.items():
        if not (
            user_ids := [uid for bid, uid in tiered if bid == budget_id]
        ):  # mutable-ok: prisma "in" filter takes a list
            continue
        writes.queue_spend_zero(
            where={"user_id": {"in": user_ids}, "spend": {"lte": cap}}
        )  # mutable-ok: prisma where filter must be a dict
        writes.queue_spend_decrement(
            where={"user_id": {"in": user_ids}, "spend": {"gt": cap}}, amount=cap
        )  # mutable-ok: prisma where filter must be a dict
    plain: Final = [
        uid for bid, uid in tiered if bid is None or bid not in cascade.rollover_caps
    ]  # mutable-ok: prisma "in" filter takes a list
    if plain:
        writes.queue_spend_zero(where={"user_id": {"in": plain}})  # mutable-ok: prisma where filter must be a dict


@dataclass(frozen=True, slots=True)
class _BudgetCascade:
    """Everything one budget-tier reset touches, resolved before any write."""

    budgets: tuple[LiteLLM_BudgetTableFull, ...] = ()
    budget_ids: tuple[str, ...] = ()
    budget_resets: tuple[tuple[str, datetime], ...] = ()
    endusers: tuple[_EndUserRow, ...] = ()
    counter_resets: tuple[tuple[str, float], ...] = ()
    cache_keys: tuple[str, ...] = ()
    rollover_caps: Mapping[str, float] = MappingProxyType({})


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


@dataclass(frozen=True, slots=True)
class _LazyJson:
    """Serialize only if a log record is actually emitted.

    ``logger.debug("... %s", json.dumps(rows))`` evaluates the dump before the
    logger decides to drop the record, so a chunk of rows is serialized on the
    event loop on every tick at any log level. Passing this instead defers the
    work to the formatter.
    """

    value: object

    def __str__(self) -> str:
        return json.dumps(self.value, indent=4, default=str)


class _Lease(Enum):
    """Whether this pod may sweep, and whether it owes a lock release."""

    LEADER = "leader"
    UNGUARDED = "unguarded"
    FOLLOWER = "follower"


async def _write_key_windows(prisma_client: PrismaClient, row_id: str, payload: str) -> None:
    await VerificationTokenRepository(prisma_client).table.update(
        where={"token": row_id},
        data={"budget_limits": payload},
    )


async def _write_team_windows(prisma_client: PrismaClient, row_id: str, payload: str) -> None:
    await TeamRepository(prisma_client).table.update(
        where={"team_id": row_id},
        data={"budget_limits": payload},
    )


@dataclass(frozen=True, slots=True)
class _WindowSource:
    """A table whose rows carry their own per-window budget limits."""

    table: str
    id_column: str
    counter_prefix: str
    log_subject: str
    retry_subject: str
    write: Callable[[PrismaClient, str, str], Awaitable[None]]

    def page_query(self) -> str:
        """One keyset page, ordered by the primary key so the cursor never repeats a row.

        prisma-client-python cannot null-filter a ``Json?`` column (no DbNull /
        JsonNull sentinel, RobertCraigie/prisma-client-py#714), so the read stays
        raw SQL; the table and column names are module constants, never input.
        Writes still go through the ORM.
        """
        return (
            f'SELECT {self.id_column}, budget_limits FROM "{self.table}" '
            f"WHERE budget_limits IS NOT NULL AND {self.id_column} > $1 "
            f"ORDER BY {self.id_column} LIMIT $2"
        )


_WINDOW_SOURCES: Final[tuple[_WindowSource, ...]] = (
    _WindowSource(
        table="LiteLLM_VerificationToken",
        id_column="token",
        counter_prefix="spend:key",
        log_subject="keys",
        retry_subject="key",
        write=_write_key_windows,
    ),
    _WindowSource(
        table="LiteLLM_TeamTable",
        id_column="team_id",
        counter_prefix="spend:team",
        log_subject="teams",
        retry_subject="team",
        write=_write_team_windows,
    ),
)


def _budget_cascade_event_metadata(cascade: _BudgetCascade) -> dict[str, object]:
    return {
        "num_budgets_found": len(cascade.budgets),
        "num_endusers_found": len(cascade.endusers),
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
        pod_lock_manager: PodLockManager | None = None,
    ):
        self.proxy_logging_obj: ProxyLogging = proxy_logging_obj
        self.prisma_client: PrismaClient = prisma_client
        self.reset_settings: BudgetResetSettings = reset_settings or get_budget_reset_settings()
        self.pod_lock_manager: PodLockManager | None = pod_lock_manager

    async def _lease_is_held(self, lock_manager: PodLockManager) -> bool:
        """True only when the lease is readable and someone holds it.

        An unreadable lock reports as unheld so the caller sweeps rather than
        skipping; being wrong here costs a duplicate sweep, and the alternative
        strands every expired budget at its cap.
        """
        if lock_manager.redis_cache is None:
            return False
        try:
            lock_key: Final = lock_manager.get_redis_lock_key(RESET_BUDGET_JOB_NAME)
            return bool(await lock_manager.redis_cache.async_get_cache(lock_key))
        except Exception as exc:  # noqa: BLE001  # an unreadable lease must not strand the sweep
            verbose_proxy_logger.warning("Reset budget job: could not read the reset lease: %s", exc)
            return False

    async def _acquire_lease(self) -> _Lease:
        """Elect one sweeper per tick.

        Every pod schedules this job, and each one otherwise re-reads the whole
        due population and writes it back at the same calendar boundary, so a
        fleet multiplies one sweep's Postgres load by its replica count. A
        deployment with no Redis-backed lock manager runs unguarded, as it
        always has.
        """
        lock_manager: Final = self.pod_lock_manager
        if lock_manager is None or lock_manager.redis_cache is None:
            return _Lease.UNGUARDED

        if await lock_manager.acquire_lock(
            cronjob_id=RESET_BUDGET_JOB_NAME,
            ttl=RESET_BUDGET_JOB_LOCK_TTL_SECONDS,
        ):
            return _Lease.LEADER

        if await self._lease_is_held(lock_manager):
            verbose_proxy_logger.debug("Reset budget job: another pod holds the reset lease, skipping this tick")
            return _Lease.FOLLOWER

        # acquire_lock reports contention and an unreachable Redis identically, so
        # treating a failed acquire as contention would skip the sweep on every pod
        # at once for as long as Redis is down. Sweeping unguarded costs duplicate
        # work; not sweeping leaves every expired budget pinned at its cap.
        verbose_proxy_logger.warning(
            "Reset budget job: could not take the reset lease and no other pod holds it, "
            "sweeping unguarded rather than skipping the tick"
        )
        return _Lease.UNGUARDED

    async def reset_budget(
        self,
    ):
        """
        Gets all the non-expired keys for a db, which need spend to be reset

        Resets their spend

        Updates db

        Runs on one pod per tick where a Redis lease is available.
        """
        if self.prisma_client is None:
            return

        lease: Final = await self._acquire_lease()
        if lease is _Lease.FOLLOWER:
            return

        try:
            await self.reset_budget_for_litellm_keys()
            await self.reset_budget_for_litellm_users()
            await self.reset_budget_for_litellm_teams()
            await self.reset_budget_for_litellm_budget_table()
            await self.reset_budget_windows()
        finally:
            if lease is _Lease.LEADER and self.pod_lock_manager is not None:
                await self.pod_lock_manager.release_lock(cronjob_id=RESET_BUDGET_JOB_NAME)

    async def _with_db_retry(self, operation: Callable[[], Awaitable[_RowT]], *, reason: str) -> _RowT:
        """Reconnect and retry once on a transport error, so a dropped connection
        costs one retry instead of the whole tick.
        """
        return await call_with_db_reconnect_retry(self.prisma_client, operation, reason=reason)

    async def _with_db_write_retry(self, operation: Callable[[], Awaitable[_RowT]], *, reason: str) -> _RowT:
        """Same, for writes: only replay when the statements provably never
        reached the database. A reset zeroes spend unconditionally, so replaying
        an ambiguous commit would erase spend accrued since it landed.
        """
        return await call_with_db_reconnect_retry(
            self.prisma_client,
            operation,
            reason=reason,
            retry_safe_error_types=DB_RETRY_SAFE_ERROR_TYPES,
        )

    @staticmethod
    async def _invalidate_spend_counter(counter_key: str, new_spend: float = 0.0) -> None:
        """Overwrite a spend counter with the post-reset value (0, or the carried
        overage when budget rollover is enabled) so a DB-row reset takes effect
        immediately.

        Call AFTER the DB write commits. Clearing Redis before the DB
        commit opens a window where get_current_spend reads 0 from Redis
        while the DB still holds the pre-reset value, allowing bypass.
        """
        try:
            from litellm.proxy.proxy_server import spend_counter_cache

            spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=new_spend, ttl=60)
            if spend_counter_cache.redis_cache is not None:
                try:
                    await spend_counter_cache.redis_cache.async_set_cache(key=counter_key, value=new_spend, ttl=60)
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
            return tuple(
                await self._with_db_retry(
                    lambda: table.find_many(where=where),
                    reason=f"reset_budget_read_{log_subject.replace(' ', '_')}_failure",
                )
            )
        except Exception as e:
            verbose_proxy_logger.warning("Failed to fetch %s for counter invalidation: %s", log_subject, e)
            return ()

    async def _collect_endusers_to_reset(self, budget_ids: Sequence[str]) -> tuple[_EndUserRow, ...]:
        linked: Final[Sequence[_EndUserRow] | None] = await self._with_db_retry(
            lambda: self.prisma_client.get_data(
                table_name="enduser",
                query_type="find_all",
                budget_id_list=list(budget_ids),
            ),
            reason="reset_budget_read_endusers_failure",
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
        rollover_caps: Final[Mapping[str, float]] = MappingProxyType(
            {  # mutable-ok: MappingProxyType wraps a one-shot dict comprehension
                b.budget_id: cap
                for b in budgets_to_reset
                if b.budget_id is not None and (cap := _rollover_cap(b.max_budget)) is not None
            }
            if _rollover_enabled()
            else {}  # mutable-ok: empty sentinel immediately frozen by MappingProxyType
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
            counter_resets=(
                *(
                    (_team_membership_counter_key(row), _row_carried_spend(row, rollover_caps))
                    for row in team_memberships
                ),
                *((_key_counter_key(row), _row_carried_spend(row, rollover_caps)) for row in keys),
                *((_org_counter_key(row), _row_carried_spend(row, rollover_caps)) for row in orgs),
                *((_tag_counter_key(row), _row_carried_spend(row, rollover_caps)) for row in tags),
            ),
            rollover_caps=rollover_caps,
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

        await self._with_db_write_retry(
            lambda: self._commit_budget_cascade_once(cascade),
            reason="reset_budget_write_budget_cascade_failure",
        )

    async def _commit_budget_cascade_once(self, cascade: _BudgetCascade) -> None:
        async with budget_cascade_unit_of_work(self.prisma_client.db.batch_) as uow:
            _queue_budget_linked_resets(uow.team_memberships, cascade)
            _queue_budget_linked_resets(uow.keys, cascade, extra=_LINKED_KEYS_WHERE)
            _queue_budget_linked_resets(uow.organizations, cascade, extra=_SPENT_ROWS_WHERE)
            _queue_budget_linked_resets(uow.tags, cascade, extra=_SPENT_ROWS_WHERE)
            _queue_enduser_resets(uow.endusers, cascade)
            for budget_id, budget_reset_at in cascade.budget_resets:
                uow.budgets.queue_window_advance(budget_id=budget_id, budget_reset_at=budget_reset_at)

    async def _invalidate_budget_cascade_caches(self, cascade: _BudgetCascade) -> None:
        for counter_key, new_spend in cascade.counter_resets:
            await self._invalidate_spend_counter(counter_key, new_spend=new_spend)
        for cache_key in cascade.cache_keys:
            await self._invalidate_user_api_key_cache_entry(cache_key)

    async def _reset_expired_budget_cascade(self) -> _BudgetCascadeCommitted | _BudgetCascadeFailed:
        now: Final = datetime.now(timezone.utc)
        try:
            budgets_to_reset: Final[Sequence[LiteLLM_BudgetTableFull] | None] = await self._with_db_retry(
                lambda: self.prisma_client.get_data(
                    table_name="budget",
                    query_type="find_all",
                    reset_at=now,
                    limit=RESET_BUDGET_JOB_BATCH_SIZE,
                ),
                reason="reset_budget_read_budgets_failure",
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
        table: Final = EndUserRepository(self.prisma_client).table
        rows: Final = await self._with_db_retry(
            lambda: table.find_many(
                where={
                    "budget_id": None,
                    "spend": {"gt": 0},
                },
            ),
            reason="reset_budget_read_endusers_without_budget_id_failure",
        )
        return [LiteLLM_EndUserTable.model_validate(row.model_dump()) for row in rows]

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
        await self._with_db_write_retry(
            lambda: self._write_key_reset_updates_once(updated_keys),
            reason="reset_budget_write_keys_failure",
        )

    async def _write_key_reset_updates_once(self, updated_keys: list[LiteLLM_VerificationToken]) -> None:
        async with spend_reset_unit_of_work(self.prisma_client.db.batch_) as uow:
            for k in updated_keys:
                if k.token is None:
                    continue
                uow.keys.queue_spend_reset(
                    token=k.token,
                    budget_reset_at=k.budget_reset_at,
                    spend_decrement=k.max_budget if (k.spend or 0.0) > 0.0 else None,
                )

    async def _write_user_reset_updates(self, updated_users: list[LiteLLM_UserTable]) -> None:
        """
        Write per-row {spend, budget_reset_at} updates for users.

        Mirrors _write_key_reset_updates — avoids the full-model update path
        that trips Prisma's DataError on rows carrying unrecognised fields
        (see #27730).
        """
        await self._with_db_write_retry(
            lambda: self._write_user_reset_updates_once(updated_users),
            reason="reset_budget_write_users_failure",
        )

    async def _write_user_reset_updates_once(self, updated_users: list[LiteLLM_UserTable]) -> None:
        async with spend_reset_unit_of_work(self.prisma_client.db.batch_) as uow:
            for u in updated_users:
                uow.users.queue_spend_reset(
                    user_id=u.user_id,
                    budget_reset_at=u.budget_reset_at,
                    spend_decrement=u.max_budget if (u.spend or 0.0) > 0.0 else None,
                )

    async def _write_team_reset_updates(self, updated_teams: list[LiteLLM_TeamTable]) -> None:
        """
        Write per-row {spend, budget_reset_at} updates for teams.

        Mirrors _write_key_reset_updates — avoids the full-model update path
        that trips Prisma's DataError on rows carrying unrecognised fields
        (see #27730).
        """
        await self._with_db_write_retry(
            lambda: self._write_team_reset_updates_once(updated_teams),
            reason="reset_budget_write_teams_failure",
        )

    async def _write_team_reset_updates_once(self, updated_teams: list[LiteLLM_TeamTable]) -> None:
        async with spend_reset_unit_of_work(self.prisma_client.db.batch_) as uow:
            for t in updated_teams:
                uow.teams.queue_spend_reset(
                    team_id=t.team_id,
                    budget_reset_at=t.budget_reset_at,
                    spend_decrement=t.max_budget if (t.spend or 0.0) > 0.0 else None,
                )

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
            keys_to_reset = await self._with_db_retry(
                lambda: self.prisma_client.get_data(
                    table_name="key",
                    query_type="find_all",
                    expires=now,
                    reset_at=now,
                    limit=RESET_BUDGET_JOB_BATCH_SIZE,
                ),
                reason="reset_budget_read_keys_failure",
            )
            verbose_proxy_logger.debug("Keys to reset %s", _LazyJson(keys_to_reset))
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

                verbose_proxy_logger.debug("Updated keys %s", _LazyJson(updated_keys))

                if updated_keys:
                    await self._write_key_reset_updates(updated_keys=updated_keys)
                    for k in updated_keys:
                        token = getattr(k, "token", None)
                        if token:
                            await self._invalidate_spend_counter(f"spend:key:{token}", new_spend=k.spend or 0.0)

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
                        "num_keys_updated": len(updated_keys),
                        "num_keys_failed": len(failed_keys),
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
            users_to_reset = await self._with_db_retry(
                lambda: self.prisma_client.get_data(
                    table_name="user",
                    query_type="find_all",
                    reset_at=now,
                    limit=RESET_BUDGET_JOB_BATCH_SIZE,
                ),
                reason="reset_budget_read_users_failure",
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

                verbose_proxy_logger.debug("Updated users %s", _LazyJson(updated_users))
                if updated_users:
                    await self._write_user_reset_updates(updated_users=updated_users)
                    for u in updated_users:
                        user_id = getattr(u, "user_id", None)
                        if user_id:
                            await self._invalidate_spend_counter(f"spend:user:{user_id}", new_spend=u.spend or 0.0)
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
                        "num_users_updated": len(updated_users),
                        "num_users_failed": len(failed_users),
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
            teams_to_reset = await self._with_db_retry(
                lambda: self.prisma_client.get_data(
                    table_name="team",
                    query_type="find_all",
                    reset_at=now,
                    limit=RESET_BUDGET_JOB_BATCH_SIZE,
                ),
                reason="reset_budget_read_teams_failure",
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

                verbose_proxy_logger.debug("Updated teams %s", _LazyJson(updated_teams))
                if updated_teams:
                    await self._write_team_reset_updates(updated_teams=updated_teams)
                    for t in updated_teams:
                        team_id = getattr(t, "team_id", None)
                        if team_id:
                            await self._invalidate_spend_counter(f"spend:team:{team_id}", new_spend=t.spend or 0.0)

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
                        "num_teams_updated": len(updated_teams),
                        "num_teams_failed": len(failed_teams),
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
        new_value: Final = await ResetBudgetJob._window_carried_spend(window, counter_key, spend_counter_cache)
        spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=new_value)
        if spend_counter_cache.redis_cache is not None:
            try:
                await spend_counter_cache.redis_cache.async_set_cache(key=counter_key, value=new_value)
            except Exception as redis_err:
                verbose_proxy_logger.warning("Failed to reset Redis counter %s: %s", counter_key, redis_err)
        window["reset_at"] = compute_budget_reset_at(
            budget_duration=window["budget_duration"], settings=reset_settings
        ).isoformat()
        return True

    @staticmethod
    async def _window_carried_spend(
        window: Mapping[str, object], counter_key: str, spend_counter_cache: DualCache
    ) -> float:
        """Per-window spend lives only in the counter, so the carried overage is
        read from it before the reset overwrites it."""
        if not _rollover_enabled():
            return 0.0
        window_max: Final = window.get("max_budget")
        cap: Final = _rollover_cap(window_max) if isinstance(window_max, (int, float)) else None
        if cap is None:
            return 0.0
        try:
            current: Final = await spend_counter_cache.async_get_cache(key=counter_key)
        except Exception as e:  # noqa: BLE001  # an unreadable counter falls back to a plain zero reset
            verbose_proxy_logger.warning("Failed to read spend counter %s for rollover: %s", counter_key, e)
            return 0.0
        if not isinstance(current, (int, float)):
            return 0.0
        return _carried_spend(float(current), cap)

    async def reset_budget_windows(self) -> None:
        """
        For keys and teams with budget_limits, reset any individual windows where
        reset_at <= now. Only the expired windows are reset; other windows are untouched.
        """

        from litellm.proxy.proxy_server import spend_counter_cache

        now: Final = datetime.utcnow()
        for source in _WINDOW_SOURCES:
            try:
                await self._reset_windows_for(source=source, now=now, spend_counter_cache=spend_counter_cache)
            except Exception as e:
                verbose_proxy_logger.exception("Failed to reset budget windows for %s: %s", source.log_subject, e)

    async def _reset_windows_for(
        self,
        source: _WindowSource,
        now: datetime,
        spend_counter_cache: DualCache,
    ) -> None:
        """Walk one table's windowed rows a page at a time, to the end.

        Paging is what bounds the memory: the previous form pulled every row
        carrying budget_limits into one result set on every tick, which grows
        with the deployment's key count and is paid on the event loop.

        The walk deliberately has no per-run page cap. A cap has to remember
        where it stopped, and that position cannot live in the process: the
        lease is released after each sweep, so the next tick can elect a
        different pod whose own position is unset. It would restart at the first
        row and never reach the tail, pinning those windows at their cap for
        good. The cursor strictly advances, so the walk terminates on its own
        without needing a bound.
        """
        cursor = ""
        while True:
            next_cursor = await self._reset_window_page(
                source=source,
                cursor=cursor,
                now=now,
                spend_counter_cache=spend_counter_cache,
            )
            if next_cursor is None:
                return
            cursor = next_cursor

    async def _reset_window_page(
        self,
        source: _WindowSource,
        cursor: str,
        now: datetime,
        spend_counter_cache: DualCache,
    ) -> str | None:
        """Reset one page of windows; return the next cursor, or None when drained."""
        rows: Final = await self._with_db_retry(
            lambda: self.prisma_client.db.query_raw(source.page_query(), cursor, RESET_BUDGET_JOB_BATCH_SIZE),
            reason=f"reset_budget_read_{source.retry_subject}_windows_failure",
        )
        for row in rows:
            raw = row["budget_limits"]
            if not raw:
                continue
            row_id: str = row[source.id_column]
            windows: list[dict[str, object]] = raw if isinstance(raw, list) else json.loads(raw)
            changed = False
            for window in windows:
                counter_key = f"{source.counter_prefix}:{row_id}:window:{window['budget_duration']}"
                if await ResetBudgetJob._reset_expired_window(
                    window,
                    counter_key,
                    spend_counter_cache,
                    now,
                    self.reset_settings,
                ):
                    changed = True
            if changed:
                await self._with_db_write_retry(
                    lambda: source.write(self.prisma_client, row_id, json.dumps(windows)),
                    reason=f"reset_budget_write_{source.retry_subject}_windows_failure",
                )

        if len(rows) < RESET_BUDGET_JOB_BATCH_SIZE:
            return None
        return rows[-1][source.id_column]

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
            item.spend = _carried_spend(item.spend, _rollover_cap(item.max_budget)) if _rollover_enabled() else 0.0
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
