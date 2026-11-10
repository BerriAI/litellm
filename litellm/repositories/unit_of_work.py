"""
Units of work over a single Prisma batch.

Each context manager here opens one ``db.batch_()`` and binds a typed write
repository per table to it, so every update queued through the yielded object
lands in the same transaction. The batch commits when the block exits cleanly
and is abandoned, writing nothing, when the block raises.

``spend_reset_unit_of_work`` covers the per-row key/user/team resets;
``budget_cascade_unit_of_work`` covers a budget tier's reset, where the
dependent spend and the tier's next window have to move together.

Each write repository queues narrow ``{spend}`` / ``{budget_reset_at}`` updates
instead of full-model writes, which trip ``prisma.errors.DataError`` on rows
carrying fields the update input type rejects (see #27730).
"""

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite, ulp
from typing import Final, Protocol, runtime_checkable

from litellm.repositories.prisma_protocols import BatchTable, PrismaBatch


def _spend_reset_data(budget_reset_at: datetime | None, spend_decrement: float) -> Mapping[str, object]:
    spend: Final[object] = {"decrement": spend_decrement}
    return {"spend": spend, "budget_reset_at": budget_reset_at}


_SPEND_RESET_SET: Final = (
    " SET spend = CASE WHEN ABS(spend - $3::text::double precision) <= $5::text::double precision"
    " THEN 0 ELSE GREATEST(spend - $3::text::double precision, 0) END, budget_reset_at = $2::timestamp,"
    " updated_at = (NOW() AT TIME ZONE 'UTC')"
)
_SPEND_RESET_WHERE: Final = " AND budget_reset_at IS NOT DISTINCT FROM $4::timestamp"
_KEY_SPEND_RESET_SQL: Final = (
    'UPDATE "LiteLLM_VerificationToken"' + _SPEND_RESET_SET + " WHERE token = $1" + _SPEND_RESET_WHERE
)
_USER_SPEND_RESET_SQL: Final = (
    'UPDATE "LiteLLM_UserTable"' + _SPEND_RESET_SET + " WHERE user_id = $1" + _SPEND_RESET_WHERE
)
_TEAM_SPEND_RESET_SQL: Final = (
    'UPDATE "LiteLLM_TeamTable"' + _SPEND_RESET_SET + " WHERE team_id = $1" + _SPEND_RESET_WHERE
)


def _reset_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    reset_at: Final = (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value).astimezone(timezone.utc)
    return reset_at.replace(tzinfo=None).isoformat(timespec="microseconds")


def _queue_guarded_spend_reset(
    batch: PrismaBatch,
    query: str,
    identity: str,
    budget_reset_at: datetime | None,
    spend_decrement: float,
    expected_reset_at: datetime | None,
    spend_after_reset: float,
) -> bool:
    if budget_reset_at is None or spend_decrement == 0 or not isfinite(spend_decrement):
        return False
    # Prisma rounds Float reads, so exact subtraction can still leave a few ULPs.
    tolerance: Final = 4 * ulp(spend_decrement) if spend_after_reset == 0.0 else 0.0
    batch.execute_raw(
        query,
        identity,
        _reset_timestamp(budget_reset_at),
        repr(spend_decrement),
        _reset_timestamp(expected_reset_at),
        str(tolerance),
    )
    return True


@dataclass(frozen=True, slots=True)
class KeySpendResetWrites:
    table: BatchTable
    batch: PrismaBatch

    def queue_spend_reset(
        self,
        token: str,
        budget_reset_at: datetime | None,
        spend_decrement: float,
        expected_reset_at: datetime | None,
        spend_after_reset: float = 0.0,
    ) -> None:
        if _queue_guarded_spend_reset(
            self.batch,
            _KEY_SPEND_RESET_SQL,
            token,
            budget_reset_at,
            spend_decrement,
            expected_reset_at,
            spend_after_reset,
        ):
            return
        self.table.update_many(
            where={"token": token, "budget_reset_at": expected_reset_at},
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )


@dataclass(frozen=True, slots=True)
class UserSpendResetWrites:
    table: BatchTable
    batch: PrismaBatch

    def queue_spend_reset(
        self,
        user_id: str,
        budget_reset_at: datetime | None,
        spend_decrement: float,
        expected_reset_at: datetime | None,
        spend_after_reset: float = 0.0,
    ) -> None:
        if _queue_guarded_spend_reset(
            self.batch,
            _USER_SPEND_RESET_SQL,
            user_id,
            budget_reset_at,
            spend_decrement,
            expected_reset_at,
            spend_after_reset,
        ):
            return
        self.table.update_many(
            where={"user_id": user_id, "budget_reset_at": expected_reset_at},
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )


@dataclass(frozen=True, slots=True)
class TeamSpendResetWrites:
    table: BatchTable
    batch: PrismaBatch

    def queue_spend_reset(
        self,
        team_id: str,
        budget_reset_at: datetime | None,
        spend_decrement: float,
        expected_reset_at: datetime | None,
        spend_after_reset: float = 0.0,
    ) -> None:
        if _queue_guarded_spend_reset(
            self.batch,
            _TEAM_SPEND_RESET_SQL,
            team_id,
            budget_reset_at,
            spend_decrement,
            expected_reset_at,
            spend_after_reset,
        ):
            return
        self.table.update_many(
            where={"team_id": team_id, "budget_reset_at": expected_reset_at},
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )


@runtime_checkable
class SpendResetWrites(Protocol):
    def queue_spend_zero(self, where: Mapping[str, object]) -> None: ...

    def queue_spend_decrement(self, where: Mapping[str, object], amount: float) -> None: ...


@dataclass(frozen=True, slots=True)
class LinkedSpendResetWrites:
    table: BatchTable

    def queue_spend_zero(self, where: Mapping[str, object]) -> None:
        self.table.update_many(where=where, data={"spend": 0})

    def queue_spend_decrement(self, where: Mapping[str, object], amount: float) -> None:
        """``decrement`` rather than a read-then-set, so spend written between the
        cascade's read and its commit survives the reset instead of being erased."""
        self.table.update_many(
            where=where,
            data={"spend": {"decrement": amount}},
        )


# Membership rows written before total_spend existed (migration 20260421135425, default 0) hold their
# history only in spend. Zeroing spend would erase it, so the floor lands in the same batch, ahead of the reset.
_TEAM_MEMBERSHIP_TOTAL_SPEND_REPAIR_SQL: Final = """
UPDATE "LiteLLM_TeamMembership"
SET total_spend = spend
WHERE budget_id = ANY($1::text[]) AND spend > total_spend
"""


@dataclass(frozen=True, slots=True)
class TeamMembershipSpendResetWrites:
    resets: LinkedSpendResetWrites
    batch: PrismaBatch

    def queue_spend_zero(self, where: Mapping[str, object]) -> None:
        self.resets.queue_spend_zero(where)

    def queue_spend_decrement(self, where: Mapping[str, object], amount: float) -> None:
        self.resets.queue_spend_decrement(where, amount)

    def queue_total_spend_repair(self, budget_ids: Sequence[str]) -> None:
        self.batch.execute_raw(_TEAM_MEMBERSHIP_TOTAL_SPEND_REPAIR_SQL, list(budget_ids))


_LOCK_BUDGET_ORGS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_OrganizationTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY organization_id COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_MEMBERSHIPS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_TeamMembership" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY team_id COLLATE "C", user_id COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_KEYS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_VerificationToken" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY token COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_PROJECTS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_ProjectTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY project_id COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_TAGS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_TagTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY tag_name COLLATE "C" FOR UPDATE'
)
_LOCK_BUDGET_ACCESS_GROUPS_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_ModelAccessGroupBudgetTable" WHERE budget_id = ANY($1::text[]) '
    'ORDER BY access_group_name COLLATE "C" FOR UPDATE'
)


@dataclass(frozen=True, slots=True)
class BudgetLinkedRowLocks:
    batch: PrismaBatch

    def queue_row_locks(self, budget_ids: Sequence[str]) -> None:
        """Locks every row a cascade may reset, in the shared row lock order, before the filtered resets run."""
        self.batch.execute_raw(_LOCK_BUDGET_ORGS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_MEMBERSHIPS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_KEYS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_PROJECTS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_TAGS_SQL, list(budget_ids))
        self.batch.execute_raw(_LOCK_BUDGET_ACCESS_GROUPS_SQL, list(budget_ids))


@dataclass(frozen=True, slots=True)
class BudgetWindowWrites:
    table: BatchTable

    def queue_window_advance(self, budget_id: str, budget_reset_at: datetime) -> None:
        """``update_many`` so a tier deleted between the read and the commit is a
        no-op row count instead of a P2025 that aborts the whole chunk."""
        self.table.update_many(where={"budget_id": budget_id}, data={"budget_reset_at": budget_reset_at})


@dataclass(frozen=True, slots=True)
class SpendResetUnitOfWork:
    keys: KeySpendResetWrites
    users: UserSpendResetWrites
    teams: TeamSpendResetWrites


@dataclass(frozen=True, slots=True)
class BudgetCascadeUnitOfWork:
    """Every write a budget-tier reset performs in its final batch.

    The dependent spend rows and the budget rows' ``budget_reset_at`` advance
    must land together: advancing the window without zeroing the spend it
    gates leaves the dependents pinned at their cap until the next window.
    End users are the exception, paged and committed ahead of this batch by
    ``enduser_spend_reset_unit_of_work`` because one tier can gate millions.
    """

    row_locks: BudgetLinkedRowLocks
    team_memberships: TeamMembershipSpendResetWrites
    keys: LinkedSpendResetWrites
    organizations: LinkedSpendResetWrites
    tags: LinkedSpendResetWrites
    model_access_groups: LinkedSpendResetWrites
    projects: LinkedSpendResetWrites
    budgets: BudgetWindowWrites


@asynccontextmanager
async def spend_reset_unit_of_work(new_batch: Callable[[], PrismaBatch]) -> AsyncGenerator[SpendResetUnitOfWork, None]:
    batch = new_batch()
    yield SpendResetUnitOfWork(
        keys=KeySpendResetWrites(table=batch.litellm_verificationtoken, batch=batch),
        users=UserSpendResetWrites(table=batch.litellm_usertable, batch=batch),
        teams=TeamSpendResetWrites(table=batch.litellm_teamtable, batch=batch),
    )
    await batch.commit()


@asynccontextmanager
async def budget_cascade_unit_of_work(
    new_batch: Callable[[], PrismaBatch],
) -> AsyncGenerator[BudgetCascadeUnitOfWork, None]:
    batch = new_batch()
    yield BudgetCascadeUnitOfWork(
        row_locks=BudgetLinkedRowLocks(batch=batch),
        team_memberships=TeamMembershipSpendResetWrites(
            resets=LinkedSpendResetWrites(table=batch.litellm_teammembership), batch=batch
        ),
        keys=LinkedSpendResetWrites(table=batch.litellm_verificationtoken),
        organizations=LinkedSpendResetWrites(table=batch.litellm_organizationtable),
        tags=LinkedSpendResetWrites(table=batch.litellm_tagtable),
        model_access_groups=LinkedSpendResetWrites(table=batch.litellm_modelaccessgroupbudgettable),
        projects=LinkedSpendResetWrites(table=batch.litellm_projecttable),
        budgets=BudgetWindowWrites(table=batch.litellm_budgettable),
    )
    await batch.commit()


@asynccontextmanager
async def enduser_spend_reset_unit_of_work(
    new_batch: Callable[[], PrismaBatch],
) -> AsyncGenerator[LinkedSpendResetWrites, None]:
    """One page of end-user spend resets, committed on its own so a statement
    never grows with the end-user population."""
    batch: Final = new_batch()
    yield LinkedSpendResetWrites(table=batch.litellm_endusertable)
    await batch.commit()
