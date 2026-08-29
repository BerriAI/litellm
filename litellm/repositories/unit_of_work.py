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

from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm.repositories.prisma_protocols import BatchTable, PrismaBatch


def _spend_reset_data(budget_reset_at: datetime | None, spend_decrement: float | None) -> Mapping[str, object]:
    spend: Final[object] = (
        {"decrement": spend_decrement}  # mutable-ok: prisma update payload must be a dict
        if spend_decrement is not None
        else 0
    )
    return {"spend": spend, "budget_reset_at": budget_reset_at}  # mutable-ok: prisma update payload must be a dict


@dataclass(frozen=True, slots=True)
class KeySpendResetWrites:
    table: BatchTable

    def queue_spend_reset(
        self, token: str, budget_reset_at: datetime | None, spend_decrement: float | None = None
    ) -> None:
        self.table.update(
            where={"token": token},  # mutable-ok: prisma where filter must be a dict
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )


@dataclass(frozen=True, slots=True)
class UserSpendResetWrites:
    table: BatchTable

    def queue_spend_reset(
        self, user_id: str, budget_reset_at: datetime | None, spend_decrement: float | None = None
    ) -> None:
        self.table.update(
            where={"user_id": user_id},  # mutable-ok: prisma where filter must be a dict
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )


@dataclass(frozen=True, slots=True)
class TeamSpendResetWrites:
    table: BatchTable

    def queue_spend_reset(
        self, team_id: str, budget_reset_at: datetime | None, spend_decrement: float | None = None
    ) -> None:
        self.table.update(
            where={"team_id": team_id},  # mutable-ok: prisma where filter must be a dict
            data=_spend_reset_data(budget_reset_at, spend_decrement),
        )


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
            data={"spend": {"decrement": amount}},  # mutable-ok: prisma update payload must be a dict
        )


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
    """Every write a budget-tier reset performs, bound to one batch.

    The dependent spend rows and the budget rows' ``budget_reset_at`` advance
    must land together: advancing the window without zeroing the spend it
    gates leaves the dependents pinned at their cap until the next window.
    """

    team_memberships: LinkedSpendResetWrites
    keys: LinkedSpendResetWrites
    organizations: LinkedSpendResetWrites
    tags: LinkedSpendResetWrites
    endusers: LinkedSpendResetWrites
    budgets: BudgetWindowWrites


@asynccontextmanager
async def spend_reset_unit_of_work(new_batch: Callable[[], PrismaBatch]) -> AsyncGenerator[SpendResetUnitOfWork, None]:
    batch = new_batch()
    yield SpendResetUnitOfWork(
        keys=KeySpendResetWrites(table=batch.litellm_verificationtoken),
        users=UserSpendResetWrites(table=batch.litellm_usertable),
        teams=TeamSpendResetWrites(table=batch.litellm_teamtable),
    )
    await batch.commit()


@asynccontextmanager
async def budget_cascade_unit_of_work(
    new_batch: Callable[[], PrismaBatch],
) -> AsyncGenerator[BudgetCascadeUnitOfWork, None]:
    batch = new_batch()
    yield BudgetCascadeUnitOfWork(
        team_memberships=LinkedSpendResetWrites(table=batch.litellm_teammembership),
        keys=LinkedSpendResetWrites(table=batch.litellm_verificationtoken),
        organizations=LinkedSpendResetWrites(table=batch.litellm_organizationtable),
        tags=LinkedSpendResetWrites(table=batch.litellm_tagtable),
        endusers=LinkedSpendResetWrites(table=batch.litellm_endusertable),
        budgets=BudgetWindowWrites(table=batch.litellm_budgettable),
    )
    await batch.commit()
