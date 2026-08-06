"""
Unit of work over a single Prisma batch.

``spend_reset_unit_of_work`` opens one ``db.batch_()`` and binds a typed write
repository per table to it, so every update queued through the yielded object
lands in the same transaction. The batch commits when the block exits cleanly
and is abandoned, writing nothing, when the block raises.

Each write repository queues narrow ``{spend, budget_reset_at}`` updates
instead of full-model writes, which trip ``prisma.errors.DataError`` on rows
carrying fields the update input type rejects (see #27730).
"""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

from litellm.repositories.prisma_protocols import BatchTable, PrismaBatch


@dataclass(frozen=True, slots=True)
class KeySpendResetWrites:
    table: BatchTable

    def queue_spend_reset(self, token: str, budget_reset_at: datetime | None) -> None:
        self.table.update(where={"token": token}, data={"spend": 0, "budget_reset_at": budget_reset_at})


@dataclass(frozen=True, slots=True)
class UserSpendResetWrites:
    table: BatchTable

    def queue_spend_reset(self, user_id: str, budget_reset_at: datetime | None) -> None:
        self.table.update(where={"user_id": user_id}, data={"spend": 0, "budget_reset_at": budget_reset_at})


@dataclass(frozen=True, slots=True)
class TeamSpendResetWrites:
    table: BatchTable

    def queue_spend_reset(self, team_id: str, budget_reset_at: datetime | None) -> None:
        self.table.update(where={"team_id": team_id}, data={"spend": 0, "budget_reset_at": budget_reset_at})


@dataclass(frozen=True, slots=True)
class SpendResetUnitOfWork:
    keys: KeySpendResetWrites
    users: UserSpendResetWrites
    teams: TeamSpendResetWrites


@asynccontextmanager
async def spend_reset_unit_of_work(new_batch: Callable[[], PrismaBatch]) -> AsyncGenerator[SpendResetUnitOfWork, None]:
    batch = new_batch()
    yield SpendResetUnitOfWork(
        keys=KeySpendResetWrites(table=batch.litellm_verificationtoken),
        users=UserSpendResetWrites(table=batch.litellm_usertable),
        teams=TeamSpendResetWrites(table=batch.litellm_teamtable),
    )
    await batch.commit()
