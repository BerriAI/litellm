import asyncio
import os
import socket
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.constants import BACKGROUND_INTERACTION_COST_POLLING_ENABLED
from litellm.interactions.background_cost_polling import (
    DEFAULT_POLL_SCHEDULE,
    BackgroundInteractionCreateContext,
    FetchInteraction,
    PendingBackgroundInteraction,
    PollSchedule,
    SettlementOutcome,
    configure_background_settlement_store,
    fetch_background_interaction,
    resume_unsettled_background_interactions,
)
from litellm.repositories.table_repositories import BackgroundInteractionSettlementRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _SettlementRow(Protocol):
    @property
    def interaction_id(self) -> str: ...
    @property
    def custom_llm_provider(self) -> str: ...
    @property
    def create_context(self) -> object: ...
    @property
    def created_at(self) -> datetime: ...
    @property
    def claimed_at(self) -> datetime | None: ...


class _NewSettlementRow(TypedDict):
    interaction_id: ReadOnly[str]
    custom_llm_provider: ReadOnly[str]
    create_context: ReadOnly[object]
    created_at: ReadOnly[datetime]


class _RowKey(TypedDict):
    interaction_id: ReadOnly[str]


class _UnclaimedRowKey(TypedDict):
    interaction_id: ReadOnly[str]
    claimed_at: ReadOnly[None]


class _UnclaimedRows(TypedDict):
    claimed_at: ReadOnly[None]


class _Claim(TypedDict):
    claimed_at: ReadOnly[datetime]
    claimed_by: ReadOnly[str]


class _Outcome(TypedDict):
    settled_at: ReadOnly[datetime]
    outcome: ReadOnly[SettlementOutcome]


class _SettlementTableActions(Protocol):
    def create(self, *, data: _NewSettlementRow) -> Awaitable[_SettlementRow]: ...

    def find_unique(self, *, where: _RowKey) -> Awaitable[_SettlementRow | None]: ...

    def find_many(self, *, where: _UnclaimedRows) -> Awaitable[Sequence[_SettlementRow]]: ...

    def update_many(self, *, data: _Claim | _Outcome, where: _RowKey | _UnclaimedRowKey) -> Awaitable[int]: ...


def _settlement_table(prisma_client: "PrismaClient") -> _SettlementTableActions:
    return BackgroundInteractionSettlementRepository(prisma_client).table


def _pending_rows(rows: Sequence[_SettlementRow]) -> tuple[PendingBackgroundInteraction, ...]:
    return tuple(pending for row in rows for pending in _pending_row(row))


def _pending_row(row: _SettlementRow) -> tuple[PendingBackgroundInteraction, ...]:
    try:
        create_context: Final = BackgroundInteractionCreateContext.model_validate(row.create_context)
    except ValidationError:
        verbose_proxy_logger.exception(
            "Background interaction %s has a settlement row this version cannot read; leaving it unsettled",
            row.interaction_id,
        )
        return ()
    return (
        PendingBackgroundInteraction(
            interaction_id=row.interaction_id,
            custom_llm_provider=row.custom_llm_provider,
            create_context=create_context,
            created_at=row.created_at,
        ),
    )


@dataclass(frozen=True, slots=True)
class PrismaBackgroundSettlementStore:
    table: _SettlementTableActions
    claimed_by: str

    async def register(self, pending: PendingBackgroundInteraction) -> None:
        from prisma import Json  # noqa: PLC0415  # local import: prisma may be ungenerated at module load in some tools

        await self.table.create(
            data=_NewSettlementRow(
                interaction_id=pending.interaction_id,
                custom_llm_provider=pending.custom_llm_provider,
                create_context=Json(pending.create_context.model_dump(mode="json")),
                created_at=pending.created_at,
            )
        )

    async def pending(self, interaction_id: str) -> PendingBackgroundInteraction | None:
        row: Final = await self.table.find_unique(where=_RowKey(interaction_id=interaction_id))
        if row is None or row.claimed_at is not None:
            return None
        return next(iter(_pending_row(row)), None)

    async def is_claimed(self, interaction_id: str) -> bool:
        row: Final = await self.table.find_unique(where=_RowKey(interaction_id=interaction_id))
        return row is not None and row.claimed_at is not None

    async def claim(self, interaction_id: str) -> bool:
        claimed_rows: Final = await self.table.update_many(
            data=_Claim(claimed_at=datetime.now(timezone.utc), claimed_by=self.claimed_by),
            where=_UnclaimedRowKey(interaction_id=interaction_id, claimed_at=None),
        )
        return claimed_rows == 1

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None:
        await self.table.update_many(
            data=_Outcome(settled_at=datetime.now(timezone.utc), outcome=outcome),
            where=_RowKey(interaction_id=interaction_id),
        )

    async def unclaimed(self) -> Sequence[PendingBackgroundInteraction]:
        return _pending_rows(await self.table.find_many(where=_UnclaimedRows(claimed_at=None)))


async def configure_background_interaction_settlement(
    table: _SettlementTableActions,
    claimed_by: str,
    fetch_interaction: FetchInteraction = fetch_background_interaction,
    schedule: PollSchedule = DEFAULT_POLL_SCHEDULE,
) -> tuple["asyncio.Task[SettlementOutcome | None]", ...]:
    if not BACKGROUND_INTERACTION_COST_POLLING_ENABLED:
        return ()
    store: Final = PrismaBackgroundSettlementStore(table=table, claimed_by=claimed_by)
    configure_background_settlement_store(store)
    resumed: Final = await resume_unsettled_background_interactions(store, fetch_interaction, schedule)
    if resumed:
        verbose_proxy_logger.info("Resumed cost polling for %s unsettled background interactions", len(resumed))
    return resumed


async def install_background_interaction_settlement(prisma_client: "PrismaClient") -> None:
    await configure_background_interaction_settlement(
        table=BackgroundInteractionSettlementRepository(prisma_client).table,
        claimed_by=f"{socket.gethostname()}:{os.getpid()}",
    )
