import asyncio
import os
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeVar

from pydantic import ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    BACKGROUND_INTERACTION_COST_POLLING_ENABLED,
    BACKGROUND_INTERACTION_SETTLEMENT_CLAIM_LEASE_SECONDS,
)
from litellm.interactions.background_cost_polling import (
    DEFAULT_POLL_SCHEDULE,
    BackgroundInteractionCreateContext,
    BackgroundInteractionPollContext,
    FetchInteraction,
    PendingBackgroundInteraction,
    PollSchedule,
    SettlementOutcome,
    configure_background_settlement_store,
    fetch_background_interaction,
    resume_unsettled_background_interactions,
)
from litellm.repositories.table_repositories import BackgroundInteractionSettlementRepository
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.router import CredentialLiteLLMParams

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router


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
    @property
    def settled_at(self) -> datetime | None: ...


class _NewSettlementRow(TypedDict):
    interaction_id: ReadOnly[str]
    custom_llm_provider: ReadOnly[str]
    create_context: ReadOnly[object]
    created_at: ReadOnly[datetime]


class _RowKey(TypedDict):
    interaction_id: ReadOnly[str]


class _Unclaimed(TypedDict):
    claimed_at: ReadOnly[None]


class _Before(TypedDict):
    lt: ReadOnly[datetime]


class _ExpiredClaim(TypedDict):
    settled_at: ReadOnly[None]
    claimed_at: ReadOnly[_Before]


class _ClaimableRowKey(TypedDict):
    interaction_id: ReadOnly[str]
    OR: ReadOnly[tuple[_Unclaimed, _ExpiredClaim]]


class _UnsettledRows(TypedDict):
    settled_at: ReadOnly[None]


class _Claim(TypedDict):
    claimed_at: ReadOnly[datetime]
    claimed_by: ReadOnly[str]


class _Outcome(TypedDict):
    settled_at: ReadOnly[datetime]
    outcome: ReadOnly[SettlementOutcome]
    create_context: ReadOnly[object]


class _SettlementTableActions(Protocol):
    def create(self, *, data: _NewSettlementRow) -> Awaitable[_SettlementRow]: ...

    def find_unique(self, *, where: _RowKey) -> Awaitable[_SettlementRow | None]: ...

    def find_many(self, *, where: _UnsettledRows) -> Awaitable[Sequence[_SettlementRow]]: ...

    def update_many(self, *, data: _Claim | _Outcome, where: _RowKey | _ClaimableRowKey) -> Awaitable[int]: ...


_CLEARED_CREATE_CONTEXT: Final[Mapping[str, object]] = MappingProxyType({})
_T = TypeVar("_T")


async def _read_from_a_table_that_may_not_exist(query: Awaitable[_T], when_missing: _T) -> _T:
    from prisma.errors import TableNotFoundError  # noqa: PLC0415  # local import: prisma may be ungenerated at load

    try:
        return await query
    except TableNotFoundError:
        return when_missing


def _json(data: Mapping[str, object]) -> object:
    from prisma import Json  # noqa: PLC0415  # local import: prisma may be ungenerated at module load in some tools

    return Json.keys(**data)


def _pending_row(row: _SettlementRow, claim_expires_at: datetime | None) -> tuple[PendingBackgroundInteraction, ...]:
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
            claim_expires_at=claim_expires_at,
        ),
    )


@dataclass(frozen=True, slots=True)
class PrismaBackgroundSettlementStore:
    """
    A claim is a lease, not a settlement: the settler that claims a row bills
    or releases it and then records the outcome, so a row claimed longer ago
    than the lease with no outcome belongs to a settler that died in between,
    and any settler may claim it again. The lease is far longer than settling
    takes, so a live settler is never overtaken.
    """

    table: _SettlementTableActions
    claimed_by: str
    claim_lease_seconds: float = BACKGROUND_INTERACTION_SETTLEMENT_CLAIM_LEASE_SECONDS

    async def register(self, pending: PendingBackgroundInteraction) -> None:
        await self.table.create(
            data=_NewSettlementRow(
                interaction_id=pending.interaction_id,
                custom_llm_provider=pending.custom_llm_provider,
                create_context=_json(pending.create_context.model_dump(mode="json")),
                created_at=pending.created_at,
            )
        )

    async def pending(self, interaction_id: str) -> PendingBackgroundInteraction | None:
        row: Final = await self._row(interaction_id)
        if row is None or self._is_owned(row):
            return None
        return next(iter(_pending_row(row, claim_expires_at=None)), None)

    async def is_claimed(self, interaction_id: str) -> bool:
        row: Final = await self._row(interaction_id)
        return row is not None and self._is_owned(row)

    async def claim(self, interaction_id: str) -> bool:
        now: Final = datetime.now(timezone.utc)
        claimed_rows: Final = await _read_from_a_table_that_may_not_exist(
            self.table.update_many(
                data=_Claim(claimed_at=now, claimed_by=self.claimed_by),
                where=_ClaimableRowKey(
                    interaction_id=interaction_id,
                    OR=(
                        _Unclaimed(claimed_at=None),
                        _ExpiredClaim(settled_at=None, claimed_at=_Before(lt=now - self._lease)),
                    ),
                ),
            ),
            when_missing=0,
        )
        return claimed_rows == 1

    @property
    def _lease(self) -> timedelta:
        return timedelta(seconds=self.claim_lease_seconds)

    def _claim_expires_at(self, row: _SettlementRow) -> datetime | None:
        return None if row.claimed_at is None else row.claimed_at + self._lease

    def _is_owned(self, row: _SettlementRow) -> bool:
        if row.settled_at is not None:
            return True
        claim_expires_at: Final = self._claim_expires_at(row)
        return claim_expires_at is not None and claim_expires_at > datetime.now(timezone.utc)

    async def _row(self, interaction_id: str) -> _SettlementRow | None:
        return await _read_from_a_table_that_may_not_exist(
            self.table.find_unique(where=_RowKey(interaction_id=interaction_id)), when_missing=None
        )

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None:
        await self.table.update_many(
            data=_Outcome(
                settled_at=datetime.now(timezone.utc), outcome=outcome, create_context=_json(_CLEARED_CREATE_CONTEXT)
            ),
            where=_RowKey(interaction_id=interaction_id),
        )

    async def unsettled(self) -> Sequence[PendingBackgroundInteraction]:
        rows: Final = await self.table.find_many(where=_UnsettledRows(settled_at=None))
        return tuple(chain.from_iterable(_pending_row(row, self._claim_expires_at(row)) for row in rows))


def _deployment_credentials(router: "Router | None", deployment_id: str | None) -> CredentialLiteLLMParams | None:
    if router is None or deployment_id is None:
        return None
    credentials: Final = router.get_deployment_credentials_with_provider(model_id=deployment_id)
    return None if credentials is None else CredentialLiteLLMParams.model_validate(credentials)


def fetch_with_deployment_credentials(
    router: Callable[[], "Router | None"], fetch_interaction: FetchInteraction = fetch_background_interaction
) -> FetchInteraction:
    """
    A resumed poll has no request to take credentials from, and they are never
    stored, so it fetches with the credentials of the deployment that served
    the create, looked up on the live router at each fetch so a deployment
    loaded from the database after startup is found too. A deployment the
    router no longer has falls back to the provider's environment credentials.
    """

    async def fetch(context: BackgroundInteractionPollContext) -> InteractionsAPIResponse:
        credentials: Final = _deployment_credentials(router(), context.deployment_id)
        if credentials is None:
            return await fetch_interaction(context)
        return await fetch_interaction(replace(context, api_key=credentials.api_key, api_base=credentials.api_base))

    return fetch


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


async def install_background_interaction_settlement(
    prisma_client: "PrismaClient", router: Callable[[], "Router | None"]
) -> None:
    try:
        await configure_background_interaction_settlement(
            table=BackgroundInteractionSettlementRepository(prisma_client).table,
            claimed_by=f"{socket.gethostname()}:{os.getpid()}",
            fetch_interaction=fetch_with_deployment_credentials(router),
        )
    except Exception as e:  # noqa: BLE001  # a boot step must survive any DB error; billing then settles in-process as before
        verbose_proxy_logger.warning(
            "Durable background interaction settlement is off on this replica, so billing settles in-process only: %s",
            e,
        )
