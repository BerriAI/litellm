import asyncio
import os
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeVar

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
from litellm.repositories.table_repositories import BackgroundInteractionSettlementRepository, SpendLogsRepository
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


class _UnclaimedRowKey(TypedDict):
    interaction_id: ReadOnly[str]
    claimed_at: ReadOnly[None]


class _Before(TypedDict):
    lt: ReadOnly[datetime]


class _ExpiredClaimRowKey(TypedDict):
    interaction_id: ReadOnly[str]
    settled_at: ReadOnly[None]
    claimed_at: ReadOnly[_Before]


class _OwnClaimRowKey(TypedDict):
    interaction_id: ReadOnly[str]
    claimed_by: ReadOnly[str]
    settled_at: ReadOnly[None]


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

    def update_many(
        self, *, data: _Claim | _Outcome, where: _RowKey | _UnclaimedRowKey | _ExpiredClaimRowKey | _OwnClaimRowKey
    ) -> Awaitable[int]: ...


class _SpendLogKey(TypedDict):
    request_id: ReadOnly[str]


class _SpendLogLookup(Protocol):
    def find_unique(self, *, where: _SpendLogKey) -> Awaitable[object | None]: ...


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
    and any settler may claim it again. A live settler renews its claim while
    it settles, so it is never overtaken. The settler before may have billed
    and then died or failed to record it, so a takeover bills only when the
    interaction has no spend row yet, and otherwise records it as billed. The
    spend row is read before the claim is taken, so a failed read takes
    nothing and the claim is retried. With spend logs off there is no row to
    tell, so a takeover bills nothing rather than risk billing twice.
    """

    table: _SettlementTableActions
    spend_logs: _SpendLogLookup
    spend_logs_kept: Callable[[], bool]
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
        if await self._claim_where(_UnclaimedRowKey(interaction_id=interaction_id, claimed_at=None)):
            return True
        settled_before: Final = await self._outcome_a_dead_settler_left(interaction_id)
        took_over: Final = await self._claim_where(
            _ExpiredClaimRowKey(
                interaction_id=interaction_id,
                settled_at=None,
                claimed_at=_Before(lt=datetime.now(timezone.utc) - self._lease),
            )
        )
        if not took_over or settled_before is None:
            return took_over
        if settled_before == "unsettled":
            verbose_proxy_logger.warning(
                "Background interaction %s lost its settler mid-settle, and with spend logs off nothing shows "
                "whether it was billed, so it is left unbilled rather than risk billing it twice",
                interaction_id,
            )
        await self.record_outcome(interaction_id, settled_before)
        return False

    async def renew_claim(self, interaction_id: str) -> None:
        await self.table.update_many(
            data=_Claim(claimed_at=datetime.now(timezone.utc), claimed_by=self.claimed_by),
            where=_OwnClaimRowKey(interaction_id=interaction_id, claimed_by=self.claimed_by, settled_at=None),
        )

    async def _claim_where(self, where: _UnclaimedRowKey | _ExpiredClaimRowKey) -> bool:
        claimed_rows: Final = await _read_from_a_table_that_may_not_exist(
            self.table.update_many(
                data=_Claim(claimed_at=datetime.now(timezone.utc), claimed_by=self.claimed_by), where=where
            ),
            when_missing=0,
        )
        return claimed_rows == 1

    async def _outcome_a_dead_settler_left(self, interaction_id: str) -> Literal["billed", "unsettled"] | None:
        if not self.spend_logs_kept():
            return "unsettled"
        if await self.spend_logs.find_unique(where=_SpendLogKey(request_id=interaction_id)) is None:
            return None
        return "billed"

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


def _provider_credentials(router: "Router", deployment_id: str, custom_llm_provider: str) -> Mapping[str, object]:
    credentials: Final = router.get_deployment_credentials_with_provider(model_id=deployment_id)
    if credentials is None or credentials.get("custom_llm_provider") != custom_llm_provider:
        return MappingProxyType({})
    return MappingProxyType(credentials)


def _group_deployment_credentials(router: "Router", context: BackgroundInteractionPollContext) -> Mapping[str, object]:
    if context.model_group is None:
        return MappingProxyType({})
    candidates: Final = tuple(
        credentials
        for deployment_id in router.get_model_ids(model_name=context.model_group)
        if (credentials := _provider_credentials(router, deployment_id, context.custom_llm_provider))
    )
    return candidates[0] if len(candidates) == 1 else MappingProxyType({})


def _deployment_credentials(
    router: "Router | None", context: BackgroundInteractionPollContext
) -> CredentialLiteLLMParams | None:
    if router is None:
        return None
    serving: Final = (
        _provider_credentials(router, context.deployment_id, context.custom_llm_provider)
        if context.deployment_id is not None and router.get_deployment(model_id=context.deployment_id) is not None
        else _group_deployment_credentials(router, context)
    )
    return CredentialLiteLLMParams.model_validate(serving) if serving else None


def fetch_with_deployment_credentials(
    router: Callable[[], "Router | None"], fetch_interaction: FetchInteraction = fetch_background_interaction
) -> FetchInteraction:
    """
    A resumed poll has no request to take credentials from, and they are never
    stored, so it fetches with the credentials of the deployment that served
    the create, looked up by its id on the live router at each fetch so a
    deployment loaded from the database after startup is found too. Editing a
    deployment changes its id, so a missing id falls back to the create's model
    group when exactly one of its deployments serves the same provider. Anything
    else falls back to the provider's environment credentials, never to a
    wildcard route, whose key belongs to whatever provider it routes to.
    """

    async def fetch(context: BackgroundInteractionPollContext) -> InteractionsAPIResponse:
        credentials: Final = _deployment_credentials(router(), context)
        if credentials is None:
            return await fetch_interaction(context)
        return await fetch_interaction(replace(context, credentials=credentials))

    return fetch


async def configure_background_interaction_settlement(
    table: _SettlementTableActions,
    spend_logs: _SpendLogLookup,
    spend_logs_kept: Callable[[], bool],
    claimed_by: str,
    fetch_interaction: FetchInteraction = fetch_background_interaction,
    schedule: PollSchedule = DEFAULT_POLL_SCHEDULE,
) -> tuple["asyncio.Task[SettlementOutcome | None]", ...]:
    if not BACKGROUND_INTERACTION_COST_POLLING_ENABLED:
        return ()
    store: Final = PrismaBackgroundSettlementStore(
        table=table, spend_logs=spend_logs, spend_logs_kept=spend_logs_kept, claimed_by=claimed_by
    )
    configure_background_settlement_store(store)
    resumed: Final = await resume_unsettled_background_interactions(store, fetch_interaction, schedule)
    if resumed:
        verbose_proxy_logger.info("Resumed cost polling for %s unsettled background interactions", len(resumed))
    return resumed


async def install_background_interaction_settlement(
    prisma_client: "PrismaClient", router: Callable[[], "Router | None"], spend_logs_kept: Callable[[], bool]
) -> None:
    try:
        await configure_background_interaction_settlement(
            table=BackgroundInteractionSettlementRepository(prisma_client).table,
            spend_logs=SpendLogsRepository(prisma_client).table,
            spend_logs_kept=spend_logs_kept,
            claimed_by=f"{socket.gethostname()}:{os.getpid()}",
            fetch_interaction=fetch_with_deployment_credentials(router),
        )
    except Exception as e:  # noqa: BLE001  # a boot step must survive any DB error; billing then settles in-process as before
        verbose_proxy_logger.warning(
            "Durable background interaction settlement is off on this replica, so billing settles in-process only: %s",
            e,
        )
