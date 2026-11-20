import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.constants import BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS
from litellm.repositories.table_repositories import BackgroundResponseSettlementRepository


class _NewSettlementRow(TypedDict):
    response_id: ReadOnly[str]


class BackgroundResponseSettlementTable(Protocol):
    def create_many(self, data: Sequence[_NewSettlementRow], *, skip_duplicates: bool) -> Awaitable[int]: ...


@dataclass(frozen=True, slots=True)
class Claimed:
    pass


@dataclass(frozen=True, slots=True)
class ClaimedBefore:
    pass


@dataclass(frozen=True, slots=True)
class ClaimUnrecorded:
    pass


ClaimOutcome: TypeAlias = Claimed | ClaimedBefore | ClaimUnrecorded

CLAIMED: Final = Claimed()
CLAIMED_BEFORE: Final = ClaimedBefore()
CLAIM_UNRECORDED: Final = ClaimUnrecorded()


class BackgroundResponseSettlementStore(Protocol):
    async def claim(self, response_id: str) -> bool: ...

    async def claim_outcome(self, response_id: str) -> ClaimOutcome: ...


def _proxy_settlement_table() -> BackgroundResponseSettlementTable | None:
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global, set after load

    if prisma_client is None:
        return None
    return BackgroundResponseSettlementRepository(prisma_client).table


@dataclass(frozen=True, slots=True)
class PrismaBackgroundResponseSettlementStore:
    table: Callable[[], BackgroundResponseSettlementTable | None] = _proxy_settlement_table
    timeout_seconds: float = BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS

    async def claim(self, response_id: str) -> bool:
        return not isinstance(await self.claim_outcome(response_id), ClaimedBefore)

    async def claim_outcome(self, response_id: str) -> ClaimOutcome:
        try:
            table: Final = self.table()
            if table is None:
                return CLAIM_UNRECORDED
            inserted: Final = await asyncio.wait_for(
                table.create_many([_NewSettlementRow(response_id=response_id)], skip_duplicates=True),
                timeout=self.timeout_seconds,
            )
        except Exception as e:  # noqa: BLE001  # a DB failure comes back as a value so each caller picks its side
            verbose_proxy_logger.warning(
                "Could not record the settlement of background response %s: %s", response_id, e
            )
            return CLAIM_UNRECORDED
        return CLAIMED if inserted > 0 else CLAIMED_BEFORE
