import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, TypeAlias

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.constants import BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS
from litellm.repositories.table_repositories import BackgroundResponseSettlementRepository


class _NewSettlementRow(TypedDict):
    response_id: ReadOnly[str]


class _SettlementKey(TypedDict):
    response_id: ReadOnly[str]


class BackgroundResponseSettlementTable(Protocol):
    def create_many(self, data: Sequence[_NewSettlementRow], *, skip_duplicates: bool) -> Awaitable[int]: ...

    def delete_many(self, *, where: _SettlementKey) -> Awaitable[int]: ...


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
    async def claim_outcome(self, response_id: str, *, release_if_late: bool = False) -> ClaimOutcome: ...

    async def release(self, response_id: str) -> None: ...


def _proxy_settlement_table() -> BackgroundResponseSettlementTable | None:
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global, set after load

    if prisma_client is None:
        return None
    return BackgroundResponseSettlementRepository(prisma_client).table


@dataclass(frozen=True, slots=True)
class PrismaBackgroundResponseSettlementStore:
    table: Callable[[], BackgroundResponseSettlementTable | None] = _proxy_settlement_table
    timeout_seconds: float = BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS
    late_claims: set["asyncio.Task[None]"] = field(  # mutable-ok: asyncio needs strong refs to late claims
        default_factory=set
    )

    async def claim_outcome(self, response_id: str, *, release_if_late: bool = False) -> ClaimOutcome:
        try:
            table: Final = self.table()
            if table is None:
                return CLAIM_UNRECORDED
            insert: Final = asyncio.ensure_future(
                table.create_many([_NewSettlementRow(response_id=response_id)], skip_duplicates=True)
            )
            if not await self._recorded_in_time(response_id, insert):
                verbose_proxy_logger.warning(
                    "Recording the settlement of background response %s took over %s s, so it is settled when it "
                    "returns",
                    response_id,
                    self.timeout_seconds,
                )
                self._settle_late_claim(response_id, insert, release_if_late)
                return CLAIM_UNRECORDED
            inserted: Final = insert.result()
        except Exception as e:  # noqa: BLE001  # a DB failure comes back as a value so each caller picks its side
            verbose_proxy_logger.warning(
                "Could not record the settlement of background response %s: %s", response_id, e
            )
            return CLAIM_UNRECORDED
        return CLAIMED if inserted > 0 else CLAIMED_BEFORE

    async def _recorded_in_time(self, response_id: str, insert: "asyncio.Future[int]") -> bool:
        try:
            finished, _ = await asyncio.wait((insert,), timeout=self.timeout_seconds)
        except asyncio.CancelledError:
            self._settle_late_claim(response_id, insert, release_if_late=True)
            raise
        return bool(finished)

    def _settle_late_claim(self, response_id: str, insert: "asyncio.Future[int]", release_if_late: bool) -> None:
        late_claim: Final = asyncio.create_task(self._release_late_claim(response_id, insert, release_if_late))
        self.late_claims.add(late_claim)
        late_claim.add_done_callback(self.late_claims.discard)

    async def _release_late_claim(self, response_id: str, insert: "asyncio.Future[int]", release_if_late: bool) -> None:
        try:
            inserted: Final = await insert
        except Exception as e:  # noqa: BLE001  # its caller already went on as if the claim was never recorded
            verbose_proxy_logger.warning(
                "Could not record the settlement of background response %s: %s", response_id, e
            )
            return
        if release_if_late and inserted > 0:
            await self.release(response_id)

    async def release(self, response_id: str) -> None:
        try:
            table: Final = self.table()
            if table is None:
                return
            await asyncio.wait_for(
                table.delete_many(where=_SettlementKey(response_id=response_id)), timeout=self.timeout_seconds
            )
        except Exception as e:  # noqa: BLE001  # an unreleased claim leaves the response unbilled, never billed twice
            verbose_proxy_logger.warning(
                "Could not release the settlement of background response %s, so it stays unbilled: %s", response_id, e
            )
