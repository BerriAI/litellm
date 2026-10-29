from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Protocol

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.repositories.table_repositories import BackgroundResponseSettlementRepository


class _NewSettlementRow(TypedDict):
    response_id: ReadOnly[str]


class BackgroundResponseSettlementTable(Protocol):
    def create(self, *, data: _NewSettlementRow) -> Awaitable[object]: ...


class BackgroundResponseSettlementStore(Protocol):
    async def claim(self, response_id: str) -> bool: ...


def _proxy_settlement_table() -> BackgroundResponseSettlementTable | None:
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global, set after load

    if prisma_client is None:
        return None
    return BackgroundResponseSettlementRepository(prisma_client).table


@dataclass(frozen=True, slots=True)
class PrismaBackgroundResponseSettlementStore:
    """
    The primary key lets exactly one call across all replicas insert a response id, and that call bills it.
    With no database or a failed insert the call still bills, since a lost charge cannot be recovered.
    """

    table: Callable[[], BackgroundResponseSettlementTable | None] = _proxy_settlement_table

    async def claim(self, response_id: str) -> bool:
        from prisma.errors import UniqueViolationError  # noqa: PLC0415  # prisma may be ungenerated at load

        try:
            table: Final = self.table()
            if table is None:
                return True
            await table.create(data=_NewSettlementRow(response_id=response_id))
        except UniqueViolationError:
            return False
        except Exception as e:  # noqa: BLE001  # any DB failure must bill as before rather than drop the charge
            verbose_proxy_logger.warning(
                "Could not record the settlement of background response %s, billing it anyway: %s", response_id, e
            )
        return True
