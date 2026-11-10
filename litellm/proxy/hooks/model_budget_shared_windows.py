from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.db.budget_window_spend_writer import open_window_spend_row
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    WindowSpendTransaction,
    build_window_spend_transaction,
)
from litellm.repositories.table_repositories import BudgetWindowSpendRepository
from litellm.router_strategy.budget_limiter import budget_window_seconds

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

MODEL_BUDGET_WINDOW_ENTITY_TYPE: Final = "model_max_budget"
_READ_TTL_SECONDS: Final = 5


@dataclass(frozen=True, slots=True)
class SharedWindow:
    started_at: float
    spend: float

    def is_open(self, budget_duration: str, now: float) -> bool:
        return now < self.started_at + budget_window_seconds(self.started_at, budget_duration)


class LocalTTLCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object, *, ttl: float) -> None: ...

    def delete_cache(self, key: str) -> None: ...


class SharedWindows(Protocol):
    async def read(self, spend_key: str, budget_duration: str) -> SharedWindow | None: ...

    async def open(self, spend_key: str, budget_duration: str, now: float) -> float | None: ...

    async def add(self, spend_key: str, budget_duration: str, started_at: float, spend: float) -> None: ...


PrismaClientSource: TypeAlias = Callable[[], "PrismaClient | None"]
WindowSpendSink: TypeAlias = Callable[[WindowSpendTransaction], Awaitable[None]]


def _proxy_prisma_client() -> "PrismaClient | None":
    try:
        from litellm.proxy.proxy_server import prisma_client
    except ImportError:
        return None
    return prisma_client


async def _queue_on_proxy_spend_writer(transaction: WindowSpendTransaction) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    await proxy_logging_obj.db_spend_update_writer.window_spend_update_queue.add_update(transaction)


@dataclass(frozen=True, slots=True)
class _Reading:
    window: SharedWindow | None


class DatabaseSharedWindows:
    def __init__(
        self,
        prisma_client_source: PrismaClientSource = _proxy_prisma_client,
        sink: WindowSpendSink = _queue_on_proxy_spend_writer,
        read_ttl_seconds: int = _READ_TTL_SECONDS,
    ) -> None:
        self._prisma_client_source = prisma_client_source
        self._sink = sink
        self._read_ttl_seconds = read_ttl_seconds
        self._readings: LocalTTLCache = InMemoryCache(max_size_in_memory=10_000, default_ttl=read_ttl_seconds)

    async def read(self, spend_key: str, budget_duration: str) -> SharedWindow | None:
        cached: Final = self._readings.get_cache(key=spend_key)
        if isinstance(cached, _Reading):
            return cached.window
        window: Final = await self._read_row(spend_key=spend_key, budget_duration=budget_duration)
        self._readings.set_cache(key=spend_key, value=_Reading(window=window), ttl=self._read_ttl_seconds)
        return window

    async def open(self, spend_key: str, budget_duration: str, now: float) -> float | None:
        """When the window every replica shares started, opening one at `now` if none is open.

        Without a database every window is this replica's own, so it starts at `now`. None means the database did
        not answer, so this replica cannot know which window the others are in yet.
        """
        prisma_client: Final = self._prisma_client_source()
        if prisma_client is None:
            return now
        try:
            seen: Final = await _fetch_row(
                prisma_client=prisma_client, spend_key=spend_key, budget_duration=budget_duration, use_writer=True
            )
            if seen is not None and seen.is_open(budget_duration=budget_duration, now=now):
                return seen.started_at
            await open_window_spend_row(
                prisma_client=prisma_client,
                entity_type=MODEL_BUDGET_WINDOW_ENTITY_TYPE,
                entity_id=spend_key,
                window_duration=budget_duration,
                window_start=datetime.fromtimestamp(now, tz=timezone.utc),
                replaces_window_start=None
                if seen is None
                else datetime.fromtimestamp(seen.started_at, tz=timezone.utc),
            )
            opened: Final = await _fetch_row(
                prisma_client=prisma_client, spend_key=spend_key, budget_duration=budget_duration, use_writer=True
            )
        except Exception:  # noqa: BLE001  # an unreachable database leaves this replica on its own window
            verbose_proxy_logger.exception("Unable to open the shared model budget window")
            return None
        if opened is None or not opened.is_open(budget_duration=budget_duration, now=now):
            return None
        return opened.started_at

    async def add(self, spend_key: str, budget_duration: str, started_at: float, spend: float) -> None:
        if self._prisma_client_source() is None:
            return
        try:
            await self._sink(
                build_window_spend_transaction(
                    entity_type=MODEL_BUDGET_WINDOW_ENTITY_TYPE,
                    entity_id=spend_key,
                    window_duration=budget_duration,
                    window_start=datetime.fromtimestamp(started_at, tz=timezone.utc),
                    spend=spend,
                )
            )
        except Exception:  # noqa: BLE001  # spend tracking must never fail the success callback
            verbose_proxy_logger.debug("Unable to queue shared model budget spend", exc_info=True)

    async def _read_row(self, spend_key: str, budget_duration: str) -> SharedWindow | None:
        prisma_client: Final = self._prisma_client_source()
        if prisma_client is None:
            return None
        try:
            return await _fetch_row(
                prisma_client=prisma_client, spend_key=spend_key, budget_duration=budget_duration, use_writer=False
            )
        except Exception:  # noqa: BLE001  # an unreadable row leaves enforcement on this replica's own counter
            verbose_proxy_logger.exception("Unable to read shared model budget spend")
            return None


async def _fetch_row(
    prisma_client: "PrismaClient", spend_key: str, budget_duration: str, use_writer: bool
) -> SharedWindow | None:
    row: Final = await BudgetWindowSpendRepository(prisma_client, use_writer=use_writer).table.find_unique(
        where={
            "entity_type_entity_id_window_duration": {
                "entity_type": MODEL_BUDGET_WINDOW_ENTITY_TYPE,
                "entity_id": spend_key,
                "window_duration": budget_duration,
            }
        }
    )
    if row is None:
        return None
    window_start: Final = row.window_start
    started_at: Final = (
        window_start if window_start.tzinfo is not None else window_start.replace(tzinfo=timezone.utc)
    ).timestamp()
    return SharedWindow(started_at=started_at, spend=float(row.spend or 0.0))


PROXY_SHARED_WINDOWS: Final = DatabaseSharedWindows()
