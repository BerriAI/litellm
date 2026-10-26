from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from litellm.repositories.user_banner_repository import UserBannerRepository, UserBannerStore

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@runtime_checkable
class Database(Protocol):
    @property
    def user_banners(self) -> UserBannerStore: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _PrismaDatabase:
    client: "PrismaClient"

    @property
    def user_banners(self) -> UserBannerStore:
        return UserBannerRepository(self.client)

    async def aclose(self) -> None:
        await self.client.disconnect()


def from_prisma(client: "PrismaClient") -> Database:
    """Adapt an initialized client without opening another pool."""
    return _PrismaDatabase(client)


@asynccontextmanager
async def lifespan(database: Database | None) -> AsyncGenerator[Database | None, None]:
    try:
        yield database
    finally:
        if database is not None:
            await database.aclose()
