import asyncio
import urllib.parse
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.native_database import (
    keep_writer_token_fresh,
    resolve_native_endpoint,
    writer_token_rotation,
)
from litellm.rust_bridge._native import NativeDatabase

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class Database:
    prisma: "PrismaClient"
    native: NativeDatabase | None
    tasks: "tuple[asyncio.Task[None], ...]" = ()

    @classmethod
    async def open(cls, prisma: "PrismaClient", writer_url: str, native_connections: int) -> "Database":
        if native_connections == 0:
            return cls(prisma=prisma, native=None)
        try:
            settings: Final = await resolve_native_endpoint(writer_url, native_connections)
            native: Final = await NativeDatabase.connect(settings.endpoint)
        except BaseException:
            await prisma.aclose()
            raise
        if settings.ignored_options:
            verbose_proxy_logger.error(
                "Native database pool cannot send the DATABASE_URL options %s; only -c name=value settings carry over",
                settings.ignored_options,
            )
        rotation: Final = writer_token_rotation(writer_url)
        token_refresh: Final = (
            ()
            if rotation is None
            else (
                asyncio.create_task(
                    keep_writer_token_fresh(
                        native, rotation, urllib.parse.unquote(urllib.parse.urlsplit(writer_url).password or "")
                    )
                ),
            )
        )
        return cls(prisma=prisma, native=native, tasks=token_refresh)

    async def aclose(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        try:
            await self.prisma.aclose()
        finally:
            if self.native is not None:
                await self.native.close()


@asynccontextmanager
async def background_task(work: Coroutine[object, object, None]) -> AsyncGenerator[None, None]:
    task: Final = asyncio.create_task(work)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
