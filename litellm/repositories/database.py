import asyncio
import urllib.parse
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.proxy.db.native_database import keep_writer_token_fresh, native_endpoint, writer_token_rotation
from litellm.rust_bridge._native import NativeDatabase

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class Database:
    prisma: "PrismaClient"
    native: NativeDatabase
    token_refresh: "asyncio.Task[None] | None"

    @classmethod
    async def open(cls, prisma: "PrismaClient", writer_url: str) -> "Database":
        try:
            native: Final = await NativeDatabase.connect(native_endpoint(writer_url))
        except BaseException:
            await prisma.aclose()
            raise
        rotation: Final = writer_token_rotation(writer_url)
        token_refresh: Final = (
            None
            if rotation is None
            else asyncio.create_task(
                keep_writer_token_fresh(
                    native, rotation, urllib.parse.unquote(urllib.parse.urlsplit(writer_url).password or "")
                )
            )
        )
        return cls(prisma=prisma, native=native, token_refresh=token_refresh)

    async def aclose(self) -> None:
        if self.token_refresh is not None:
            self.token_refresh.cancel()
            await asyncio.gather(self.token_refresh, return_exceptions=True)
        try:
            await self.prisma.aclose()
        finally:
            await self.native.close()


@asynccontextmanager
async def background_task(work: Coroutine[object, object, None]) -> AsyncGenerator[None, None]:
    task: Final = asyncio.create_task(work)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
