import asyncio
import urllib.parse
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.proxy.db.db_span import proxy_service_logging
from litellm.proxy.db.native_database import (
    ServiceLoggingEvents,
    TokenRotation,
    keep_token_fresh,
    native_endpoint,
    reader_token_rotation,
    writer_token_rotation,
)
from litellm.rust_bridge._native import NativeDatabase

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def _token_refresh(
    install: Callable[[str], None], rotation: TokenRotation | None, url: str
) -> "asyncio.Task[None] | None":
    if rotation is None:
        return None
    current: Final = urllib.parse.unquote(urllib.parse.urlsplit(url).password or "")
    return asyncio.create_task(keep_token_fresh(install, rotation, current))


@dataclass(frozen=True, slots=True)
class Database:
    prisma: "PrismaClient"
    native: NativeDatabase
    token_refreshes: "tuple[asyncio.Task[None], ...]"

    @classmethod
    async def open(cls, prisma: "PrismaClient", writer_url: str, reader_url: str | None) -> "Database":
        try:
            native: Final = await NativeDatabase.connect(
                ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=proxy_service_logging),
                native_endpoint(writer_url),
                None if reader_url is None else native_endpoint(reader_url),
            )
        except BaseException:
            await prisma.aclose()
            raise
        refreshes: Final = (
            _token_refresh(native.set_writer_password, writer_token_rotation(writer_url), writer_url),
            None
            if reader_url is None
            else _token_refresh(native.set_reader_password, reader_token_rotation(reader_url), reader_url),
        )
        return cls(
            prisma=prisma,
            native=native,
            token_refreshes=tuple(task for task in refreshes if task is not None),
        )

    async def aclose(self) -> None:
        for task in self.token_refreshes:
            task.cancel()
        await asyncio.gather(*self.token_refreshes, return_exceptions=True)
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
