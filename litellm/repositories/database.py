import asyncio
import urllib.parse
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.db_span import proxy_service_logging
from litellm.proxy.db.native_database import (
    NativeEndpointSettings,
    ServiceLoggingEvents,
    TokenRotation,
    keep_token_fresh,
    reader_token_rotation,
    resolve_native_endpoint,
    writer_token_rotation,
)
from litellm.rust_bridge._native import NativeDatabase, NativeDatabaseFailure

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def _token_refresh(
    install: Callable[[str], None], rotation: TokenRotation | None, url: str
) -> "tuple[asyncio.Task[None], ...]":
    if rotation is None:
        return ()
    current: Final = urllib.parse.unquote(urllib.parse.urlsplit(url).password or "")
    return (asyncio.create_task(keep_token_fresh(install, rotation, current)),)


def _log_ignored_options(env_var: str, settings: NativeEndpointSettings) -> None:
    if settings.ignored_options:
        verbose_proxy_logger.error(
            "Native database pool cannot send the %s options %s; only -c name=value settings carry over",
            env_var,
            settings.ignored_options,
        )


async def _warn_if_unreachable(native: NativeDatabase) -> None:
    try:
        await native.health_check()
    except NativeDatabaseFailure as error:
        verbose_proxy_logger.warning(
            "Native database pool cannot reach the database, so operations ported to it will fail until it"
            " connects. Prisma is checked separately. Cause: %s",
            error,
        )


@dataclass(frozen=True, slots=True)
class Database:
    prisma: "PrismaClient"
    native: NativeDatabase | None
    tasks: "tuple[asyncio.Task[None], ...]" = ()
    boot_probe: "asyncio.Task[None] | None" = None

    @classmethod
    async def open(
        cls, prisma: "PrismaClient", writer_url: str, reader_url: str | None, native_connections: int
    ) -> "Database":
        if native_connections == 0:
            return cls(prisma=prisma, native=None)
        try:
            writer: Final = await resolve_native_endpoint(writer_url, native_connections)
            reader: Final = (
                None if reader_url is None else await resolve_native_endpoint(reader_url, native_connections)
            )
            native: Final = await NativeDatabase.connect(
                ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=proxy_service_logging),
                writer.endpoint,
                None if reader is None else reader.endpoint,
            )
        except BaseException:
            await prisma.aclose()
            raise
        _log_ignored_options("DATABASE_URL", writer)
        if reader is not None:
            _log_ignored_options("DATABASE_URL_READ_REPLICA", reader)
        writer_refresh: Final = _token_refresh(
            native.set_writer_password, writer_token_rotation(writer_url), writer_url
        )
        reader_refresh: Final = (
            ()
            if reader_url is None
            else _token_refresh(native.set_reader_password, reader_token_rotation(reader_url), reader_url)
        )
        return cls(
            prisma=prisma,
            native=native,
            tasks=(*writer_refresh, *reader_refresh),
            boot_probe=asyncio.create_task(_warn_if_unreachable(native)),
        )

    async def aclose(self) -> None:
        background: Final = (*self.tasks, *(() if self.boot_probe is None else (self.boot_probe,)))
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
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
