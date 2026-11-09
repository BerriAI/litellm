import asyncio
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.token_auth import parse_database_token_expiration
from litellm.repositories.database_settings import NativeDatabaseSettings
from litellm.rust_bridge._native import NativeDatabase

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class Database:
    prisma: "PrismaClient"
    native: NativeDatabase
    password_tasks: tuple[asyncio.Task[None], ...] = ()

    @classmethod
    async def open(cls, prisma: "PrismaClient") -> "Database":
        try:
            configured: Final = getattr(prisma, "native_writer_settings", None)
            writer: Final = (
                await _initial_password(configured) if isinstance(configured, NativeDatabaseSettings) else None
            )
            reader_configured: Final = getattr(prisma, "native_reader_settings", None)
            reader: Final = (
                await _initial_password(reader_configured)
                if isinstance(reader_configured, NativeDatabaseSettings)
                else None
            )
            native: Final = await NativeDatabase.connect(writer, reader)
        except BaseException:
            await prisma.aclose()
            raise
        prisma.set_native_auth_database(native if writer is not None else None)
        return cls(
            prisma=prisma,
            native=native,
            password_tasks=tuple(
                asyncio.create_task(_refresh_password(native, settings, is_reader))
                for settings, is_reader in ((writer, False), (reader, True))
                if settings is not None and settings.password_provider is not None
            ),
        )

    async def aclose(self) -> None:
        self.prisma.set_native_auth_database(None)
        for task in self.password_tasks:
            task.cancel()
        await asyncio.gather(*self.password_tasks, return_exceptions=True)
        try:
            await self.prisma.aclose()
        finally:
            await self.native.close()


async def _initial_password(settings: NativeDatabaseSettings) -> NativeDatabaseSettings:
    if settings.password_provider is None:
        return settings
    return replace(settings, password=await asyncio.to_thread(settings.password_provider))


def _refresh_delay(settings: NativeDatabaseSettings, password: str | None) -> float:
    expires: Final = (
        parse_database_token_expiration(settings.token_auth, password)
        if settings.token_auth is not None and password is not None
        else None
    )
    return (
        600.0
        if expires is None
        else max(10.0, (expires - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds() - 180.0)
    )


async def _refresh_password(native: NativeDatabase, settings: NativeDatabaseSettings, reader: bool) -> None:
    if settings.password_provider is None:
        return
    await asyncio.sleep(_refresh_delay(settings, settings.password))
    while True:
        await asyncio.sleep(await _rotate_password(native, settings, reader))


async def _rotate_password(native: NativeDatabase, settings: NativeDatabaseSettings, reader: bool) -> float:
    if settings.password_provider is None:
        return 600.0
    try:
        password: Final = await asyncio.to_thread(settings.password_provider)
        if reader:
            native.set_passwords(reader_password=password)
        else:
            native.set_passwords(writer_password=password)
    except Exception:
        verbose_proxy_logger.warning("Native database credential refresh failed")
        return 30.0
    return _refresh_delay(settings, password)


@asynccontextmanager
async def background_task(work: Coroutine[object, object, None]) -> AsyncGenerator[None, None]:
    task: Final = asyncio.create_task(work)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
