import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from prisma import Prisma
from prisma.types import DatasourceOverride

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging


@asynccontextmanager
async def connected_prisma(database_url: str | None = None) -> AsyncIterator[Prisma]:
    client: Final = Prisma(
        datasource=DatasourceOverride(url=database_url)
    ) if database_url is not None else Prisma()
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


@asynccontextmanager
async def connected_proxy_prisma(database_url: str | None = None) -> AsyncIterator[PrismaClient]:
    client: Final = PrismaClient(
        database_url=database_url or os.environ["DATABASE_URL"],
        proxy_logging_obj=ProxyLogging(UserApiKeyCache()),
    )
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()
