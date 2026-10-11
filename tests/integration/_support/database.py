import os
import uuid
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Final, LiteralString
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from prisma import Prisma
from prisma.types import DatasourceOverride
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging

ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])


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


def read_rows(
    query: str, parameters: tuple[str, ...], *, database_url: str | None = None
) -> list[dict[str, JsonValue]]:
    with psycopg.connect(database_url or os.environ["DATABASE_URL"], row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        return ROWS.validate_python(connection.execute(query, parameters).fetchall())


def write_rows(query: LiteralString, parameters: tuple[str, ...], *, database_url: str | None = None) -> None:
    with psycopg.connect(database_url or os.environ["DATABASE_URL"]) as connection:
        connection.execute(query, parameters)


@contextmanager
def scratch_database() -> Generator[str]:
    name: Final = f"integration_{uuid.uuid4().hex}"
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{name}"))
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
