import os
import uuid
from datetime import timedelta
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from prisma import Prisma
from prisma.errors import DataError
from psycopg import sql

from litellm.proxy.spend_tracking.key_metadata_recovery import attach_user_details
from litellm.repositories.chunked_in import count_in, delete_many_in, find_many_in, update_many_in

ROWS: Final = 40_000
OUTSIDE: Final = 25


def _scoped_url(url: str, schema: str) -> str:
    parsed: Final = urlsplit(url)
    return urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))


@asynccontextmanager
async def _user_table(users: int) -> AsyncIterator[Prisma]:
    """A private schema holding a copy of the migrated `LiteLLM_UserTable`, seeded with `users` rows."""
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    table: Final = sql.Identifier(schema, "LiteLLM_UserTable")
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL('CREATE TABLE {} (LIKE "LiteLLM_UserTable" INCLUDING DEFAULTS INCLUDING CONSTRAINTS)').format(
                    table
                )
            )
            setup.execute(
                sql.SQL(
                    "INSERT INTO {} (user_id, user_email) "
                    "SELECT 'user-' || n, 'user-' || n || '@example.com' FROM generate_series(0, %s) n"
                ).format(table),
                (users - 1,),
            )
            database: Final = Prisma(datasource={"url": _scoped_url(url, schema)})
            await database.connect()
            try:
                yield database
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@asynccontextmanager
async def _config_table() -> AsyncIterator[tuple[Prisma, str]]:
    """A private schema holding only `LiteLLM_Config`, seeded with ROWS listed and OUTSIDE unlisted rows."""
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    scoped_url: Final = _scoped_url(url, schema)
    table: Final = sql.Identifier(schema, "LiteLLM_Config")
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL(
                    "CREATE TABLE {} (param_name text PRIMARY KEY, param_value jsonb, "
                    "last_run_at timestamp(3), reload_revision bigint NOT NULL DEFAULT 0)"
                ).format(table)
            )
            setup.execute(
                sql.SQL(
                    "INSERT INTO {} (param_name) SELECT 'listed-' || n FROM generate_series(0, %s) n "
                    "UNION ALL SELECT 'outside-' || n FROM generate_series(0, %s) n"
                ).format(table),
                (ROWS - 1, OUTSIDE - 1),
            )
            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                yield database, schema
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _listed() -> list[str]:
    return [f"listed-{n}" for n in range(ROWS)]


def _count(schema: str, condition: sql.Composable) -> int:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        row: Final = connection.execute(
            sql.SQL("SELECT count(*) FROM {} WHERE ").format(sql.Identifier(schema, "LiteLLM_Config")) + condition
        ).fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.covers("other.database.chunked_in.raw_in_list_over_bind_cap_fails")
async def test_a_raw_in_filter_over_the_bind_parameter_cap_is_rejected_by_postgres() -> None:
    async with _config_table() as (database, schema):
        where: Final = {"param_name": {"in": _listed()}}
        with pytest.raises(DataError, match="too many bind variables"):
            await database.litellm_config.count(where=where)
        with pytest.raises(DataError, match="too many bind variables"):
            await database.litellm_config.update_many(where=where, data={"reload_revision": 1})
        with pytest.raises(DataError, match="too many bind variables"):
            await database.litellm_config.delete_many(where=where)
        assert _count(schema, sql.SQL("reload_revision = 0")) == ROWS + OUTSIDE


@pytest.mark.covers(
    "other.database.chunked_in.find_many_in_returns_every_row",
    "other.database.chunked_in.count_in_counts_every_row",
)
async def test_find_many_in_and_count_in_read_every_row_past_the_bind_parameter_cap() -> None:
    async with _config_table() as (database, _):
        values: Final = [*_listed(), *_listed()[:100], "missing"]
        rows: Final = await find_many_in(database.litellm_config, "param_name", values)
        assert sorted(row.param_name for row in rows) == sorted(_listed())
        assert await count_in(database.litellm_config, "param_name", values) == ROWS
        assert await count_in(database.litellm_config, "param_name", values, where={"reload_revision": 1}) == 0


@pytest.mark.covers("other.database.chunked_in.update_many_in_updates_every_row_in_a_transaction")
async def test_update_many_in_updates_every_row_inside_one_transaction() -> None:
    async with _config_table() as (database, schema):
        async with database.tx(timeout=timedelta(seconds=60)) as transaction:
            updated: Final = await update_many_in(
                transaction.litellm_config,
                "param_name",
                _listed(),
                data={"reload_revision": 7},
                atomicity="caller_transaction",
            )
        assert updated == ROWS
        assert _count(schema, sql.SQL("reload_revision = 7 AND param_name LIKE 'listed-%'")) == ROWS
        assert _count(schema, sql.SQL("reload_revision = 0 AND param_name LIKE 'outside-%'")) == OUTSIDE


@pytest.mark.covers("other.database.chunked_in.delete_many_in_deletes_every_row")
async def test_delete_many_in_deletes_every_listed_row_and_nothing_else() -> None:
    async with _config_table() as (database, schema):
        deleted: Final = await delete_many_in(
            database.litellm_config, "param_name", _listed(), atomicity="per_chunk_ok", where={"reload_revision": 0}
        )
        assert deleted == ROWS
        assert _count(schema, sql.SQL("TRUE")) == OUTSIDE


@pytest.mark.covers("other.database.chunked_in.key_metadata_recovery_attaches_details_past_the_bind_parameter_cap")
async def test_key_metadata_recovery_attaches_user_details_for_more_users_than_the_bind_parameter_cap() -> None:
    async with _user_table(ROWS) as database:
        recovered: Final = {f"key-{n}": {"key_alias": f"alias-{n}", "user_id": f"user-{n}"} for n in range(ROWS)}
        attached: Final = await attach_user_details(SimpleNamespace(db=database), recovered)  # pyright: ignore[reportArgumentType]  # only .db is read
        assert all(attached[f"key-{n}"].get("user_email") == f"user-{n}@example.com" for n in range(ROWS))
