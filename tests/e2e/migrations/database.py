from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, LiteralString
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql
from pydantic import TypeAdapter

Scalar = str | int | bool | None
ROWS: Final = TypeAdapter(tuple[tuple[Scalar, ...], ...])
GATE_KEY: Final = 39178002
PRISMA_LOCK: Final = 72707369
COORDINATOR_LOCK: Final = int.from_bytes(b"llm_mig2", "big")


def connect_url(url: str, name: str) -> str:
    return urlunsplit(urlsplit(url)._replace(path=f"/{name}", query=""))


def prisma_url(url: str, schema: str) -> str:
    parsed: Final = urlsplit(url)
    query: Final = tuple((key, value) for key, value in parse_qsl(parsed.query) if key != "schema")
    return urlunsplit(parsed._replace(query=urlencode((*query, ("schema", schema)))))


@dataclass(frozen=True, slots=True)
class Database:
    name: str
    url: str
    container_url: str
    schema: str = "public"

    @contextmanager
    def connection(self) -> Generator[psycopg.Connection[tuple[object, ...]]]:
        with psycopg.connect(self.url, autocommit=True, connect_timeout=5) as connection:
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
            connection.execute("SET statement_timeout = '15s'")
            yield connection

    def execute(self, statement: LiteralString | sql.Composed, params: tuple[Scalar, ...] = ()) -> None:
        with self.connection() as connection:
            connection.execute(statement, params or None)

    def query(
        self, statement: LiteralString | sql.Composed, params: tuple[Scalar, ...] = ()
    ) -> tuple[tuple[Scalar, ...], ...]:
        with self.connection() as connection:
            return ROWS.validate_python(connection.execute(statement, params or None).fetchall())

    def exists(self, name: str) -> bool:
        return self.query("SELECT to_regclass(%s) IS NOT NULL", (name,)) == ((True,),)

    def history(self) -> tuple[tuple[Scalar, ...], ...]:
        if not self.exists("_prisma_migrations"):
            return ()
        return self.query(
            "SELECT id, migration_name, checksum, started_at::text, finished_at::text, rolled_back_at::text, "
            "applied_steps_count, logs FROM _prisma_migrations ORDER BY id"
        )

    def blocked(self, key: int = GATE_KEY) -> tuple[tuple[Scalar, ...], ...]:
        return self.query(
            "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND NOT granted "
            "AND database = (SELECT oid FROM pg_database WHERE datname = current_database()) "
            "AND classid = %s AND objid = %s ORDER BY pid",
            (key >> 32, key & 0xFFFFFFFF),
        )

    @contextmanager
    def lock(self, key: int = GATE_KEY) -> Generator[None]:
        with self.connection() as connection:
            connection.execute("SELECT pg_advisory_lock(%s)", (key,))
            try:
                yield
            finally:
                connection.execute("SELECT pg_advisory_unlock(%s)", (key,))


@dataclass(frozen=True, slots=True)
class Databases:
    admin_url: str
    container_admin_url: str

    @contextmanager
    def create(self, template: Database | None = None, schema: str = "public") -> Generator[Database]:
        name: Final = f"litellm_migration_test_{uuid4().hex[:20]}"
        database: Final = Database(
            name, connect_url(self.admin_url, name), connect_url(self.container_admin_url, name), schema
        )
        with psycopg.connect(self.admin_url, autocommit=True, connect_timeout=5) as connection:
            connection.execute(
                sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                    sql.Identifier(name), sql.Identifier(template.name if template else "template0")
                )
            )
        try:
            yield database
        finally:
            with psycopg.connect(self.admin_url, autocommit=True, connect_timeout=5) as connection:
                connection.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@contextmanager
def restricted_user(database: Database) -> Generator[Database]:
    role: Final = f"migration_reader_{uuid4().hex[:16]}"
    password: Final = "migration-test-password"
    with database.connection() as connection:
        connection.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password))
        )
    try:
        database.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(database.schema), sql.Identifier(role))
        )
        database.execute(
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
                sql.Identifier(database.schema), sql.Identifier(role)
            )
        )
        local: Final = urlsplit(database.url)
        remote: Final = urlsplit(database.container_url)
        yield Database(
            database.name,
            urlunsplit(local._replace(netloc=f"{role}:{password}@{local.hostname}:{local.port}")),
            urlunsplit(remote._replace(netloc=f"{role}:{password}@{remote.hostname}:{remote.port}")),
            database.schema,
        )
    finally:
        database.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
        database.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
