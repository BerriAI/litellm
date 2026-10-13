import random
import re
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from litellm_proxy_extras._logging import logger
from litellm_proxy_extras.prisma_toolchain import (
    MIGRATION_LOCK_TIMEOUT_ENV_VAR,
    migration_ddl_lock_timeout,
    migration_lock_timeout,
)

MIGRATION_LOCK_KEY: Final = int.from_bytes(b"llm_mig2", "big")
_LOCK_TIMEOUT_PINNED_RE: Final = re.compile(r"(?:-c\s*|--)lock_timeout=")

if TYPE_CHECKING:
    import psycopg


def _migration_url(database_url: str, direct_url: "str | None") -> str:
    if not direct_url:
        return database_url
    schema: Final = next((value for key, value in parse_qsl(urlsplit(database_url).query) if key == "schema"), "public")
    direct: Final = urlsplit(direct_url)
    parameters: Final = tuple((key, value) for key, value in parse_qsl(direct.query) if key != "schema")
    return urlunsplit(direct._replace(query=urlencode((*parameters, ("schema", schema)))))


def _with_ddl_lock_timeout(url: str) -> str:
    split: Final = urlsplit(url)
    pairs: Final = tuple(parse_qsl(split.query))
    if any(key == "pgbouncer" and value == "true" for key, value in pairs):
        return url
    existing: Final = next((value for key, value in pairs if key == "options"), "")
    if _LOCK_TIMEOUT_PINNED_RE.search(existing):
        return url
    timeout_ms: Final = max(1, int(migration_ddl_lock_timeout() * 1000))
    options: Final = f"{existing} -c lock_timeout={timeout_ms}".strip()
    rewritten: Final = tuple((key, value) for key, value in pairs if key != "options") + (("options", options),)
    return urlunsplit(split._replace(query=urlencode(rewritten)))


def migration_environment(environment: Mapping[str, str]) -> Mapping[str, str]:
    database_url: Final = environment.get("DATABASE_URL")
    if not database_url:
        return environment
    migration_url: Final = _migration_url(database_url, environment.get("DIRECT_URL"))
    return {
        **environment,
        "DATABASE_URL": _with_ddl_lock_timeout(migration_url),
    }


@dataclass(frozen=True, slots=True)
class _LockResult:
    acquired: bool


def _try_lock(connection: "psycopg.Connection[tuple[object, ...]]", key: int = MIGRATION_LOCK_KEY) -> bool:
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(_LockResult)) as cursor:
        row: Final = cursor.execute("SELECT pg_try_advisory_xact_lock(%s) AS acquired", (key,)).fetchone()
    return row is not None and row.acquired


@dataclass(frozen=True, slots=True)
class MigrationCoordinator:
    connection: "psycopg.Connection[tuple[object, ...]]"

    def check_connection(self) -> None:
        self.connection.execute("SELECT 1")

    def acquire_prisma_lock(self) -> None:
        deadline: Final = time.monotonic() + migration_lock_timeout()
        while time.monotonic() < deadline:
            if _try_lock(self.connection, 72707369):
                return
            time.sleep(min(random.uniform(0.5, 1.5), max(0.0, deadline - time.monotonic())))
        raise RuntimeError(
            "Timed out waiting for Prisma's lock to recover migration history. LiteLLM startup has stopped. "
            "Another migration or a pooled database session may still hold the lock. Check the database lock holder. "
            "When using a transaction pooler, configure DIRECT_URL to reach the same database without the pooler."
        )


@contextmanager
def migration_lock(database_url: str) -> Generator[MigrationCoordinator, None, None]:
    import psycopg

    wait_seconds: Final = migration_lock_timeout()
    deadline: Final = time.monotonic() + wait_seconds
    try:
        with psycopg.connect(database_url, connect_timeout=10, autocommit=True) as connection:
            coordinator: Final = MigrationCoordinator(connection)
            logger.info("Waiting for the v2 migration coordinator lock (up to %ss)", wait_seconds)
            while time.monotonic() < deadline:
                with connection.transaction():
                    if _try_lock(connection):
                        logger.info("Acquired the v2 migration coordinator lock")

                        yield coordinator
                        coordinator.check_connection()
                        return
                time.sleep(min(random.uniform(0.5, 1.5), max(0.0, deadline - time.monotonic())))
    except psycopg.Error as exc:
        raise RuntimeError(f"Lost or could not establish v2 migration coordination with the database: {exc}") from exc
    raise RuntimeError(
        f"Timed out waiting for another v2 migration resolver after {wait_seconds}s. "
        f"Check the running migration or increase {MIGRATION_LOCK_TIMEOUT_ENV_VAR}."
    )


@contextmanager
def held_migration_lock(connection: "psycopg.Connection[tuple[object, ...]]") -> Generator[bool, None, None]:
    """A session-level, non-blocking hold of the migration coordinator lock on an autocommit
    connection, for DDL that cannot run inside a transaction (`CREATE INDEX CONCURRENTLY`).
    Yields whether the lock was acquired; a v2 resolver or another migration job's index build
    holding it yields False. Released on exit."""
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(_LockResult)) as cursor:
        row: Final = cursor.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (MIGRATION_LOCK_KEY,)).fetchone()
    acquired: Final = row is not None and row.acquired
    try:
        yield acquired
    finally:
        if acquired:
            connection.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
