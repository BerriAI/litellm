import random
import re
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from litellm_proxy_extras._logging import logger
from litellm_proxy_extras._redaction import _redact_credentials
from litellm_proxy_extras.prisma_toolchain import (
    MIGRATION_LOCK_TIMEOUT_ENV_VAR,
    migration_lock_timeout,
)

MIGRATION_LOCK_KEY: Final = int.from_bytes(b"llm_mig2", "big")
PRISMA_ADVISORY_LOCK_KEY: Final = 72707369
_SERVER_ANSWERED_MARKERS: Final = ("FATAL:", "SSL", "certificate")


@dataclass(frozen=True, slots=True)
class AdvisoryLock:
    key: int
    name: str
    holder: str


COORDINATOR_LOCK: Final = AdvisoryLock(
    MIGRATION_LOCK_KEY,
    "v2 migration coordinator lock",
    "another v2 migration resolver",
)
PRISMA_LOCK: Final = AdvisoryLock(
    PRISMA_ADVISORY_LOCK_KEY,
    "Prisma migration lock",
    "another Prisma migration",
)

if TYPE_CHECKING:
    import psycopg


def _migration_url(database_url: str, direct_url: "str | None") -> str:
    if not direct_url:
        return database_url
    schema: Final = next((value for key, value in parse_qsl(urlsplit(database_url).query) if key == "schema"), "public")
    direct: Final = urlsplit(direct_url)
    parameters: Final = tuple((key, value) for key, value in parse_qsl(direct.query) if key != "schema")
    return urlunsplit(direct._replace(query=urlencode((*parameters, ("schema", schema)))))


def migration_environment(environment: Mapping[str, str]) -> Mapping[str, str]:
    database_url: Final = environment.get("DATABASE_URL")
    if not database_url:
        return environment
    migration_url: Final = _migration_url(database_url, environment.get("DIRECT_URL"))
    return {**environment, "DATABASE_URL": migration_url, "PRISMA_SCHEMA_DISABLE_ADVISORY_LOCK": "1"}


@dataclass(frozen=True, slots=True)
class _LockResult:
    acquired: bool


def _try_lock(connection: "psycopg.Connection[tuple[object, ...]]", key: int = MIGRATION_LOCK_KEY) -> bool:
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(_LockResult)) as cursor:
        row: Final = cursor.execute("SELECT pg_try_advisory_xact_lock(%s) AS acquired", (key,)).fetchone()
    return row is not None and row.acquired


def _coordination_failure(database_url: str, exc: "psycopg.Error", *, connecting: bool) -> str:
    import psycopg

    prefix: Final = "Lost or could not establish v2 migration coordination with the database: "
    raw: Final = str(exc)
    unquoted: Final = re.sub(r"\"[^\"]*\"|'[^']*'", "", raw)
    detail: Final = _redact_credentials(raw).strip()
    if (
        not connecting
        or not isinstance(exc, psycopg.OperationalError)
        or any(marker in unquoted for marker in _SERVER_ANSWERED_MARKERS)
    ):
        return prefix + detail
    parsed_url: Final = urlsplit(database_url)
    try:
        port: Final = parsed_url.port
    except ValueError:
        return prefix + f"P1001: Can't reach database server. {detail}"
    host: Final = parsed_url.hostname or "localhost"
    return prefix + f"P1001: Can't reach database server at `{host}:{port or 5432}`. {detail}"


@dataclass(frozen=True, slots=True)
class MigrationCoordinator:
    connection: "psycopg.Connection[tuple[object, ...]]"

    def check_connection(self) -> None:
        self.connection.execute("SELECT 1")

    def release_statement_snapshot(self) -> None:
        # a parameterized statement leaves its portal snapshot registered on the backend until the
        # next statement, and CREATE INDEX CONCURRENTLY waits for every such snapshot to go away
        self.connection.execute("SELECT 1")

    def acquire_prisma_lock(self) -> None:
        deadline: Final = time.monotonic() + migration_lock_timeout()
        while time.monotonic() < deadline:
            if _try_lock(self.connection, PRISMA_ADVISORY_LOCK_KEY):
                return
            time.sleep(min(random.uniform(0.5, 1.5), max(0.0, deadline - time.monotonic())))
        raise RuntimeError(
            "Timed out waiting for Prisma's lock to recover migration history. LiteLLM startup has stopped. "
            "Another migration or a pooled database session may still hold the lock. Check the database lock holder. "
            "When using a transaction pooler, configure DIRECT_URL to reach the same database without the pooler."
        )


@contextmanager
def migration_lock(
    database_url: str,
    lock: AdvisoryLock = COORDINATOR_LOCK,
) -> Generator[MigrationCoordinator, None, None]:
    import psycopg

    wait_seconds: Final = migration_lock_timeout()
    deadline: Final = time.monotonic() + wait_seconds
    try:
        connection: Final = psycopg.connect(database_url, connect_timeout=10, autocommit=True)
    except psycopg.Error as exc:
        raise RuntimeError(_coordination_failure(database_url, exc, connecting=True)) from None
    try:
        with connection:
            coordinator: Final = MigrationCoordinator(connection)
            logger.info("Waiting for the %s (up to %ss)", lock.name, wait_seconds)
            while time.monotonic() < deadline:
                with connection.transaction():
                    if _try_lock(connection, lock.key):
                        logger.info("Acquired the %s", lock.name)

                        yield coordinator
                        coordinator.check_connection()
                        return
                time.sleep(min(random.uniform(0.5, 1.5), max(0.0, deadline - time.monotonic())))
    except psycopg.Error as exc:
        raise RuntimeError(_coordination_failure(database_url, exc, connecting=False)) from None
    raise RuntimeError(
        f"Timed out waiting for {lock.holder} after {wait_seconds}s. "
        f"Check the running migration or increase {MIGRATION_LOCK_TIMEOUT_ENV_VAR}."
    )


@contextmanager
def prisma_migration_lock(database_url: str) -> Generator[None, None, None]:
    """Hold only Prisma's key as an xact lock on a pinned connection with Prisma's session lock disabled.

    This keeps deploy independent of the coordinator key held by request-log index builds."""
    with migration_lock(database_url, PRISMA_LOCK) as coordinator:
        coordinator.release_statement_snapshot()
        yield


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
