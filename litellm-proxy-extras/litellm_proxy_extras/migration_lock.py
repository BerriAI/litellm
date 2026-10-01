import random
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from litellm_proxy_extras._logging import logger
from litellm_proxy_extras.prisma_toolchain import MIGRATION_LOCK_TIMEOUT_ENV_VAR, migration_lock_timeout

MIGRATION_LOCK_KEY: Final = int.from_bytes(b"llm_mig2", "big")

if TYPE_CHECKING:
    import psycopg


def migration_environment(environment: Mapping[str, str]) -> Mapping[str, str]:
    database_url: Final = environment.get("DATABASE_URL")
    direct_url: Final = environment.get("DIRECT_URL")
    if not database_url or not direct_url:
        return environment
    schema: Final = next((value for key, value in parse_qsl(urlsplit(database_url).query) if key == "schema"), "public")
    direct: Final = urlsplit(direct_url)
    parameters: Final = tuple((key, value) for key, value in parse_qsl(direct.query) if key != "schema")
    return {
        **environment,
        "DATABASE_URL": urlunsplit(direct._replace(query=urlencode((*parameters, ("schema", schema))))),
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
