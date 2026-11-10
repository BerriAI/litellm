from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

MIGRATION_LOCK_KEY: Final = int.from_bytes(b"llm_mig2", "big")

if TYPE_CHECKING:
    import psycopg


@dataclass(frozen=True, slots=True)
class _LockResult:
    acquired: bool


@contextmanager
def held_migration_lock(connection: "psycopg.Connection[tuple[object, ...]]") -> Generator[bool, None, None]:
    """A session-level, non-blocking hold of the migration coordinator lock on an autocommit
    connection, for DDL that cannot run inside a transaction (`CREATE INDEX CONCURRENTLY`).
    Yields whether the lock was acquired; another migration job's index build or invalid
    index repair holding it yields False. Released on exit."""
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(_LockResult)) as cursor:
        row: Final = cursor.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (MIGRATION_LOCK_KEY,)).fetchone()
    acquired: Final = row is not None and row.acquired
    try:
        yield acquired
    finally:
        if acquired:
            connection.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
