"""`litellm --check_migrations_against_existing_data`: run the pending migrations for real,
inside one transaction that is always rolled back, so Postgres itself says whether this
database's rows and schema take them. It holds the same locks the real upgrade will."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from litellm_proxy_extras.migration_data_checks import pending_migrations

if TYPE_CHECKING:
    import psycopg

_CONCURRENTLY: Final = re.compile(
    r"\b((?:CREATE\s+(?:UNIQUE\s+)?INDEX|DROP\s+INDEX|REINDEX\s+\w+)\s+)CONCURRENTLY\s+", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class Applied:
    name: str
    seconds: float


@dataclass(frozen=True, slots=True)
class Rejected:
    name: str
    error: str
    seconds: float


@dataclass(frozen=True, slots=True)
class DryRun:
    applied: tuple[Applied, ...]
    rejected: Rejected | None

    @property
    def ok(self) -> bool:
        return self.rejected is None


def transactional(migration_sql: str) -> str:
    """The migration as it can run inside a transaction: `CONCURRENTLY` index builds become plain ones."""
    return _CONCURRENTLY.sub(r"\1", migration_sql)


def _run(
    connection: psycopg.Connection[tuple[object, ...]],
    done: DryRun,
    migration: tuple[str, str],
    clock: Callable[[], float],
) -> DryRun:
    import psycopg

    if done.rejected is not None:
        return done
    name, migration_sql = migration
    started: Final = clock()
    try:
        connection.execute(transactional(migration_sql))
    except psycopg.Error as exc:
        return DryRun(done.applied, Rejected(name, str(exc).strip(), clock() - started))
    return DryRun(done.applied + (Applied(name, clock() - started),), None)


def dry_run(
    connection: psycopg.Connection[tuple[object, ...]],
    schema: str,
    migrations_dir: Path,
    clock: Callable[[], float] = time.monotonic,
) -> DryRun:
    """Apply every pending migration in order on `connection`, then roll all of it back,
    whatever happened. The first migration Postgres rejects ends the run with its error."""
    from psycopg import sql

    pending: Final = pending_migrations(connection, schema, migrations_dir)
    with connection.transaction(force_rollback=True):
        connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
        result: Final = _fold(connection, pending, clock)
    return result


def _fold(
    connection: psycopg.Connection[tuple[object, ...]],
    pending: tuple[tuple[str, str], ...],
    clock: Callable[[], float],
) -> DryRun:
    from functools import reduce

    return reduce(lambda done, migration: _run(connection, done, migration, clock), pending, DryRun((), None))


def report(result: DryRun) -> str:
    """What the operator reads: the per-migration time the real upgrade will hold its locks,
    or the migration Postgres rejected with Postgres's own words. Nothing was applied either way."""
    timings: Final = "".join(f"\n  {applied.name}: {applied.seconds:.1f} s" for applied in result.applied)
    if result.rejected is None:
        count: Final = len(result.applied)
        if count == 0:
            return "No pending migrations, this database is on this version's schema."
        return f"{count} pending migration(s) applied in a dry run and rolled back, nothing changed:{timings}"
    return (
        f"Migration {result.rejected.name} fails on this database after {result.rejected.seconds:.1f} s, "
        f"nothing was applied.\n{result.rejected.error}"
        + (f"\nApplied before it in the dry run:{timings}" if timings else "")
    )
