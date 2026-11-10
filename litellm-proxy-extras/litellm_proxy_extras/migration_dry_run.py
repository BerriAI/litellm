"""`litellm --check_migrations_against_existing_data`: run the pending migrations for real, each
inside its own transaction that is always rolled back, so Postgres itself says whether this
database's rows and schema take them. Each migration holds the locks the real upgrade will,
and releases them before the next one runs."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Final

from litellm_proxy_extras.migration_data_checks import pending_migrations

if TYPE_CHECKING:
    import psycopg

_CONCURRENTLY: Final = re.compile(
    r"\b((?:CREATE\s+(?:UNIQUE\s+)?INDEX|DROP\s+INDEX|REINDEX\s+\w+)\s+)CONCURRENTLY\s+", re.IGNORECASE
)
UNDEFINED_SQLSTATES: Final = frozenset({"42703", "42P01", "42704", "42883"})

Migration = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Applied:
    name: str
    seconds: float
    replayed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Rejected:
    name: str
    error: str
    seconds: float


@dataclass(frozen=True, slots=True)
class Baseline:
    """A database with tables but no `_prisma_migrations`: the boot baselines it instead of replaying
    history, so the dry run asks the same question the boot does. `refusal` is None when the boot
    will baseline, otherwise its reason (a custom migration folder, or the schema diff)."""

    refusal: str | None

    @property
    def matches(self) -> bool:
        return self.refusal is None


@dataclass(frozen=True, slots=True)
class DryRun:
    applied: tuple[Applied, ...]
    rejected: Rejected | None
    baseline: Baseline | None = None

    @property
    def ok(self) -> bool:
        return self.rejected is None and (self.baseline is None or self.baseline.matches)


def transactional(migration_sql: str) -> str:
    """The migration as it can run inside a transaction: `CONCURRENTLY` index builds become plain ones."""
    return _CONCURRENTLY.sub(r"\1", migration_sql)


def _attempt(
    connection: psycopg.Connection[tuple[object, ...]],
    schema: str,
    chain: tuple[Migration, ...],
    clock: Callable[[], float],
) -> tuple[float, psycopg.Error | None]:
    import psycopg
    from psycopg import sql

    started: Final = clock()
    try:
        with connection.transaction(force_rollback=True):
            connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            for _, migration_sql in chain:
                connection.execute(transactional(migration_sql))
    except psycopg.Error as exc:
        return clock() - started, exc
    return clock() - started, None


def _run(
    connection: psycopg.Connection[tuple[object, ...]],
    schema: str,
    pending: tuple[Migration, ...],
    position: int,
    done: DryRun,
    clock: Callable[[], float],
) -> DryRun:
    """One migration in its own rolled-back transaction. A migration that fails because something
    an earlier pending migration creates does not exist yet is rerun with those replayed in front."""
    if done.rejected is not None:
        return done
    name: Final = pending[position][0]
    seconds, error = _attempt(connection, schema, (pending[position],), clock)
    if error is None:
        return DryRun(done.applied + (Applied(name, seconds),), None, done.baseline)
    if error.sqlstate not in UNDEFINED_SQLSTATES or position == 0:
        return DryRun(done.applied, Rejected(name, str(error).strip(), seconds), done.baseline)
    earlier: Final = pending[:position]
    chained_seconds, chained_error = _attempt(connection, schema, earlier + (pending[position],), clock)
    if chained_error is None:
        replayed: Final = tuple(earlier_name for earlier_name, _ in earlier)
        return DryRun(done.applied + (Applied(name, chained_seconds, replayed),), None, done.baseline)
    return DryRun(done.applied, Rejected(name, str(chained_error).strip(), chained_seconds), done.baseline)


def _has_ledger(connection: psycopg.Connection[tuple[object, ...]], schema: str) -> bool:
    from psycopg import sql

    row: Final = connection.execute(
        "SELECT to_regclass(%s)", (sql.Identifier(schema, "_prisma_migrations").as_string(connection),)
    ).fetchone()
    return row is not None and row[0] is not None


def _has_litellm_tables(connection: psycopg.Connection[tuple[object, ...]], schema: str) -> bool:
    row: Final = connection.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name LIKE 'LiteLLM\\_%%' LIMIT 1",
        (schema,),
    ).fetchone()
    return row is not None


def dry_run(
    connection: psycopg.Connection[tuple[object, ...]],
    schema: str,
    migrations_dir: Path,
    baseline_check: Callable[[], str | None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> DryRun:
    """Apply every pending migration in order, each in its own transaction that is rolled back
    before the next one starts. The first migration Postgres rejects ends the run with its error.
    `baseline_check` (None when the boot would baseline, otherwise why it refuses) stands in for
    the replay on a database that has tables but no migration history, which the boot baselines."""
    if baseline_check is not None and not _has_ledger(connection, schema) and _has_litellm_tables(connection, schema):
        return DryRun((), None, Baseline(baseline_check()))
    pending: Final = pending_migrations(connection, schema, migrations_dir)
    return reduce(
        lambda done, position: _run(connection, schema, pending, position, done, clock),
        range(len(pending)),
        DryRun((), None),
    )


def _timing(applied: Applied) -> str:
    replay: Final = f" (depends on {', '.join(applied.replayed)}, replayed in front of it)" if applied.replayed else ""
    return f"\n  {applied.name}: {applied.seconds:.1f} s{replay}"


def report(result: DryRun) -> str:
    """What the operator reads: the per-migration time the real upgrade will hold its locks,
    or the migration Postgres rejected with Postgres's own words. Nothing was applied either way."""
    if result.baseline is not None:
        if result.baseline.matches:
            return (
                "No migration history and the schema matches this build, "
                "the proxy baselines it at boot, nothing to dry-run."
            )
        return f"No migration history and the proxy cannot baseline it at boot: {result.baseline.refusal}"
    timings: Final = "".join(_timing(applied) for applied in result.applied)
    if result.rejected is None:
        count: Final = len(result.applied)
        if count == 0:
            return "No pending migrations, this database is on this version's schema."
        return (
            f"{count} pending migration(s) applied one at a time in a dry run, each rolled back before the next "
            f"so its locks are released, nothing changed:{timings}"
        )
    return (
        f"Migration {result.rejected.name} fails on this database after {result.rejected.seconds:.1f} s, "
        f"nothing was applied.\n{result.rejected.error}"
        + (f"\nApplied before it in the dry run:{timings}" if timings else "")
    )
