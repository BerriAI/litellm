import hashlib
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from litellm_proxy_extras import prisma_toolchain
from litellm_proxy_extras._logging import logger
from litellm_proxy_extras._redaction import _redact_credentials
from litellm_proxy_extras.migration_lock import MigrationCoordinator

if TYPE_CHECKING:
    import psycopg


# SHA-256 of earlier shipped files whose successful run applied everything the current file does,
# and whose failed run the current file can replay.
SUPERSEDED_CHECKSUMS: Final = MappingProxyType(
    {
        "20260226120000_add_spend_log_tool_index": frozenset(
            {"019ae3e4e71f35763d6d515e1d463d97e75b8c50e7637c842e4e16ce59009a4b"}
        )
    }
)


@dataclass(frozen=True, slots=True)
class MigrationProgress:
    checksum: str
    applied_steps_count: int
    logs: str
    id: str = ""
    finished: bool = False

    def confirms_completion(self, checksums: frozenset[str]) -> bool:
        return self.applied_steps_count == 1 and not self.logs.strip() and self.checksum in checksums


def _migration_records(
    connection: "psycopg.Connection[tuple[object, ...]]", schema: str, migration: Path
) -> tuple[MigrationProgress, ...]:
    from psycopg import sql
    from psycopg.rows import class_row

    with connection.cursor(row_factory=class_row(MigrationProgress)) as cursor:
        records: Final = cursor.execute(
            sql.SQL(
                "SELECT id, checksum, applied_steps_count, coalesce(logs, '') AS logs, "
                "finished_at IS NOT NULL AS finished FROM {} "
                "WHERE migration_name = %s AND rolled_back_at IS NULL"
            ).format(sql.Identifier(schema, "_prisma_migrations")),
            (migration.parent.name,),
        ).fetchall()
    return tuple(records)


def recover_completed_migration(coordinator: MigrationCoordinator, schema: str, migration: Path) -> bool:
    """Finish a proven successful row without erasing its durable completion evidence.

    The caller commits this checkpoint before running another Prisma command.
    """
    from psycopg import sql

    coordinator.acquire_prisma_lock()
    records: Final = _migration_records(coordinator.connection, schema, migration)
    unfinished: Final = tuple(record for record in records if not record.finished)
    script: Final = migration.read_bytes()
    checksums: Final = frozenset({hashlib.sha256(script).hexdigest()}) | SUPERSEDED_CHECKSUMS.get(
        migration.parent.name, frozenset()
    )
    if not unfinished:
        return any(record.checksum in checksums for record in records)
    if len(unfinished) != 1 or not unfinished[0].confirms_completion(checksums):
        return False
    progress: Final = unfinished[0]
    result: Final = coordinator.connection.execute(
        sql.SQL(
            "UPDATE {} SET finished_at = current_timestamp "
            "WHERE id = %s AND checksum = %s AND applied_steps_count = 1 "
            "AND finished_at IS NULL AND rolled_back_at IS NULL AND coalesce(logs, '') = %s"
        ).format(sql.Identifier(schema, "_prisma_migrations")),
        (progress.id, progress.checksum, progress.logs),
    )
    if result.rowcount != 1:
        raise RuntimeError("Could not complete the confirmed migration history row; retry startup.")
    logger.info("Completed migration %s using its successful SQL step and matching checksum", migration.parent.name)
    return True


def migration_files(directory: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path.parent.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted((directory / "migrations").glob("*/migration.sql"))
    )


def has_packaged_history(migrations_dir: Path) -> bool:
    """Whether `migrations_dir` carries exactly the migration files and schema.prisma this package ships,
    the only history the boot baselines an existing database onto."""
    packaged_dir: Final = Path(__file__).parent
    migrations: Final = migration_files(migrations_dir)
    return (
        bool(migrations)
        and migrations == migration_files(packaged_dir)
        and (migrations_dir / "schema.prisma").read_bytes() == (packaged_dir / "schema.prisma").read_bytes()
    )


def baseline_current_schema(
    coordinator: MigrationCoordinator,
    schema: str,
    migrations_dir: Path,
    prisma_command: str,
    prisma_env: Mapping[str, str],
) -> None:
    from psycopg import sql

    if not has_packaged_history(migrations_dir):
        raise RuntimeError("Cannot automatically baseline an existing database with custom migration history.")
    migrations: Final = migration_files(migrations_dir)

    coordinator.acquire_prisma_lock()
    existing: Final = coordinator.connection.execute(
        "SELECT to_regclass(%s)", (sql.Identifier(schema, "_prisma_migrations").as_string(coordinator.connection),)
    ).fetchone()
    if existing is not None and existing[0] is not None:
        return
    try:
        prisma_toolchain.run_prisma(
            (
                prisma_command,
                "migrate",
                "diff",
                "--from-schema-datasource",
                str(migrations_dir / "schema.prisma"),
                "--to-schema-datamodel",
                str(migrations_dir / "schema.prisma"),
                "--exit-code",
            ),
            timeout=prisma_toolchain.prisma_command_timeout(),
            env=prisma_env,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Cannot automatically baseline this database: its schema has not been verified to match this build. "
            "Establish the existing migration history before retrying. No schema reconciliation was performed. "
            "If using a transaction pooler, configure DIRECT_URL to reach the same database without the pooler. "
            f"Schema verification detail: {exc.stderr}"
        ) from exc

    coordinator.check_connection()
    ledger: Final = sql.Identifier(schema, "_prisma_migrations")
    coordinator.connection.execute(
        sql.SQL(
            "CREATE TABLE {} (id varchar(36) PRIMARY KEY NOT NULL, checksum varchar(64) NOT NULL, "
            "finished_at timestamptz, migration_name varchar(255) NOT NULL, logs text, rolled_back_at timestamptz, "
            "started_at timestamptz NOT NULL DEFAULT now(), applied_steps_count integer NOT NULL DEFAULT 0)"
        ).format(ledger)
    )
    with coordinator.connection.cursor() as cursor:
        cursor.executemany(
            sql.SQL(
                "INSERT INTO {} (id, checksum, migration_name, logs, started_at, finished_at) "
                "VALUES (%s, %s, %s, '', current_timestamp, current_timestamp)"
            ).format(ledger),
            tuple((str(uuid4()), checksum, name) for name, checksum in migrations),
        )
    logger.warning(
        "Legacy migration history was missing. The existing Prisma schema matches this build; "
        "adopted %s packaged migrations as a baseline. No schema changes were applied, and "
        "historical data backfills were not replayed or verified. Continuing startup; "
        "review any feature-specific backfill requirements.",
        len(migrations),
    )


_LINE_COMMENT_RE: Final = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE: Final = re.compile(r"/\*.*?\*/", re.DOTALL)
_NO_OP_STATEMENT_RE: Final = re.compile(r"^\s*SELECT\s+1\s*$", re.IGNORECASE)


def is_inert_migration(script: str) -> bool:
    """Whether a migration file changes nothing: only comments and `SELECT 1`, so
    applying it can neither repeat nor skip a database change."""
    stripped: Final = _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", script))
    return all(not part.strip() or _NO_OP_STATEMENT_RE.match(part) for part in stripped.split(";"))


def roll_back_replayable_failed_migration(coordinator: MigrationCoordinator, schema: str, migration: Path) -> bool:
    """Roll back the failed ledger row of a migration this build can safely replay, so
    `migrate deploy` applies the current file on its next pass. Either the file is now
    inert, or the row recorded an earlier shipped file listed in SUPERSEDED_CHECKSUMS
    whose index the migration job now builds. A failed run of the current file is left
    for the operator. The caller commits this checkpoint before the next Prisma command.
    """
    from psycopg import sql

    inert: Final = is_inert_migration(migration.read_text(encoding="utf-8"))
    superseded: Final = SUPERSEDED_CHECKSUMS.get(migration.parent.name, frozenset())
    if not inert and not superseded:
        return False
    coordinator.acquire_prisma_lock()
    records: Final = _migration_records(coordinator.connection, schema, migration)
    unfinished: Final = tuple(record for record in records if not record.finished)
    if len(unfinished) != 1 or not (inert or unfinished[0].checksum in superseded):
        return False
    result: Final = coordinator.connection.execute(
        sql.SQL(
            "UPDATE {} SET rolled_back_at = current_timestamp "
            "WHERE id = %s AND finished_at IS NULL AND rolled_back_at IS NULL"
        ).format(sql.Identifier(schema, "_prisma_migrations")),
        (unfinished[0].id,),
    )
    if result.rowcount != 1:
        raise RuntimeError("Could not roll back the failed migration history row; rerun the database setup.")
    logger.info(
        "Rolled back the failed history row of %s: this build replays it safely, "
        "its index is built by the migration job",
        migration.parent.name,
    )
    return True


KILLED_SESSION_WAIT_SECONDS: Final = 30.0


def roll_back_killed_migration(
    database_url: str, schema: str, application_name: str, attempt_started: float
) -> tuple[str, ...]:
    """Roll back the history row of a migration that was still running when its `prisma migrate deploy`
    was killed at the deploy time limit, and return its name.

    Postgres keeps running a killed client's statement, so the session Prisma left behind can still
    commit the migration once the lock it waits on is released. The row is only rolled back with
    evidence that the migration did not commit: a session named `application_name` was seen still
    running its statement, every such session was terminated and is gone, and every transaction
    those sessions had open is recorded as aborted. The caller holds Prisma's migration lock and
    `attempt_started` is its `time.monotonic()` from before the deploy, so the unfinished row is the
    killed command's own. Without that evidence nothing is changed.
    """
    try:
        import psycopg
    except ImportError:
        return ()
    from psycopg import sql

    unchanged: Final = "Its migration history row was left unchanged."
    try:
        with psycopg.connect(database_url, connect_timeout=10, autocommit=True, client_encoding="utf8") as connection:
            deadline: Final = time.monotonic() + KILLED_SESSION_WAIT_SECONDS
            sessions: Final = _terminate_sessions(connection, application_name)
            while _terminate_sessions(connection, application_name):
                if time.monotonic() >= deadline:
                    logger.warning(
                        "The database session of the prisma migrate deploy that was stopped at its time limit "
                        "is still running after %ss. %s",
                        KILLED_SESSION_WAIT_SECONDS,
                        unchanged,
                    )
                    return ()
                time.sleep(0.2)
            if not any(state == "active" for state, _ in sessions):
                logger.warning(
                    "The prisma migrate deploy that was stopped at its time limit had no database session still "
                    "running a statement, so its migration may have committed. %s",
                    unchanged,
                )
                return ()
            if any(
                connection.execute(
                    "SELECT txid_status(((txid_snapshot_xmax(txid_current_snapshot()) >> 32) << 32) + %s)", (int(xid),)
                ).fetchone()
                != ("aborted",)
                for _, xid in sessions
                if xid is not None
            ):
                logger.warning(
                    "The transaction of the prisma migrate deploy that was stopped at its time limit is not "
                    "recorded as aborted, so its migration may have committed. %s",
                    unchanged,
                )
                return ()
            rolled_back: Final = connection.execute(
                sql.SQL(
                    "UPDATE {} SET rolled_back_at = current_timestamp "
                    "WHERE finished_at IS NULL AND rolled_back_at IS NULL AND logs IS NULL "
                    "AND applied_steps_count = 0 AND started_at >= current_timestamp - make_interval(secs => %s) "
                    "RETURNING migration_name"
                ).format(sql.Identifier(schema, "_prisma_migrations")),
                (time.monotonic() - attempt_started,),
            ).fetchall()
    except psycopg.Error as exc:
        logger.warning(
            "Could not confirm that the database session of the prisma migrate deploy that was stopped at its "
            "time limit is gone (%s). %s",
            _redact_credentials(str(exc)).strip(),
            unchanged,
        )
        return ()
    names: Final = tuple(str(row[0]) for row in rolled_back)
    for name in names:
        logger.warning(
            "Migration %s was still running when prisma migrate deploy was stopped at its time limit. "
            "Its database session was ended, which undid its changes, and its migration history row was "
            "marked rolled back so it can be applied again.",
            name,
        )
    return names


def _terminate_sessions(
    connection: "psycopg.Connection[tuple[object, ...]]", application_name: str
) -> tuple[tuple[str | None, str | None], ...]:
    """Signal every other session named `application_name` to end, returning each one's (state, open xid)."""
    rows: Final = connection.execute(
        "SELECT state, backend_xid::text, pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = current_database() AND application_name = %s AND pid <> pg_backend_pid()",
        (application_name,),
    ).fetchall()
    return tuple((None if state is None else str(state), None if xid is None else str(xid)) for state, xid, _ in rows)
