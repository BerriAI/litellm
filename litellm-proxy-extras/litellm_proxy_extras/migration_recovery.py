import hashlib
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from litellm_proxy_extras import prisma_toolchain
from litellm_proxy_extras._logging import logger
from litellm_proxy_extras.migration_lock import MigrationCoordinator

if TYPE_CHECKING:
    import psycopg


@dataclass(frozen=True, slots=True)
class MigrationProgress:
    checksum: str
    applied_steps_count: int
    logs: str
    id: str = ""
    finished: bool = False

    def confirms_completion(self, script: bytes) -> bool:
        return (
            self.applied_steps_count == 1
            and not self.logs.strip()
            and self.checksum == hashlib.sha256(script).hexdigest()
        )


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
    if not unfinished:
        return any(record.checksum == hashlib.sha256(script).hexdigest() for record in records)
    if len(unfinished) != 1 or not unfinished[0].confirms_completion(script):
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


def baseline_current_schema(
    coordinator: MigrationCoordinator,
    schema: str,
    migrations_dir: Path,
    prisma_command: str,
    prisma_env: Mapping[str, str],
) -> None:
    from psycopg import sql

    packaged_dir: Final = Path(__file__).parent
    migrations: Final = migration_files(migrations_dir)
    if (
        not migrations
        or migrations != migration_files(packaged_dir)
        or (migrations_dir / "schema.prisma").read_bytes() != (packaged_dir / "schema.prisma").read_bytes()
    ):
        raise RuntimeError("Cannot automatically baseline an existing database with custom migration history.")

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
