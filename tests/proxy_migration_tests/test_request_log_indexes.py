import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import psycopg
import pytest
from litellm_proxy_extras import request_log_indexes
from litellm_proxy_extras.migration_lock import MIGRATION_LOCK_KEY, migration_lock
from litellm_proxy_extras.migration_recovery import roll_back_failed_inert_migration
from litellm_proxy_extras.request_log_indexes import (
    REQUEST_LOG_INDEXES,
    RequestLogIndex,
    build_index_on_partitioned_table,
    ensure_request_log_indexes,
)
from litellm_proxy_extras.utils import ProxyExtrasDBManager
from psycopg import sql
from psycopg.abc import Params, QueryNoTemplate
from psycopg.rows import class_row

pytestmark = pytest.mark.timeout(900)

requires_db: Final = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)

REPO: Final = Path(__file__).resolve().parents[2]
PACKAGE: Final = REPO / "litellm-proxy-extras" / "litellm_proxy_extras"
PARTITION_SCRIPT: Final = REPO / "db_scripts" / "partition_spend_logs.sql"
API_KEY_INDEX_MIGRATION: Final = "20260823000000_add_spend_logs_api_key_starttime_index"
CALL_ID_INDEX_MIGRATION: Final = "20260831120001_spend_logs_litellm_call_id_index"
API_KEY_INDEX: Final = "LiteLLM_SpendLogs_api_key_startTime_idx"
CALL_ID_INDEX: Final = "LiteLLM_SpendLogs_litellm_call_id_idx"
PARTITIONED_PARENT_ERROR: Final = 'cannot create index on partitioned table "LiteLLM_SpendLogs" concurrently'
ORIGINAL_MIGRATION_SQL: Final = MappingProxyType(
    {
        API_KEY_INDEX_MIGRATION: (
            "-- CreateIndex\n"
            'CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx" '
            'ON "LiteLLM_SpendLogs"("api_key", "startTime");\n'
        ),
        CALL_ID_INDEX_MIGRATION: (
            "-- CreateIndex\n"
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_SpendLogs_litellm_call_id_idx" '
            'ON "LiteLLM_SpendLogs"("litellm_call_id");\n'
        ),
    }
)
CALL_ID_INDEX_DEFINITION: Final = next(index for index in REQUEST_LOG_INDEXES if index.name == CALL_ID_INDEX)
RELEASES: Final = pytest.mark.parametrize(
    "release", (API_KEY_INDEX_MIGRATION, CALL_ID_INDEX_MIGRATION), ids=("v1.102.1", "v1.103.0")
)
PARTITIONS: Final = MappingProxyType(
    {
        "LiteLLM_SpendLogs_p2026_08": ("2026-08-01", "2026-09-01"),
        "LiteLLM_SpendLogs_p2026_09": ("2026-09-01", "2026-10-01"),
    }
)
DEFAULT_PARTITION: Final = "LiteLLM_SpendLogs_pdefault"
ROWS_PER_PARTITION: Final = 200
RESOLVERS: Final = pytest.mark.parametrize("use_v2_resolver", (True, False), ids=("v2", "v1"))


def _base_url() -> str:
    return os.environ["DATABASE_URL"].split("?")[0]


def _migrate_deploy(database_url: str, schema: Path) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(schema)],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_URL": database_url},
    )


def _release_layout(prisma_dir: Path, before: str) -> Path:
    """The shipped migrations older than `before`, with the two index migrations written
    the way the releases that shipped them did: the Prisma layout of a proxy on that release."""
    (prisma_dir / "migrations").mkdir(parents=True)
    shutil.copy(PACKAGE / "schema.prisma", prisma_dir / "schema.prisma")
    for migration in sorted((PACKAGE / "migrations").iterdir()):
        if migration.is_dir() and migration.name < before:
            shutil.copytree(migration, prisma_dir / "migrations" / migration.name)
    for name, original in ORIGINAL_MIGRATION_SQL.items():
        if (prisma_dir / "migrations" / name).is_dir():
            (prisma_dir / "migrations" / name / "migration.sql").write_text(original)
    return prisma_dir / "schema.prisma"


def _deploy_release(database_url: str, prisma_dir: Path, before: str) -> None:
    deployed: Final = _migrate_deploy(database_url, _release_layout(prisma_dir, before))
    assert deployed.returncode == 0, deployed.stderr


def _insert_spend_log(
    conn: "psycopg.Connection[tuple[object, ...]]", request_id: str, day: str, table: str = "LiteLLM_SpendLogs"
) -> None:
    conn.execute(
        sql.SQL(
            'INSERT INTO {} ("request_id", "call_type", "startTime", "endTime", "api_key") VALUES (%s, %s, %s, %s, %s)'
        ).format(sql.Identifier(table)),
        (request_id, "acompletion", day, day, f"key-{request_id[-1]}"),
    )


def _partition_spend_logs(database_url: str) -> None:
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(PARTITION_SCRIPT.read_bytes())
        for partition, (start, stop) in PARTITIONS.items():
            conn.execute(
                sql.SQL('CREATE TABLE {} PARTITION OF "LiteLLM_SpendLogs" FOR VALUES FROM ({}) TO ({})').format(
                    sql.Identifier(partition), sql.Literal(start), sql.Literal(stop)
                )
            )
            for row in range(ROWS_PER_PARTITION):
                _insert_spend_log(conn, f"{partition}-{row}", start)
        for row in range(ROWS_PER_PARTITION):
            _insert_spend_log(conn, f"default-{row}", "2020-01-01")


@pytest.fixture
def release() -> str:
    """The first migration a database has not applied yet; the v1.103.0 shape unless a test parametrizes it."""
    return CALL_ID_INDEX_MIGRATION


@pytest.fixture
def scratch_database(release: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[str]:
    """A deployment stopped before `release`, with DATABASE_URL pointed at it so
    ProxyExtrasDBManager upgrades it like a booting proxy."""
    admin_url: Final = _base_url()
    name: Final = f"spend_logs_index_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    database_url: Final = f"{admin_url.rsplit('/', 1)[0]}/{name}"
    try:
        _deploy_release(database_url, tmp_path / "prisma", release)
        monkeypatch.delenv("DIRECT_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", database_url)
        yield database_url
    finally:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def partitioned_database(scratch_database: str) -> str:
    _partition_spend_logs(scratch_database)
    return scratch_database


def _fail_the_call_id_migration_like_the_shipped_release(database_url: str, tmp_path: Path) -> None:
    """Boot the original v1.103.0 layout once: its CONCURRENTLY statement fails on the
    partitioned parent and leaves the call_id ledger row unfinished."""
    failed: Final = _migrate_deploy(database_url, _release_layout(tmp_path / "v1.103.0", "99999999999999"))
    assert failed.returncode != 0 and PARTITIONED_PARENT_ERROR in failed.stderr, failed.stderr
    assert _ledger(database_url)[CALL_ID_INDEX_MIGRATION] == (False, False)


@dataclass(frozen=True, slots=True)
class _IndexRow:
    name: str
    valid: bool


@dataclass(frozen=True, slots=True)
class _AttachedRow:
    table: str
    index: str


@dataclass(frozen=True, slots=True)
class _LedgerRow:
    name: str
    finished: bool
    rolled_back: bool


@dataclass(frozen=True, slots=True)
class _OidRow:
    name: str
    oid: int


def _index_validity(database_url: str, suffix: str) -> Mapping[str, bool]:
    """index name -> indisvalid for every index ending in `suffix` on the SpendLogs parent or one of its partitions."""
    with psycopg.connect(database_url) as conn, conn.cursor(row_factory=class_row(_IndexRow)) as cursor:
        rows: Final = cursor.execute(
            "SELECT c.relname AS name, i.indisvalid AS valid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname LIKE %s AND (i.indrelid = to_regclass('\"LiteLLM_SpendLogs\"') OR i.indrelid IN "
            "(SELECT inhrelid FROM pg_inherits WHERE inhparent = to_regclass('\"LiteLLM_SpendLogs\"'))) "
            "ORDER BY c.relname",
            (f"%{suffix}",),
        ).fetchall()
    return MappingProxyType({row.name: row.valid for row in rows})


def _attached_children(database_url: str, parent_index: str) -> frozenset[tuple[str, str]]:
    """(partition, child index) pairs attached under the parent index."""
    with psycopg.connect(database_url) as conn, conn.cursor(row_factory=class_row(_AttachedRow)) as cursor:
        rows: Final = cursor.execute(
            'SELECT t.relname AS "table", c.relname AS index FROM pg_inherits i '
            "JOIN pg_class c ON c.oid = i.inhrelid JOIN pg_index x ON x.indexrelid = c.oid "
            "JOIN pg_class t ON t.oid = x.indrelid "
            "WHERE i.inhparent = to_regclass(%s)",
            (f'"{parent_index}"',),
        ).fetchall()
    return frozenset((row.table, row.index) for row in rows)


@dataclass(frozen=True, slots=True)
class _TableRow:
    name: str


def _indexed_table(database_url: str, index: str) -> "str | None":
    with psycopg.connect(database_url) as conn, conn.cursor(row_factory=class_row(_TableRow)) as cursor:
        row: Final = cursor.execute(
            "SELECT t.relname AS name FROM pg_index x JOIN pg_class t ON t.oid = x.indrelid "
            "WHERE x.indexrelid = to_regclass(%s)",
            (f'"{index}"',),
        ).fetchone()
    return None if row is None else row.name


def _ledger(database_url: str) -> Mapping[str, tuple[bool, bool]]:
    """migration name -> (finished, rolled back) for the newest ledger row of each migration."""
    with psycopg.connect(database_url) as conn, conn.cursor(row_factory=class_row(_LedgerRow)) as cursor:
        rows: Final = cursor.execute(
            "SELECT DISTINCT ON (migration_name) migration_name AS name, finished_at IS NOT NULL AS finished, "
            "rolled_back_at IS NOT NULL AS rolled_back FROM _prisma_migrations ORDER BY migration_name, started_at DESC"
        ).fetchall()
    return MappingProxyType({row.name: (row.finished, row.rolled_back) for row in rows})


def _index_oids(database_url: str) -> Mapping[str, int]:
    """index name -> oid for every index on the SpendLogs parent or one of its partitions; a rebuild changes the oid."""
    with psycopg.connect(database_url) as conn, conn.cursor(row_factory=class_row(_OidRow)) as cursor:
        rows: Final = cursor.execute(
            "SELECT c.relname AS name, c.oid::int AS oid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indrelid = to_regclass('\"LiteLLM_SpendLogs\"') OR i.indrelid IN "
            "(SELECT inhrelid FROM pg_inherits WHERE inhparent = to_regclass('\"LiteLLM_SpendLogs\"'))"
        ).fetchall()
    return MappingProxyType({row.name: row.oid for row in rows})


def _migration_job(use_v2_resolver: bool) -> bool:
    return ProxyExtrasDBManager.run_migration_job(use_migrate=True, use_v2_resolver=use_v2_resolver)


def _assert_no_pending_migrations(database_url: str) -> None:
    status: Final = _migrate_deploy(database_url, PACKAGE / "schema.prisma")
    assert status.returncode == 0 and "No pending migrations" in status.stdout, status.stdout + status.stderr


def _assert_every_ledger_row_is_finished(database_url: str) -> Mapping[str, tuple[bool, bool]]:
    ledger: Final = _ledger(database_url)
    assert ledger[API_KEY_INDEX_MIGRATION] == (True, False) and ledger[CALL_ID_INDEX_MIGRATION] == (True, False)
    assert all(finished and not rolled_back for finished, rolled_back in ledger.values()), ledger
    return ledger


def _expected_children(partitions: tuple[str, ...], suffix: str) -> frozenset[tuple[str, str]]:
    return frozenset((partition, f"{partition}_{suffix}") for partition in partitions)


def _assert_index_covers_every_partition(database_url: str, parent_index: str, suffix: str) -> None:
    partitions: Final = (*PARTITIONS, DEFAULT_PARTITION)
    assert _index_validity(database_url, suffix) == {parent_index: True} | {f"{p}_{suffix}": True for p in partitions}
    assert _attached_children(database_url, parent_index) == _expected_children(partitions, suffix)


@requires_db
@RESOLVERS
@RELEASES
def test_a_partitioned_spend_logs_upgrade_builds_both_indexes_per_partition_and_a_rerun_is_idempotent(
    partitioned_database: str, use_v2_resolver: bool
) -> None:
    assert _migration_job(use_v2_resolver) is True

    _assert_index_covers_every_partition(partitioned_database, API_KEY_INDEX, "api_key_startTime_idx")
    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")
    ledger: Final = _assert_every_ledger_row_is_finished(partitioned_database)
    _assert_no_pending_migrations(partitioned_database)
    oids: Final = _index_oids(partitioned_database)

    with psycopg.connect(partitioned_database, autocommit=True) as conn:
        conn.execute(
            'CREATE TABLE "LiteLLM_SpendLogs_p2026_10" PARTITION OF "LiteLLM_SpendLogs" '
            "FOR VALUES FROM ('2026-10-01') TO ('2026-11-01')"
        )
    inherited: Final = frozenset(
        ("LiteLLM_SpendLogs_p2026_10", f"LiteLLM_SpendLogs_p2026_10_{suffix}")
        for suffix in ("api_key_startTime_idx", "litellm_call_id_idx")
    )
    attached: Final = _attached_children(partitioned_database, API_KEY_INDEX) | _attached_children(
        partitioned_database, CALL_ID_INDEX
    )
    assert inherited <= attached, attached

    assert _migration_job(use_v2_resolver) is True
    assert _ledger(partitioned_database) == ledger
    assert {name: oid for name, oid in _index_oids(partitioned_database).items() if name in oids} == oids


@requires_db
@RESOLVERS
@RELEASES
def test_a_plain_spend_logs_upgrade_builds_both_indexes_and_a_second_job_run_rebuilds_nothing(
    scratch_database: str, use_v2_resolver: bool
) -> None:
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        for row in range(ROWS_PER_PARTITION):
            _insert_spend_log(conn, f"flat-{row}", "2026-09-01")

    assert _migration_job(use_v2_resolver) is True

    assert _index_validity(scratch_database, "api_key_startTime_idx") == {API_KEY_INDEX: True}
    assert _index_validity(scratch_database, "litellm_call_id_idx") == {CALL_ID_INDEX: True}
    _assert_every_ledger_row_is_finished(scratch_database)
    _assert_no_pending_migrations(scratch_database)
    oids: Final = _index_oids(scratch_database)

    assert _migration_job(use_v2_resolver) is True
    assert _index_oids(scratch_database) == oids


@requires_db
@RESOLVERS
def test_a_database_that_applied_the_original_migration_files_sees_no_pending_migrations_and_no_rebuild(
    scratch_database: str, use_v2_resolver: bool, tmp_path: Path
) -> None:
    """A plain table upgraded on v1.103.0 applied both original files. The inert files in
    this build must neither re-run nor fail those rows, and the migration job must keep the
    indexes the migrations built."""
    deployed: Final = _migrate_deploy(scratch_database, _release_layout(tmp_path / "v1.103.0", "99999999999999"))
    assert deployed.returncode == 0, deployed.stderr
    before: Final = _ledger(scratch_database)
    assert before[API_KEY_INDEX_MIGRATION] == (True, False) and before[CALL_ID_INDEX_MIGRATION] == (True, False)
    oids: Final = _index_oids(scratch_database)
    assert {API_KEY_INDEX, CALL_ID_INDEX} <= set(oids)

    _assert_no_pending_migrations(scratch_database)
    assert _migration_job(use_v2_resolver) is True

    assert _ledger(scratch_database) == before
    assert _index_oids(scratch_database) == oids


@requires_db
@RESOLVERS
def test_a_failed_call_id_ledger_row_from_a_v1_103_boot_is_rolled_back_and_the_inert_file_applied(
    partitioned_database: str, use_v2_resolver: bool, tmp_path: Path
) -> None:
    _fail_the_call_id_migration_like_the_shipped_release(partitioned_database, tmp_path)

    assert _migration_job(use_v2_resolver) is True

    _assert_every_ledger_row_is_finished(partitioned_database)
    _assert_index_covers_every_partition(partitioned_database, API_KEY_INDEX, "api_key_startTime_idx")
    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")
    _assert_no_pending_migrations(partitioned_database)
    with psycopg.connect(partitioned_database) as conn:
        rows: Final = conn.execute(
            "SELECT finished_at IS NOT NULL, rolled_back_at IS NOT NULL FROM _prisma_migrations "
            "WHERE migration_name = %s ORDER BY started_at",
            (CALL_ID_INDEX_MIGRATION,),
        ).fetchall()
    assert rows == [(False, True), (True, False)], rows


@requires_db
def test_a_failed_row_whose_migration_still_runs_sql_in_this_build_is_left_for_the_operator(
    partitioned_database: str, tmp_path: Path
) -> None:
    _fail_the_call_id_migration_like_the_shipped_release(partitioned_database, tmp_path)
    still_building: Final = tmp_path / "edited" / CALL_ID_INDEX_MIGRATION / "migration.sql"
    still_building.parent.mkdir(parents=True)
    still_building.write_text(ORIGINAL_MIGRATION_SQL[CALL_ID_INDEX_MIGRATION])

    with migration_lock(partitioned_database) as coordinator:
        assert roll_back_failed_inert_migration(coordinator, "public", still_building) is False

    assert _ledger(partitioned_database)[CALL_ID_INDEX_MIGRATION] == (False, False)


@requires_db
def test_a_migration_without_a_failed_row_is_not_touched(partitioned_database: str) -> None:
    inert: Final = PACKAGE / "migrations" / CALL_ID_INDEX_MIGRATION / "migration.sql"
    before: Final = _ledger(partitioned_database)

    with migration_lock(partitioned_database) as coordinator:
        assert roll_back_failed_inert_migration(coordinator, "public", inert) is False

    assert _ledger(partitioned_database) == before


def _pin_a_snapshot_on(database_url: str, table: str) -> "psycopg.Connection[tuple[object, ...]]":
    pin: Final = psycopg.connect(database_url)
    pin.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    pin.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
    return pin


def _leave_an_invalid_index(database_url: str, name: str, table: str, column: str) -> None:
    with _pin_a_snapshot_on(database_url, table):
        with psycopg.connect(database_url, autocommit=True) as builder:
            builder.execute("SET statement_timeout = '1s'")
            with pytest.raises(psycopg.errors.QueryCanceled):
                builder.execute(
                    sql.SQL("CREATE INDEX CONCURRENTLY {} ON {} ({})").format(
                        sql.Identifier(name), sql.Identifier(table), sql.Identifier(column)
                    )
                )


@requires_db
def test_an_invalid_index_of_the_managed_name_on_a_plain_table_is_rebuilt(scratch_database: str) -> None:
    _leave_an_invalid_index(scratch_database, CALL_ID_INDEX, "LiteLLM_SpendLogs", "litellm_call_id")
    assert _index_validity(scratch_database, "litellm_call_id_idx") == {CALL_ID_INDEX: False}

    assert ensure_request_log_indexes(scratch_database, "public") is True

    assert _index_validity(scratch_database, "litellm_call_id_idx") == {CALL_ID_INDEX: True}


def _rebuild_as_another_replica(database_url: str, name: str, table: str, column: str) -> int:
    """Drop and rebuild the index from a second connection, as a replica that won the
    race would, and return the oid of the index it built."""
    with psycopg.connect(database_url, autocommit=True) as other_replica:
        other_replica.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(name)))
        other_replica.execute(
            sql.SQL("CREATE INDEX {} ON {} ({})").format(
                sql.Identifier(name), sql.Identifier(table), sql.Identifier(column)
            )
        )
    return _index_oids(database_url)[name]


def _connecting_with_another_replica_acting_first(
    statement: str, other_replica: Callable[[QueryNoTemplate], None]
) -> Callable[[str], "psycopg.Connection[tuple[object, ...]]"]:
    """A connect function whose cursors let `other_replica` act, once, right before the
    first statement containing `statement` runs: the interleaving two replicas booting
    together can produce, made deterministic."""
    raced: Final = threading.Event()

    class _RacedCursor(psycopg.Cursor[tuple[object, ...]]):
        def execute(  # pyright: ignore[reportIncompatibleMethodOverride]  # the builder never runs a Template query
            self,
            query: QueryNoTemplate,
            params: "Params | None" = None,
            *,
            prepare: "bool | None" = None,
            binary: "bool | None" = None,
        ) -> "_RacedCursor":
            text: Final = query.as_string(self.connection) if isinstance(query, sql.Composable) else query
            if isinstance(text, str) and statement in text and not raced.is_set():
                raced.set()
                other_replica(query)
            return super().execute(query, params, prepare=prepare, binary=binary)

    def connect(database_url: str) -> "psycopg.Connection[tuple[object, ...]]":
        return psycopg.connect(database_url, autocommit=True, cursor_factory=_RacedCursor)

    return connect


@requires_db
def test_an_index_another_replica_made_valid_before_the_lock_was_taken_is_kept(scratch_database: str) -> None:
    """Two replicas boot against the same invalid index. The one that takes the lock
    second must read the catalog again under it, or it drops the valid index the first
    one just finished and starts the whole build over."""
    _leave_an_invalid_index(scratch_database, CALL_ID_INDEX, "LiteLLM_SpendLogs", "litellm_call_id")
    theirs: Final[queue.SimpleQueue[int]] = queue.SimpleQueue()
    connect: Final = _connecting_with_another_replica_acting_first(
        "pg_try_advisory_lock",
        lambda _: theirs.put(
            _rebuild_as_another_replica(scratch_database, CALL_ID_INDEX, "LiteLLM_SpendLogs", "litellm_call_id")
        ),
    )

    assert ensure_request_log_indexes(scratch_database, "public", (CALL_ID_INDEX_DEFINITION,), connect) is True

    assert _index_oids(scratch_database)[CALL_ID_INDEX] == theirs.get_nowait()
    assert _index_validity(scratch_database, "litellm_call_id_idx") == {CALL_ID_INDEX: True}


@requires_db
def test_a_child_index_another_replica_attached_first_is_not_attached_twice(partitioned_database: str) -> None:
    """A replica that reaches the attach step after another one attached the same child
    relies on ATTACH PARTITION being a no-op for an index already under that parent
    (PostgreSQL 14 ALTER INDEX, ATExecAttachPartitionIdx, checked 2026-10-01); this test
    is where that would surface if a future version or a code change made it an error."""

    def attach_as_another_replica(statement: QueryNoTemplate) -> None:
        with psycopg.connect(partitioned_database, autocommit=True) as other_replica:
            other_replica.execute(statement)

    connect: Final = _connecting_with_another_replica_acting_first("ATTACH PARTITION", attach_as_another_replica)

    assert ensure_request_log_indexes(partitioned_database, "public", (CALL_ID_INDEX_DEFINITION,), connect) is True

    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")


@requires_db
def test_an_invalid_child_index_left_by_an_interrupted_build_is_rebuilt_and_attached(
    partitioned_database: str,
) -> None:
    partition: Final = "LiteLLM_SpendLogs_p2026_08"
    child: Final = f"{partition}_litellm_call_id_idx"
    _leave_an_invalid_index(partitioned_database, child, partition, "litellm_call_id")
    assert _index_validity(partitioned_database, "litellm_call_id_idx") == {child: False}

    assert ensure_request_log_indexes(partitioned_database, "public") is True

    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")


@requires_db
def test_an_index_of_that_name_on_another_table_is_left_alone_and_reported(partitioned_database: str) -> None:
    with psycopg.connect(partitioned_database, autocommit=True) as conn:
        conn.execute(f'CREATE INDEX "{CALL_ID_INDEX}" ON "LiteLLM_ErrorLogs" ("request_id")')

    assert ensure_request_log_indexes(partitioned_database, "public") is False

    assert _index_validity(partitioned_database, "litellm_call_id_idx") == {}
    assert _indexed_table(partitioned_database, CALL_ID_INDEX) == "LiteLLM_ErrorLogs"
    _assert_index_covers_every_partition(partitioned_database, API_KEY_INDEX, "api_key_startTime_idx")


@requires_db
def test_an_invalid_index_of_a_child_name_on_another_table_is_not_dropped(partitioned_database: str) -> None:
    child: Final = "LiteLLM_SpendLogs_p2026_08_litellm_call_id_idx"
    _leave_an_invalid_index(partitioned_database, child, "LiteLLM_ErrorLogs", "request_id")

    assert ensure_request_log_indexes(partitioned_database, "public") is False

    assert _indexed_table(partitioned_database, child) == "LiteLLM_ErrorLogs"
    assert _attached_children(partitioned_database, CALL_ID_INDEX) == frozenset()


@requires_db
def test_a_process_holding_the_migration_lock_makes_the_build_wait_for_the_next_job_run(scratch_database: str) -> None:
    with psycopg.connect(scratch_database, autocommit=True) as other_replica:
        other_replica.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
        assert ensure_request_log_indexes(scratch_database, "public") is False
        assert _index_validity(scratch_database, "litellm_call_id_idx") == {}

    assert ensure_request_log_indexes(scratch_database, "public") is True
    assert _index_validity(scratch_database, "litellm_call_id_idx") == {CALL_ID_INDEX: True}


@requires_db
@RESOLVERS
def test_a_migration_job_that_could_not_build_the_indexes_reports_failure_and_succeeds_when_rerun(
    scratch_database: str, use_v2_resolver: bool
) -> None:
    """The migration job waits for the build and exits by run_migration_job's result; a job
    that exits 0 with the indexes missing would leave the table unindexed until the next
    deploy or until a serving proxy's background build gets to them."""
    with psycopg.connect(scratch_database, autocommit=True) as other_replica:
        other_replica.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
        assert _migration_job(use_v2_resolver) is False
        _assert_every_ledger_row_is_finished(scratch_database)
        assert _index_validity(scratch_database, "litellm_call_id_idx") == {}

    assert _migration_job(use_v2_resolver) is True
    assert _index_validity(scratch_database, "litellm_call_id_idx") == {CALL_ID_INDEX: True}
    assert _index_validity(scratch_database, "api_key_startTime_idx") == {API_KEY_INDEX: True}


@requires_db
@RESOLVERS
def test_the_serving_proxy_setup_applies_the_inert_migrations_and_builds_no_index(
    partitioned_database: str, use_v2_resolver: bool
) -> None:
    """setup_database alone applies the inert files and builds nothing, so a serving proxy's
    readiness is never held up by an index build; the build it starts afterwards, or the
    migration job, is what puts the indexes in place."""
    api_key_index_before: Final = _index_validity(partitioned_database, "api_key_startTime_idx")
    assert ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=use_v2_resolver) is True

    _assert_every_ledger_row_is_finished(partitioned_database)
    _assert_no_pending_migrations(partitioned_database)
    assert _index_validity(partitioned_database, "litellm_call_id_idx") == {}
    assert _index_validity(partitioned_database, "api_key_startTime_idx") == api_key_index_before

    assert _migration_job(use_v2_resolver) is True
    _assert_index_covers_every_partition(partitioned_database, API_KEY_INDEX, "api_key_startTime_idx")
    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")


@requires_db
def test_a_role_that_may_not_create_indexes_is_logged_and_left_for_the_next_job_run(
    scratch_database: str, caplog: pytest.LogCaptureFixture
) -> None:
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        conn.execute("CREATE ROLE spend_logs_reader LOGIN PASSWORD 'reader'")
        conn.execute("GRANT USAGE ON SCHEMA public TO spend_logs_reader")
        conn.execute('GRANT SELECT ON "LiteLLM_SpendLogs" TO spend_logs_reader')
    reader_url: Final = scratch_database.replace("postgres:postgres@", "spend_logs_reader:reader@", 1)
    try:
        with caplog.at_level("WARNING", logger="litellm_proxy_extras"):
            assert ensure_request_log_indexes(reader_url, "public") is False
    finally:
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute("DROP OWNED BY spend_logs_reader")
            conn.execute("DROP ROLE spend_logs_reader")
    assert "leaving them for the next index build" in caplog.text
    assert _index_validity(scratch_database, "litellm_call_id_idx") == {}


@requires_db
def test_inserts_keep_flowing_while_the_partition_indexes_build(partitioned_database: str) -> None:
    """With a write open on one partition, the parent index goes on ONLY the parent and
    the CONCURRENTLY child build waits for that write without blocking new INSERTs. A
    plain CREATE INDEX on the parent would wait for the same write while holding SHARE
    on the parent, queueing every new INSERT behind it."""
    outcome: Final[list[bool]] = []  # mutable-ok: the builder thread hands its result back through it
    with psycopg.connect(partitioned_database) as writer:
        _insert_spend_log(writer, "LiteLLM_SpendLogs_p2026_08-open", "2026-08-15", table="LiteLLM_SpendLogs_p2026_08")
        builder_thread: Final = threading.Thread(
            target=lambda: outcome.append(_build_in_its_own_session(partitioned_database, CALL_ID_INDEX_DEFINITION))
        )
        builder_thread.start()
        try:
            _wait_until_the_build_is_waiting(partitioned_database)
            with psycopg.connect(partitioned_database, autocommit=True) as late_writer:
                late_writer.execute("SET lock_timeout = '1s'")
                _insert_spend_log(late_writer, "LiteLLM_SpendLogs_p2026_08-late", "2026-08-16")
        finally:
            writer.commit()
            builder_thread.join()
    assert outcome == [True]
    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")


def _insert_for(database_url: str, seconds: float) -> None:
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("SET lock_timeout = '1s'")
        deadline: Final = time.monotonic() + seconds
        while time.monotonic() < deadline:
            _insert_spend_log(conn, f"lock-test-{uuid.uuid4().hex}", "2026-08-16")
            time.sleep(0.05)


def _wait_for_blocked_ddl(database_url: str, query_pattern: str) -> bool:
    with psycopg.connect(database_url, autocommit=True) as conn:
        deadline: Final = time.monotonic() + 10
        while time.monotonic() < deadline:
            if conn.execute(
                "SELECT 1 FROM pg_stat_activity WHERE wait_event_type = 'Lock' AND query ILIKE %s",
                (query_pattern,),
            ).fetchone():
                return True
            time.sleep(0.01)
    return False


@requires_db
def test_inserts_are_never_held_back_while_the_parent_index_waits_for_an_open_write(
    partitioned_database: str,
) -> None:
    outcome: Final[list[bool]] = []  # mutable-ok: the builder thread hands its result back through it
    with psycopg.connect(partitioned_database) as writer:
        _insert_spend_log(writer, "parent-index-lock-owner", "2026-08-15")
        builder_thread: Final = threading.Thread(
            target=lambda: outcome.append(_build_in_its_own_session(partitioned_database, CALL_ID_INDEX_DEFINITION))
        )
        builder_thread.start()
        try:
            assert _wait_for_blocked_ddl(partitioned_database, "%CREATE INDEX%ON ONLY%")
            _insert_for(partitioned_database, 3)
        finally:
            try:
                writer.commit()
            finally:
                builder_thread.join()
    assert outcome == [True]
    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")


@requires_db
def test_inserts_are_never_held_back_while_attach_partition_waits_for_a_reader_of_the_child_index(
    partitioned_database: str,
) -> None:
    partition: Final = "LiteLLM_SpendLogs_p2026_08"
    child_index: Final = CALL_ID_INDEX_DEFINITION.partition_index_name(partition)
    assert child_index == "LiteLLM_SpendLogs_p2026_08_litellm_call_id_idx"
    with psycopg.connect(partitioned_database, autocommit=True) as conn:
        conn.execute(
            'CREATE INDEX "LiteLLM_SpendLogs_litellm_call_id_idx" ON ONLY "LiteLLM_SpendLogs" ("litellm_call_id")'
        )
        conn.execute(
            sql.SQL('CREATE INDEX {} ON {} ("litellm_call_id")').format(
                sql.Identifier(CALL_ID_INDEX_DEFINITION.partition_index_name(partition)), sql.Identifier(partition)
            )
        )

    outcome: Final[list[bool]] = []  # mutable-ok: the builder thread hands its result back through it
    with psycopg.connect(partitioned_database) as reader:
        reader.execute("SET enable_seqscan = off")
        reader.execute(
            sql.SQL('SELECT count(*) FROM {} WHERE "litellm_call_id" IS NULL').format(sql.Identifier(partition))
        ).fetchone()
        reader_pid: Final = reader.execute("SELECT pg_backend_pid()").fetchone()[0]
        with psycopg.connect(partitioned_database, autocommit=True) as inspector:
            child_lock: Final = inspector.execute(
                "SELECT 1 FROM pg_locks WHERE pid = %s AND relation = to_regclass(%s) "
                "AND mode = 'AccessShareLock' AND granted",
                (reader_pid, f'"{child_index}"'),
            ).fetchone()
        assert child_lock is not None
        builder_thread: Final = threading.Thread(
            target=lambda: outcome.append(_build_in_its_own_session(partitioned_database, CALL_ID_INDEX_DEFINITION))
        )
        builder_thread.start()
        try:
            assert _wait_for_blocked_ddl(partitioned_database, "%ATTACH PARTITION%")
            _insert_for(partitioned_database, 3)
        finally:
            try:
                reader.commit()
            finally:
                builder_thread.join()
    assert outcome == [True]
    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")


@requires_db
def test_a_parent_index_that_never_gets_its_lock_is_left_for_the_next_index_build(
    partitioned_database: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(request_log_indexes, "_DDL_LOCK_ATTEMPTS", 2)
    with psycopg.connect(partitioned_database) as writer:
        _insert_spend_log(writer, "parent-index-lock-owner", "2026-08-15")
        with caplog.at_level("WARNING", logger="litellm_proxy_extras"):
            assert _build_in_its_own_session(partitioned_database, CALL_ID_INDEX_DEFINITION) is False
        assert "leaving it for the next index build" in caplog.text
        assert _indexed_table(partitioned_database, CALL_ID_INDEX) is None
        writer.commit()
        assert _build_in_its_own_session(partitioned_database, CALL_ID_INDEX_DEFINITION) is True
    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")


@requires_db
def test_an_attach_that_never_gets_its_lock_is_left_for_the_next_index_build(
    partitioned_database: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    partition: Final = "LiteLLM_SpendLogs_p2026_08"
    child_index: Final = CALL_ID_INDEX_DEFINITION.partition_index_name(partition)
    assert child_index == "LiteLLM_SpendLogs_p2026_08_litellm_call_id_idx"
    with psycopg.connect(partitioned_database, autocommit=True) as conn:
        conn.execute(
            'CREATE INDEX "LiteLLM_SpendLogs_litellm_call_id_idx" ON ONLY "LiteLLM_SpendLogs" ("litellm_call_id")'
        )
        conn.execute(
            sql.SQL('CREATE INDEX {} ON {} ("litellm_call_id")').format(
                sql.Identifier(child_index), sql.Identifier(partition)
            )
        )

    monkeypatch.setattr(request_log_indexes, "_DDL_LOCK_ATTEMPTS", 2)
    with psycopg.connect(partitioned_database) as reader:
        reader.execute("SET enable_seqscan = off")
        reader.execute(
            sql.SQL('SELECT count(*) FROM {} WHERE "litellm_call_id" IS NULL').format(sql.Identifier(partition))
        ).fetchone()
        reader_pid: Final = reader.execute("SELECT pg_backend_pid()").fetchone()[0]
        with psycopg.connect(partitioned_database, autocommit=True) as inspector:
            child_lock: Final = inspector.execute(
                "SELECT 1 FROM pg_locks WHERE pid = %s AND relation = to_regclass(%s) "
                "AND mode = 'AccessShareLock' AND granted",
                (reader_pid, f'"{child_index}"'),
            ).fetchone()
        assert child_lock is not None
        with caplog.at_level("WARNING", logger="litellm_proxy_extras"):
            assert _build_in_its_own_session(partitioned_database, CALL_ID_INDEX_DEFINITION) is False
        assert "Could not get the lock for attaching" in caplog.text
        reader.commit()

    assert _build_in_its_own_session(partitioned_database, CALL_ID_INDEX_DEFINITION) is True
    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")


def _build_in_its_own_session(database_url: str, index: RequestLogIndex) -> bool:
    with psycopg.connect(database_url, autocommit=True) as builder:
        return build_index_on_partitioned_table(builder, "public", index)


def _wait_until_the_build_is_waiting(database_url: str) -> None:
    deadline: Final = time.monotonic() + 30
    with psycopg.connect(database_url, autocommit=True) as conn:
        while time.monotonic() < deadline:
            waiting = conn.execute(
                "SELECT 1 FROM pg_stat_activity WHERE query LIKE 'CREATE INDEX%' AND wait_event_type IS NOT NULL"
            ).fetchone()
            if waiting is not None:
                return
            time.sleep(0.05)
    pytest.fail("the partition index build never started waiting on the open write")


def _create_index(database_url: str, name: str, table: str, columns: str) -> int:
    """Create a plain index by hand, the way an operator's workaround would, and return its oid."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("CREATE INDEX {} ON {} {}").format(sql.Identifier(name), sql.Identifier(table), sql.SQL(columns))
        )
    return _index_oids(database_url)[name]


@requires_db
def test_a_valid_index_of_the_same_definition_under_another_name_is_renamed_instead_of_rebuilt(
    scratch_database: str,
) -> None:
    hand_built: Final = _create_index(scratch_database, "call_id_by_hand", "LiteLLM_SpendLogs", '("litellm_call_id")')

    assert ensure_request_log_indexes(scratch_database, "public") is True

    oids: Final = _index_oids(scratch_database)
    assert "call_id_by_hand" not in oids and oids[CALL_ID_INDEX] == hand_built
    assert _index_validity(scratch_database, "litellm_call_id_idx") == {CALL_ID_INDEX: True}


@requires_db
def test_a_hand_built_child_index_under_another_name_is_renamed_and_attached(partitioned_database: str) -> None:
    partition: Final = "LiteLLM_SpendLogs_p2026_08"
    hand_built: Final = _create_index(
        partitioned_database, "p2026_08_call_id_by_hand", partition, '("litellm_call_id")'
    )

    assert ensure_request_log_indexes(partitioned_database, "public") is True

    _assert_index_covers_every_partition(partitioned_database, CALL_ID_INDEX, "litellm_call_id_idx")
    oids: Final = _index_oids(partitioned_database)
    assert "p2026_08_call_id_by_hand" not in oids and oids[f"{partition}_litellm_call_id_idx"] == hand_built


@requires_db
def test_an_index_with_another_definition_is_not_taken_for_the_managed_one(scratch_database: str) -> None:
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(API_KEY_INDEX)))
    others: Final = {
        "time_then_key": _create_index(
            scratch_database, "time_then_key", "LiteLLM_SpendLogs", '("startTime", "api_key")'
        ),
        "call_id_desc": _create_index(
            scratch_database, "call_id_desc", "LiteLLM_SpendLogs", '("litellm_call_id" DESC)'
        ),
        "call_id_then_key": _create_index(
            scratch_database, "call_id_then_key", "LiteLLM_SpendLogs", '("litellm_call_id", "api_key")'
        ),
        "call_id_pattern": _create_index(
            scratch_database, "call_id_pattern", "LiteLLM_SpendLogs", '("litellm_call_id" text_pattern_ops)'
        ),
    }

    assert ensure_request_log_indexes(scratch_database, "public") is True

    oids: Final = _index_oids(scratch_database)
    assert {name: oids[name] for name in others} == others
    assert _index_validity(scratch_database, "litellm_call_id_idx") == {CALL_ID_INDEX: True}
    assert _index_validity(scratch_database, "api_key_startTime_idx") == {API_KEY_INDEX: True}


@requires_db
def test_a_valid_partitioned_parent_index_under_another_name_is_renamed_with_its_children_kept(
    partitioned_database: str, caplog: pytest.LogCaptureFixture
) -> None:
    hand_built: Final = _create_index(
        partitioned_database, "call_id_parent_by_hand", "LiteLLM_SpendLogs", '("litellm_call_id")'
    )
    children_before: Final = _attached_children(partitioned_database, "call_id_parent_by_hand")

    with caplog.at_level("INFO", logger="litellm_proxy_extras"):
        assert ensure_request_log_indexes(partitioned_database, "public") is True

    assert "Building index" not in caplog.text
    oids: Final = _index_oids(partitioned_database)
    assert "call_id_parent_by_hand" not in oids and oids[CALL_ID_INDEX] == hand_built
    assert _attached_children(partitioned_database, CALL_ID_INDEX) == children_before
    assert _index_validity(partitioned_database, "litellm_call_id_idx")[CALL_ID_INDEX] is True


@requires_db
def test_a_second_copy_of_a_managed_index_is_reported_with_its_drop_statement_and_left_in_place(
    scratch_database: str, caplog: pytest.LogCaptureFixture
) -> None:
    assert ensure_request_log_indexes(scratch_database, "public") is True
    copy: Final = _create_index(scratch_database, "call_id_copy", "LiteLLM_SpendLogs", '("litellm_call_id")')

    with caplog.at_level("WARNING", logger="litellm_proxy_extras"):
        assert ensure_request_log_indexes(scratch_database, "public") is True

    assert 'remove it with: DROP INDEX CONCURRENTLY "public"."call_id_copy"' in caplog.text
    assert _index_oids(scratch_database)["call_id_copy"] == copy
