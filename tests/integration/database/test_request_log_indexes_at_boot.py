import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import scratch_database
from integration._support.process import LEGACY_MIGRATE_DEPLOY, MIGRATE_DEPLOY, owned_proxy_process
from psycopg import sql
from psycopg.rows import class_row

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
PRISMA_DIR: Final = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
PARTITION_SCRIPT: Final = REPO_ROOT / "db_scripts" / "partition_spend_logs.sql"
SHIPPED_MIGRATIONS: Final = tuple(sorted(path.name for path in (PRISMA_DIR / "migrations").iterdir() if path.is_dir()))
API_KEY_INDEX_MIGRATION: Final = "20260823000000_add_spend_logs_api_key_starttime_index"
CALL_ID_INDEX_MIGRATION: Final = "20260831120001_spend_logs_litellm_call_id_index"
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
MIGRATION_JOB_SECONDS: Final = 300
INDEXES_IN_PLACE: Final = "Request-log indexes are all in place"
INDEX_BUILD_LINES: Final = ("Building index", "Attached index")
BUILD_SECONDS: Final = 60
SPEND_LOGS_INDEXES: Final = ("LiteLLM_SpendLogs_api_key_startTime_idx", "LiteLLM_SpendLogs_litellm_call_id_idx")
POPULATED_PARTITIONS: Final = MappingProxyType(
    {
        "LiteLLM_SpendLogs_p2026_08": ("2026-08-01", "2026-09-01"),
        "LiteLLM_SpendLogs_p2026_09": ("2026-09-01", "2026-10-01"),
    }
)
DEFAULT_PARTITION: Final = "LiteLLM_SpendLogs_pdefault"
ROWS_PER_PARTITION: Final = 500
PARTITIONED_PARENT_ERROR: Final = 'cannot create index on partitioned table "LiteLLM_SpendLogs" concurrently'


@dataclass(frozen=True, slots=True)
class Resolver:
    """One migration resolver as the serving proxy selects it (CLI flags) and as the
    migration job selects it (environment)."""

    proxy_flags: tuple[str, ...]
    job_environment: Mapping[str, str]


V2: Final = Resolver(MIGRATE_DEPLOY, MappingProxyType({"USE_V2_MIGRATION_RESOLVER": "true"}))
LEGACY: Final = Resolver(LEGACY_MIGRATE_DEPLOY, MappingProxyType({"USE_V2_MIGRATION_RESOLVER": "false"}))
RESOLVERS: Final = pytest.mark.parametrize("resolver", (V2, LEGACY), ids=("v2", "legacy"))


def release_layout(directory: Path, migrations: tuple[str, ...]) -> Path:
    """The Prisma layout of the release that shipped `migrations`: the two index migrations
    carry the SQL they shipped with, not the inert files of this build."""
    (directory / "migrations").mkdir(parents=True)
    shutil.copy(PRISMA_DIR / "schema.prisma", directory / "schema.prisma")
    shutil.copy(PRISMA_DIR / "migrations" / "migration_lock.toml", directory / "migrations" / "migration_lock.toml")
    for name in migrations:
        shutil.copytree(PRISMA_DIR / "migrations" / name, directory / "migrations" / name)
    for name, original in ORIGINAL_MIGRATION_SQL.items():
        if name in migrations:
            (directory / "migrations" / name / "migration.sql").write_text(original)
    return directory / "schema.prisma"


def migrate_deploy(database_url: str, schema: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(schema)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "DATABASE_URL": database_url},
    )


def migration_job(database_url: str, resolver: Resolver) -> subprocess.CompletedProcess[str]:
    """The migrations image entrypoint, as the helm migration Job runs it."""
    return subprocess.run(
        [sys.executable, "-I", str(REPO_ROOT / "migrations" / "run.py")],
        capture_output=True,
        text=True,
        timeout=MIGRATION_JOB_SECONDS,
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": database_url, **resolver.job_environment},
    )


def migration_cli(database_url: str, gateway: Gateway, resolver: Resolver) -> subprocess.CompletedProcess[str]:
    """The proxy CLI as a migration job: `--skip_server_startup` migrates, builds the
    indexes and exits by their result."""
    return subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "integration._support.proxy",
            "--config",
            "tests/integration/proxy_config.yaml",
            *resolver.proxy_flags,
            "--skip_server_startup",
        ],
        capture_output=True,
        text=True,
        timeout=MIGRATION_JOB_SECONDS,
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": database_url, "LITELLM_MASTER_KEY": gateway.key},
    )


def deploy_schema_before_the_index_migrations(database_url: str, directory: Path) -> None:
    older: Final = tuple(name for name in SHIPPED_MIGRATIONS if name < API_KEY_INDEX_MIGRATION)
    deployed: Final = migrate_deploy(database_url, release_layout(directory / "older-release", older))
    assert deployed.returncode == 0, deployed.stdout + deployed.stderr


def deploy_the_original_index_migrations(database_url: str, directory: Path) -> subprocess.CompletedProcess[str]:
    """Boot the v1.103.0 layout once: on a partitioned table its CONCURRENTLY call_id index
    fails with P3018 and leaves the ledger row unfinished, on a plain table both apply."""
    return migrate_deploy(database_url, release_layout(directory / "v1.103.0", SHIPPED_MIGRATIONS))


def fail_the_call_id_index_migration_like_the_shipped_release(database_url: str, directory: Path) -> None:
    deployed: Final = deploy_the_original_index_migrations(database_url, directory)
    assert deployed.returncode != 0, deployed.stdout
    assert "P3018" in deployed.stderr and PARTITIONED_PARENT_ERROR in deployed.stderr, deployed.stderr
    assert ledger(database_url)[CALL_ID_INDEX_MIGRATION] is False


def partition_spend_logs(database_url: str) -> None:
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(PARTITION_SCRIPT.read_bytes())
        for partition, (start, stop) in POPULATED_PARTITIONS.items():
            add_partition(connection, partition, start, stop)
            connection.execute(
                'INSERT INTO "LiteLLM_SpendLogs" ("request_id", "call_type", "api_key", "startTime", "endTime") '
                "SELECT %s || n, 'acompletion', 'sk-' || (n %% 7), %s::timestamp + (n * interval '1 minute'), "
                "%s::timestamp + (n * interval '1 minute') + interval '1 second' FROM generate_series(1, %s) AS n",
                (partition, start, start, ROWS_PER_PARTITION),
            )
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogs" ("request_id", "call_type", "startTime", "endTime") '
            "SELECT 'default-' || n, 'acompletion', '2026-07-01'::timestamp + (n * interval '1 minute'), "
            "'2026-07-01'::timestamp + (n * interval '1 minute') FROM generate_series(1, %s) AS n",
            (ROWS_PER_PARTITION,),
        )


@dataclass(frozen=True, slots=True)
class _LedgerRow:
    name: str
    finished: bool


@dataclass(frozen=True, slots=True)
class _IndexRow:
    index: str
    valid: bool


@dataclass(frozen=True, slots=True)
class _OidRow:
    index: str
    oid: int


@dataclass(frozen=True, slots=True)
class _AttachedRow:
    partition: str
    parent_index: str
    valid: bool


def ledger(database_url: str) -> Mapping[str, bool]:
    """Every migration in the ledger that was not rolled back, mapped to whether it finished."""
    with psycopg.connect(database_url) as connection, connection.cursor(row_factory=class_row(_LedgerRow)) as cursor:
        rows: Final = cursor.execute(
            'SELECT migration_name AS name, finished_at IS NOT NULL AS finished FROM "_prisma_migrations" '
            "WHERE rolled_back_at IS NULL ORDER BY migration_name"
        ).fetchall()
    return MappingProxyType({row.name: row.finished for row in rows})


def parent_index_validity(database_url: str) -> Mapping[str, bool]:
    with psycopg.connect(database_url) as connection, connection.cursor(row_factory=class_row(_IndexRow)) as cursor:
        rows: Final = cursor.execute(
            "SELECT c.relname AS index, x.indisvalid AS valid FROM pg_index x JOIN pg_class c ON c.oid = x.indexrelid "
            "WHERE x.indrelid = '\"LiteLLM_SpendLogs\"'::regclass AND c.relname = ANY(%s)",
            (list(SPEND_LOGS_INDEXES),),
        ).fetchall()
    return MappingProxyType({row.index: row.valid for row in rows})


def index_oids(database_url: str) -> Mapping[str, int]:
    """index name -> oid for the SpendLogs indexes on the parent or any partition; a rebuild changes the oid."""
    with psycopg.connect(database_url) as connection, connection.cursor(row_factory=class_row(_OidRow)) as cursor:
        rows: Final = cursor.execute(
            "SELECT c.relname AS index, c.oid::int AS oid FROM pg_index x JOIN pg_class c ON c.oid = x.indexrelid "
            "WHERE x.indrelid = '\"LiteLLM_SpendLogs\"'::regclass OR x.indrelid IN "
            "(SELECT inhrelid FROM pg_inherits WHERE inhparent = '\"LiteLLM_SpendLogs\"'::regclass)"
        ).fetchall()
    return MappingProxyType({row.index: row.oid for row in rows})


def attached_partition_indexes(database_url: str) -> frozenset[tuple[str, str, bool]]:
    """(partition, parent index, child is valid) for every child index attached under a SpendLogs parent index."""
    with psycopg.connect(database_url) as connection, connection.cursor(row_factory=class_row(_AttachedRow)) as cursor:
        rows: Final = cursor.execute(
            "SELECT part.relname AS partition, parent_index.relname AS parent_index, child.indisvalid AS valid "
            "FROM pg_inherits attached "
            "JOIN pg_class parent_index ON parent_index.oid = attached.inhparent "
            "JOIN pg_index child ON child.indexrelid = attached.inhrelid "
            "JOIN pg_class part ON part.oid = child.indrelid "
            "WHERE parent_index.relname = ANY(%s)",
            (list(SPEND_LOGS_INDEXES),),
        ).fetchall()
    return frozenset((row.partition, row.parent_index, row.valid) for row in rows)


def expected_attachments(partitions: tuple[str, ...]) -> frozenset[tuple[str, str, bool]]:
    return frozenset((partition, index, True) for partition, index in product(partitions, SPEND_LOGS_INDEXES))


def add_partition(connection: psycopg.Connection[tuple[object, ...]], partition: str, start: str, stop: str) -> None:
    connection.execute(
        sql.SQL('CREATE TABLE {} PARTITION OF "LiteLLM_SpendLogs" FOR VALUES FROM ({}) TO ({})').format(
            sql.Identifier(partition), sql.Literal(start), sql.Literal(stop)
        )
    )


def assert_ready(booted_gateway: Gateway) -> None:
    readiness: Final = booted_gateway.request("GET", "/health/readiness")
    assert readiness.status_code == 200, readiness.text
    assert readiness.json()["db"] == "connected", readiness.text


def assert_both_indexes_cover_every_partition(database_url: str) -> None:
    """Every populated partition's index is attached and valid, both parents are valid, and
    a partition created afterwards inherits both indexes."""
    populated: Final = (*POPULATED_PARTITIONS, DEFAULT_PARTITION)
    assert ledger(database_url) == {name: True for name in SHIPPED_MIGRATIONS}
    assert attached_partition_indexes(database_url) == expected_attachments(populated)
    assert parent_index_validity(database_url) == {index: True for index in SPEND_LOGS_INDEXES}
    with psycopg.connect(database_url, autocommit=True) as connection:
        add_partition(connection, "LiteLLM_SpendLogs_p2026_10", "2026-10-01", "2026-11-01")
    assert attached_partition_indexes(database_url) == expected_attachments((*populated, "LiteLLM_SpendLogs_p2026_10"))


def assert_the_serving_proxy_boots_and_finds_the_indexes_in_place(
    gateway: Gateway, directory: Path, database_url: str, resolver: Resolver
) -> None:
    """The serving proxy applies the inert files, reports ready, and its background build
    finds every index already there, so it builds nothing and the catalog is untouched."""
    oids: Final = index_oids(database_url)
    with owned_proxy_process(
        gateway, directory, {"DATABASE_URL": database_url}, database_setup=resolver.proxy_flags
    ) as booted:
        assert_ready(booted.gateway)
        log: Final = eventually(
            lambda: booted.log.read_text(errors="replace"), lambda text: INDEXES_IN_PLACE in text, seconds=BUILD_SECONDS
        )
    assert ledger(database_url) == {name: True for name in SHIPPED_MIGRATIONS}
    assert not any(line in log for line in INDEX_BUILD_LINES), log[-4000:]
    assert index_oids(database_url) == oids


@RESOLVERS
def test_the_migration_job_gives_a_partitioned_table_at_the_pre_index_schema_both_indexes_per_partition(
    gateway: Gateway, tmp_path: Path, resolver: Resolver
) -> None:
    with scratch_database() as database_url:
        deploy_schema_before_the_index_migrations(database_url, tmp_path)
        partition_spend_logs(database_url)
        job: Final = migration_job(database_url, resolver)
        assert job.returncode == 0, job.stdout + job.stderr
        assert INDEXES_IN_PLACE in job.stderr + job.stdout, job.stdout + job.stderr
        assert_both_indexes_cover_every_partition(database_url)
        assert_the_serving_proxy_boots_and_finds_the_indexes_in_place(gateway, tmp_path, database_url, resolver)


@RESOLVERS
def test_the_migration_job_heals_a_partitioned_table_left_with_the_failed_call_id_ledger_row(
    gateway: Gateway, tmp_path: Path, resolver: Resolver
) -> None:
    with scratch_database() as database_url:
        deploy_schema_before_the_index_migrations(database_url, tmp_path)
        partition_spend_logs(database_url)
        fail_the_call_id_index_migration_like_the_shipped_release(database_url, tmp_path)
        job: Final = migration_job(database_url, resolver)
        assert job.returncode == 0, job.stdout + job.stderr
        assert_both_indexes_cover_every_partition(database_url)
        assert_the_serving_proxy_boots_and_finds_the_indexes_in_place(gateway, tmp_path, database_url, resolver)


@RESOLVERS
def test_the_migration_job_leaves_a_plain_table_that_applied_the_original_index_migrations_alone(
    gateway: Gateway, tmp_path: Path, resolver: Resolver
) -> None:
    with scratch_database() as database_url:
        deploy_schema_before_the_index_migrations(database_url, tmp_path)
        deployed: Final = deploy_the_original_index_migrations(database_url, tmp_path)
        assert deployed.returncode == 0, deployed.stdout + deployed.stderr
        before: Final = index_oids(database_url)
        assert set(SPEND_LOGS_INDEXES) <= set(before), before
        job: Final = migration_job(database_url, resolver)
        assert job.returncode == 0, job.stdout + job.stderr
        assert INDEXES_IN_PLACE in job.stderr + job.stdout, job.stdout + job.stderr
        assert "Building index" not in job.stderr + job.stdout, job.stdout + job.stderr
        assert ledger(database_url) == {name: True for name in SHIPPED_MIGRATIONS}
        assert index_oids(database_url) == before
        assert_the_serving_proxy_boots_and_finds_the_indexes_in_place(gateway, tmp_path, database_url, resolver)


@RESOLVERS
def test_a_serving_proxy_that_runs_the_migrations_itself_builds_both_indexes_after_it_is_ready(
    gateway: Gateway, tmp_path: Path, resolver: Resolver
) -> None:
    """A deployment that runs migrate deploy from the serving proxy and never runs the
    migration job answers readiness with the inert files applied, then its background build
    puts both indexes on every partition."""
    with scratch_database() as database_url:
        deploy_schema_before_the_index_migrations(database_url, tmp_path)
        partition_spend_logs(database_url)
        assert "LiteLLM_SpendLogs_litellm_call_id_idx" not in parent_index_validity(database_url)
        with owned_proxy_process(
            gateway, tmp_path, {"DATABASE_URL": database_url}, database_setup=resolver.proxy_flags
        ) as booted:
            assert_ready(booted.gateway)
            assert ledger(database_url) == {name: True for name in SHIPPED_MIGRATIONS}
            log: Final = eventually(
                lambda: booted.log.read_text(errors="replace"),
                lambda text: INDEXES_IN_PLACE in text,
                seconds=BUILD_SECONDS,
            )
        assert "Building index" in log and "Attached index" in log, log[-4000:]
        assert_both_indexes_cover_every_partition(database_url)


def test_the_cli_run_as_a_migration_job_builds_both_indexes_before_it_exits(gateway: Gateway, tmp_path: Path) -> None:
    with scratch_database() as database_url:
        deploy_schema_before_the_index_migrations(database_url, tmp_path)
        partition_spend_logs(database_url)
        job: Final = migration_cli(database_url, gateway, V2)
        assert job.returncode == 0, job.stdout + job.stderr
        assert_both_indexes_cover_every_partition(database_url)
