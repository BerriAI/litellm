import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from litellm_proxy_extras import utils as utils_module
from litellm_proxy_extras.migration_recovery import recover_partitioned_index_migration
from litellm_proxy_extras.partitioned_index import PARTITIONED_INDEX_LOCK_KEY
from litellm_proxy_extras.prisma_toolchain import MIGRATION_LOCK_TIMEOUT_ENV_VAR
from litellm_proxy_extras.utils import ProxyExtrasDBManager

psycopg = pytest.importorskip("psycopg")

pytestmark = [
    pytest.mark.timeout(600),
    pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="requires a postgres database (DATABASE_URL)"),
]

REPO: Final = Path(__file__).resolve().parents[2]
PACKAGE: Final = REPO / "litellm-proxy-extras" / "litellm_proxy_extras"
PARTITION_SCRIPT: Final = REPO / "db_scripts" / "partition_spend_logs.sql"
INDEX_MIGRATION: Final = "20260831120001_spend_logs_litellm_call_id_index"
INDEX: Final = "LiteLLM_SpendLogs_litellm_call_id_idx"
PARTITIONS: Final = MappingProxyType(
    {
        "LiteLLM_SpendLogs_p2026_09": ("2026-09-01", "2026-10-01"),
        "LiteLLM_SpendLogs_p2026_10": ("2026-10-01", "2026-11-01"),
    }
)
NESTED_SCHEMA: Final = "spend_archive"
NESTED_PARTITION: Final = "LiteLLM_SpendLogs_p2026_11"
NESTED_LEAVES: Final = ("LiteLLM_SpendLogs_p2026_11_h0", "LiteLLM_SpendLogs_p2026_11_h1")


def _base_url() -> str:
    return os.environ["DATABASE_URL"].split("?")[0]


def _deploy_release_before_the_index_migration(database_url: str, prisma_dir: Path) -> None:
    (prisma_dir / "migrations").mkdir(parents=True)
    shutil.copy(PACKAGE / "schema.prisma", prisma_dir / "schema.prisma")
    for migration in sorted((PACKAGE / "migrations").iterdir()):
        if migration.is_dir() and migration.name < INDEX_MIGRATION:
            shutil.copytree(migration, prisma_dir / "migrations" / migration.name)
    subprocess.run(
        ["prisma", "migrate", "deploy", "--schema", str(prisma_dir / "schema.prisma")],
        check=True,
        capture_output=True,
        env={**os.environ, "DATABASE_URL": database_url},
    )


@pytest.fixture
def partitioned_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[str]:
    """A deployment on the release before the concurrent index migration whose
    LiteLLM_SpendLogs was converted with db_scripts/partition_spend_logs.sql."""
    admin_url: Final = _base_url()
    name: Final = f"partitioned_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    database_url: Final = f"{admin_url.rsplit('/', 1)[0]}/{name}"
    try:
        _deploy_release_before_the_index_migration(database_url, tmp_path / "prisma")
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute(PARTITION_SCRIPT.read_text())
            for partition, (start, end) in PARTITIONS.items():
                conn.execute(
                    f'CREATE TABLE "{partition}" PARTITION OF "LiteLLM_SpendLogs" '
                    f"FOR VALUES FROM ('{start}') TO ('{end}')"
                )
                _insert_spend_log(conn, partition, start)
            conn.execute(f'CREATE SCHEMA "{NESTED_SCHEMA}"')
            conn.execute(
                f'CREATE TABLE "{NESTED_SCHEMA}"."{NESTED_PARTITION}" PARTITION OF "LiteLLM_SpendLogs" '
                "FOR VALUES FROM ('2026-11-01') TO ('2026-12-01') PARTITION BY HASH (\"request_id\")"
            )
            for remainder, leaf in enumerate(NESTED_LEAVES):
                conn.execute(
                    f'CREATE TABLE "{NESTED_SCHEMA}"."{leaf}" PARTITION OF "{NESTED_SCHEMA}"."{NESTED_PARTITION}" '
                    f"FOR VALUES WITH (MODULUS {len(NESTED_LEAVES)}, REMAINDER {remainder})"
                )
            _insert_spend_log(conn, NESTED_PARTITION, "2026-11-01")
        monkeypatch.delenv("DIRECT_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", database_url)
        monkeypatch.setattr(utils_module.time, "sleep", lambda seconds: None)
        yield database_url
    finally:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


def _insert_spend_log(conn: psycopg.Connection[tuple[object, ...]], partition: str, start: str) -> None:
    conn.execute(
        'INSERT INTO "LiteLLM_SpendLogs" ("request_id", "call_type", "startTime", "endTime", "litellm_call_id") '
        "VALUES (%s, 'acompletion', %s, %s, %s)",
        (f"req-{partition}", start, start, f"call-{partition}"),
    )


def _call_id_index_validity(database_url: str) -> Mapping[str, bool]:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT c.relname, i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname LIKE '%%litellm_call_id_idx' ORDER BY c.relname"
        ).fetchall()
    return MappingProxyType(dict(rows))


def _ledger_rows(database_url: str) -> list[tuple[bool, bool]]:
    with psycopg.connect(database_url) as conn:
        return conn.execute(
            "SELECT finished_at IS NOT NULL, rolled_back_at IS NOT NULL FROM _prisma_migrations "
            "WHERE migration_name = %s ORDER BY started_at",
            (INDEX_MIGRATION,),
        ).fetchall()


def _call_id_lookup_uses_the_index(database_url: str, partition: str, *leaves: str) -> bool:
    with psycopg.connect(database_url) as conn:
        conn.execute("SET enable_seqscan = off")
        plan = conn.execute(
            'EXPLAIN SELECT request_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s', (f"call-{partition}",)
        ).fetchall()
    return all(any(f"{leaf}_litellm_call_id_idx" in line for (line,) in plan) for leaf in leaves or (partition,))


@pytest.mark.parametrize("use_v2_resolver", [False, True])
@pytest.mark.parametrize("failed_deploy_first", [False, True])
def test_upgrade_builds_the_call_id_index_per_partition(
    partitioned_database: str, use_v2_resolver: bool, failed_deploy_first: bool
) -> None:
    if failed_deploy_first:
        failed: Final = subprocess.run(
            ["prisma", "migrate", "deploy", "--schema", str(PACKAGE / "schema.prisma")],
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": partitioned_database},
        )
        assert failed.returncode != 0
        assert "cannot create index on partitioned table" in failed.stderr
        assert _ledger_rows(partitioned_database) == [(False, False)]

    assert ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=use_v2_resolver) is True

    expected: Final = {
        f"{table}_litellm_call_id_idx": True
        for table in ("LiteLLM_SpendLogs", "LiteLLM_SpendLogs_pdefault", *PARTITIONS, NESTED_PARTITION, *NESTED_LEAVES)
    }
    assert _call_id_index_validity(partitioned_database) == expected
    ledger: Final = _ledger_rows(partitioned_database)
    assert ledger[-1] == (True, False)
    assert all(rolled_back for _, rolled_back in ledger[:-1])
    assert all(_call_id_lookup_uses_the_index(partitioned_database, partition) for partition in PARTITIONS)
    assert _call_id_lookup_uses_the_index(partitioned_database, NESTED_PARTITION, *NESTED_LEAVES)

    assert ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=use_v2_resolver) is True
    assert _call_id_index_validity(partitioned_database) == expected
    assert _ledger_rows(partitioned_database) == ledger


def test_a_failed_row_without_the_partitioned_error_is_left_alone(partitioned_database: str) -> None:
    subprocess.run(
        ["prisma", "migrate", "deploy", "--schema", str(PACKAGE / "schema.prisma")],
        capture_output=True,
        env={**os.environ, "DATABASE_URL": partitioned_database},
    )
    with psycopg.connect(partitioned_database, autocommit=True) as conn:
        conn.execute(
            "UPDATE _prisma_migrations SET logs = 'ERROR: permission denied for table LiteLLM_SpendLogs' "
            "WHERE migration_name = %s",
            (INDEX_MIGRATION,),
        )
    migration: Final = PACKAGE / "migrations" / INDEX_MIGRATION / "migration.sql"

    assert recover_partitioned_index_migration("public", migration, partitioned_database) is False

    assert _call_id_index_validity(partitioned_database) == {}
    assert _ledger_rows(partitioned_database) == [(False, False)]


def test_a_failed_row_whose_checksum_does_not_match_the_file_is_left_alone(partitioned_database: str) -> None:
    subprocess.run(
        ["prisma", "migrate", "deploy", "--schema", str(PACKAGE / "schema.prisma")],
        capture_output=True,
        env={**os.environ, "DATABASE_URL": partitioned_database},
    )
    with psycopg.connect(partitioned_database, autocommit=True) as conn:
        conn.execute(
            "UPDATE _prisma_migrations SET checksum = repeat('0', 64) WHERE migration_name = %s",
            (INDEX_MIGRATION,),
        )
    migration: Final = PACKAGE / "migrations" / INDEX_MIGRATION / "migration.sql"

    assert recover_partitioned_index_migration("public", migration, partitioned_database) is False

    assert _call_id_index_validity(partitioned_database) == {}
    assert _ledger_rows(partitioned_database) == [(False, False)]


def test_recovery_outwaits_another_runner_holding_the_build_lock_under_a_statement_timeout(
    partitioned_database: str,
) -> None:
    subprocess.run(
        ["prisma", "migrate", "deploy", "--schema", str(PACKAGE / "schema.prisma")],
        capture_output=True,
        env={**os.environ, "DATABASE_URL": partitioned_database},
    )
    migration: Final = PACKAGE / "migrations" / INDEX_MIGRATION / "migration.sql"
    timed_out_url: Final = f"{partitioned_database}?options=-c%20statement_timeout%3D1000"
    outcome: list[bool] = []
    with psycopg.connect(partitioned_database, autocommit=True) as other_runner:
        other_runner.execute("SELECT pg_advisory_lock(%s)", (PARTITIONED_INDEX_LOCK_KEY,))
        recovery = threading.Thread(
            target=lambda: outcome.append(recover_partitioned_index_migration("public", migration, timed_out_url))
        )
        recovery.start()
        recovery.join(timeout=3)
        assert recovery.is_alive()
        other_runner.execute("SELECT pg_advisory_unlock(%s)", (PARTITIONED_INDEX_LOCK_KEY,))
    recovery.join(timeout=120)

    assert outcome == [True]
    assert _call_id_index_validity(partitioned_database)[INDEX] is True
    assert _ledger_rows(partitioned_database) == [(True, False)]


def test_recovery_gives_up_when_another_runner_keeps_the_build_lock_past_the_deadline(
    partitioned_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(
        ["prisma", "migrate", "deploy", "--schema", str(PACKAGE / "schema.prisma")],
        capture_output=True,
        env={**os.environ, "DATABASE_URL": partitioned_database},
    )
    migration: Final = PACKAGE / "migrations" / INDEX_MIGRATION / "migration.sql"
    monkeypatch.setenv(MIGRATION_LOCK_TIMEOUT_ENV_VAR, "1")
    outcome: Final = _while_another_runner_keeps_the_build_lock(
        partitioned_database, lambda: recover_partitioned_index_migration("public", migration, partitioned_database)
    )

    assert isinstance(outcome, RuntimeError)
    assert "still building" in str(outcome)
    assert INDEX not in _call_id_index_validity(partitioned_database)
    assert _ledger_rows(partitioned_database) == [(False, False)]


@pytest.mark.parametrize("use_v2_resolver", [False, True])
def test_a_runner_that_outlives_the_build_lock_deadline_fails_without_rolling_the_builder_back(
    partitioned_database: str, use_v2_resolver: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(
        ["prisma", "migrate", "deploy", "--schema", str(PACKAGE / "schema.prisma")],
        capture_output=True,
        env={**os.environ, "DATABASE_URL": partitioned_database},
    )
    monkeypatch.setenv(MIGRATION_LOCK_TIMEOUT_ENV_VAR, "1")
    outcome: Final = _while_another_runner_keeps_the_build_lock(
        partitioned_database,
        lambda: ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=use_v2_resolver),
    )

    assert isinstance(outcome, RuntimeError)
    assert "still building" in str(outcome)
    assert INDEX not in _call_id_index_validity(partitioned_database)
    assert _ledger_rows(partitioned_database) == [(False, False)]


def _while_another_runner_keeps_the_build_lock(
    database_url: str, attempt: Callable[[], bool]
) -> bool | RuntimeError | None:
    """Run ``attempt`` while a second session holds the build lock for up to 60s;
    its return value or RuntimeError, None when it was still running."""
    outcome: list[bool | RuntimeError] = []

    def run() -> None:
        try:
            outcome.append(attempt())
        except RuntimeError as exc:
            outcome.append(exc)

    with psycopg.connect(database_url, autocommit=True) as other_runner:
        other_runner.execute("SELECT pg_advisory_lock(%s)", (PARTITIONED_INDEX_LOCK_KEY,))
        runner = threading.Thread(target=run)
        runner.start()
        runner.join(timeout=60)
        finished: Final = not runner.is_alive()
        other_runner.execute("SELECT pg_advisory_unlock(%s)", (PARTITIONED_INDEX_LOCK_KEY,))
    runner.join(timeout=120)
    return outcome[0] if finished and outcome else None
