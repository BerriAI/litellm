import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)
def test_schema_migration_in_sync():
    """Fail if schema.prisma has changes not captured by the committed migrations.

    Applies every committed migration to an empty database, then diffs the result
    against schema.prisma. A non-empty diff means the schema was changed without a
    matching migration being generated.
    """
    db_url = os.environ["DATABASE_URL"]
    source_migrations_dir = Path(
        "./litellm-proxy-extras/litellm_proxy_extras/migrations"
    )
    source_schema_path = Path("./schema.prisma")

    temp_base = Path(tempfile.mkdtemp(prefix="litellm_schema_migration_"))
    schema_path = temp_base / "schema.prisma"
    migrations_dir = temp_base / "migrations"

    try:
        shutil.copy(source_schema_path, schema_path)
        shutil.copytree(source_migrations_dir, migrations_dir)

        if not any(migrations_dir.iterdir()):
            pytest.fail(
                "No existing migrations found. Run `python litellm/ci_cd/baseline_db_migration.py`."
            )

        subprocess.run(
            ["prisma", "migrate", "deploy", "--schema", str(schema_path)],
            check=True,
            env={**os.environ, "DATABASE_URL": db_url},
        )

        diff = subprocess.run(
            [
                "prisma",
                "migrate",
                "diff",
                "--from-url",
                db_url,
                "--to-schema-datamodel",
                str(schema_path),
                "--script",
                "--exit-code",
            ],
            capture_output=True,
            text=True,
        )

        if diff.returncode == 2:
            pytest.fail(
                "Schema changes detected that no migration captures. Run "
                "`python litellm/ci_cd/run_migration.py <migration_name>`.\n\n"
                + diff.stdout
            )
        assert diff.returncode == 0, f"prisma migrate diff errored: {diff.stderr}"
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)


SPEND_LOGS_API_KEY_INDEX = "LiteLLM_SpendLogs_api_key_startTime_idx"

SCHEMA_PATHS = (
    Path("./schema.prisma"),
    Path("./litellm-proxy-extras/litellm_proxy_extras/schema.prisma"),
)


def _spend_logs_model_block(schema: str) -> str:
    start = schema.index("model LiteLLM_SpendLogs {")
    return schema[start : schema.index("\n}", start)]


@pytest.mark.parametrize("schema_path", SCHEMA_PATHS, ids=lambda p: str(p))
def test_spend_logs_declares_api_key_index(schema_path: Path):
    """The Logs tab filters ``LiteLLM_SpendLogs`` by ``api_key`` inside a ``startTime``
    window. Without an index led by ``api_key`` that degrades to a full scan of the
    window on every page turn, which pinned a customer's Aurora writer at ~99% CPU.
    """
    block = _spend_logs_model_block(schema_path.read_text())
    assert f'@@index([api_key, startTime(sort: Desc)], map: "{SPEND_LOGS_API_KEY_INDEX}")' in block, (
        f"{schema_path} must keep the api_key-led index on LiteLLM_SpendLogs"
    )


MIGRATIONS_DIR = Path("./litellm-proxy-extras/litellm_proxy_extras/migrations")

PARTITION_PROBE_SCHEMA = "litellm_spendlogs_partition_probe"


def _api_key_index_migration() -> str:
    creating = [
        sql
        for sql in (path.read_text() for path in MIGRATIONS_DIR.glob("*/migration.sql"))
        if SPEND_LOGS_API_KEY_INDEX in sql
    ]
    assert len(creating) == 1, f"expected exactly one migration creating {SPEND_LOGS_API_KEY_INDEX}, found {len(creating)}"
    return creating[0]


def test_spend_logs_api_key_index_has_a_migration():
    """A schema-only index never reaches deployments that run ``prisma migrate deploy``."""
    assert 'ON "LiteLLM_SpendLogs"("api_key", "startTime" DESC)' in _api_key_index_migration()


@pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)
def test_spend_logs_api_key_index_migration_applies_to_a_partitioned_table():
    """``db_scripts/partition_spend_logs.sql`` leaves ``LiteLLM_SpendLogs`` a partitioned parent,
    and Postgres rejects both a concurrent index build on a partitioned parent and any concurrent
    build inside a transaction, so a ``CONCURRENTLY`` migration would abort the rollout of every
    partitioned deployment. This applies the shipped statement against that shape for real.
    """
    probe_sql = f"""
DROP SCHEMA IF EXISTS "{PARTITION_PROBE_SCHEMA}" CASCADE;
CREATE SCHEMA "{PARTITION_PROBE_SCHEMA}";
SET search_path TO "{PARTITION_PROBE_SCHEMA}";

CREATE TABLE "LiteLLM_SpendLogs" (
    request_id TEXT NOT NULL,
    api_key TEXT,
    "startTime" TIMESTAMP NOT NULL,
    PRIMARY KEY (request_id, "startTime")
) PARTITION BY RANGE ("startTime");
CREATE TABLE "LiteLLM_SpendLogs_pdefault" PARTITION OF "LiteLLM_SpendLogs" DEFAULT;

{_api_key_index_migration()}

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = '{PARTITION_PROBE_SCHEMA}'
          AND tablename = 'LiteLLM_SpendLogs'
          AND indexname = '{SPEND_LOGS_API_KEY_INDEX}'
    ) THEN
        RAISE EXCEPTION 'migration did not create {SPEND_LOGS_API_KEY_INDEX} on the partitioned parent';
    END IF;
END
$$;

DROP SCHEMA "{PARTITION_PROBE_SCHEMA}" CASCADE;
"""

    temp_base = Path(tempfile.mkdtemp(prefix="litellm_partition_probe_"))
    try:
        sql_path = temp_base / "probe.sql"
        sql_path.write_text(probe_sql)
        applied = subprocess.run(
            ["prisma", "db", "execute", "--url", os.environ["DATABASE_URL"], "--file", str(sql_path)],
            capture_output=True,
            text=True,
        )
        assert applied.returncode == 0, (
            f"migration for {SPEND_LOGS_API_KEY_INDEX} failed on a partitioned LiteLLM_SpendLogs:\n"
            f"{applied.stdout}\n{applied.stderr}"
        )
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)
