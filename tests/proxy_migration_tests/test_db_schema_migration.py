import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Final

import pytest

psycopg = pytest.importorskip("psycopg")

CALL_ID_MIGRATION: Final = Path(
    "./litellm-proxy-extras/litellm_proxy_extras/migrations/20260831120001_spend_logs_litellm_call_id_index/migration.sql"
)


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


@pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)
def test_spend_logs_call_id_index_migration_applies_to_a_partitioned_table() -> None:
    schema: Final = f"partitioned_spend_logs_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(os.environ["DATABASE_URL"].split("?")[0], autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
        conn.execute(f'SET search_path TO "{schema}"')
        conn.execute(
            'CREATE TABLE "LiteLLM_SpendLogs" (request_id TEXT, "startTime" TIMESTAMPTZ, litellm_call_id TEXT) '
            'PARTITION BY RANGE ("startTime")'
        )
        conn.execute('CREATE TABLE "LiteLLM_SpendLogs_pdefault" PARTITION OF "LiteLLM_SpendLogs" DEFAULT')
        try:
            conn.execute(CALL_ID_MIGRATION.read_bytes())
            indexed: Final = conn.execute(
                "SELECT tablename FROM pg_indexes WHERE schemaname = %s AND indexdef LIKE '%%(litellm_call_id)' "
                "ORDER BY tablename",
                (schema,),
            ).fetchall()
        finally:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
    assert indexed == [("LiteLLM_SpendLogs",), ("LiteLLM_SpendLogs_pdefault",)], (
        "the call id index was not built on the partitioned parent and its partition"
    )
