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


def test_spend_logs_api_key_index_has_a_migration():
    """A schema-only index never reaches deployments that run ``prisma migrate deploy``."""
    migrations = Path("./litellm-proxy-extras/litellm_proxy_extras/migrations")
    creating = [
        sql
        for sql in (path.read_text() for path in migrations.glob("*/migration.sql"))
        if SPEND_LOGS_API_KEY_INDEX in sql
    ]
    assert len(creating) == 1, f"expected exactly one migration creating {SPEND_LOGS_API_KEY_INDEX}, found {len(creating)}"
    assert 'ON "LiteLLM_SpendLogs"("api_key", "startTime" DESC)' in creating[0]
    assert "CONCURRENTLY" in creating[0], (
        "LiteLLM_SpendLogs is the largest table on most deployments; a blocking build stalls spend-log inserts"
    )
