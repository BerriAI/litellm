import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from litellm_proxy_extras.migration_sources import materialize_prisma_dir
from litellm_proxy_extras.spend_logs.request_log_indexes import filter_request_log_index_diff


@pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)
def test_schema_migration_in_sync():
    """Fail if schema.prisma has changes not captured by the committed migrations.

    Applies every committed migration to an empty database, then diffs the result
    against schema.prisma. A non-empty diff means the schema was changed without a
    matching migration being generated. The request-log indexes the migration job
    builds are declared in the schema and deliberately absent from the migrations,
    so those statements are filtered out before the diff is judged.
    """
    db_url = os.environ["DATABASE_URL"]

    temp_base = Path(tempfile.mkdtemp(prefix="litellm_schema_migration_"))
    schema_path = temp_base / "schema.prisma"
    migrations_dir = temp_base / "migrations"

    try:
        materialize_prisma_dir(temp_base)

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
            drift = filter_request_log_index_diff(diff.stdout)
            if drift.strip():
                pytest.fail(
                    "Schema changes detected that no migration captures. Run "
                    "`python litellm/ci_cd/run_migration.py <migration_name>`.\n\n"
                    + drift
                )
        else:
            assert diff.returncode == 0, f"prisma migrate diff errored: {diff.stderr}"
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)
