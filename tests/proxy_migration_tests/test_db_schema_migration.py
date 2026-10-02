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
