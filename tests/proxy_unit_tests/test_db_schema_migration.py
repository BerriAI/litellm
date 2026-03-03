import pytest
import os
import subprocess
from pathlib import Path
from pytest_postgresql import factories
import shutil
import tempfile

# Create postgresql fixture
postgresql_my_proc = factories.postgresql_proc(port=None)
postgresql_my = factories.postgresql("postgresql_my_proc")


@pytest.fixture(scope="function")
def schema_setup(postgresql_my):
    """Fixture to provide a test postgres database"""
    return postgresql_my


@pytest.mark.xdist_group("proxy_heavy")
def test_aaaasschema_migration_check(schema_setup, monkeypatch):
    """Test to check if schema requires migration"""
    # Set test database URL
    test_db_url = f"postgresql://{schema_setup.info.user}:@{schema_setup.info.host}:{schema_setup.info.port}/{schema_setup.info.dbname}"
    # test_db_url = "postgresql://test-user:test-password@test-host.example.com/test-db?sslmode=require"
    monkeypatch.setenv("DATABASE_URL", test_db_url)

    deploy_dir = Path("./litellm-proxy-extras/litellm_proxy_extras")
    source_migrations_dir = deploy_dir / "migrations"
    source_schema_path = Path("./schema.prisma")

    # Use worker-specific temp directory to avoid races when running with -n 8.
    # Prisma expects migrations in <schema_dir>/migrations, so we create that layout.
    temp_base = Path(tempfile.mkdtemp(prefix="litellm_schema_migration_"))
    temp_migrations_dir = temp_base / "migrations"
    schema_path = temp_base / "schema.prisma"

    try:
        shutil.copy(source_schema_path, schema_path)
        shutil.copytree(source_migrations_dir, temp_migrations_dir)

        if not temp_migrations_dir.exists() or not any(temp_migrations_dir.iterdir()):
            print("No existing migrations found - first migration needed")
            pytest.fail(
                "No existing migrations found - first migration needed. Run `litellm/ci_cd/baseline_db.py` to create new migration -E.g. `python litellm/ci_cd/baseline_db_migration.py`."
            )

        # Apply all existing migrations
        subprocess.run(
            ["prisma", "migrate", "deploy", "--schema", str(schema_path)], check=True
        )

        # Compare current database state against schema
        diff_result = subprocess.run(
            [
                "prisma",
                "migrate",
                "diff",
                "--from-url",
                test_db_url,
                "--to-schema-datamodel",
                str(schema_path),
                "--script",  # Show the SQL diff
                "--exit-code",  # Return exit code 2 if there are differences
            ],
            capture_output=True,
            text=True,
        )

        print("Exit code:", diff_result.returncode)
        print("Stdout:", diff_result.stdout)
        print("Stderr:", diff_result.stderr)

        if diff_result.returncode == 2:
            print("Schema changes detected. New migration needed.")
            print("Schema differences:")
            print(diff_result.stdout)
            pytest.fail(
                "Schema changes detected - new migration required. Run `litellm/ci_cd/run_migration.py` to create new migration -E.g. `python litellm/ci_cd/run_migration.py <migration_name>`."
            )
        else:
            print("No schema changes detected. Migration not needed.")

    finally:
        # Clean up: remove temporary directory
        if temp_base.exists():
            shutil.rmtree(temp_base)
