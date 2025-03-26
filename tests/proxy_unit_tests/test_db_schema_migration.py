import pytest
import os
import subprocess
from pathlib import Path
from pytest_postgresql import factories

# Create postgresql fixture
postgresql_my_proc = factories.postgresql_proc(port=None)
postgresql_my = factories.postgresql("postgresql_my_proc")


@pytest.fixture(scope="function")
def schema_setup(postgresql_my):
    """Fixture to provide a test postgres database"""
    return postgresql_my


def test_schema_migration_check(schema_setup):
    """Test to check if schema requires migration"""
    # Set test database URL
    test_db_url = f"postgresql://{schema_setup.info.user}:@{schema_setup.info.host}:{schema_setup.info.port}/{schema_setup.info.dbname}"
    os.environ["DATABASE_URL"] = test_db_url

    deploy_dir = Path("./deploy")
    migrations_dir = deploy_dir / "migrations"

    print(migrations_dir)
    if not migrations_dir.exists() or not any(migrations_dir.iterdir()):
        print("No existing migrations found - first migration needed")
        pytest.fail("No existing migrations found - first migration needed")

    # If migrations exist, check for changes
    result = subprocess.run(
        ["prisma", "migrate", "status"], capture_output=True, text=True
    )

    status_output = result.stdout.lower()
    needs_migration = any(
        state in status_output for state in ["drift detected", "pending"]
    )

    if needs_migration:
        print("Schema changes detected. New migration needed.")
        # Show the differences
        diff_result = subprocess.run(
            [
                "prisma",
                "migrate",
                "diff",
                "--from-migrations",
                "--to-schema-datamodel",
                "schema.prisma",
            ],
            capture_output=True,
            text=True,
        )
        print("Schema differences:")
        print(diff_result.stdout)
    else:
        print("No schema changes detected. Migration not needed.")

    assert not needs_migration, "Schema changes detected - new migration required"
