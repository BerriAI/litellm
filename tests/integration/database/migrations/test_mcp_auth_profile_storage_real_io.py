import os
import shutil
import sys
from pathlib import Path
from typing import Final

import psycopg
from litellm_proxy_extras.prisma_toolchain import prisma_migrate_deploy_timeout, run_prisma

MIGRATION: Final = (
    Path(__file__).resolve().parents[4]
    / "litellm/proxy-extras/litellm_proxy_extras/migrations/20261121010000_mcp_auth_profile_storage/migration.sql"
)


def test_profile_migration_preserves_existing_rows_and_accepts_legacy_inserts(tmp_path: Path) -> None:
    migrations: Final = MIGRATION.parent.parent
    schema: Final = tmp_path / "schema.prisma"
    schema.write_bytes((migrations.parent / "schema.prisma").read_bytes())
    previous: Final = tmp_path / "migrations"
    previous.mkdir()
    shutil.copy(migrations / "migration_lock.toml", previous)
    for item in migrations.iterdir():
        if item.is_dir() and item.name < MIGRATION.parent.name:
            shutil.copytree(item, previous / item.name)

    deploy: Final = (sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(schema))
    run_prisma(deploy, env=os.environ, timeout=prisma_migrate_deploy_timeout())
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute("INSERT INTO \"LiteLLM_MCPServerTable\" (server_id) VALUES ('original')")
        connection.execute("INSERT INTO \"LiteLLM_ObjectPermissionTable\" (object_permission_id) VALUES ('original')")

        shutil.copytree(MIGRATION.parent, previous / MIGRATION.parent.name)
        run_prisma(deploy, env=os.environ, timeout=prisma_migrate_deploy_timeout())
        connection.execute("INSERT INTO \"LiteLLM_MCPServerTable\" (server_id) VALUES ('later')")
        connection.execute("INSERT INTO \"LiteLLM_ObjectPermissionTable\" (object_permission_id) VALUES ('later')")

        servers: Final = connection.execute(
            "SELECT server_id, additional_auth IS NULL, pg_typeof(additional_auth)::text "
            'FROM "LiteLLM_MCPServerTable" ORDER BY 1'
        ).fetchall()
        permissions: Final = connection.execute(
            "SELECT object_permission_id, mcp_auth_profiles IS NULL, pg_typeof(mcp_auth_profiles)::text "
            'FROM "LiteLLM_ObjectPermissionTable" ORDER BY 1'
        ).fetchall()
        assert servers == permissions == [("later", True, "jsonb"), ("original", True, "jsonb")]
