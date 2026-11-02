import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from psycopg import sql
from psycopg.rows import dict_row

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
PRISMA_DIR: Final = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
MISSING_MIGRATION: Final = "20260626120000_add_mcp_tool_search_enabled"
SHIPPED_MIGRATIONS: Final = tuple(sorted(path.name for path in (PRISMA_DIR / "migrations").iterdir() if path.is_dir()))


@contextmanager
def fresh_database() -> Iterator[str]:
    name: Final = f"integration_upgrade_{uuid.uuid4().hex}"
    admin_url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(admin_url)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield urlunsplit(parsed._replace(path=f"/{name}"))
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def deploy_older_schema(database_url: str, directory: Path) -> None:
    older: Final = directory / "older-release"
    (older / "migrations").mkdir(parents=True)
    shutil.copy(PRISMA_DIR / "schema.prisma", older / "schema.prisma")
    shutil.copy(PRISMA_DIR / "migrations" / "migration_lock.toml", older / "migrations" / "migration_lock.toml")
    for name in (name for name in SHIPPED_MIGRATIONS if name < MISSING_MIGRATION):
        shutil.copytree(PRISMA_DIR / "migrations" / name, older / "migrations" / name)
    subprocess.run(
        [sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(older / "schema.prisma")],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "DATABASE_URL": database_url},
    )


def applied_migrations(database_url: str) -> tuple[str, ...]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows: Final = connection.execute(
            'SELECT migration_name FROM "_prisma_migrations" '
            "WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL ORDER BY migration_name"
        ).fetchall()
    return tuple(str(row["migration_name"]) for row in rows)


def object_permission_columns(database_url: str) -> tuple[str, ...]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows: Final = connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'LiteLLM_ObjectPermissionTable' AND column_name = 'mcp_tool_search_enabled'"
        ).fetchall()
    return tuple(str(row["column_name"]) for row in rows)


@pytest.mark.covers("other.database.migrations.entrypoint_deploys_pending_migrations_before_startup")
def test_migration_entrypoint_upgrades_an_older_schema_so_the_proxy_serves_mcp_tools(
    gateway: Gateway, tmp_path: Path
) -> None:
    with fresh_database() as database_url:
        deploy_older_schema(database_url, tmp_path)
        assert object_permission_columns(database_url) == ()
        assert applied_migrations(database_url) == tuple(
            name for name in SHIPPED_MIGRATIONS if name < MISSING_MIGRATION
        )
        entrypoint: Final = subprocess.run(
            [sys.executable, "-I", "-m", "litellm.proxy.prisma_migration"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=REPO_ROOT,
            env={**os.environ, "DATABASE_URL": database_url},
        )
        assert entrypoint.returncode == 0, entrypoint.stdout + entrypoint.stderr
        assert object_permission_columns(database_url) == ("mcp_tool_search_enabled",), entrypoint.stdout
        assert applied_migrations(database_url) == SHIPPED_MIGRATIONS, entrypoint.stdout
        with owned_proxy(
            gateway, tmp_path, {"DATABASE_URL": database_url, "DISABLE_SCHEMA_UPDATE": "true"}
        ) as upgraded:
            tools: Final = upgraded.request("GET", "/mcp-rest/tools/list")
            assert tools.status_code == 200, tools.text
            assert tools.json()["tools"] == [], tools.text
