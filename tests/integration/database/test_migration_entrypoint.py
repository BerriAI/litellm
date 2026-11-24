import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.database import read_rows
from integration._support.mcp import paginated_mcp_peer
from integration._support.process import owned_proxy
from psycopg import sql
from psycopg.rows import dict_row

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
PRISMA_DIR: Final = REPO_ROOT / "litellm" / "proxy-extras" / "litellm_proxy_extras"
MISSING_MIGRATION: Final = "20260626120000_add_mcp_tool_search_enabled"
MCP_CATALOG_SEED: Final = "20261102000000_seed_mcp_catalog_revision"
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


def deploy_older_schema(database_url: str, directory: Path, migrations: tuple[str, ...]) -> None:
    older: Final = directory / "older-release"
    (older / "migrations").mkdir(parents=True)
    shutil.copy(PRISMA_DIR / "schema.prisma", older / "schema.prisma")
    shutil.copy(PRISMA_DIR / "migrations" / "migration_lock.toml", older / "migrations" / "migration_lock.toml")
    for name in migrations:
        shutil.copytree(PRISMA_DIR / "migrations" / name, older / "migrations" / name)
    subprocess.run(
        [sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(older / "schema.prisma")],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "DATABASE_URL": database_url, "DIRECT_URL": database_url},
    )


def applied_migrations(database_url: str) -> tuple[str, ...]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows: Final = connection.execute(
            'SELECT migration_name FROM "_prisma_migrations" '
            "WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL ORDER BY migration_name"
        ).fetchall()
    return tuple(str(row["migration_name"]) for row in rows)


def deploy_pending_migrations(database_url: str) -> None:
    entrypoint: Final = subprocess.run(
        [sys.executable, "-I", "-m", "litellm.proxy.prisma_migration"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": database_url, "DIRECT_URL": database_url},
    )
    assert entrypoint.returncode == 0, entrypoint.stdout + entrypoint.stderr


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
        older_migrations: Final = tuple(name for name in SHIPPED_MIGRATIONS if name < MISSING_MIGRATION)
        deploy_older_schema(database_url, tmp_path, older_migrations)
        assert object_permission_columns(database_url) == ()
        assert applied_migrations(database_url) == older_migrations
        deploy_pending_migrations(database_url)
        assert object_permission_columns(database_url) == ("mcp_tool_search_enabled",)
        assert applied_migrations(database_url) == SHIPPED_MIGRATIONS
        with owned_proxy(
            gateway, tmp_path, {"DATABASE_URL": database_url, "DISABLE_SCHEMA_UPDATE": "true"}
        ) as upgraded:
            tools: Final = upgraded.request("GET", "/mcp-rest/tools/list")
            assert tools.status_code == 200, tools.text
            assert tools.json()["tools"] == [], tools.text


def catalog_query_count(database_url: str) -> int:
    rows: Final = read_rows(
        "SELECT coalesce(sum(calls), 0)::bigint AS calls FROM pg_stat_statements "
        "WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database()) "
        "AND query LIKE %s",
        ('SELECT%FROM%"LiteLLM_MCPServerTable"%',),
        database_url=database_url,
    )
    calls: Final = rows[0]["calls"]
    assert isinstance(calls, int)
    return calls


def yaml_catalog_reloads(gateway: Gateway, database_url: str) -> int:
    initial_reloads: Final = catalog_query_count(database_url)
    for _ in range(3):
        tools: Final = gateway.request("GET", "/mcp-rest/tools/list", params={"mcp_server_name": "pages"})
        assert tools.status_code == 200, tools.text
        assert len(tools.json()["tools"]) == 3, tools.text
    return catalog_query_count(database_url) - initial_reloads


def test_upgrade_seeds_the_catalog_for_yaml_only_mcp_servers(
    gateway: Gateway, tmp_path: Path, record_property: Callable[[str, int], None]
) -> None:
    revision_sql: Final = 'SELECT "reload_revision" FROM "LiteLLM_Config" WHERE "param_name" = %s'
    catalog: Final = ("mcp_catalog",)
    with fresh_database() as database_url, paginated_mcp_peer(page_size=3) as peer:
        environment: Final = {"DATABASE_URL": database_url, "DISABLE_SCHEMA_UPDATE": "true"}
        server_id: Final = str(uuid.uuid4())
        older_migrations: Final = tuple(name for name in SHIPPED_MIGRATIONS if name != MCP_CATALOG_SEED)
        deploy_older_schema(database_url, tmp_path, older_migrations)
        config: Final = tmp_path / "yaml-mcp.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {"pages": {"url": peer.url, "transport": "http", "server_id": server_id}},
                    "general_settings": {
                        "master_key": gateway.key,
                        "store_model_in_db": True,
                        "proxy_config_reload_interval_seconds": 3600,
                    },
                }
            ),
            encoding="utf-8",
        )
        with psycopg.connect(database_url, autocommit=True) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            assert connection.execute(revision_sql, catalog).fetchone() is None
            with owned_proxy(
                gateway, tmp_path / "before", environment, config=config, database_setup=()
            ) as before_proxy:
                before: Final = yaml_catalog_reloads(before_proxy, database_url)
            record_property("catalog_queries_before_seed", before)
            assert before >= 3
            assert connection.execute(revision_sql, catalog).fetchone() is None
            deploy_pending_migrations(database_url)
            initial_revision: Final = connection.execute(revision_sql, catalog).fetchone()
            assert initial_revision is not None

            with owned_proxy(gateway, tmp_path / "after", environment, config=config, database_setup=()) as after_proxy:
                after: Final = yaml_catalog_reloads(after_proxy, database_url)
                record_property("catalog_queries_after_seed", after)
                assert after <= 1
                assert connection.execute(revision_sql, catalog).fetchone() == initial_revision
                connection.execute('INSERT INTO "LiteLLM_MCPServerOAuthClient" ("server_id") VALUES (%s)', (server_id,))
                assert connection.execute(revision_sql, catalog).fetchone() == (initial_revision[0] + 1,)
                after_oauth_write: Final = yaml_catalog_reloads(after_proxy, database_url)
                record_property("catalog_queries_after_oauth_write", after_oauth_write)
                assert after_oauth_write == 1
                assert yaml_catalog_reloads(after_proxy, database_url) == 0
                connection.execute(
                    'UPDATE "LiteLLM_MCPServerOAuthClient" SET "credentials" = \'{}\' WHERE "server_id" = %s',
                    (server_id,),
                )
                assert connection.execute(revision_sql, catalog).fetchone() == (initial_revision[0] + 2,)
                connection.execute('DELETE FROM "LiteLLM_MCPServerOAuthClient" WHERE "server_id" = %s', (server_id,))
                assert connection.execute(revision_sql, catalog).fetchone() == (initial_revision[0] + 3,)
            assert connection.execute('SELECT count(*) FROM "LiteLLM_MCPServerTable"').fetchone() == (0,)

            connection.execute('DELETE FROM "LiteLLM_MCPServerTable" WHERE FALSE')
            bumped_revision: Final = connection.execute(revision_sql, catalog).fetchone()
            assert bumped_revision == (initial_revision[0] + 4,)
            seed_sql: Final = (PRISMA_DIR / "migrations" / MCP_CATALOG_SEED / "migration.sql").read_text(
                encoding="utf-8"
            )
            with connection.transaction():
                connection.execute(seed_sql)
            assert connection.execute(revision_sql, catalog).fetchone() == bumped_revision

            connection.execute('ALTER TABLE "LiteLLM_MCPServerTable" DISABLE TRIGGER "litellm_mcp_catalog_revision"')
            connection.execute('DELETE FROM "LiteLLM_Config" WHERE "param_name" = %s', catalog)
            with connection.transaction():
                connection.execute(seed_sql)
            assert connection.execute(revision_sql, catalog).fetchone() is None
