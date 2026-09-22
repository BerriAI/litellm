import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from psycopg import sql

ROOT: Final = Path(__file__).resolve().parents[3]
EXTRAS: Final = ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
PARTITION_SCRIPT: Final = ROOT / "db_scripts" / "partition_spend_logs.sql"
INDEX_MIGRATION: Final = "20260831120001_spend_logs_litellm_call_id_index"
SEED_ROWS: Final = 500


@pytest.mark.timeout(180)
@pytest.mark.covers("other.database.partitions.upgrade_builds_call_id_index_on_partitioned_parent")
def test_upgrade_builds_the_call_id_index_on_a_partitioned_spend_log(gateway: Gateway, tmp_path: Path) -> None:
    database: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    database_url: Final = urlunsplit(parsed._replace(path=f"/{database}"))
    scoped_url: Final = urlunsplit(
        parsed._replace(path=f"/{database}", query=urlencode({**dict(parse_qsl(parsed.query)), "schema": "upgraded"}))
    )
    project: Final = tmp_path / "prisma"
    staged_migrations: Final = project / "migrations"
    staged_migrations.mkdir(parents=True)
    shutil.copy2(EXTRAS / "schema.prisma", project / "schema.prisma")
    shutil.copy2(EXTRAS / "migrations" / "migration_lock.toml", staged_migrations / "migration_lock.toml")
    earlier: Final = tuple(
        sorted(
            source for source in (EXTRAS / "migrations").iterdir() if source.is_dir() and source.name < "20260831120001"
        )
    )
    for source in earlier:
        shutil.copytree(source, staged_migrations / source.name)
    prisma_bin: Final = Path(sys.executable).with_name("prisma")
    assert prisma_bin.exists(), f"Prisma CLI missing next to the test interpreter {sys.executable}"
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        try:
            deploy: Final = subprocess.run(
                [str(prisma_bin), "migrate", "deploy", "--schema", str(project / "schema.prisma")],
                capture_output=True,
                text=True,
                env={**os.environ, "DATABASE_URL": scoped_url},
            )
            assert deploy.returncode == 0, f"prisma migrate deploy failed before the upgrade: {deploy.stderr}"
            with psycopg.connect(database_url, autocommit=True) as setup:
                setup.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier("upgraded")))
                setup.execute(PARTITION_SCRIPT.read_bytes())
                assert setup.execute(
                    "SELECT c.relkind::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'upgraded' AND c.relname = 'LiteLLM_SpendLogs'"
                ).fetchone() == ("p",), "The partition script did not leave a partitioned LiteLLM_SpendLogs behind"
                setup.execute(
                    'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", "endTime") '
                    "SELECT 'upgrade-shape-' || g, 'acompletion', now() - (g || ' seconds')::interval, "
                    "now() - (g || ' seconds')::interval FROM generate_series(1, %s) AS g",
                    (SEED_ROWS,),
                )
                assert setup.execute('SELECT count(*) FROM "LiteLLM_SpendLogs"').fetchone() == (SEED_ROWS,), (
                    "The seeded spend rows did not land on the partitioned table"
                )
                with owned_proxy(
                    gateway, tmp_path, {"DATABASE_URL": scoped_url}, use_prisma_db_push=False
                ) as candidate:
                    assert candidate.client.get("/health/readiness").status_code == 200, (
                        "The upgraded proxy did not become ready against the partitioned database"
                    )
                assert setup.execute(
                    "SELECT c.relkind::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'upgraded' AND c.relname = 'LiteLLM_SpendLogs'"
                ).fetchone() == ("p",), "The upgrade replaced the partitioned LiteLLM_SpendLogs with a plain table"
                assert setup.execute('SELECT count(*) FROM "LiteLLM_SpendLogs"').fetchone() == (SEED_ROWS,), (
                    "The upgrade lost spend rows from the partitioned table"
                )
                assert setup.execute(
                    "SELECT count(*) FROM pg_indexes WHERE schemaname = 'upgraded' "
                    "AND tablename IN ('LiteLLM_SpendLogs', 'LiteLLM_SpendLogs_pdefault') "
                    "AND indexdef LIKE '%(litellm_call_id)'"
                ).fetchone() == (2,), (
                    "The litellm_call_id index was not built on the partitioned parent and propagated to its partition"
                )
                assert setup.execute(
                    "SELECT finished_at IS NOT NULL FROM _prisma_migrations WHERE migration_name = %s",
                    (INDEX_MIGRATION,),
                ).fetchall() == [(True,)], "The upgrade did not record the litellm_call_id index migration as applied"
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
