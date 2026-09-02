import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

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


requires_db: Final = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)

MIGRATIONS_DIR: Final = Path("./litellm-proxy-extras/litellm_proxy_extras/migrations")

SPEND_LOGS_FILTER_INDEXES: Final = (
    ('"user"', "user-7", "LiteLLM_SpendLogs_user_startTime_idx"),
    ("api_key", "key-7", "LiteLLM_SpendLogs_api_key_startTime_idx"),
)

SPEND_LOGS_PROBE_SQL: Final = """
CREATE TABLE "LiteLLM_SpendLogs" (
    request_id TEXT PRIMARY KEY,
    api_key TEXT NOT NULL DEFAULT '',
    "startTime" TIMESTAMP(3) NOT NULL,
    "user" TEXT DEFAULT ''
);
CREATE INDEX "LiteLLM_SpendLogs_startTime_idx" ON "LiteLLM_SpendLogs"("startTime");
CREATE INDEX "LiteLLM_SpendLogs_startTime_request_id_idx" ON "LiteLLM_SpendLogs"("startTime", "request_id");
INSERT INTO "LiteLLM_SpendLogs" (request_id, api_key, "startTime", "user")
SELECT 'req-' || g, 'key-' || (g % 200), TIMESTAMP '2026-08-01' + g * INTERVAL '1 minute', 'user-' || (g % 200)
FROM generate_series(1, 60000) AS g;
ANALYZE "LiteLLM_SpendLogs";
"""

SPEND_LOGS_PARTITIONED_PROBE_SQL: Final = """
CREATE TABLE "LiteLLM_SpendLogs" (
    request_id TEXT NOT NULL,
    api_key TEXT NOT NULL DEFAULT '',
    "startTime" TIMESTAMP(3) NOT NULL,
    "user" TEXT DEFAULT '',
    PRIMARY KEY (request_id, "startTime")
) PARTITION BY RANGE ("startTime");
CREATE TABLE "LiteLLM_SpendLogs_pdefault" PARTITION OF "LiteLLM_SpendLogs" DEFAULT;
"""


def _in_schema(schema: str, sql: str) -> list[tuple[object, ...]]:
    psycopg: Final = pytest.importorskip("psycopg")
    with psycopg.connect(os.environ["DATABASE_URL"].split("?")[0], autocommit=True) as conn:
        conn.execute(f'SET search_path TO "{schema}"')
        cursor: Final = conn.execute(sql)
        return cursor.fetchall() if cursor.description is not None else []


def _migration_creating(index_name: str) -> str:
    creating: Final = tuple(
        sql
        for sql in (path.read_text() for path in sorted(MIGRATIONS_DIR.glob("*/migration.sql")))
        if index_name in sql
    )
    assert len(creating) == 1, f"expected exactly one migration creating {index_name}, found {len(creating)}"
    return creating[0]


def _bounded_count_plan(schema: str, column: str, value: str) -> str:
    rows: Final = _in_schema(
        schema,
        f"""
        EXPLAIN (FORMAT JSON)
        SELECT COUNT(*) AS total_count
        FROM (
            SELECT 1
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" >= ('2026-08-10T00:00:00Z'::timestamptz AT TIME ZONE 'UTC')
              AND "startTime" <= ('2026-08-17T00:00:00Z'::timestamptz AT TIME ZONE 'UTC')
              AND {column} = '{value}'
            LIMIT 10001
        ) AS bounded_matches
        """,
    )
    return json.dumps(rows[0][0])


@pytest.fixture
def spend_logs_scratch_schema() -> Iterator[str]:
    schema: Final = f"spend_logs_idx_{uuid.uuid4().hex[:8]}"
    _in_schema("public", f'CREATE SCHEMA "{schema}"')
    yield schema
    _in_schema("public", f'DROP SCHEMA "{schema}" CASCADE')


@requires_db
@pytest.mark.parametrize(("column", "value", "index_name"), SPEND_LOGS_FILTER_INDEXES, ids=("user", "api_key"))
def test_spend_logs_bounded_count_is_served_by_the_filter_column_index(
    spend_logs_scratch_schema: str, column: str, value: str, index_name: str
) -> None:
    """The /spend/logs/v2 count for one user or key inside a startTime window must read that
    user's rows only. Without a filter-column-led index it range-scans the whole window and
    post-filters every row, which times out on multi-day windows at production volume.
    """
    _in_schema(spend_logs_scratch_schema, SPEND_LOGS_PROBE_SQL)
    _in_schema(spend_logs_scratch_schema, _migration_creating(index_name))
    _in_schema(spend_logs_scratch_schema, 'ANALYZE "LiteLLM_SpendLogs"')

    plan: Final = _bounded_count_plan(spend_logs_scratch_schema, column, value)

    assert f'"Index Name": "{index_name}"' in plan, plan
    assert '"Filter"' not in plan, plan


@requires_db
def test_spend_logs_filter_index_migrations_apply_to_a_partitioned_table(spend_logs_scratch_schema: str) -> None:
    """db_scripts/partition_spend_logs.sql leaves LiteLLM_SpendLogs a partitioned parent, and
    Postgres rejects CONCURRENTLY there, so a concurrent build would abort every partitioned
    deployment's rollout. Apply the shipped statements against that shape for real.
    """
    _in_schema(spend_logs_scratch_schema, SPEND_LOGS_PARTITIONED_PROBE_SQL)
    for _column, _value, index_name in SPEND_LOGS_FILTER_INDEXES:
        _in_schema(spend_logs_scratch_schema, _migration_creating(index_name))

    parent_indexes: Final = frozenset(
        str(row[0])
        for row in _in_schema(
            spend_logs_scratch_schema,
            "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() AND tablename = 'LiteLLM_SpendLogs'",
        )
    )

    assert {index_name for _column, _value, index_name in SPEND_LOGS_FILTER_INDEXES} <= parent_indexes, parent_indexes
