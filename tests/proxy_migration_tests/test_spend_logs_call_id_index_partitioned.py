"""Regression for BerriAI/litellm#41548.

PostgreSQL rejects CREATE INDEX CONCURRENTLY on partitioned tables
(SQLSTATE 0A000). Migration 20260831120001 must use a plain CREATE INDEX
so prisma migrate deploy works when LiteLLM_SpendLogs is partitioned.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import pytest

psycopg = pytest.importorskip("psycopg")

requires_db: Final = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)

MIGRATION: Final = (
    Path(__file__).resolve().parents[2]
    / "litellm-proxy-extras"
    / "litellm_proxy_extras"
    / "migrations"
    / "20260831120001_spend_logs_litellm_call_id_index"
    / "migration.sql"
)

INDEX_SQL: Final = (
    'CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_litellm_call_id_idx" '
    'ON "LiteLLM_SpendLogs"("litellm_call_id");'
)


def _sql_statements(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("--")
    )


def test_spend_logs_call_id_index_migration_avoids_concurrently() -> None:
    text = MIGRATION.read_text()
    statements = _sql_statements(text)
    assert "CONCURRENTLY" not in statements.upper()
    assert statements.strip() == INDEX_SQL


@requires_db
def test_spend_logs_call_id_index_builds_on_partitioned_table() -> None:
    """The migration statement must succeed on a partitioned parent (issue #41548)."""
    admin_url = os.environ["DATABASE_URL"].split("?")[0]
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute('DROP TABLE IF EXISTS "LiteLLM_SpendLogs" CASCADE')
        conn.execute(
            """
            CREATE TABLE "LiteLLM_SpendLogs" (
                "request_id" TEXT NOT NULL,
                "startTime" TIMESTAMP(3) NOT NULL,
                "litellm_call_id" TEXT
            ) PARTITION BY RANGE ("startTime")
            """
        )
        conn.execute(
            """
            CREATE TABLE "LiteLLM_SpendLogs_pdefault"
                PARTITION OF "LiteLLM_SpendLogs" DEFAULT
            """
        )
        conn.execute(_sql_statements(MIGRATION.read_text()))
        rows = conn.execute(
            """
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'LiteLLM_SpendLogs'
              AND indexname = 'LiteLLM_SpendLogs_litellm_call_id_idx'
            """
        ).fetchall()
        assert rows
        conn.execute('DROP TABLE IF EXISTS "LiteLLM_SpendLogs" CASCADE')
