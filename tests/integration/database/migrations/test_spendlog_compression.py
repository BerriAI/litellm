import os
from pathlib import Path
from typing import Final

import psycopg

_MIGRATION: Final = (
    Path(__file__).resolve().parents[4]
    / "litellm/proxy-extras/litellm_proxy_extras/migrations/20261118140000_spendlogs_lz4_compression/migration.sql"
)


def test_spendlog_migration_compresses_new_payloads_without_rewriting_old_rows() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            'CREATE TABLE "LiteLLM_SpendLogs" ('
            "request_id integer PRIMARY KEY, metadata jsonb, messages jsonb, response jsonb, proxy_server_request jsonb)"
        )
        supports_lz4: Final = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_settings WHERE name = 'default_toast_compression' AND 'lz4' = ANY(enumvals))"
        ).fetchone() == (True,)
        if supports_lz4:
            connection.execute("SET LOCAL default_toast_compression = pglz")
        insert: Final = (
            'INSERT INTO "LiteLLM_SpendLogs" '
            "SELECT %s, payload, payload, payload, payload "
            "FROM (SELECT jsonb_build_object('body', repeat('abcde', 10000)) AS payload) AS input"
        )
        connection.execute(insert, (1,))
        connection.execute(_MIGRATION.read_text())
        connection.execute(insert, (2,))
        payload: Final = {"body": "abcde" * 10000}
        assert connection.execute(
            'SELECT request_id, metadata, messages, response, proxy_server_request FROM "LiteLLM_SpendLogs" '
            "ORDER BY request_id"
        ).fetchall() == [(row_id, payload, payload, payload, payload) for row_id in (1, 2)]
        if supports_lz4:
            assert connection.execute(
                "SELECT request_id, pg_column_compression(metadata), pg_column_compression(messages), "
                'pg_column_compression(response), pg_column_compression(proxy_server_request) FROM "LiteLLM_SpendLogs" '
                "ORDER BY request_id"
            ).fetchall() == [(1, "pglz", "pglz", "pglz", "pglz"), (2, "lz4", "lz4", "lz4", "lz4")]
