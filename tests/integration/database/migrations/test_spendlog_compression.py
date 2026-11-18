import os
from pathlib import Path
from typing import Final

import psycopg
import pytest

_MIGRATION: Final = (
    Path(__file__).resolve().parents[4]
    / "litellm/proxy-extras/litellm_proxy_extras/migrations/20261118140000_spendlogs_lz4_compression/migration.sql"
)


@pytest.mark.parametrize("partitioned", (False, True), ids=("plain", "partitioned"))
def test_spendlog_migration_compresses_new_payloads_without_rewriting_old_rows(partitioned: bool) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        partition_by: Final = " PARTITION BY RANGE (request_id)" if partitioned else ""
        connection.execute(
            'CREATE TABLE "LiteLLM_SpendLogs" ('
            "request_id integer PRIMARY KEY, metadata jsonb, messages jsonb, response jsonb, proxy_server_request jsonb)"
            + partition_by
        )
        if partitioned:
            connection.execute(
                'CREATE TABLE spend_existing PARTITION OF "LiteLLM_SpendLogs" '
                "FOR VALUES FROM (1) TO (3) PARTITION BY RANGE (request_id)"
            )
            connection.execute("CREATE TABLE spend_leaf PARTITION OF spend_existing FOR VALUES FROM (1) TO (3)")
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
        if partitioned:
            connection.execute('CREATE TABLE spend_future PARTITION OF "LiteLLM_SpendLogs" FOR VALUES FROM (3) TO (4)')
        connection.execute(insert, (2,))
        connection.execute(insert, (3,))
        payload: Final = {"body": "abcde" * 10000}
        assert connection.execute(
            'SELECT request_id, metadata, messages, response, proxy_server_request FROM "LiteLLM_SpendLogs" '
            "ORDER BY request_id"
        ).fetchall() == [(row_id, payload, payload, payload, payload) for row_id in (1, 2, 3)]
        if supports_lz4:
            assert connection.execute(
                "SELECT request_id, pg_column_compression(metadata), pg_column_compression(messages), "
                'pg_column_compression(response), pg_column_compression(proxy_server_request) FROM "LiteLLM_SpendLogs" '
                "ORDER BY request_id"
            ).fetchall() == [
                (1, "pglz", "pglz", "pglz", "pglz"),
                (2, "lz4", "lz4", "lz4", "lz4"),
                (3, "lz4", "lz4", "lz4", "lz4"),
            ]
