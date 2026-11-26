import json
import os
from datetime import datetime, timezone
from typing import Final
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from pydantic import SecretStr

from litellm.integrations.postgres_olap import PostgresOlapConfig, PostgresOlapLogger


def test_pg_protocol_batch_roundtrip_with_quoted_identifiers_and_content() -> None:
    dsn: Final = os.environ["DATABASE_URL"]
    table: Final = "logs_" + uuid4().hex + '"quoted'
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload: Final = {
        "id": "request-'中文",
        "model": "synthetic",
        "total_tokens": 5,
        "response_cost": 0.125,
        "messages": [{"role": "user", "content": "'quoted 中文"}],
    }
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("""CREATE TABLE {} (
            event_id TEXT NOT NULL, request_id TEXT, status TEXT, model TEXT,
            start_time TIMESTAMPTZ, end_time TIMESTAMPTZ, prompt_tokens BIGINT,
            completion_tokens BIGINT, total_tokens BIGINT, response_cost DOUBLE PRECISION, payload TEXT
        )""").format(sql.Identifier(table))
        )
        logger: Final = PostgresOlapLogger(
            PostgresOlapConfig(
                dsn=SecretStr(dsn),
                table_name=table,
                batch_size=2,
                turn_off_message_logging=False,
            )
        )
        try:
            logger.log_success_event({"standard_logging_object": {**payload, "status": "success"}}, None, now, now)
            logger.log_failure_event({"standard_logging_object": {**payload, "status": "failure"}}, None, now, now)
            assert logger.flush(20)
            assert logger.statistics == {"accepted": 2, "delivered": 2, "dropped": 0, "pending": 0}
            rows: Final = connection.execute(
                sql.SQL("SELECT request_id,status,total_tokens,response_cost,payload FROM {} ORDER BY status").format(
                    sql.Identifier(table)
                )
            ).fetchall()
            assert len(rows) == 2
            assert tuple(row[:4] for row in rows) == (
                (payload["id"], "failure", 5, 0.125),
                (payload["id"], "success", 5, 0.125),
            )
            assert tuple(json.loads(row[4]) for row in rows) == (
                {**payload, "status": "failure"},
                {**payload, "status": "success"},
            )
        finally:
            logger.close()
            connection.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))


def test_missing_table_reports_dropped_batch_without_payload_disclosure(caplog: pytest.LogCaptureFixture) -> None:
    logger: Final = PostgresOlapLogger(
        PostgresOlapConfig(
            dsn=SecretStr(os.environ["DATABASE_URL"]),
            table_name="missing_" + uuid4().hex,
            retries=0,
        )
    )
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    try:
        logger.log_success_event({"standard_logging_object": {"id": "private-synthetic-id"}}, None, now, now)
        assert logger.flush(20)
        assert logger.statistics == {"accepted": 1, "delivered": 0, "dropped": 1, "pending": 0}
        assert "private-synthetic-id" not in caplog.text
    finally:
        logger.close()
