"""
Tests for the agent tracing ClickHouse DDL.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.clickhouse.schema import (
    AGENT_TRACES_TABLE,
    OTEL_TRACES_TABLE,
    SPEND_LOGS_TABLE,
    schema_statements,
)


def test_schema_statements_cover_all_tables_and_retention():
    statements = schema_statements("litellm", trace_retention_days=7, spend_log_retention_days=45)
    ddl = "\n".join(statements)

    for table in (OTEL_TRACES_TABLE, AGENT_TRACES_TABLE, SPEND_LOGS_TABLE):
        assert f"litellm.{table}" in ddl
    assert f"litellm.{AGENT_TRACES_TABLE}_mv TO litellm.{AGENT_TRACES_TABLE}" in ddl
    assert "INTERVAL 7 DAY" in ddl
    assert "INTERVAL 45 DAY" in ddl


def test_otel_traces_keeps_collector_compatible_columns():
    otel_ddl = next(
        s for s in schema_statements("litellm", 30, 90) if f"TABLE IF NOT EXISTS litellm.{OTEL_TRACES_TABLE}" in s
    )
    for column in (
        "Timestamp",
        "TraceId",
        "SpanId",
        "ParentSpanId",
        "SpanName",
        "ServiceName",
        "ResourceAttributes",
        "SpanAttributes",
        "Duration",
        "StatusCode",
    ):
        assert f"{column} " in otel_ddl
    # join key to spend logs
    assert "LiteLLMRequestId" in otel_ddl
