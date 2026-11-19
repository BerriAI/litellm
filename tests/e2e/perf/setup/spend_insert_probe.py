from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Final
from uuid import uuid4

import psycopg
from psycopg import sql
from pydantic import TypeAdapter

from setup.datastore import DbCounters, statement_deltas

_PLAN: Final = TypeAdapter(list[dict[str, object]])
_ROWS: Final = TypeAdapter(list[tuple[dict[str, object], str | None, str | None]])
_NODE: Final = TypeAdapter(dict[str, object])
_NODES: Final = TypeAdapter(list[dict[str, object]])
_FLOAT: Final = TypeAdapter(float)


def spend_insert_probe(
    before: DbCounters | None, after: DbCounters | None, started: datetime, wall_seconds: float
) -> tuple[tuple[str, str], ...]:
    writes: Final = tuple(
        statement for statement in statement_deltas(before, after) or ()
        if statement.table == "LiteLLM_SpendLogs" and statement.query.lstrip().upper().startswith("INSERT")
    )
    total: Final = sum(write.total_exec_ms for write in writes)
    calls: Final = sum(write.calls for write in writes)
    summary: Final = (
        ("perf_probe_spendlogs_insert_calls", str(calls)),
        ("perf_probe_spendlogs_insert_mean_ms", str(total / calls if calls else 0)),
        ("perf_probe_spendlogs_insert_active_sessions", str(total / 1000 / wall_seconds)),
    )
    try:
        return (*summary, ("perf_probe_spendlogs_plan", json.dumps(_profile(started))))
    except psycopg.Error as error:
        return (*summary, ("perf_probe_spendlogs_plan_error", type(error).__name__))


def _profile(started: datetime) -> tuple[dict[str, object], ...]:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        _ = conn.execute("SET statement_timeout = '5s'")
        rows: Final = _ROWS.validate_python(conn.execute(
            'SELECT to_jsonb(s), pg_column_compression(messages), pg_column_compression(proxy_server_request) '
            'FROM "LiteLLM_SpendLogs" AS s WHERE "startTime" >= %s ORDER BY md5(request_id) LIMIT 24',
            (started.replace(tzinfo=None),),
        ).fetchall())
        return tuple(_explain(conn, rows[offset:offset + 2]) for offset in range(0, len(rows), 2))


def _explain(
    conn: psycopg.Connection[tuple[object, ...]], rows: list[tuple[dict[str, object], str | None, str | None]]
) -> dict[str, object]:
    columns: Final = sql.SQL(", ").join(sql.Identifier(key) for key in rows[0][0])
    query: Final = sql.SQL(
        'EXPLAIN (ANALYZE, BUFFERS, WAL, FORMAT JSON) '
        'INSERT INTO "LiteLLM_SpendLogs" ({columns}) SELECT {columns} '
        'FROM jsonb_populate_recordset(NULL::"LiteLLM_SpendLogs", %s::jsonb) '
        'ON CONFLICT DO NOTHING RETURNING request_id'
    ).format(columns=columns)
    body: Final = json.dumps([dict(row, request_id=f"perf-insert-probe-{uuid4()}") for row, _, _ in rows])
    with conn.transaction(force_rollback=True):
        result: Final = conn.execute(query, (body,)).fetchone()
        assert result is not None
        plan: Final = _PLAN.validate_python(result[0])[0]
        root: Final = _NODE.validate_python(plan["Plan"])
        scan: Final = _NODES.validate_python(root["Plans"])[0]
        return {
            "rows": len(rows),
            "input_bytes": len(body.encode()),
            "compression": [[messages, request] for _, messages, request in rows],
            "planning_ms": plan["Planning Time"],
            "execution_ms": plan["Execution Time"],
            "recordset_ms": _FLOAT.validate_python(scan["Actual Total Time"]),
            "insert_wal_bytes": root["WAL Bytes"],
            "inserted": root["Tuples Inserted"],
        }
