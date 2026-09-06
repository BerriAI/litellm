#!/usr/bin/env python3
"""Preview or execute the optional team_id backfill using the existing proxy runtime."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Final, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
from psycopg import Connection, sql
from psycopg.rows import class_row

Row = tuple[object, ...]
Emit = Callable[[Mapping[str, object]], None]
Status = Literal["pending", "global", "already_set", "invalid_metadata", "invalid_team_id", "conflict", "missing_team"]
PRISMA_OPTIONS: Final = frozenset({"schema", "connection_limit", "pool_timeout", "pgbouncer"})


@dataclass(frozen=True, slots=True)
class Options:
    execute: bool = False
    schema: str | None = None
    batch_size: int = 500
    after_model_id: str | None = None
    max_batches: int | None = None
    lock_timeout_ms: int = 2000
    statement_timeout_ms: int = 30000


@dataclass(frozen=True, slots=True)
class Snapshot:
    model_id: str
    model_name: str
    raw_info: str | None
    team_id: str | None


@dataclass(frozen=True, slots=True)
class Decision:
    row: Snapshot
    status: Status
    proposed_team_id: str | None = None


@dataclass(frozen=True, slots=True)
class Progress:
    scanned: int = 0
    pending: int = 0
    updated: int = 0
    issues: int = 0
    batches: int = 0
    cursor: str | None = None


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    data_type: str


@dataclass(frozen=True, slots=True)
class Target:
    schema: str
    replica: bool
    read_only: bool


def classify(row: Snapshot) -> Decision:
    try:
        decoded: Final[object] = (
            json.loads(row.raw_info, object_pairs_hook=unique_object) if row.raw_info is not None else None
        )
        info: Final[object] = (
            json.loads(decoded, object_pairs_hook=unique_object) if isinstance(decoded, str) else decoded
        )
    except (ValueError, RecursionError):
        return Decision(row, "invalid_metadata")
    if info is not None and not isinstance(info, dict):
        return Decision(row, "invalid_metadata")
    metadata: Final = cast("dict[str, object] | None", info)  # cast-ok: decoded JSON object keys are strings
    owner: Final = metadata.get("team_id") if metadata is not None else None
    if owner is None and metadata is not None and "team_public_model_name" in metadata:
        return Decision(row, "invalid_team_id")
    if owner is not None and (
        not isinstance(owner, str)
        or not owner.strip()
        or "\x00" in owner
        or any(0xD800 <= ord(character) <= 0xDFFF for character in owner)
    ):
        return Decision(row, "invalid_team_id")
    proposed: Final = owner if isinstance(owner, str) else None
    if row.team_id is not None and row.team_id != proposed:
        return Decision(row, "conflict", proposed)
    if proposed is None:
        return Decision(row, "global")
    return Decision(row, "already_set" if row.team_id is not None else "pending", proposed)


def unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: Final = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate metadata keys")
    return result


def connection_parameters(dsn: str, schema: str | None) -> tuple[str, str | None]:
    if not dsn.startswith(("postgres://", "postgresql://")):
        return dsn, schema
    parsed: Final = urlsplit(dsn)
    pairs: Final = tuple(parse_qsl(parsed.query, keep_blank_values=True))
    schemas: Final = tuple(value for key, value in pairs if key == "schema")
    if len(schemas) > 1 or (schemas and (not schemas[0] or (schema is not None and schema != schemas[0]))):
        raise ValueError("Conflicting or empty schema settings")
    cleaned: Final = urlunsplit(parsed._replace(query=urlencode(tuple(p for p in pairs if p[0] not in PRISMA_OPTIONS))))
    return cleaned, schema if schema is not None else (schemas[0] if schemas else None)


def _one(conn: Connection[Row], query: sql.SQL | sql.Composed, params: Sequence[object] = ()) -> Row:
    result: Final = conn.execute(query, params).fetchone()
    if result is None:
        raise ValueError("Expected database result")
    return result


def inspect_target(conn: Connection[Row], options: Options) -> Target:
    observed: Final = _one(
        conn, sql.SQL("SELECT current_schema(), pg_is_in_recovery(), current_setting('transaction_read_only')")
    )
    schema: Final = options.schema if options.schema is not None else observed[0]
    if not isinstance(schema, str) or not schema or "\x00" in schema:
        raise ValueError("No valid schema selected")
    return Target(schema, observed[1] is True, observed[2] == "on")


def check_schema(conn: Connection[Row], schema: str) -> None:
    for table, expected in (
        (
            "LiteLLM_ProxyModelTable",
            {"model_id": {"text"}, "model_name": {"text"}, "model_info": {"json", "jsonb"}, "team_id": {"text"}},
        ),
        ("LiteLLM_TeamTable", {"team_id": {"text"}}),
    ):
        check_table(conn, schema, table, expected)


def check_table(conn: Connection[Row], schema: str, table: str, expected: Mapping[str, set[str]]) -> None:
    relation: Final = sql.Identifier(schema, table).as_string(conn)
    flags: Final = _one(
        conn,
        sql.SQL("SELECT c.relkind, row_security_active(c.oid) FROM pg_class c WHERE c.oid = to_regclass(%s)"),
        (relation,),
    )
    if flags[0] != "r" or flags[1] is not False:
        raise ValueError("Expected an ordinary table with unrestricted row visibility")
    with conn.cursor(row_factory=class_row(Column)) as cursor:
        cursor.execute(
            "SELECT attname AS name, typname AS data_type FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid "
            "WHERE attrelid=to_regclass(%s) AND attnum>0 AND NOT attisdropped",
            (relation,),
        )
        columns: Final = {column.name: column.data_type for column in cursor.fetchall()}
    if any(columns.get(name) not in accepted for name, accepted in expected.items()):
        raise ValueError("Required columns or types are missing; apply the schema migration first")
    id_column: Final = "model_id" if table == "LiteLLM_ProxyModelTable" else "team_id"
    unique: Final = _one(
        conn,
        sql.SQL(
            "SELECT EXISTS (SELECT 1 FROM pg_index i JOIN pg_attribute a "
            "ON a.attrelid=i.indrelid AND a.attnum=i.indkey[0] "
            "WHERE i.indrelid=to_regclass(%s) AND i.indisunique AND i.indisvalid "
            "AND i.indnkeyatts=1 AND i.indpred IS NULL AND a.attname=%s AND a.attnotnull)"
        ),
        (relation, id_column),
    )
    if unique[0] is not True:
        raise ValueError("Expected a non-null unique identifier")


def read_batch(
    conn: Connection[Row], table: sql.Identifier, after: str | None, upper: str, batch_size: int
) -> tuple[Snapshot, ...]:
    query: Final = sql.SQL(
        'SELECT "model_id", "model_name", "model_info"::text AS raw_info, "team_id" FROM {} '
        'WHERE (%s::text IS NULL OR "model_id" > %s) AND "model_id" <= %s ORDER BY "model_id" LIMIT %s'
    ).format(table)
    with conn.cursor(row_factory=class_row(Snapshot)) as cursor:
        cursor.execute(query, (after, after, upper, batch_size))
        return tuple(cursor.fetchall())


def check_teams(conn: Connection[Row], table: sql.Identifier, decisions: tuple[Decision, ...]) -> tuple[Decision, ...]:
    owners: Final = tuple({d.proposed_team_id for d in decisions if d.status in {"pending", "already_set"}})
    if not owners:
        return decisions
    existing: Final = frozenset(
        row[0]
        for row in conn.execute(
            sql.SQL('SELECT "team_id" FROM {} WHERE "team_id" = ANY(%s)').format(table), (list(owners),)
        )
    )
    return tuple(
        replace(d, status="missing_team")
        if d.status in {"pending", "already_set"} and d.proposed_team_id not in existing
        else d
        for d in decisions
    )


def execute_batch(
    conn: Connection[Row], models: sql.Identifier, teams: sql.Identifier, decisions: tuple[Decision, ...]
) -> frozenset[str]:
    candidates: Final = tuple(d for d in decisions if d.status == "pending")
    if not candidates:
        return frozenset()
    values: Final = sql.SQL(", ").join(sql.SQL("(%s::text, %s::text, %s::text)") for _ in candidates)
    query: Final = sql.SQL(
        'UPDATE {} AS m SET "team_id" = v.team_id FROM (VALUES {}) AS v(model_id, raw_info, team_id) '
        'WHERE m."model_id" = v.model_id AND m."team_id" IS NULL '
        'AND m."model_info"::text IS NOT DISTINCT FROM v.raw_info '
        'AND EXISTS (SELECT 1 FROM {} t WHERE t."team_id" = v.team_id) RETURNING m."model_id"'
    ).format(models, values, teams)
    params: Final = tuple(value for d in candidates for value in (d.row.model_id, d.row.raw_info, d.proposed_team_id))
    return frozenset(str(row[0]) for row in conn.execute(query, params))


def report_batch(decisions: tuple[Decision, ...], updated: frozenset[str], execute: bool, emit: Emit) -> int:
    for decision in decisions:
        emit(model_event(decision, updated, execute))
    return sum(d.status not in {"pending", "global", "already_set"} for d in decisions) + (
        sum(d.status == "pending" for d in decisions) - len(updated) if execute else 0
    )


def model_event(decision: Decision, updated: frozenset[str], execute: bool) -> Mapping[str, object]:
    status: Final = (
        ("updated" if decision.row.model_id in updated else "concurrent_change")
        if execute and decision.status == "pending"
        else decision.status
    )
    return {
        "event": "model",
        "model_id": decision.row.model_id,
        "model_name": decision.row.model_name,
        "status": status,
        "from_team_id": decision.row.team_id,
        "to_team_id": decision.proposed_team_id,
        "action": "set_team_id" if status in {"pending", "updated"} else "none",
    }


def process_batch(
    conn: Connection[Row],
    models: sql.Identifier,
    teams: sql.Identifier,
    state: Progress,
    upper: str,
    options: Options,
    emit: Emit,
) -> Progress | None:
    with conn.transaction():
        if not options.execute:
            conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SELECT set_config('lock_timeout', %s, true)", (f"{options.lock_timeout_ms}ms",))
        conn.execute("SELECT set_config('statement_timeout', %s, true)", (f"{options.statement_timeout_ms}ms",))
        rows: Final = read_batch(conn, models, state.cursor, upper, options.batch_size)
        decisions: Final = check_teams(conn, teams, tuple(classify(row) for row in rows))
        updated: Final = execute_batch(conn, models, teams, decisions) if options.execute else frozenset[str]()
    if not rows:
        return None
    issues: Final = report_batch(decisions, updated, options.execute, emit)
    return Progress(
        scanned=state.scanned + len(rows),
        pending=state.pending + sum(d.status == "pending" for d in decisions),
        updated=state.updated + len(updated),
        issues=state.issues + issues,
        batches=state.batches + 1,
        cursor=rows[-1].model_id,
    )


def run(conn: Connection[Row], options: Options, emit: Emit) -> int:
    target: Final = inspect_target(conn, options)
    emit({"event": "target", "mode": "execute" if options.execute else "dry_run", **asdict(target)})
    if options.execute and (target.replica or target.read_only):
        emit({"event": "error", "reason": "execute_requires_writable_primary"})
        return 1
    if not options.execute:
        conn.execute("SET default_transaction_read_only = on")
    conn.execute("SELECT set_config('lock_timeout', %s, false)", (f"{options.lock_timeout_ms}ms",))
    conn.execute("SELECT set_config('statement_timeout', %s, false)", (f"{options.statement_timeout_ms}ms",))
    check_schema(conn, target.schema)
    models: Final = sql.Identifier(target.schema, "LiteLLM_ProxyModelTable")
    teams: Final = sql.Identifier(target.schema, "LiteLLM_TeamTable")
    upper_row: Final = conn.execute(
        sql.SQL('SELECT "model_id" FROM {} ORDER BY "model_id" DESC LIMIT 1').format(models)
    ).fetchone()
    upper: Final = str(upper_row[0]) if upper_row else None
    state = Progress(
        cursor=options.after_model_id
    )  # rebind-ok: bounded scan progress advances after each committed batch
    while upper is not None:
        if (next_state := process_batch(conn, models, teams, state, upper, options, emit)) is None:
            break
        state = next_state  # rebind-ok: only committed batches contribute to progress
        emit({"event": "progress", **asdict(state)})
        if options.max_batches is not None and state.batches >= options.max_batches and state.cursor != upper:
            emit({"event": "summary", "scan_finished": False, **asdict(state)})
            return 2 if state.issues else 3
    emit({"event": "summary", "scan_finished": True, "resumed": options.after_model_id is not None, **asdict(state)})
    return 2 if state.issues else 0


def positive_int(value: str) -> int:
    parsed: Final = int(value)
    if not 1 <= parsed <= 2_147_483_647:
        raise argparse.ArgumentTypeError("Must be between 1 and 2147483647")
    return parsed


def batch_size(value: str) -> int:
    parsed: Final = positive_int(value)
    if parsed > 10000:
        raise argparse.ArgumentTypeError("Batch size must not exceed 10000")
    return parsed


def emit_json(event: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(dict(event), ensure_ascii=True) + "\n")
    sys.stdout.flush()


class Arguments(argparse.Namespace):
    execute: bool
    schema: str | None
    batch_size: int
    after_model_id: str | None
    max_batches: int | None
    lock_timeout_ms: int
    statement_timeout_ms: int


def main(argv: Sequence[str] | None = None) -> int:
    driver_logger: Final = logging.getLogger("psycopg")
    driver_logger.addHandler(logging.NullHandler())
    driver_logger.propagate = False
    parser: Final = argparse.ArgumentParser(description=__doc__)
    mode: Final = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Apply changes; the default is a read-only dry run")
    mode.add_argument("--dry-run", action="store_true", help="Explicitly request the default read-only preview")
    parser.add_argument("--schema", help="Database schema; otherwise use DATABASE_URL's schema or current_schema()")
    parser.add_argument("--batch-size", type=batch_size, default=500)
    parser.add_argument(
        "--after-model-id", help="Resume after a committed progress cursor; rerun a full dry run afterward"
    )
    parser.add_argument(
        "--max-batches", type=positive_int, help="Stop after this many batches and print a resume cursor"
    )
    parser.add_argument("--lock-timeout-ms", type=positive_int, default=2000)
    parser.add_argument("--statement-timeout-ms", type=positive_int, default=30000)
    args: Final = parser.parse_args(argv, namespace=Arguments())
    dsn: Final = os.environ.get("DATABASE_URL")
    if not dsn:
        emit_json({"event": "error", "reason": "DATABASE_URL_is_required"})
        return 1
    try:
        connection_string, schema = connection_parameters(dsn, args.schema)
        options: Final = Options(
            execute=args.execute,
            schema=schema,
            batch_size=args.batch_size,
            after_model_id=args.after_model_id,
            max_batches=args.max_batches,
            lock_timeout_ms=args.lock_timeout_ms,
            statement_timeout_ms=args.statement_timeout_ms,
        )
        with psycopg.connect(connection_string, autocommit=True, connect_timeout=10) as conn:
            return run(conn, options, emit_json)
    except KeyboardInterrupt:
        emit_json({"event": "error", "reason": "interrupted_rerun_safely"})
        return 130
    except (psycopg.Error, ValueError, OSError) as error:
        emit_json(
            {
                "event": "error",
                "reason": "database_or_configuration_error_rerun_safely",
                "sqlstate": error.sqlstate if isinstance(error, psycopg.Error) else None,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
