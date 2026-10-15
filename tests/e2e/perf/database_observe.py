from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime
from threading import Event
from typing import Final

import psycopg
from pydantic import BaseModel, ConfigDict, TypeAdapter

from .database_queries import LogRow
from .database_seed import SeedManifest, scalar

INTEGER: Final = TypeAdapter(int)
NUMBER: Final = TypeAdapter(float)
TEXT: Final = TypeAdapter(str)
FINGERPRINTS: Final = TypeAdapter(tuple[tuple[str, str], ...])


class Migration(BaseModel):
    model_config = ConfigDict(frozen=True)

    migration_name: str
    checksum: str
    started_at: datetime
    finished_at: datetime | None
    rolled_back_at: datetime | None


class DatabaseSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    started: float
    blocked_sessions: int
    longest_lock_wait_seconds: float
    read_seconds: float
    failure: str | None = None


class IndexSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    observed: float
    pending: tuple[str, ...]
    active_builds: int
    target_probe_persisted: bool


def index_sample(
    connection: psycopg.Connection[tuple[object, ...]], names: tuple[str, ...], request_id: str | None = None
) -> IndexSample:
    written: Final = request_id is not None and persisted(connection, (request_id,)) == 1
    rows: Final = connection.execute(
        "SELECT name FROM unnest(%s::text[]) name LEFT JOIN pg_index i "
        "ON i.indexrelid = to_regclass(format('%%I.%%I', current_schema(), name)) "
        "WHERE i.indisvalid IS DISTINCT FROM true ORDER BY name",
        (list(names),),
    ).fetchall()
    active: Final = scalar(
        connection,
        b"SELECT count(*) FROM pg_stat_progress_create_index "
        b"WHERE datname = current_database() AND index_relid IN "
        b"(SELECT to_regclass(format('%%I.%%I', current_schema(), name)) "
        b"FROM unnest(%s::text[]) name)",
        (list(names),),
    )
    return IndexSample(
        observed=time.time(),
        pending=tuple(TEXT.validate_python(row[0]) for row in rows),
        active_builds=INTEGER.validate_python(active),
        target_probe_persisted=written,
    )


def manifest(connection: psycopg.Connection[tuple[object, ...]], expected_bytes: int) -> SeedManifest:
    value: Final = SeedManifest.model_validate(scalar(connection, b"SELECT manifest FROM litellm_perf_seed.checkpoint"))
    assert value.progress.total_bytes >= expected_bytes, "Snapshot is smaller than the requested physical dataset"
    assert value.progress.rows > 100, "Snapshot does not contain enough rows for pagination"
    assert value.progress.rows == scalar(connection, b"SELECT rows FROM litellm_perf_seed.checkpoint"), (
        "Seed has unmanifested changes"
    )
    return value


def log_sample(connection: psycopg.Connection[tuple[object, ...]]) -> LogRow:
    return LogRow.model_validate(
        scalar(connection, b'SELECT row_to_json(l) FROM "LiteLLM_SpendLogs" l WHERE request_id = %s', ("perf-seed-17",))
    )


def history(connection: psycopg.Connection[tuple[object, ...]]) -> tuple[Migration, ...]:
    rows: Final = connection.execute(
        'SELECT row_to_json(m) FROM "_prisma_migrations" m ORDER BY migration_name'
    ).fetchall()
    migrations: Final = tuple(Migration.model_validate(row[0]) for row in rows)
    assert migrations, "No migration history"
    assert all(row.finished_at is not None and row.rolled_back_at is None for row in migrations), (
        "Unfinished or rolled-back migration"
    )
    assert len({row.migration_name for row in migrations}) == len(migrations), "Duplicate migration history"
    return migrations


def fingerprint(connection: psycopg.Connection[tuple[object, ...]], rows: int) -> tuple[tuple[str, str], ...]:
    ids: Final = tuple(f"perf-seed-{1 + index * (rows - 1) // 63}" for index in range(64))
    values: Final = connection.execute(
        "SELECT request_id, md5(jsonb_build_array(api_key, spend, total_tokens, prompt_tokens, completion_tokens, "
        '"startTime", "endTime", metadata, session_id, status, messages, response, proxy_server_request)::text) '
        'FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s) ORDER BY request_id',
        (list(ids),),
    ).fetchall()
    assert len(values) == 64, "Missing deterministic seed samples"
    return FINGERPRINTS.validate_python(values)


def count_logs(connection: psycopg.Connection[tuple[object, ...]]) -> int:
    return INTEGER.validate_python(scalar(connection, b'SELECT count(*) FROM "LiteLLM_SpendLogs"'))


def persisted(connection: psycopg.Connection[tuple[object, ...]], request_ids: tuple[str, ...]) -> int:
    row: Final = connection.execute(
        'SELECT count(*) FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)', (list(request_ids),)
    ).fetchone()
    assert row is not None
    return INTEGER.validate_python(row[0])


def await_spend_logs(
    connection: psycopg.Connection[tuple[object, ...]], request_ids: tuple[str, ...], *, timeout: float = 120
) -> int:
    assert request_ids and len(set(request_ids)) == len(request_ids), "Missing or duplicate response IDs"
    deadline: Final = time.monotonic() + timeout
    while (written := persisted(connection, request_ids)) != len(request_ids):
        assert time.monotonic() < deadline, f"Only {written}/{len(request_ids)} successful requests reached spend logs"
        time.sleep(0.2)
    return written


def observe_once(connection: psycopg.Connection[tuple[object, ...]]) -> DatabaseSample:
    started: Final = time.time()
    try:
        locks: Final = connection.execute(
            "SELECT count(*), coalesce(max(extract(epoch FROM clock_timestamp() - query_start)), 0) "
            "FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'"
        ).fetchone()
        assert locks is not None
        reading: Final = time.monotonic()
        row: Final = connection.execute(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', ("perf-seed-17",)
        ).fetchone()
        return DatabaseSample(
            started=started,
            blocked_sessions=INTEGER.validate_python(locks[0]),
            longest_lock_wait_seconds=NUMBER.validate_python(locks[1]),
            read_seconds=time.monotonic() - reading,
            failure=None if row == ("perf-seed-17",) else "Seed row disappeared",
        )
    except psycopg.Error as exc:
        return DatabaseSample(
            started=started,
            blocked_sessions=0,
            longest_lock_wait_seconds=0,
            read_seconds=time.time() - started,
            failure=type(exc).__name__,
        )


def observations(url: str, stop: Event) -> Iterator[DatabaseSample]:
    with psycopg.connect(url, autocommit=True, connect_timeout=5) as connection:
        connection.execute("SET statement_timeout = '5s'")
        while not stop.is_set():
            yield observe_once(connection)
            stop.wait(0.5)
