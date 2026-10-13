"""Postgres and Redis work counters, read before and after the traffic phase so a case can put a ceiling
on how much database and cache work the gateway does per request. Both come from the servers' own
statistics (`pg_stat_database`, Redis `INFO`), so nothing here depends on how the gateway talks to them.
`DATABASE_URL` and `REDIS_HOST` are the runner's own env, as in the other suites; without them the
counters are None and any Limit on them reports as missing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter

_DB_QUERY: Final = (
    "SELECT xact_commit, tup_inserted + tup_updated + tup_deleted, tup_returned + tup_fetched "
    "FROM pg_stat_database WHERE datname = current_database()"
)
_DB_ROW: Final = TypeAdapter(tuple[int, int, int])
_INFO: Final = TypeAdapter(dict[str, object])
_INT: Final = TypeAdapter(int)


@dataclass(frozen=True, slots=True)
class DbCounters:
    commits: int
    rows_written: int
    rows_read: int


@dataclass(frozen=True, slots=True)
class RedisCounters:
    commands: int
    connected_clients: int


def read_db_counters() -> DbCounters | None:
    url: Final = os.environ.get("DATABASE_URL")
    if url is None:
        return None
    import psycopg

    with psycopg.connect(url) as conn:
        row: Final = conn.execute(_DB_QUERY).fetchone()
    commits, written, read = _DB_ROW.validate_python(row)
    return DbCounters(commits=commits, rows_written=written, rows_read=read)


def read_redis_counters() -> RedisCounters | None:
    """Standalone Redis only, which is what the perf lane's chart-bundled Redis and a local run are."""
    host: Final = os.environ.get("REDIS_HOST")
    if host is None:
        return None
    import redis

    client: Final = redis.Redis(
        host=host,
        port=int(os.environ.get("REDIS_PORT", "6379")),
        ssl=os.environ.get("REDIS_SSL", "false").lower() == "true",
        password=os.environ.get("REDIS_PASSWORD") or None,
        decode_responses=True,
        socket_connect_timeout=2,
    )
    stats: Final = _INFO.validate_python(client.info("stats"))
    clients: Final = _INFO.validate_python(client.info("clients"))
    return RedisCounters(
        commands=_INT.validate_python(stats["total_commands_processed"]),
        connected_clients=_INT.validate_python(clients["connected_clients"]),
    )
