"""Postgres and Redis work the gateway did during the traffic phase, split into what a request waits on and
what runs after the response. Both come from the servers' own accounting, so nothing here depends on how the
gateway talks to them: Postgres statement counts from `pg_stat_statements` (read before and after the run) and
Redis commands from a `MONITOR` connection held open for the run. `DATABASE_URL` and `REDIS_HOST` are the
runner's own env, as in the other suites; without them the counters are None and any Limit on them reports as
missing. `E2E_DATABASE_ADMIN_URL`, when set, is used once to create the `pg_stat_statements` extension."""

from __future__ import annotations

import os
import re
import threading
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter

if TYPE_CHECKING:
    import redis

Bucket = Literal["awaited", "background", "plumbing", "unclassified"]

_COMMITS_QUERY: Final = "SELECT xact_commit FROM pg_stat_database WHERE datname = current_database()"
_STATEMENTS_QUERY: Final = (
    "SELECT queryid, calls, query FROM pg_stat_statements "
    "WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())"
)
_CREATE_EXTENSION: Final = "CREATE EXTENSION IF NOT EXISTS pg_stat_statements"
_COMMITS_ROW: Final = TypeAdapter(tuple[int])
_STATEMENT_ROWS: Final = TypeAdapter(list[tuple[int, int, str]])
_INFO: Final = TypeAdapter(dict[str, object])
_INT: Final = TypeAdapter(int)
_MONITOR_LINE: Final = TypeAdapter(dict[str, object])

_TABLE: Final = re.compile(r'"(LiteLLM_\w+)"')
_AWAITED_TABLES: Final = frozenset(
    {
        "LiteLLM_VerificationToken",
        "LiteLLM_UserTable",
        "LiteLLM_TeamTable",
        "LiteLLM_TeamMembership",
        "LiteLLM_OrganizationTable",
        "LiteLLM_OrganizationMembership",
        "LiteLLM_EndUserTable",
        "LiteLLM_BudgetTable",
        "LiteLLM_ObjectPermissionTable",
    }
)
_PLUMBING_VERBS: Final = frozenset({"BEGIN", "COMMIT", "ROLLBACK", "DEALLOCATE", "SET", "SHOW"})
_REDIS_READS: Final = frozenset({"GET", "MGET", "HGET", "HGETALL", "EXISTS", "TTL", "PTTL"})


@dataclass(frozen=True, slots=True)
class StatementCalls:
    queryid: int
    query: str
    calls: int

    @property
    def bucket(self) -> Bucket:
        """A SELECT on the key, user, team, org, end-user or budget tables is what a request waits on at auth
        and routing. Transaction control and session statements are plumbing. Everything else (spend and
        audit writes, daily aggregates, config and model reloads) runs after the response or on a timer."""
        verb: Final = self.query.lstrip("(").split(maxsplit=1)[0].upper() if self.query.strip() else ""
        tables: Final = _TABLE.findall(self.query)
        if verb in _PLUMBING_VERBS or not tables:
            return "plumbing"
        return "awaited" if verb == "SELECT" and tables[0] in _AWAITED_TABLES else "background"

    @property
    def summary(self) -> str:
        return re.sub(r"\s+", " ", self.query)[:110]


@dataclass(frozen=True, slots=True)
class DbCounters:
    commits: int
    statements: tuple[StatementCalls, ...] | None


@dataclass(frozen=True, slots=True)
class RedisCounters:
    commands: int
    connected_clients: int


@dataclass(frozen=True, slots=True)
class RedisCommand:
    command: str
    key: str
    inside_script: bool
    count: int

    @property
    def bucket(self) -> Bucket:
        """Budget reads, cooldown reads and the rate-limiter script (with every command it issues) happen
        before the request is forwarded. Spend increments, token counters, config rewrites and cache
        invalidations (PUBLISH, DEL) happen after the response or on a timer."""
        if self.inside_script or self.command in {"EVALSHA", "EVAL"}:
            return "awaited"
        if self.command in {"PUBLISH", "DEL", "SCRIPT", "INFO", "MONITOR"} or self.key.startswith(
            ("litellm_config:", "daily_")
        ):
            return "background"
        if self.key.startswith("spend:"):
            return "awaited" if self.command in _REDIS_READS else "background"
        if self.key.startswith("deployment:") or self.key.endswith(("}:requests", "}:window")):
            return "awaited"
        if self.key.endswith("}:tokens"):
            return "background"
        return "unclassified"

    @property
    def summary(self) -> str:
        key: Final = re.sub(r"[0-9a-f]{16,}|sk-[A-Za-z0-9_-]+|[0-9a-f-]{36}", "*", self.key)[:70]
        return f"{self.command} {key}{' (in script)' if self.inside_script else ''}".rstrip()


def statement_deltas(before: DbCounters | None, after: DbCounters | None) -> tuple[StatementCalls, ...] | None:
    """Calls per statement during the run: `after` minus `before` by queryid, new statements whole."""
    if before is None or after is None or before.statements is None or after.statements is None:
        return None
    earlier: Final = {statement.queryid: statement.calls for statement in before.statements}
    return tuple(
        StatementCalls(
            queryid=statement.queryid, query=statement.query, calls=statement.calls - earlier.get(statement.queryid, 0)
        )
        for statement in after.statements
        if statement.calls > earlier.get(statement.queryid, 0)
    )


def calls_in(bucket: Bucket, items: tuple[StatementCalls, ...] | tuple[RedisCommand, ...] | None) -> int | None:
    if items is None:
        return None
    return sum(_count(item) for item in items if item.bucket == bucket)


def _count(item: StatementCalls | RedisCommand) -> int:
    return item.calls if isinstance(item, StatementCalls) else item.count


def read_db_counters() -> DbCounters | None:
    url: Final = os.environ.get("DATABASE_URL")
    if url is None:
        return None
    import psycopg

    _ensure_statements_extension(url)
    with psycopg.connect(url) as conn:
        (commits,) = _COMMITS_ROW.validate_python(conn.execute(_COMMITS_QUERY).fetchone())
        try:
            rows = _STATEMENT_ROWS.validate_python(conn.execute(_STATEMENTS_QUERY).fetchall())
        except psycopg.errors.UndefinedTable:
            return DbCounters(commits=commits, statements=None)
    return DbCounters(
        commits=commits,
        statements=tuple(StatementCalls(queryid=queryid, query=query, calls=calls) for queryid, calls, query in rows),
    )


def _ensure_statements_extension(url: str) -> None:
    """Creating the extension needs a superuser, so prefer the admin URL; the runtime user's attempt is
    allowed to fail (a database without `shared_preload_libraries = pg_stat_statements` cannot have it)."""
    import psycopg

    try:
        with psycopg.connect(os.environ.get("E2E_DATABASE_ADMIN_URL", url), autocommit=True) as conn:
            _ = conn.execute(_CREATE_EXTENSION)
    except psycopg.Error:
        return


def _redis_client() -> redis.Redis[str]:
    import redis

    return redis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ.get("REDIS_PORT", "6379")),
        ssl=os.environ.get("REDIS_SSL", "false").lower() == "true",
        password=os.environ.get("REDIS_PASSWORD") or None,
        decode_responses=True,
        socket_connect_timeout=2,
    )


def read_redis_counters() -> RedisCounters | None:
    """Standalone Redis only, which is what the perf lane's chart-bundled Redis and a local run are."""
    if os.environ.get("REDIS_HOST") is None:
        return None
    client: Final = _redis_client()
    stats: Final = _INFO.validate_python(client.info("stats"))
    clients: Final = _INFO.validate_python(client.info("clients"))
    return RedisCounters(
        commands=_INT.validate_python(stats["total_commands_processed"]),
        connected_clients=_INT.validate_python(clients["connected_clients"]),
    )


class RedisMonitor:
    """Every command Redis executed while the monitor was open, counted by command, key and whether a Lua
    script issued it. `MONITOR` echoes each command to this connection, so it costs Redis one write per
    command; fine at the rates these cases run."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, str, bool]] = Counter()
        self._thread: threading.Thread | None = None
        self._pool: redis.ConnectionPool | None = None
        self.commands: tuple[RedisCommand, ...] | None = None

    def _run(self) -> None:
        import redis

        client: Final = _redis_client()
        self._pool = client.connection_pool  # pyright: ignore[reportAny]  # redis.Redis leaves connection_pool untyped
        with client.monitor() as monitor:
            try:
                while True:
                    line = _MONITOR_LINE.validate_python(monitor.next_command())
                    words = str(line["command"]).split(" ", 2)
                    key = words[1].strip('"') if len(words) > 1 else ""
                    self._counts[(words[0].upper(), key, line["client_type"] == "lua")] += 1
            except (redis.ConnectionError, OSError, ValueError):
                return

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="redis-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> tuple[RedisCommand, ...]:
        if self._pool is not None:
            self._pool.disconnect()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return tuple(
            RedisCommand(command=command, key=key, inside_script=inside_script, count=count)
            for (command, key, inside_script), count in self._counts.items()
        )


@contextmanager
def redis_monitor() -> Generator[RedisMonitor, None, None]:
    """Counts Redis commands for the duration of the block; `.commands` is set once the block ends."""
    monitor: Final = RedisMonitor()
    if os.environ.get("REDIS_HOST") is not None:
        monitor.start()
    try:
        yield monitor
    finally:
        monitor.commands = monitor.stop() if os.environ.get("REDIS_HOST") is not None else None
