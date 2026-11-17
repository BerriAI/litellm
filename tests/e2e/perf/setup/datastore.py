"""Postgres and Redis work the gateway did during the traffic phase, split into what a request waits on and
what runs after the response. Both come from the servers' own accounting, so nothing here depends on how the
gateway talks to them: Postgres statement counts from `pg_stat_statements` (read before and after the run) and
Redis commands from a `MONITOR` connection held open for the run, Postgres execution time per statement from the
same `pg_stat_statements` rows (`postgres_active_sessions`, the average-active-sessions number RDS Performance
Insights calls DB load, and the slowest statement's mean) and the connection count from `pg_stat_activity` sampled
once a second (`postgres_peak_connections`). `DATABASE_URL` and `REDIS_HOST` are the
runner's own env, as in the other suites; without them the counters are None and any Limit on them reports as
missing. `E2E_DATABASE_ADMIN_URL`, when set, is used once to create the `pg_stat_statements` extension. Only the
runtime user's own statements are read: other users' rows come back with a NULL queryid and hidden text."""

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

from setup.series import Recorder, Series

if TYPE_CHECKING:
    import redis

Bucket = Literal["awaited", "background", "plumbing", "unclassified"]

_COMMITS_QUERY: Final = "SELECT xact_commit FROM pg_stat_database WHERE datname = current_database()"
_STATEMENTS_QUERY: Final = (
    "SELECT queryid, calls, query, total_exec_time FROM pg_stat_statements "
    "WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database()) "
    "AND userid = (SELECT usesysid FROM pg_user WHERE usename = current_user) AND queryid IS NOT NULL"
)
_CREATE_EXTENSION: Final = "CREATE EXTENSION IF NOT EXISTS pg_stat_statements"
_CONNECTIONS_QUERY: Final = (
    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
    "AND backend_type = 'client backend' AND pid <> pg_backend_pid()"
)
_CONNECTIONS_SAMPLE_SECONDS: Final = 1.0
_COMMITS_ROW: Final = TypeAdapter(tuple[int])
_STATEMENT_ROWS: Final = TypeAdapter(list[tuple[int, int, str, float]])
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
_REDIS_NAMESPACE: Final = "litellm.caching:"
_REDIS_WRITES: Final = frozenset(
    {"SET", "SETEX", "DEL", "EXPIRE", "INCRBYFLOAT", "INCR", "RPUSH", "LPUSH", "LPOP", "RPOP", "LTRIM", "HSET", "SADD"}
)


@dataclass(frozen=True, slots=True)
class StatementCalls:
    queryid: int
    query: str
    calls: int
    total_exec_ms: float = 0.0

    @property
    def mean_exec_seconds(self) -> float | None:
        return self.total_exec_ms / 1000 / self.calls if self.calls > 0 else None

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
    def table(self) -> str:
        tables: Final = _TABLE.findall(self.query)
        return tables[0] if tables else ""

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
        before the request is forwarded. Spend increments, token counters, config rewrites, cache
        invalidations (PUBLISH, DEL), the spend buffers the pods flush and any other write happen after the
        response or on a timer. Reads on keys these rules do not know stay unclassified and count in neither."""
        key: Final = self.key.removeprefix(_REDIS_NAMESPACE)
        if key.startswith(("litellm_config:", "daily_", "cronjob_lock:", "litellm_spend", "litellm_daily")):
            return "background"
        if self.inside_script or self.command in {"EVALSHA", "EVAL"}:
            return "awaited"
        if self.command in {"PUBLISH", "DEL", "SCRIPT", "INFO", "MONITOR"}:
            return "background"
        if key.startswith("spend:"):
            return "awaited" if self.command in _REDIS_READS else "background"
        if key.startswith("deployment:") or key.endswith(("}:requests", "}:window")):
            return "awaited"
        if key.endswith("}:tokens") or self.command in _REDIS_WRITES:
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
    earlier: Final = {statement.queryid: statement for statement in before.statements}
    return tuple(
        _statement_delta(statement, earlier.get(statement.queryid))
        for statement in after.statements
        if statement.calls > (0 if statement.queryid not in earlier else earlier[statement.queryid].calls)
    )


def _statement_delta(statement: StatementCalls, earlier: StatementCalls | None) -> StatementCalls:
    return StatementCalls(
        queryid=statement.queryid,
        query=statement.query,
        calls=statement.calls - (earlier.calls if earlier is not None else 0),
        total_exec_ms=statement.total_exec_ms - (earlier.total_exec_ms if earlier is not None else 0.0),
    )


def active_sessions(statements: tuple[StatementCalls, ...] | None, wall_seconds: float) -> float | None:
    """Seconds Postgres spent executing the run's statements per second of wall time, the average active
    sessions figure RDS Performance Insights reports as DB load."""
    if statements is None or wall_seconds <= 0:
        return None
    return sum(statement.total_exec_ms for statement in statements) / 1000 / wall_seconds


def slowest_statement(statements: tuple[StatementCalls, ...] | None) -> StatementCalls | None:
    """The run's statement with the highest mean execution time."""
    timed: Final = tuple(statement for statement in statements or () if statement.mean_exec_seconds is not None)
    return max(timed, key=lambda statement: statement.mean_exec_seconds or 0.0, default=None)


class ConnectionsSampler:
    """`pg_stat_activity` client backends on the gateway's database once a second on a background thread, minus
    this sampler's own connection, so the peak is what the gateway (and the test's own reads) held open.
    `series` keeps every count with its time for a ramped run's per-step peaks."""

    def __init__(self, url: str) -> None:
        self._url: Final = url
        self._stop: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, name="postgres-connections", daemon=True)
        self._counts: Final[Recorder[int]] = Recorder()

    @property
    def peak(self) -> int | None:
        return max((sample.value for sample in self._counts.series.samples), default=None)

    @property
    def series(self) -> Series[int]:
        return self._counts.series

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        import psycopg

        with psycopg.connect(self._url, autocommit=True) as conn:
            while not self._stop.is_set():
                (count,) = _COMMITS_ROW.validate_python(conn.execute(_CONNECTIONS_QUERY).fetchone())
                self._counts.record(count)
                _ = self._stop.wait(_CONNECTIONS_SAMPLE_SECONDS)


class NoConnectionsSampler:
    peak: None = None
    series: None = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


@contextmanager
def connections_sampler() -> Generator[ConnectionsSampler | NoConnectionsSampler]:
    url: Final = os.environ.get("DATABASE_URL")
    sampler: Final = ConnectionsSampler(url) if url is not None else NoConnectionsSampler()
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()


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
        statements=tuple(
            StatementCalls(queryid=queryid, query=query, calls=calls, total_exec_ms=total_ms)
            for queryid, calls, query, total_ms in rows
        ),
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
