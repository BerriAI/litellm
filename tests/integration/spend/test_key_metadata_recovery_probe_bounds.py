from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import psycopg
import pytest
from litellm_proxy_extras.request_log_indexes import REQUEST_LOG_INDEXES
from psycopg.types.json import Jsonb
from pydantic import JsonValue

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.key_metadata_recovery import recover_key_metadata_from_spend_logs
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token
from tests.integration._support.client import eventually
from tests.integration._support.database import scratch_database, write_rows

_SPEND_LOGS_DDL: Final = """
    CREATE TABLE "LiteLLM_SpendLogs" (
        request_id TEXT PRIMARY KEY,
        api_key TEXT NOT NULL DEFAULT '',
        "startTime" TIMESTAMP(3) NOT NULL,
        "user" TEXT DEFAULT '',
        team_id TEXT,
        metadata JSONB DEFAULT '{}'
    )
"""

_API_KEY_START_TIME_INDEX: Final = next(
    index for index in REQUEST_LOG_INDEXES if index.name == "LiteLLM_SpendLogs_api_key_startTime_idx"
)

_STATS_SQL: Final = """
    SELECT seq_scan, idx_scan, seq_tup_read, idx_tup_fetch, n_tup_ins
    FROM pg_stat_user_tables
    WHERE relname = 'LiteLLM_SpendLogs'
"""

_OTHER_BACKENDS_SQL: Final = """
    SELECT count(*) FROM pg_stat_activity
    WHERE datname = current_database() AND pid <> pg_backend_pid() AND backend_type = 'client backend'
"""


@dataclass(frozen=True)
class _Settle:
    previous: Mapping[str, int] | None
    count: int


def _create_spend_logs_table(database_url: str) -> None:
    write_rows(_SPEND_LOGS_DDL, (), database_url=database_url)
    write_rows(
        f'CREATE INDEX "{_API_KEY_START_TIME_INDEX.name}" ON "{_API_KEY_START_TIME_INDEX.table}" '  # pyright: ignore[reportArgumentType]  # DDL from the migration job index list
        f"{_API_KEY_START_TIME_INDEX.definition}",
        (),
        database_url=database_url,
    )


def _spend_log_stats(database_url: str) -> dict[str, int]:
    with psycopg.connect(database_url) as connection:
        row: Final = connection.execute(_STATS_SQL).fetchone()
    if row is None:
        return {"seq_scan": 0, "idx_scan": 0, "seq_tup_read": 0, "idx_tup_fetch": 0, "n_tup_ins": 0}
    return {
        "seq_scan": row[0],
        "idx_scan": row[1],
        "seq_tup_read": row[2],
        "idx_tup_fetch": row[3],
        "n_tup_ins": row[4],
    }


def _other_client_backends(database_url: str) -> int:
    with psycopg.connect(database_url) as connection:
        row: Final = connection.execute(_OTHER_BACKENDS_SQL).fetchone()
    return 0 if row is None else int(row[0])


def _settled_stats(database_url: str, seeded_rows: int | None = None) -> dict[str, int]:
    eventually(
        lambda: _other_client_backends(database_url),
        lambda backends: backends == 0,
        seconds=60,
    )
    settle = _Settle(previous=None, count=0)

    def probe() -> dict[str, int]:
        nonlocal settle
        current: Final = _spend_log_stats(database_url)
        if current == settle.previous:
            settle = _Settle(previous=current, count=settle.count + 1)
        else:
            settle = _Settle(previous=current, count=0)
        return current

    settled: Final = eventually(
        probe,
        lambda stats: settle.count >= 5 and (seeded_rows is None or stats["n_tup_ins"] >= seeded_rows),
        seconds=60,
    )
    return settled


def _rows_read_since(database_url: str, baseline: Mapping[str, int]) -> int:
    settled: Final = _settled_stats(database_url)
    return (settled["seq_tup_read"] + settled["idx_tup_fetch"]) - (baseline["seq_tup_read"] + baseline["idx_tup_fetch"])


def _insert_nameless_spend_logs(connection: psycopg.Connection[tuple[object, ...]], digest: str, rows: int) -> None:
    connection.execute(
        """
        INSERT INTO "LiteLLM_SpendLogs" (request_id, api_key, "startTime")
        SELECT %(digest)s || '-' || g, %(digest)s, %(start)s + g * interval '1 minute'
        FROM generate_series(1, %(rows)s) g
        """,
        {"digest": digest, "start": datetime(2026, 9, 7), "rows": rows},
    )


def _named_spend_log(
    digest: str, logged_at: datetime, alias: str | None, user: str | None, team: str | None = None
) -> tuple[str, str, datetime, str, str | None, Jsonb]:
    return (
        f"{digest}-{logged_at.isoformat()}",
        digest,
        logged_at,
        user or "",
        team,
        Jsonb({"user_api_key_alias": alias} if alias else {}),
    )


def _insert_spend_logs(database_url: str, rows: Sequence[tuple[str, str, datetime, str, str | None, Jsonb]]) -> None:
    with psycopg.connect(database_url) as connection:
        connection.cursor().executemany(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, api_key, "startTime", "user", team_id, metadata)'
            " VALUES (%s, %s, %s, %s, %s, %s)",
            list(rows),
        )


def _analyze(database_url: str, vacuum: bool) -> None:
    with psycopg.connect(database_url, autocommit=True) as connection:
        if vacuum:
            connection.execute('VACUUM (ANALYZE) "LiteLLM_SpendLogs"')
        else:
            connection.execute('ANALYZE "LiteLLM_SpendLogs"')


async def _recover(
    monkeypatch: pytest.MonkeyPatch,
    database_url: str,
    digests: set[str] | frozenset[str],
    window: tuple[datetime, datetime],
) -> Mapping[str, JsonValue]:
    monkeypatch.setenv("DATABASE_URL", database_url)
    client: Final = PrismaClient(database_url, ProxyLogging(UserApiKeyCache()))
    await client.connect()
    try:
        return await recover_key_metadata_from_spend_logs(client, digests, window, cache=InMemoryCache())
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_names_a_key_by_its_oldest_and_newest_named_rows_in_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        _create_spend_logs_table(database_url)
        unnamed_edges, owner_logged_late, reowned, outside_window, never_named = (
            hash_token(f"cli-session-{name}") for name in ("edges", "late", "reowned", "window", "never")
        )
        _insert_spend_logs(
            database_url,
            (
                _named_spend_log(unnamed_edges, datetime(2026, 9, 7, 1), None, None),
                _named_spend_log(unnamed_edges, datetime(2026, 9, 8), "cli-a", "alice", "team-a"),
                _named_spend_log(unnamed_edges, datetime(2026, 9, 9), "cli-a", "alice", "team-a"),
                _named_spend_log(unnamed_edges, datetime(2026, 9, 9, 23), None, None),
                _named_spend_log(owner_logged_late, datetime(2026, 9, 7, 1), "cli-b", None),
                _named_spend_log(owner_logged_late, datetime(2026, 9, 9), "cli-b", "bob"),
                _named_spend_log(reowned, datetime(2026, 9, 7, 1), "cli-c", "carol"),
                _named_spend_log(reowned, datetime(2026, 9, 9), "cli-c", "dave"),
                _named_spend_log(outside_window, datetime(2026, 9, 6), "stale-alias", "erin"),
                _named_spend_log(outside_window, datetime(2026, 9, 8), "cli-d", "erin"),
                _named_spend_log(outside_window, datetime(2026, 9, 10), "later-alias", "erin"),
                _named_spend_log(never_named, datetime(2026, 9, 8), None, None),
            ),
        )

        result: Final = await _recover(
            monkeypatch,
            database_url,
            {unnamed_edges, owner_logged_late, reowned, outside_window, never_named},
            (datetime(2026, 9, 7), datetime(2026, 9, 10)),
        )

        assert dict(result) == {
            unnamed_edges: {"key_alias": "cli-a", "team_id": "team-a", "user_id": "alice"},
            owner_logged_late: {"key_alias": "cli-b", "team_id": None, "user_id": "bob"},
            reowned: {"key_alias": "cli-c", "team_id": None, "user_id": None},
            outside_window: {"key_alias": "cli-d", "team_id": None, "user_id": "erin"},
        }


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_reads_two_rows_per_key_however_many_the_key_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        _create_spend_logs_table(database_url)
        owners: Final[Mapping[str, str]] = {hash_token(f"cli-session-busy-{i}"): f"user-{i}" for i in range(5)}
        with psycopg.connect(database_url) as connection:
            for digest, owner in owners.items():
                connection.execute(
                    """
                    INSERT INTO "LiteLLM_SpendLogs" (request_id, api_key, "startTime", "user", metadata)
                    SELECT %(digest)s || '-' || g, %(digest)s, %(start)s + g * interval '1 minute', %(owner)s,
                        jsonb_build_object('user_api_key_alias', 'cli-session-' || %(owner)s)
                    FROM generate_series(1, 2000) g
                    """,
                    {"digest": digest, "owner": owner, "start": datetime(2026, 9, 7)},
                )
        _analyze(database_url, vacuum=False)
        baseline: Final = _settled_stats(database_url, seeded_rows=10000)

        result: Final = await _recover(
            monkeypatch, database_url, frozenset(owners), (datetime(2026, 9, 7), datetime(2026, 9, 10))
        )

        assert {digest: meta.get("user_id") for digest, meta in result.items()} == owners
        rows_read: Final = _rows_read_since(database_url, baseline)
        assert len(owners) <= rows_read <= 10


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_walks_a_bounded_number_of_nameless_rows_per_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        _create_spend_logs_table(database_url)
        named_late: Final[Mapping[str, str]] = {hash_token(f"cli-session-late-{i}"): f"user-{i}" for i in range(3)}
        never_named: Final = frozenset(hash_token(f"cli-session-never-{i}") for i in range(3))
        with psycopg.connect(database_url) as connection:
            for digest in (*named_late, *never_named):
                _insert_nameless_spend_logs(connection, digest, 3 * SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE)
            for digest, owner in named_late.items():
                connection.execute(
                    """
                    INSERT INTO "LiteLLM_SpendLogs" (request_id, api_key, "startTime", "user", metadata)
                    VALUES (%(digest)s || '-newest', %(digest)s, %(logged_at)s, %(owner)s,
                        jsonb_build_object('user_api_key_alias', 'cli-session-' || %(owner)s))
                    """,
                    {"digest": digest, "owner": owner, "logged_at": datetime(2026, 9, 9)},
                )
        _analyze(database_url, vacuum=False)
        baseline: Final = _settled_stats(database_url)

        result: Final = await _recover(
            monkeypatch,
            database_url,
            frozenset(named_late) | never_named,
            (datetime(2026, 9, 7), datetime(2026, 9, 10)),
        )

        assert {digest: meta.get("user_id") for digest, meta in result.items()} == named_late
        rows_read: Final = _rows_read_since(database_url, baseline)
        assert len(frozenset(named_late) | never_named) <= rows_read <= 1800


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_reads_a_short_nameless_key_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        _create_spend_logs_table(database_url)
        rows_per_key: Final = SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE // 2
        never_named: Final = frozenset(hash_token(f"cli-session-short-{i}") for i in range(20))
        with psycopg.connect(database_url) as connection:
            for digest in never_named:
                _insert_nameless_spend_logs(connection, digest, rows_per_key)
        _analyze(database_url, vacuum=True)
        baseline: Final = _settled_stats(database_url)

        result: Final = await _recover(
            monkeypatch, database_url, never_named, (datetime(2026, 9, 7), datetime(2026, 9, 10))
        )

        assert dict(result) == {}
        rows_read: Final = _rows_read_since(database_url, baseline)
        assert len(never_named) <= rows_read <= 1000


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_bounds_a_busy_nameless_key_among_short_keys_before_any_vacuum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        _create_spend_logs_table(database_url)
        busy: Final = frozenset(hash_token(f"cli-session-busy-nameless-{i}") for i in range(3))
        with psycopg.connect(database_url) as connection:
            for short_key in range(200):
                _insert_nameless_spend_logs(
                    connection,
                    hash_token(f"cli-session-short-{short_key}"),
                    SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE // 5,
                )
            for digest in busy:
                _insert_nameless_spend_logs(connection, digest, 30 * SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE)
        _analyze(database_url, vacuum=False)
        baseline: Final = _settled_stats(database_url)

        result: Final = await _recover(monkeypatch, database_url, busy, (datetime(2026, 9, 7), datetime(2026, 9, 10)))

        assert dict(result) == {}
        rows_read: Final = _rows_read_since(database_url, baseline)
        assert len(busy) <= rows_read <= 900


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_finds_a_name_logged_where_the_oldest_probe_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        _create_spend_logs_table(database_url)
        start: Final = datetime(2026, 9, 7)
        past_the_stop, tied_with_the_stop = (hash_token(f"cli-session-{name}") for name in ("past", "tied"))
        same_millisecond: Final = tuple(
            start + timedelta(minutes=SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE, microseconds=n) for n in (100, 200, 300)
        )
        _insert_spend_logs(
            database_url,
            (
                *(
                    _named_spend_log(past_the_stop, start + timedelta(minutes=minute), None, None)
                    for minute in range(1, SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE + 20)
                ),
                _named_spend_log(
                    past_the_stop,
                    start + timedelta(minutes=SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE + 20),
                    "cli-p",
                    "pat",
                ),
                *(
                    _named_spend_log(past_the_stop, start + timedelta(minutes=minute), None, None)
                    for minute in range(
                        SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE + 21,
                        SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE + 51,
                    )
                ),
                *(
                    _named_spend_log(tied_with_the_stop, start + timedelta(minutes=minute), None, None)
                    for minute in range(1, SPEND_LOG_KEY_METADATA_ROWS_PER_PROBE)
                ),
                _named_spend_log(tied_with_the_stop, same_millisecond[0], None, None),
                _named_spend_log(tied_with_the_stop, same_millisecond[1], None, None),
                _named_spend_log(tied_with_the_stop, same_millisecond[2], "cli-t", "tess"),
            ),
        )

        _analyze(database_url, vacuum=False)
        baseline: Final = _settled_stats(database_url)
        digests: Final = {past_the_stop, tied_with_the_stop}

        result: Final = await _recover(
            monkeypatch,
            database_url,
            digests,
            (start, datetime(2026, 9, 10)),
        )

        assert dict(result) == {
            past_the_stop: {"key_alias": "cli-p", "team_id": None, "user_id": "pat"},
            tied_with_the_stop: {"key_alias": "cli-t", "team_id": None, "user_id": "tess"},
        }
        assert len(digests) <= _rows_read_since(database_url, baseline)
