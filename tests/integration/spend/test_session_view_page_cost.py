"""The session-grouped /spend/logs/ui page lists each session once and reads the window at most once.

The page itself stops after page_size + 1 sessions; the capped session count is
the one full pass over the window. Tuple reads come from pg_stat, and the
ordering, representative and total assertions pin what the page returns.
"""

import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, string_value
from integration._support.database import ROWS
from pydantic import JsonValue

from litellm.proxy.spend_tracking.spend_management_endpoints import SPEND_LOGS_PAGINATION_COUNT_CAP

SEED_ROWS: Final = 40_000
LONG_SESSION_ROWS: Final = 5_000
SESSION_LEN: Final = 5
BULK_ROWS: Final = SEED_ROWS - LONG_SESSION_ROWS
PAGE_SIZE: Final = 50
MAX_TUPLES_PER_PAGE: Final = SEED_ROWS * 2
_BASE: Final = datetime(2001, 1, 1)
M1_NEWEST_SECOND: Final = SEED_ROWS + 8
M2_NEWEST_SECOND: Final = SEED_ROWS + 5
BLOCKER_NEWEST_SECOND: Final = SEED_ROWS + 3
WINDOW_START: Final = _BASE.strftime("%Y-%m-%d %H:%M:%S")
WINDOW_END: Final = (_BASE + timedelta(seconds=M1_NEWEST_SECOND)).strftime("%Y-%m-%d %H:%M:%S")
_BULK_SESSIONS: Final = len({n // SESSION_LEN for n in range(1, BULK_ROWS + 1)})
EXPECTED_SESSIONS: Final = _BULK_SESSIONS + 5

_STATS_SETTLE_S: Final = 4.0
_STATS_STABLE_POLLS: Final = 4
_STATS_TIMEOUT_S: Final = 30.0
_QUIET_SETTLE_S: Final = 12.0
_QUIET_STABLE_POLLS: Final = 5
_QUIET_TIMEOUT_S: Final = 60.0


def _at(seconds: int) -> str:
    return (_BASE + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


KEY_ALIAS: Final = "intg-sesswin-alias"
_SMALL_BASE: Final = datetime(2002, 6, 1)
SMALL_WINDOW_START: Final = _SMALL_BASE.strftime("%Y-%m-%d %H:%M:%S")
SMALL_WINDOW_END: Final = (_SMALL_BASE + timedelta(seconds=100)).strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True, slots=True)
class _SeedRow:
    request_id: str
    session_id: str | None
    start_time: str
    api_key: str = "K"
    call_type: str = "acompletion"


def _hand_placed_rows(marker: str) -> tuple[_SeedRow, ...]:
    prefix: Final = f"intg-sesswin-{marker}-"
    blocker_request_id: Final = f"{prefix}x"
    return (
        _SeedRow(blocker_request_id, None, _at(SEED_ROWS + 1)),
        _SeedRow(f"{prefix}x1", blocker_request_id, _at(SEED_ROWS + 2)),
        _SeedRow(f"{prefix}x2", blocker_request_id, _at(BLOCKER_NEWEST_SECOND)),
        _SeedRow(f"{prefix}m2a", f"{marker}-m2", _at(SEED_ROWS + 4)),
        _SeedRow(f"{prefix}m2b", f"{marker}-m2", _at(M2_NEWEST_SECOND)),
        _SeedRow(f"{prefix}m1a", f"{marker}-m1", _at(SEED_ROWS + 6)),
        _SeedRow(f"{prefix}m1b", f"{marker}-m1", _at(SEED_ROWS + 7)),
        _SeedRow(f"{prefix}m1c", f"{marker}-m1", _at(M1_NEWEST_SECOND)),
        _SeedRow(f"intg-straddle-{marker}-in", f"{marker}-straddle", _at(1000)),
        _SeedRow(f"intg-straddle-{marker}-out", f"{marker}-straddle", _at(SEED_ROWS + 2_000_000)),
    )


def _seed(marker: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogs" ('
            'request_id, call_type, api_key, session_id, "startTime", "endTime", model, metadata, messages, response'
            ") "
            "SELECT %s || n, 'acompletion', 'K', "
            "CASE WHEN n > %s THEN %s || '-long' ELSE %s || '-s-' || (n / %s) END, "
            "ts, ts, 'e2e-sesswin', jsonb_build_object('user_api_key_alias', %s::text), '{}'::jsonb, '{}'::jsonb "
            "FROM generate_series(1, %s) AS g(n) "
            "CROSS JOIN LATERAL (SELECT %s::timestamp + (n || ' seconds')::interval AS ts) AS t",
            (f"intg-sesswin-{marker}-", BULK_ROWS, marker, marker, SESSION_LEN, KEY_ALIAS, SEED_ROWS, WINDOW_START),
        )
    _insert_rows(_hand_placed_rows(marker))


def _insert_rows(rows: tuple[_SeedRow, ...]) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.cursor().executemany(
            'INSERT INTO "LiteLLM_SpendLogs" ('
            'request_id, call_type, api_key, session_id, "startTime", "endTime", model, metadata, messages, response'
            ") VALUES (%s, %s, %s, %s, %s, %s, 'm', jsonb_build_object('user_api_key_alias', %s::text), '{}', '{}')",
            [
                (row.request_id, row.call_type, row.api_key, row.session_id, row.start_time, row.start_time, KEY_ALIAS)
                for row in rows
            ],
        )
        connection.execute('VACUUM ANALYZE "LiteLLM_SpendLogs"')


def _settle_autovacuum() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute('VACUUM ANALYZE "LiteLLM_SpendLogs"')
        connection.execute('ALTER TABLE "LiteLLM_SpendLogs" SET (autovacuum_enabled = false)')


def _cleanup(marker: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        try:
            connection.execute(
                'DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s OR request_id LIKE %s',
                (f"intg-sesswin-{marker}-%", f"intg-straddle-{marker}-%"),
            )
            connection.execute('VACUUM "LiteLLM_SpendLogs"')
        finally:
            connection.execute('ALTER TABLE "LiteLLM_SpendLogs" RESET (autovacuum_enabled)')
    _wait_for_stats_quiet()


def _wait_for_stats_quiet() -> None:
    _wait_until_stable(
        _tuples_read,
        settle_s=_STATS_SETTLE_S,
        stable_polls=_STATS_STABLE_POLLS,
        timeout_s=_STATS_TIMEOUT_S,
    )


def _tuples_read() -> int:
    """Heap tuples Postgres reports read on LiteLLM_SpendLogs: seq_tup_read
    plus idx_tup_fetch (heap rows fetched through index scans). Index entries
    traversed do not count, so bulk INSERTs never inflate the reading."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        row = connection.execute(
            "SELECT t.seq_tup_read, t.idx_tup_fetch FROM pg_stat_user_tables t WHERE t.relname = 'LiteLLM_SpendLogs'",
        ).fetchone()
    assert row is not None
    return int(row[0] or 0) + int(row[1] or 0)


def _wait_until_stable(read: Callable[[], int], *, settle_s: float, stable_polls: int, timeout_s: float) -> int:
    """pg_stat readings once the collector has gone quiet. Each pooled backend
    flushes its stats independently, so the counter climbs in delayed jumps:
    wait at least ``settle_s`` seconds, then take the reading once it is
    unchanged for ``stable_polls`` consecutive polls."""
    deadline: Final = time.monotonic() + timeout_s
    min_wait: Final = time.monotonic() + settle_s
    stable: int = 0
    last: int = read()
    while time.monotonic() < deadline:
        current: int = read()
        stable = stable + 1 if current == last else 0
        last = current
        if stable >= stable_polls and time.monotonic() > min_wait:
            return current
        time.sleep(0.5)
    raise AssertionError("pg_stat counters never settled")


def _tuples_read_quiet() -> int:
    return _wait_until_stable(
        _tuples_read,
        settle_s=_QUIET_SETTLE_S,
        stable_polls=_QUIET_STABLE_POLLS,
        timeout_s=_QUIET_TIMEOUT_S,
    )


def _data_rows(page: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return ROWS.validate_python(page["data"])


def _session_page(gateway: Gateway, **extra: str) -> dict[str, JsonValue]:
    return gateway.get(
        "/spend/logs/ui",
        params={
            "group_by_session": "true",
            "start_date": WINDOW_START,
            "end_date": WINDOW_END,
            "sort_by": "startTime",
            "sort_order": "desc",
            "page_size": str(PAGE_SIZE),
            **extra,
        },
    )


@pytest.fixture
def seeded(gateway: Gateway) -> Iterator[str]:
    marker: Final = uuid.uuid4().hex
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(_cleanup, marker)
        _settle_autovacuum()
        _seed(marker)
        _wait_for_stats_quiet()
        yield marker


def test_session_view_first_page_reads_a_bounded_slice(gateway: Gateway, seeded: str) -> None:
    prefix: Final = f"intg-sesswin-{seeded}-"
    before: Final = _tuples_read_quiet()
    page: Final = _session_page(gateway, page="1")
    tuples_read: Final = _tuples_read_quiet() - before

    assert page["has_more"] is True
    assert page["total"] == EXPECTED_SESSIONS
    assert EXPECTED_SESSIONS < SPEND_LOGS_PAGINATION_COUNT_CAP
    assert page["total_is_capped"] is False
    data: Final = _data_rows(page)
    assert len(data) == PAGE_SIZE
    assert data[0]["request_id"] == f"{prefix}m1c", "multi-row session must be represented by its newest row"
    assert data[1]["request_id"] == f"{prefix}m2b"
    assert data[2]["request_id"] == f"{prefix}x2", (
        "a NULL session_id row and the rows adopting its request_id as session_id form one session"
    )
    blocker_keys: Final = {f"{prefix}x", f"{prefix}x1", f"{prefix}x2"}
    assert sum(1 for row in data if row["request_id"] in blocker_keys) == 1
    assert data[3]["request_id"] == f"{prefix}{SEED_ROWS}", "the long session must be represented by its newest row"
    expected_fivers: Final = [BULK_ROWS, BULK_ROWS - 1, BULK_ROWS - SESSION_LEN - 1]
    assert [row["request_id"] for row in data[4:7]] == [f"{prefix}{n}" for n in expected_fivers], (
        "each 5-row session must be represented by its newest row, newest session first"
    )
    start_times: Final = [string_value(row["startTime"]) for row in data]
    assert start_times == sorted(start_times, reverse=True)
    assert tuples_read < MAX_TUPLES_PER_PAGE, (
        f"first page read {tuples_read} tuples for a window of {SEED_ROWS} rows and page_size {PAGE_SIZE}; "
        "the page must stop after page_size + 1 sessions, leaving the capped count as the only full pass"
    )


def test_session_view_cursor_page_reads_a_bounded_slice(gateway: Gateway, seeded: str) -> None:
    first: Final = _session_page(gateway, page="1")
    cursor: Final = first["next_session_cursor"]
    assert isinstance(cursor, str)

    before: Final = _tuples_read_quiet()
    second: Final = _session_page(gateway, page="2", session_cursor=cursor)
    tuples_read: Final = _tuples_read_quiet() - before

    second_data: Final = _data_rows(second)
    assert len(second_data) == PAGE_SIZE
    oldest_first: Final = string_value(_data_rows(first)[-1]["startTime"])
    assert all(string_value(row["startTime"]) < oldest_first for row in second_data)
    assert tuples_read < MAX_TUPLES_PER_PAGE, (
        f"cursor page read {tuples_read} tuples for a window of {SEED_ROWS} rows and page_size {PAGE_SIZE}; "
        "the page must stop after page_size + 1 sessions, leaving the capped count as the only full pass"
    )


def test_session_view_repeated_page_stays_bounded_under_the_generic_plan(gateway: Gateway, seeded: str) -> None:
    """The proxy runs these as prepared statements and Postgres switches to a generic plan after five executions, so a plan that only looks good with literal values shows up here."""
    for _ in range(19):
        _session_page(gateway, page="1")
    before: Final = _tuples_read_quiet()
    _session_page(gateway, page="1")
    tuples_read: Final = _tuples_read_quiet() - before
    assert tuples_read < MAX_TUPLES_PER_PAGE, (
        f"the 20th page read {tuples_read} tuples for a window of {SEED_ROWS} rows and page_size {PAGE_SIZE}"
    )


def test_session_view_uses_newest_row_inside_the_window(gateway: Gateway, seeded: str) -> None:
    page: Final = _session_page(gateway, page="1", session_id=f"{seeded}-straddle")
    assert page["total"] == 1
    data: Final = _data_rows(page)
    assert [row["request_id"] for row in data] == [f"intg-straddle-{seeded}-in"]


def test_session_view_filtered_first_page_stays_bounded(gateway: Gateway, seeded: str) -> None:
    """A filter on a column no index covers is where the planner once drove every per-row probe off the wrong index."""
    before: Final = _tuples_read_quiet()
    page: Final = _session_page(gateway, page="1", key_alias=KEY_ALIAS)
    tuples_read: Final = _tuples_read_quiet() - before

    assert page["total"] == EXPECTED_SESSIONS
    assert _data_rows(page)[0]["request_id"] == f"intg-sesswin-{seeded}-m1c"
    assert tuples_read < MAX_TUPLES_PER_PAGE, (
        f"filtered first page read {tuples_read} tuples for a window of {SEED_ROWS} rows and page_size {PAGE_SIZE}"
    )


def _small_rows(marker: str) -> tuple[_SeedRow, ...]:
    def at(seconds: int) -> str:
        return (_SMALL_BASE + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")

    return (
        _SeedRow(f"{marker}-adopter", f"{marker}-root", at(10)),
        _SeedRow(f"{marker}-root", None, at(20)),
        _SeedRow(f"{marker}-adopter2", f"{marker}-root2", at(30)),
        _SeedRow(f"{marker}-root2", "", at(40)),
        _SeedRow(f"{marker}-shared-a", f"{marker}-shared", at(50), api_key=f"{marker}-key-a"),
        _SeedRow(f"{marker}-shared-b", f"{marker}-shared", at(60), api_key=f"{marker}-key-b"),
        _SeedRow(f"{marker}-tie-1", f"{marker}-tie", at(70)),
        _SeedRow(f"{marker}-tie-2", f"{marker}-tie", at(70)),
        _SeedRow(f"{marker}-mcp-llm", f"{marker}-mcp", at(80)),
        _SeedRow(f"{marker}-mcp-tool", f"{marker}-mcp", at(90), call_type="call_mcp_tool"),
    )


@pytest.fixture
def small_seed(gateway: Gateway) -> Iterator[str]:
    marker: Final = f"intg-sesssmall-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(_delete_small_rows, marker)
        _insert_rows(_small_rows(marker))
        yield marker


def _delete_small_rows(marker: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (f"{marker}-%",))


def _small_page(gateway: Gateway, **extra: str) -> dict[str, JsonValue]:
    return gateway.get(
        "/spend/logs/ui",
        params={
            "group_by_session": "true",
            "start_date": SMALL_WINDOW_START,
            "end_date": SMALL_WINDOW_END,
            "sort_by": "startTime",
            **extra,
        },
    )


def _sessions(page: Mapping[str, JsonValue]) -> list[tuple[str, str]]:
    return [(_tie_as_one(string_value(row["request_id"])), string_value(row["api_key"])) for row in _data_rows(page)]


def _tie_as_one(request_id: str) -> str:
    return request_id.removesuffix("-1").removesuffix("-2") if "-tie-" in request_id else request_id


def _newest_first(marker: str) -> list[tuple[str, str]]:
    return [
        (f"{marker}-mcp-llm", "K"),
        (f"{marker}-tie", "K"),
        (f"{marker}-shared-b", f"{marker}-key-b"),
        (f"{marker}-shared-a", f"{marker}-key-a"),
        (f"{marker}-root2", "K"),
        (f"{marker}-root", "K"),
    ]


def test_session_view_lists_each_session_once(gateway: Gateway, small_seed: str) -> None:
    page: Final = _small_page(gateway, sort_order="desc", page_size="50", page="1")

    assert _sessions(page) == _newest_first(small_seed), (
        "each session once, represented by its newest non-MCP row: a row without a session_id newer than the rows "
        "adopting its request_id joins them, two rows at one startTime are one session, and one session_id under two "
        "api_keys is two sessions"
    )
    assert page["total"] == 6
    assert page["has_more"] is False


def test_session_view_cursor_walk_counts_sessions_per_api_key(gateway: Gateway, small_seed: str) -> None:
    first: Final = _small_page(gateway, sort_order="desc", page_size="2", page="1")
    cursor_pages: Final = [first]
    while cursor_pages[-1]["has_more"] is True:
        cursor_pages.append(
            _small_page(
                gateway,
                sort_order="desc",
                page_size="2",
                page=str(len(cursor_pages) + 1),
                session_cursor=string_value(cursor_pages[-1]["next_session_cursor"]),
            )
        )
    walked: Final = [session for page in cursor_pages for session in _sessions(page)]

    assert first["total"] == 6, "the capped count must count one session_id under two api_keys as two sessions"
    assert walked == _newest_first(small_seed)
    assert [page["has_more"] for page in cursor_pages] == [True, True, False]


def test_session_view_ascending_and_offset_pages_match_the_newest_first_order(
    gateway: Gateway, small_seed: str
) -> None:
    ascending: Final = _small_page(gateway, sort_order="asc", page_size="50", page="1")
    offset_page: Final = _small_page(gateway, sort_order="desc", page_size="2", page="2")

    assert _sessions(ascending) == _newest_first(small_seed)[::-1]
    assert ascending["total"] == 6
    assert _sessions(offset_page) == _newest_first(small_seed)[2:4]
    assert offset_page["has_more"] is True
