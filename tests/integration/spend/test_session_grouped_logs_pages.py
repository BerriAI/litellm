import itertools
import json
import os
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from prisma import Prisma
from psycopg import sql
from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy.spend_tracking.session_page import read_session_page
from litellm.proxy.spend_tracking.spend_management_endpoints import _fetch_session_representatives
from tests.integration._support.client import Gateway
from tests.integration._support.database import write_rows

_URL: Final = "/spend/logs/ui"
_WINDOW_SECONDS: Final = 3600


@dataclass(frozen=True, slots=True)
class _Row:
    request: str
    session: str | None
    api_key: str
    second: int
    status: str = "success"
    model: str = "m-a"


_ROWS: Final = (
    _Row("z1", "z", "key-1", -1),
    _Row("a1", "a", "key-1", 10),
    _Row("a2", "a", "key-1", 50),
    _Row("b1", "b", "key-1", 20, model="m-b"),
    _Row("b2", "b", "key-1", 60),
    _Row("b3", "b", "key-1", 30),
    _Row("c1", "c", "key-2", 60),
    _Row("d1", "b", "key-2", 40, model="m-b"),
    _Row("e1", None, "key-1", 55, model="m-b"),
    _Row("f1", "", "key-1", 45),
    _Row("g1", "g", "key-3", 5, status="failure"),
    _Row("g2", "g", "key-3", 15, status="failure"),
    *(_Row(f"h{second - 69}", "h", "key-1", second) for second in range(70, 76)),
    _Row("i1", "i", "key-3", 35, status="failure"),
    _Row("i2", "i", "key-3", _WINDOW_SECONDS + 1),
)
_NEWEST_FIRST: Final = (
    ("h", "key-1"),
    ("c", "key-2"),
    ("b", "key-1"),
    ("e1", "key-1"),
    ("a", "key-1"),
    ("f1", "key-1"),
    ("b", "key-2"),
    ("i", "key-3"),
    ("g", "key-3"),
)


class _LogRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    session_id: str | None
    api_key: str
    session_total_count: int


class _Page(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_LogRow, ...]
    total: int
    total_is_capped: bool
    has_more: bool
    next_session_cursor: str | None


@dataclass(frozen=True, slots=True)
class _Seeded:
    prefix: str
    window: Mapping[str, str]

    def session(self, row: _LogRow) -> tuple[str, str]:
        key: Final = row.session_id or row.request_id
        assert key.startswith(self.prefix), key
        return key.removeprefix(self.prefix), row.api_key


@pytest.fixture
def seeded(gateway: Gateway) -> Iterator[_Seeded]:
    prefix: Final = f"sess{uuid.uuid4().hex[:12]}-"
    start: Final = datetime(1900, 1, 1) + timedelta(days=int(prefix[4:16], 16) % 200000)
    rows: Final = [
        {
            "request_id": f"{prefix}{row.request}",
            "session_id": None if row.session is None else f"{prefix}{row.session}" if row.session else "",
            "api_key": row.api_key,
            "start_time": (start + timedelta(seconds=row.second)).isoformat(),
            "status": row.status,
            "model": row.model,
        }
        for row in _ROWS
    ]
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(
            write_rows, 'DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (f"{prefix}%",)
        )
        write_rows(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, "startTime", "endTime", model, '
            "session_id, status) SELECT request_id, 'acompletion', api_key, start_time, start_time, model, session_id, "
            "status FROM jsonb_to_recordset(%s::jsonb) AS seeded(request_id text, session_id text, api_key text, "
            "start_time timestamp, status text, model text)",
            (json.dumps(rows),),
        )
        yield _Seeded(
            prefix,
            {
                "start_date": f"{start:%Y-%m-%d %H:%M:%S}",
                "end_date": f"{start + timedelta(seconds=_WINDOW_SECONDS):%Y-%m-%d %H:%M:%S}",
                "group_by_session": "true",
            },
        )


def _page(gateway: Gateway, seeded: _Seeded, params: Mapping[str, str]) -> _Page:
    return _Page.model_validate(gateway.get(_URL, {**seeded.window, **params}))


def _cursor_walk(gateway: Gateway, seeded: _Seeded, params: Mapping[str, str], number: int = 1) -> tuple[_Page, ...]:
    page: Final = _page(gateway, seeded, {**params, "page_size": "2", "page": str(number)})
    if page.next_session_cursor is None or number > len(_NEWEST_FIRST):
        return (page,)
    following: Final = {**params, "session_cursor": page.next_session_cursor}
    return (page, *_cursor_walk(gateway, seeded, following, number + 1))


def _rows(pages: tuple[_Page, ...]) -> Iterator[_LogRow]:
    return itertools.chain.from_iterable(page.data for page in pages)


def _sessions(seeded: _Seeded, pages: tuple[_Page, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(seeded.session(row) for row in _rows(pages))


def test_cursor_pages_show_every_session_once_newest_first_with_its_newest_row(
    gateway: Gateway, seeded: _Seeded
) -> None:
    pages: Final = _cursor_walk(gateway, seeded, {})
    assert _sessions(seeded, pages) == _NEWEST_FIRST
    assert [(page.total, page.total_is_capped) for page in pages] == [(9, False)] * 5
    assert [page.has_more for page in pages] == [True, True, True, True, False]
    representatives: Final = tuple(
        (row.request_id.removeprefix(seeded.prefix), row.session_total_count) for row in _rows(pages)
    )
    assert representatives == (
        ("h6", 6),
        ("c1", 1),
        ("b2", 3),
        ("e1", 1),
        ("a2", 2),
        ("f1", 1),
        ("d1", 1),
        ("i1", 2),
        ("g2", 2),
    )


@pytest.mark.parametrize(
    "sort_order, expected",
    [("desc", _NEWEST_FIRST), ("asc", tuple(reversed(_NEWEST_FIRST)))],
    ids=["newest-first", "oldest-first"],
)
def test_numbered_pages_show_the_same_sessions_as_the_cursor_walk(
    gateway: Gateway, seeded: _Seeded, sort_order: str, expected: tuple[tuple[str, str], ...]
) -> None:
    pages: Final = tuple(
        _page(gateway, seeded, {"page_size": "2", "page": str(number), "sort_order": sort_order})
        for number in range(1, 7)
    )
    assert _sessions(seeded, pages) == expected
    assert [len(page.data) for page in pages] == [2, 2, 2, 2, 1, 0]
    assert [page.has_more for page in pages] == [True, True, True, True, False, False]
    assert {page.total for page in pages} == {len(expected)}


def test_failure_filter_keeps_only_sessions_whose_outcome_failed(gateway: Gateway, seeded: _Seeded) -> None:
    pages: Final = _cursor_walk(gateway, seeded, {"status_filter": "failure"})
    assert _sessions(seeded, pages) == (("i", "key-3"), ("g", "key-3"))
    assert pages[0].total == 2


def test_model_filter_ranks_each_session_by_its_matching_rows_only(gateway: Gateway, seeded: _Seeded) -> None:
    pages: Final = _cursor_walk(gateway, seeded, {"model": "m-b"})
    assert _sessions(seeded, pages) == (("e1", "key-1"), ("b", "key-2"), ("b", "key-1"))
    assert tuple(row.request_id.removeprefix(seeded.prefix) for row in _rows(pages)) == ("e1", "d1", "b1")
    assert pages[0].total == 3


_COPIED_ROWS: Final = 20_000
_COPY_START: Final = datetime(2000, 1, 1, tzinfo=UTC)
_COPY_WINDOW: Final = (
    "\"startTime\" >= ($1::timestamptz AT TIME ZONE 'UTC') AND \"startTime\" <= ($2::timestamptz AT TIME ZONE 'UTC')"
)
_COPY_WINDOW_PARAMS: Final = (_COPY_START, _COPY_START + timedelta(days=1))


def _generated_sessions(rows_per_session: int) -> tuple[sql.SQL, tuple[object, ...]]:
    """`_COPIED_ROWS` rows a second apart in sessions of `rows_per_session` rows, a one-row session keyed by request_id."""
    return sql.SQL(
        'INSERT INTO {table} (request_id, call_type, api_key, "startTime", "endTime", session_id) '
        "SELECT 'r' || n, 'acompletion', 'key-1', moment, moment, "
        "CASE WHEN %s > 1 THEN 's' || (n - 1) / %s END FROM generate_series(1, %s) n, "
        "LATERAL (SELECT %s::timestamp + n * INTERVAL '1 second' AS moment) seconds"
    ), (rows_per_session, rows_per_session, _COPIED_ROWS, _COPY_START.replace(tzinfo=None))


def _sessions_tied_at_the_newest_moment() -> tuple[sql.SQL, tuple[object, ...]]:
    """`_generated_sessions(1)` plus thirty one-row sessions sharing a newer moment, inserted and keyed against session
    order so a read cut inside the tie misses the sessions that rank first."""
    start: Final = _COPY_START.replace(tzinfo=None)
    return sql.SQL(
        'INSERT INTO {table} (request_id, call_type, api_key, "startTime", "endTime", session_id) '
        "SELECT request_id, 'acompletion', 'key-1', moment, moment, session_id FROM ("
        "SELECT n AS ordinal, 'r' || n AS request_id, %s::timestamp + n * INTERVAL '1 second' AS moment, "
        "NULL AS session_id FROM generate_series(1, %s) n UNION ALL "
        "SELECT %s + 30 - n, 't' || lpad((29 - n)::text, 2, '0'), %s::timestamp + %s * INTERVAL '1 second', "
        "'s' || lpad(n::text, 2, '0') FROM generate_series(0, 29) n) seeded ORDER BY ordinal"
    ), (start, _COPIED_ROWS, _COPIED_ROWS, start, _COPIED_ROWS + 1)


@dataclass(frozen=True, slots=True)
class _Call:
    session: str
    call_type: str
    status: str | None
    second: int


def _calls(calls: tuple[_Call, ...]) -> tuple[sql.SQL, tuple[object, ...]]:
    """One row per call, `second` seconds into the window, ending a second later."""
    seeded: Final = tuple(
        {
            "request_id": f"{call.session}-{call.second}",
            "call_type": call.call_type,
            "session_id": call.session,
            "status": call.status,
            "start_time": (_COPY_START + timedelta(seconds=call.second)).replace(tzinfo=None).isoformat(),
        }
        for call in calls
    )
    return sql.SQL(
        'INSERT INTO {table} (request_id, call_type, api_key, "startTime", "endTime", session_id, status) '
        "SELECT request_id, call_type, 'key-1', start_time, start_time + INTERVAL '1 second', session_id, status "
        "FROM jsonb_to_recordset(%s::jsonb) AS calls(request_id text, call_type text, session_id text, status text, "
        "start_time timestamp)"
    ), (json.dumps(seeded),)


@asynccontextmanager
async def _spend_logs_copy(insert: sql.SQL, args: tuple[object, ...]) -> AsyncIterator[Prisma]:
    """A private schema holding a copy of the migrated `LiteLLM_SpendLogs` with its indexes, seeded by `insert`."""
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    table: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(sql.SQL('CREATE TABLE {} (LIKE "LiteLLM_SpendLogs" INCLUDING ALL)').format(table))
            setup.execute(insert.format(table=table), args)
            setup.execute(sql.SQL("ANALYZE {}").format(table))
            parsed: Final = urlsplit(url)
            query: Final = urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})
            database: Final = Prisma(datasource={"url": urlunsplit(parsed._replace(query=query))})
            await database.connect()
            try:
                yield database
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


class _PlanNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation_name: str | None = Field(default=None, alias="Relation Name")
    actual_rows: float = Field(alias="Actual Rows")
    rows_removed_by_filter: float = Field(default=0, alias="Rows Removed by Filter")
    actual_loops: int = Field(alias="Actual Loops")
    plans: tuple["_PlanNode", ...] = Field(default=(), alias="Plans")

    def spend_log_rows_scanned(self) -> int:
        own: Final = (
            (self.actual_rows + self.rows_removed_by_filter) * self.actual_loops
            if self.relation_name == "LiteLLM_SpendLogs"
            else 0
        )
        return round(own) + sum(child.spend_log_rows_scanned() for child in self.plans)


class _Explained(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan: _PlanNode = Field(alias="Plan")


class _ExplainedReads:
    """Runs each read through Prisma, first under EXPLAIN ANALYZE to count the `LiteLLM_SpendLogs` rows it scans."""

    def __init__(self, database: Prisma) -> None:
        self.database: Final = database
        self.rows_scanned: tuple[int, ...] = ()

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        explained: Final = await self.database.query_raw(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}", *args)
        plan: Final = _Explained.model_validate(explained[0]["QUERY PLAN"][0]).plan
        self.rows_scanned = (*self.rows_scanned, plan.spend_log_rows_scanned())
        return await self.database.query_raw(query, *args)


@pytest.mark.parametrize(
    "rows_per_session, page, newest_session, rows_scanned_per_read, total",
    [
        pytest.param(1, 1, ("r", 20_000), (102,), (100, True), id="first-page"),
        pytest.param(1, 3, ("r", 19_980), (102,), (100, True), id="third-page"),
        pytest.param(10, 1, ("s", 1_999), (102, 1_171), (100, True), id="second-read-sized-by-the-first-reads-density"),
        pytest.param(100, 1, ("s", 199), (102, 6_439, 12_878), (100, True), id="third-read-at-least-doubled"),
        pytest.param(1_000, 1, ("s", 19), (102, 12_878, 20_000), (20, False), id="last-read-runs-out-of-rows"),
    ],
)
async def test_a_newest_first_page_scans_only_the_rows_its_reads_ask_for(
    rows_per_session: int,
    page: int,
    newest_session: tuple[str, int],
    rows_scanned_per_read: tuple[int, ...],
    total: tuple[int, bool],
) -> None:
    async with _spend_logs_copy(*_generated_sessions(rows_per_session)) as database:
        reads: Final = _ExplainedReads(database)
        session_page: Final = await read_session_page(
            reads,
            where_clause=_COPY_WINDOW,
            params=_COPY_WINDOW_PARAMS,
            page=page,
            page_size=10,
            status=None,
            sort_desc=True,
            session_cursor=None,
            cap=100,
        )
        prefix, newest = newest_session
        assert tuple(session.key for session in session_page.sessions) == tuple(
            f"{prefix}{newest - n}" for n in range(10)
        )
        assert (session_page.total, session_page.total_is_capped) == total
        assert reads.rows_scanned == rows_scanned_per_read


async def test_a_page_cut_inside_a_tie_reads_on_until_the_tie_is_whole() -> None:
    async with _spend_logs_copy(*_sessions_tied_at_the_newest_moment()) as database:
        reads: Final = _ExplainedReads(database)
        session_page: Final = await read_session_page(
            reads,
            where_clause=_COPY_WINDOW,
            params=_COPY_WINDOW_PARAMS,
            page=1,
            page_size=5,
            status=None,
            sort_desc=True,
            session_cursor=None,
            cap=10,
        )
        assert tuple(session.key for session in session_page.sessions) == ("s29", "s28", "s27", "s26", "s25")
        assert reads.rows_scanned == (12, 24, 48)


@dataclass(frozen=True, slots=True)
class _ExplainedPrisma:
    db: _ExplainedReads


async def test_a_pages_representatives_are_found_by_session_not_by_scanning_the_window() -> None:
    async with _spend_logs_copy(*_generated_sessions(rows_per_session=10)) as database:
        reads: Final = _ExplainedReads(database)
        representatives: Final = await _fetch_session_representatives(
            prisma_client=_ExplainedPrisma(reads),  # pyright: ignore[reportArgumentType]  # only .db is used
            where_clause=_COPY_WINDOW,
            sql_params=_COPY_WINDOW_PARAMS,
            next_param_index=len(_COPY_WINDOW_PARAMS) + 1,
            session_keys=(("s1999", "key-1"), ("s1998", "key-1")),
        )
        assert tuple(row["request_id"] for row in representatives) == ("r20000", "r19990")
        assert reads.rows_scanned == (20,), "only the two sessions' rows, not the window's 20,000"


async def test_a_page_number_far_past_the_cap_shows_no_session() -> None:
    async with _spend_logs_copy(*_generated_sessions(rows_per_session=1)) as database:
        session_page: Final = await read_session_page(
            _ExplainedReads(database),
            where_clause=_COPY_WINDOW,
            params=_COPY_WINDOW_PARAMS,
            page=9_300_000_000_000_000,
            page_size=1000,
            status=None,
            sort_desc=True,
            session_cursor=None,
            cap=100,
        )
        assert (session_page.sessions, session_page.total, session_page.total_is_capped) == ((), 100, True)


_OUTCOME_CALLS: Final = (
    _Call("agent", "asend_message", "failure", 10),
    _Call("agent", "acompletion", "success", 20),
    _Call("mcp", "acompletion", "failure", 5),
    _Call("mcp", "call_mcp_tool", "success", 30),
    _Call("unknown", "acompletion", None, 40),
    _Call("failed", "acompletion", "success", 1),
    _Call("failed", "acompletion", "failure", 50),
    _Call("recovered", "acompletion", "failure", 2),
    _Call("recovered", "acompletion", "success", 45),
)


@pytest.mark.parametrize(
    "status, expected",
    [("failure", ("failed", "mcp", "agent")), ("success", ("recovered", "unknown"))],
)
async def test_a_status_filter_keeps_sessions_whose_representative_call_has_that_status(
    status: str, expected: tuple[str, ...]
) -> None:
    """An agent message stands for its session, then the newest call that is not an MCP tool call, and a missing status
    counts as success. The page stays ordered by each session's newest call."""
    async with _spend_logs_copy(*_calls(_OUTCOME_CALLS)) as database:
        session_page: Final = await read_session_page(
            _ExplainedReads(database),
            where_clause=_COPY_WINDOW,
            params=_COPY_WINDOW_PARAMS,
            status=status,
            page=1,
            page_size=10,
            sort_desc=True,
            session_cursor=None,
            cap=100,
        )
        assert tuple(session.key for session in session_page.sessions) == expected
        assert (session_page.total, session_page.total_is_capped) == (len(expected), False)
