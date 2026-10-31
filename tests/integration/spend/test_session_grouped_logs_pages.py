import json
import os
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from prisma import Prisma
from psycopg import sql
from pydantic import BaseModel, ConfigDict

from litellm.proxy.spend_tracking.session_page import read_session_page
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


def _page(gateway: Gateway, seeded: _Seeded, **params: str) -> _Page:
    return _Page.model_validate(gateway.get(_URL, {**seeded.window, **params}))


def _cursor_walk(gateway: Gateway, seeded: _Seeded, number: int = 1, **params: str) -> tuple[_Page, ...]:
    page: Final = _page(gateway, seeded, page_size="2", page=str(number), **params)
    if page.next_session_cursor is None or number > len(_NEWEST_FIRST):
        return (page,)
    following: Final = {**params, "session_cursor": page.next_session_cursor}
    return (page, *_cursor_walk(gateway, seeded, number + 1, **following))


def _sessions(seeded: _Seeded, pages: tuple[_Page, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(seeded.session(row) for page in pages for row in page.data)


def test_cursor_pages_show_every_session_once_newest_first_with_its_newest_row(
    gateway: Gateway, seeded: _Seeded
) -> None:
    pages: Final = _cursor_walk(gateway, seeded)
    assert _sessions(seeded, pages) == _NEWEST_FIRST
    assert [(page.total, page.total_is_capped) for page in pages] == [(len(_NEWEST_FIRST), False)] * len(pages)
    assert [page.has_more for page in pages] == [True] * (len(pages) - 1) + [False]
    representatives: Final = tuple(
        (row.request_id.removeprefix(seeded.prefix), row.session_total_count) for page in pages for row in page.data
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
        _page(gateway, seeded, page_size="2", page=str(number), sort_order=sort_order) for number in range(1, 7)
    )
    assert _sessions(seeded, pages) == expected
    assert [len(page.data) for page in pages] == [2, 2, 2, 2, 1, 0]
    assert [page.has_more for page in pages] == [True, True, True, True, False, False]
    assert {page.total for page in pages} == {len(expected)}


def test_failure_filter_keeps_only_sessions_whose_outcome_failed(gateway: Gateway, seeded: _Seeded) -> None:
    pages: Final = _cursor_walk(gateway, seeded, status_filter="failure")
    assert _sessions(seeded, pages) == (("i", "key-3"), ("g", "key-3"))
    assert pages[0].total == 2


def test_model_filter_ranks_each_session_by_its_matching_rows_only(gateway: Gateway, seeded: _Seeded) -> None:
    pages: Final = _cursor_walk(gateway, seeded, model="m-b")
    assert _sessions(seeded, pages) == (("e1", "key-1"), ("b", "key-2"), ("b", "key-1"))
    assert [row.request_id.removeprefix(seeded.prefix) for page in pages for row in page.data] == ["e1", "d1", "b1"]
    assert pages[0].total == 3


_COPIED_ROWS: Final = 20_000
_COPY_START: Final = datetime(2000, 1, 1, tzinfo=UTC)


@asynccontextmanager
async def _spend_logs_copy(rows_per_session: int) -> AsyncIterator[Prisma]:
    """A private schema holding a copy of the migrated `LiteLLM_SpendLogs` with its indexes, seeded with
    `_COPIED_ROWS` rows a second apart in sessions of `rows_per_session` rows, a one-row session keyed by request_id."""
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    table: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(sql.SQL('CREATE TABLE {} (LIKE "LiteLLM_SpendLogs" INCLUDING ALL)').format(table))
            setup.execute(
                sql.SQL(
                    'INSERT INTO {} (request_id, call_type, api_key, "startTime", "endTime", session_id) '
                    "SELECT 'r' || n, 'acompletion', 'key-1', moment, moment, "
                    "CASE WHEN %s > 1 THEN 's' || (n - 1) / %s END FROM generate_series(1, %s) n, "
                    "LATERAL (SELECT %s::timestamp + n * INTERVAL '1 second' AS moment) seconds"
                ).format(table),
                (rows_per_session, rows_per_session, _COPIED_ROWS, _COPY_START.replace(tzinfo=None)),
            )
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


def _spend_log_rows_scanned(node: Mapping[str, Any]) -> int:
    own: Final = (
        (node["Actual Rows"] + node.get("Rows Removed by Filter", 0)) * node["Actual Loops"]
        if node.get("Relation Name") == "LiteLLM_SpendLogs"
        else 0
    )
    return own + sum(_spend_log_rows_scanned(child) for child in node.get("Plans", ()))


@dataclass(frozen=True, slots=True)
class _ExplainedReads:
    """Runs each read through Prisma, first under EXPLAIN ANALYZE to count the `LiteLLM_SpendLogs` rows it scans."""

    database: Prisma
    rows_scanned: list[int]

    async def query_raw(self, query: str, *args: object) -> list[dict[str, Any]]:
        explained: Final = await self.database.query_raw(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}", *args)
        self.rows_scanned.append(_spend_log_rows_scanned(explained[0]["QUERY PLAN"][0]["Plan"]))
        return await self.database.query_raw(query, *args)


@pytest.mark.parametrize(
    "rows_per_session, page, newest_session, rows_asked_for",
    [
        pytest.param(1, 1, ("r", 20_000), [112], id="first-page"),
        pytest.param(1, 3, ("r", 19_980), [132], id="third-page"),
        pytest.param(10, 1, ("s", 1_999), [112, 1_179], id="second-read-sized-by-the-first-reads-density"),
        pytest.param(100, 1, ("s", 199), [112, 7_070, 20_000], id="every-row-once-a-read-would-take-half-the-window"),
    ],
)
async def test_a_newest_first_page_scans_only_the_rows_its_reads_ask_for(
    rows_per_session: int, page: int, newest_session: tuple[str, int], rows_asked_for: list[int]
) -> None:
    async with _spend_logs_copy(rows_per_session) as database:
        rows_scanned: Final[list[int]] = []
        session_page: Final = await read_session_page(
            _ExplainedReads(database, rows_scanned),
            "\"startTime\" >= ($1::timestamptz AT TIME ZONE 'UTC') AND \"startTime\" <= ($2::timestamptz AT TIME ZONE 'UTC')",
            (_COPY_START, _COPY_START + timedelta(days=1)),
            "",
            page=page,
            page_size=10,
            sort_desc=True,
            session_cursor=None,
            cap=100,
            window_start=_COPY_START,
        )
        prefix, newest = newest_session
        assert [session.key for session in session_page.sessions] == [f"{prefix}{newest - n}" for n in range(10)]
        assert (session_page.total, session_page.total_is_capped) == (100, True)
        assert rows_scanned == rows_asked_for
