import itertools
import random
import re
from collections.abc import Sequence
from typing import Final, TypeAlias

import pytest

from litellm.proxy.spend_tracking.session_page import (
    Session,
    _parse_session_cursor,
    _Read,
    _ReadRow,
    _Request,
    _session_page,
    read_session_page,
)

_Row: TypeAlias = tuple[int, str, str]  # (startTime, session key, api_key)
_Ranked: TypeAlias = tuple[int, str, str]  # (last activity, session key, api_key)


def _session_of(row: _Row) -> tuple[str, str]:
    return row[1], row[2]


def _ranked(rows: Sequence[_Row], request: _Request) -> tuple[_Ranked, ...]:
    """The sessions in ``rows`` in page order, after the request's cursor or offset."""
    by_session: Final = itertools.groupby(sorted(rows, key=_session_of), key=_session_of)
    ranked: Final = tuple(
        sorted(
            ((max(start for start, _, _ in group), key, api_key) for (key, api_key), group in by_session),
            reverse=request.sort_desc,
        )
    )
    if request.cursor is None:
        return ranked[request.offset :]
    cursor: Final = (int(request.cursor[0]), request.cursor[1], request.cursor[2])
    return tuple(session for session in ranked if (session < cursor if request.sort_desc else session > cursor))


class _Reads:
    """Reads like the page query, whose LIMIT cuts tied rows lowest key first, the opposite of page order. It fails a
    read no larger than the one before, since only growing reads end the loop."""

    def __init__(self, rows: Sequence[_Row], request: _Request, every_row: bool = False) -> None:
        self.newest_first: Final = tuple(sorted(rows, key=lambda row: (-row[0], row[1], row[2])))
        self.request: Final = request
        self.every_row: Final = every_row
        self.budgets: tuple[int | None, ...] = ()

    async def __call__(self, budget: int | None) -> _Read:
        assert not self.budgets or (self.budgets[-1] is not None and (budget is None or budget > self.budgets[-1])), (
            self.budgets,
            budget,
        )
        self.budgets = (*self.budgets, budget)
        taken: Final = self.newest_first if budget is None or self.every_row else self.newest_first[:budget]
        page: Final = _ranked(taken, self.request)[: self.request.page_limit + 1] if self.request.page_limit > 0 else ()
        return _Read(
            candidates=tuple(Session(key, api_key, f"{at:06d}") for at, key, api_key in page),
            last_settled=bool(page) and page[-1][0] > taken[-1][0],
            rows_read=len(taken),
            session_count=len(frozenset(_session_of(row) for row in taken)),
            saw_every_row=budget is None or self.every_row or len(taken) < budget,
        )


def _session_rows(rng: random.Random, index: int, clock: int) -> tuple[_Row, ...]:
    start, length = rng.randrange(0, clock), rng.choice((1, 1, 2, 3, 8, 40))
    return tuple((start + rng.randrange(0, 30), f"s{index:03d}", rng.choice(("k1", "k2"))) for _ in range(length))


def _sessions(rng: random.Random) -> tuple[_Row, ...]:
    """Sessions of one to forty rows over a short clock, so many rows share a time."""
    clock: Final = rng.choice((3, 40, 400))
    return tuple(
        itertools.chain.from_iterable(_session_rows(rng, index, clock) for index in range(rng.randrange(1, 120)))
    )


def _random_cursor(rng: random.Random, ranked: Sequence[_Ranked]) -> tuple[str, str, str]:
    at, key, api_key = ranked[rng.randrange(len(ranked))]
    return f"{at:06d}", key, api_key


def _request(rng: random.Random, rows: Sequence[_Row]) -> _Request:
    cap, page_size, sort_desc = rng.randrange(2, 60), rng.randrange(1, 12), rng.random() < 0.8
    mode: Final = rng.choice(("first", "offset", "cursor"))
    full: Final = _ranked(rows, _Request(None, 0, 0, sort_desc, None, cap))
    cursor: Final = _random_cursor(rng, full) if mode == "cursor" and full else None
    offset: Final = (rng.randrange(1, 2 * cap // page_size + 3) - 1) * page_size if mode == "offset" else 0
    return _Request(cursor, offset, min(page_size, cap - offset), sort_desc, None, cap)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.asyncio
async def test_a_page_read_from_the_newest_rows_matches_grouping_every_row(seed: int) -> None:
    rng: Final = random.Random(seed)
    rows: Final = _sessions(rng)
    for request in tuple(_request(rng, rows) for _ in range(12)):
        grouped = await _session_page(_Reads(rows, request, every_row=True), request)
        assert await _session_page(_Reads(rows, request), request) == grouped, request


@pytest.mark.parametrize(
    "rows_per_session, expected_budgets",
    [
        pytest.param(1, (102,), id="one-read-holds-the-cap"),
        pytest.param(2, (102, 253), id="sized-by-the-first-reads-density"),
        pytest.param(50, (102, 4293, 8586), id="doubles-when-the-density-misleads"),
        pytest.param(500, (102, 12878, 62533), id="re-sized-by-each-reads-density"),
    ],
)
@pytest.mark.asyncio
async def test_reads_grow_by_the_last_reads_rows_per_session_until_one_settles_the_page(
    rows_per_session: int, expected_budgets: tuple[int, ...]
) -> None:
    """150 sessions a row per second apart. A cap of 100 starts at 100 + 2 rows."""
    rows: Final = tuple((index, f"s{index // rows_per_session:05d}", "k") for index in range(150 * rows_per_session))
    request: Final = _Request(cursor=None, offset=0, page_limit=10, sort_desc=True, status=None, cap=100)
    reads: Final = _Reads(rows, request)
    await _session_page(reads, request)
    assert reads.budgets == expected_budgets


@pytest.mark.asyncio
async def test_a_session_tied_with_the_reads_oldest_row_is_not_trusted() -> None:
    """Thirty one-row sessions share one time and the first read takes 12 of them. The page's sixth session sits at the
    read's oldest time, where the LIMIT may have cut a session that ranks before it, so the page reads on."""
    rows: Final = tuple((0, f"s{index:02d}", "k") for index in range(30))
    request: Final = _Request(cursor=None, offset=0, page_limit=5, sort_desc=True, status=None, cap=10)
    reads: Final = _Reads(rows, request)
    page: Final = await _session_page(reads, request)
    assert tuple(session.key for session in page.sessions) == ("s29", "s28", "s27", "s26", "s25")
    assert reads.budgets == (12, 24, 48)


@pytest.mark.parametrize("session_count, expected_total", [(100, (100, False)), (101, (100, True))])
@pytest.mark.asyncio
async def test_the_total_is_capped_only_past_the_cap(session_count: int, expected_total: tuple[int, bool]) -> None:
    rows: Final = tuple((index, f"s{index:05d}", "k") for index in range(session_count))
    request: Final = _Request(cursor=None, offset=0, page_limit=10, sort_desc=True, status=None, cap=100)
    page: Final = await _session_page(_Reads(rows, request), request)
    assert (page.total, page.total_is_capped) == expected_total


@pytest.mark.asyncio
async def test_a_page_number_near_the_bigint_limit_reads_once() -> None:
    rows: Final = tuple((index, f"s{index:05d}", "k") for index in range(500))
    offset: Final = (9_300_000_000_000_000 - 1) * 1000
    request: Final = _Request(
        cursor=None, offset=offset, page_limit=min(1000, 100 - offset), sort_desc=True, status=None, cap=100
    )
    reads: Final = _Reads(rows, request)
    page: Final = await _session_page(reads, request)
    assert reads.budgets == (102,)
    assert (page.sessions, page.total, page.total_is_capped) == ((), 100, True)


@pytest.mark.parametrize(
    "sort_desc, status",
    [pytest.param(False, None, id="oldest-first"), pytest.param(True, "failure", id="filtered-by-outcome")],
)
@pytest.mark.asyncio
async def test_a_page_that_needs_every_row_reads_them_once(sort_desc: bool, status: str | None) -> None:
    rows: Final = tuple((index, f"s{index:05d}", "k") for index in range(500))
    request: Final = _Request(cursor=None, offset=0, page_limit=10, sort_desc=sort_desc, status=status, cap=100)
    reads: Final = _Reads(rows, request)
    await _session_page(reads, request)
    assert reads.budgets == (None,)


def test_parse_session_cursor() -> None:
    assert _parse_session_cursor(None) is None
    assert _parse_session_cursor("no-delimiter") is None
    assert _parse_session_cursor("2026-08-29 10:00:00|sess-1") is None
    assert _parse_session_cursor("|hashed-key|sess-1") is None
    assert _parse_session_cursor("2026-08-29 10:00:00|hashed-key|") is None
    assert _parse_session_cursor("2026-08-29 10:00:00|hashed-key|sess-1") == (
        "2026-08-29 10:00:00",
        "sess-1",
        "hashed-key",
    )
    assert _parse_session_cursor("2026-08-29 10:00:00.123|hashed-key|sess|with|pipes") == (
        "2026-08-29 10:00:00.123",
        "sess|with|pipes",
        "hashed-key",
    )


def _bound(sql: str, args: tuple[object, ...], pattern: str) -> tuple[object, ...]:
    """The values bound to the ``$N`` placeholders that ``pattern``'s groups capture in ``sql``."""
    found: Final = re.search(pattern, sql)
    assert found, f"{pattern!r} not found in:\n{sql}"
    assert all(1 <= int(index) <= len(args) for index in found.groups()), f"{found.groups()} outside $1..${len(args)}"
    return tuple(args[int(index) - 1] for index in found.groups())


class _TableDb:
    """A table of ``rows`` matching rows in ``sessions`` sessions, whose reads return no page row."""

    def __init__(self, rows: int, sessions: int) -> None:
        self.rows: Final = rows
        self.sessions: Final = sessions
        self.calls: tuple[tuple[str, tuple[object, ...]], ...] = ()

    async def query_raw(self, query: str, *args: object) -> tuple[_ReadRow, ...]:
        self.calls = (*self.calls, (query, args))
        budget: Final = re.search(r"DESC LIMIT (\d+)", query)
        rows_read: Final = min(int(budget.group(1)), self.rows) if budget else self.rows
        return (
            _ReadRow(
                session_key=None,
                api_key=None,
                last_activity=None,
                settled=None,
                rows_read=rows_read,
                session_count=self.sessions,
            ),
        )


def _page_bounds(sql: str) -> tuple[int, int]:
    """The LIMIT and OFFSET of the read's sessions."""
    found: Final = re.search(r"LIMIT (\d+)(?: OFFSET (\d+))?\s*\) AS page", sql)
    assert found, sql
    return int(found.group(1)), int(found.group(2) or 0)


def _read_size(sql: str) -> int | None:
    found: Final = re.search(r"DESC LIMIT (\d+)", sql)
    return int(found.group(1)) if found else None


@pytest.mark.asyncio
async def test_a_read_that_fills_its_size_is_not_taken_for_every_row() -> None:
    """A read that took every row it asked for may have stopped short, so its 50 sessions are no total: the reads
    grow until one takes fewer rows than it asked for."""
    db: Final = _TableDb(rows=9_000, sessions=50)
    page: Final = await read_session_page(
        db,
        where_clause="TRUE",
        params=(),
        page=1,
        page_size=10,
        status=None,
        sort_desc=True,
        session_cursor=None,
        cap=100,
    )
    assert tuple(_read_size(sql) for sql, _ in db.calls) == (102, 258, 652, 1647, 4159, 10502)
    assert (page.total, page.total_is_capped) == (50, False)


@pytest.mark.parametrize("page_number", [pytest.param(5, id="at-the-cap"), pytest.param(6, id="past-the-cap")])
@pytest.mark.asyncio
async def test_a_page_starting_at_or_past_the_cap_reads_once_and_shows_no_session(page_number: int) -> None:
    db: Final = _TableDb(rows=1_000, sessions=150)
    page: Final = await read_session_page(
        db,
        where_clause="TRUE",
        params=(),
        page=page_number,
        page_size=25,
        status=None,
        sort_desc=True,
        session_cursor=None,
        cap=100,
    )
    assert len(db.calls) == 1, "the first read already holds more sessions than the cap"
    assert _page_bounds(db.calls[0][0])[0] == 0
    assert (page.total, page.total_is_capped) == (100, True)


@pytest.mark.asyncio
async def test_the_read_binds_the_filters_and_the_cursor_to_their_placeholders() -> None:
    db: Final = _TableDb(rows=50, sessions=40)
    where, params = "model = $1 AND api_key = $2", ("gpt-4o", "key-a")
    for page, page_size, session_cursor in ((1, 3, "2026-10-28 09:00:00|key-a|sess|9"), (3, 1, None)):
        await read_session_page(
            db,
            where_clause=where,
            params=params,
            page=page,
            page_size=page_size,
            status=None,
            sort_desc=True,
            session_cursor=session_cursor,
            cap=100,
        )
    (cursor_sql, cursor_args), (offset_sql, _) = db.calls
    for sql, args in db.calls:
        assert {int(index) for index in re.findall(r"\$(\d+)", sql)} == set(range(1, len(args) + 1)), (
            "every argument needs a placeholder and every placeholder an argument"
        )
        assert _bound(sql, args, r"model = \$(\d+) AND api_key = \$(\d+)") == params
    assert _bound(
        cursor_sql, cursor_args, r"\(last_activity, session_key, api_key\) < \(\$(\d+)::timestamp, \$(\d+), \$(\d+)\)"
    ) == ("2026-10-28 09:00:00", "sess|9", "key-a")
    assert _page_bounds(cursor_sql) == (4, 0)
    assert _page_bounds(offset_sql) == (2, 2)


@pytest.mark.asyncio
async def test_a_status_filtered_read_binds_the_status_between_the_filters_and_the_cursor() -> None:
    db: Final = _TableDb(rows=50, sessions=40)
    await read_session_page(
        db,
        where_clause="model = $1",
        params=("gpt-4o",),
        status="failure",
        page=1,
        page_size=3,
        sort_desc=True,
        session_cursor="2026-10-28 09:00:00|key-a|sess-9",
        cap=100,
    )
    ((sql, args),) = db.calls
    assert {int(index) for index in re.findall(r"\$(\d+)", sql)} == set(range(1, len(args) + 1))
    assert _bound(sql, args, r"model = \$(\d+)") == ("gpt-4o",)
    assert _bound(sql, args, r"COALESCE\(status, 'success'\) = \$(\d+)") == ("failure",)
    assert _bound(sql, args, r"\(last_activity, session_key, api_key\) < \(\$(\d+)::timestamp, \$(\d+), \$(\d+)\)") == (
        "2026-10-28 09:00:00",
        "sess-9",
        "key-a",
    )
