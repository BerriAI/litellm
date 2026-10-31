import random
import re

import pytest

from litellm.proxy.spend_tracking.session_page import (
    Session,
    _parse_session_cursor,
    _Read,
    _Request,
    _session_page,
    read_session_page,
)

_Row = tuple[int, str, str]  # (startTime, session key, api_key)


def _ranked(rows: list[_Row], request: _Request) -> list[tuple[int, str, str]]:
    """The sessions in ``rows`` in page order, after the request's cursor or offset."""
    last_activity: dict[tuple[str, str], int] = {}
    for start, key, api_key in rows:
        last_activity[(key, api_key)] = max(last_activity.get((key, api_key), start), start)
    ranked = sorted(((at, key, api_key) for (key, api_key), at in last_activity.items()), reverse=request.sort_desc)
    if request.cursor is None:
        return ranked[request.offset :]
    cursor = (int(request.cursor[0]), request.cursor[1], request.cursor[2])
    return [session for session in ranked if (session < cursor if request.sort_desc else session > cursor)]


def _reader(rows: list[_Row], request: _Request, budgets: list[int | None], every_row: bool = False):
    """Reads like the page query, whose LIMIT cuts tied rows lowest key first, the opposite of page order. It fails a
    read no larger than the one before, since only growing reads end the loop."""
    newest_first = sorted(rows, key=lambda row: (-row[0], row[1], row[2]))

    async def read(budget: int | None) -> _Read:
        assert not budgets or (budgets[-1] is not None and (budget is None or budget > budgets[-1])), (budgets, budget)
        budgets.append(budget)
        taken = newest_first if budget is None or every_row else newest_first[:budget]
        page = _ranked(taken, request)[: request.page_limit + 1] if request.page_limit > 0 else []
        return _Read(
            candidates=tuple(Session(key, api_key, f"{at:06d}") for at, key, api_key in page),
            last_settled=bool(page) and page[-1][0] > taken[-1][0],
            rows_read=len(taken),
            session_count=len({(key, api_key) for _, key, api_key in taken}),
            saw_every_row=budget is None or every_row or len(taken) < budget,
        )

    return read


def _sessions(rng: random.Random) -> list[_Row]:
    """Sessions of one to forty rows over a short clock, so many rows share a time."""
    rows: list[_Row] = []
    clock = rng.choice((3, 40, 400))
    for index in range(rng.randrange(1, 120)):
        start, length = rng.randrange(0, clock), rng.choice((1, 1, 2, 3, 8, 40))
        rows.extend((start + rng.randrange(0, 30), f"s{index:03d}", rng.choice(("k1", "k2"))) for _ in range(length))
    return rows


def _requests(rng: random.Random, rows: list[_Row]):
    for _ in range(12):
        cap, page_size, sort_desc = rng.randrange(2, 60), rng.randrange(1, 12), rng.random() < 0.8
        mode = rng.choice(("first", "offset", "cursor"))
        full = _ranked(rows, _Request(None, 0, 0, sort_desc, None, cap))
        cursor = None
        if mode == "cursor" and full:
            at, key, api_key = full[rng.randrange(len(full))]
            cursor = (f"{at:06d}", key, api_key)
        offset = (rng.randrange(1, 2 * cap // page_size + 3) - 1) * page_size if mode == "offset" else 0
        yield _Request(cursor, offset, min(page_size, cap - offset), sort_desc, None, cap)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.asyncio
async def test_a_page_read_from_the_newest_rows_matches_grouping_every_row(seed):
    rng = random.Random(seed)
    rows = _sessions(rng)
    for request in _requests(rng, rows):
        grouped = await _session_page(_reader(rows, request, [], every_row=True), request)
        assert await _session_page(_reader(rows, request, []), request) == grouped, request


@pytest.mark.parametrize(
    "rows_per_session, expected_budgets",
    [
        pytest.param(1, [102], id="one-read-holds-the-cap"),
        pytest.param(2, [102, 253], id="sized-by-the-first-reads-density"),
        pytest.param(50, [102, 4293, 8586], id="doubles-when-the-density-misleads"),
        pytest.param(500, [102, 12878, 62533], id="re-sized-by-each-reads-density"),
    ],
)
@pytest.mark.asyncio
async def test_reads_grow_by_the_last_reads_rows_per_session_until_one_settles_the_page(
    rows_per_session, expected_budgets
):
    """150 sessions a row per second apart. A cap of 100 starts at 100 + 2 rows."""
    rows = [(index, f"s{index // rows_per_session:05d}", "k") for index in range(150 * rows_per_session)]
    request = _Request(cursor=None, offset=0, page_limit=10, sort_desc=True, status=None, cap=100)
    budgets: list[int | None] = []
    await _session_page(_reader(rows, request, budgets), request)
    assert budgets == expected_budgets


@pytest.mark.parametrize(
    "rows_read, session_count, expected",
    [
        pytest.param(1000, 10, 12625, id="rows-per-session-with-a-quarter-to-spare"),
        pytest.param(1000, 7, 18036, id="rounded-up"),
        pytest.param(1000, 900, 2000, id="at-least-twice-the-last-read"),
    ],
)
def test_the_next_read_is_sized_by_rows_per_session_and_at_least_doubled(rows_read, session_count, expected):
    read = _Read((), False, rows_read, session_count, saw_every_row=False)
    assert (
        read.next_size(_Request(cursor=None, offset=0, page_limit=10, sort_desc=True, status=None, cap=100)) == expected
    )


@pytest.mark.asyncio
async def test_a_session_tied_with_the_reads_oldest_row_is_not_trusted():
    """Thirty one-row sessions share one time and the first read takes 12 of them. The page's sixth session sits at the
    read's oldest time, where the LIMIT may have cut a session that ranks before it, so the page reads on."""
    rows = [(0, f"s{index:02d}", "k") for index in range(30)]
    request = _Request(cursor=None, offset=0, page_limit=5, sort_desc=True, status=None, cap=10)
    budgets: list[int | None] = []
    page = await _session_page(_reader(rows, request, budgets), request)
    assert [session.key for session in page.sessions] == ["s29", "s28", "s27", "s26", "s25"]
    assert budgets == [12, 24, 48]


@pytest.mark.parametrize("session_count, expected_total", [(100, (100, False)), (101, (100, True))])
@pytest.mark.asyncio
async def test_the_total_is_capped_only_past_the_cap(session_count, expected_total):
    rows = [(index, f"s{index:05d}", "k") for index in range(session_count)]
    request = _Request(cursor=None, offset=0, page_limit=10, sort_desc=True, status=None, cap=100)
    page = await _session_page(_reader(rows, request, []), request)
    assert (page.total, page.total_is_capped) == expected_total


@pytest.mark.asyncio
async def test_a_page_number_near_the_bigint_limit_reads_once():
    rows = [(index, f"s{index:05d}", "k") for index in range(500)]
    offset = (9_300_000_000_000_000 - 1) * 1000
    request = _Request(
        cursor=None, offset=offset, page_limit=min(1000, 100 - offset), sort_desc=True, status=None, cap=100
    )
    budgets: list[int | None] = []
    page = await _session_page(_reader(rows, request, budgets), request)
    assert budgets == [102]
    assert (page.sessions, page.total, page.total_is_capped) == ((), 100, True)


@pytest.mark.parametrize(
    "sort_desc, status",
    [pytest.param(False, None, id="oldest-first"), pytest.param(True, "failure", id="filtered-by-outcome")],
)
@pytest.mark.asyncio
async def test_a_page_that_needs_every_row_reads_them_once(sort_desc, status):
    rows = [(index, f"s{index:05d}", "k") for index in range(500)]
    request = _Request(cursor=None, offset=0, page_limit=10, sort_desc=sort_desc, status=status, cap=100)
    budgets: list[int | None] = []
    await _session_page(_reader(rows, request, budgets), request)
    assert budgets == [None]


def test_parse_session_cursor():
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
    found = re.search(pattern, sql)
    assert found, f"{pattern!r} not found in:\n{sql}"
    assert all(1 <= int(index) <= len(args) for index in found.groups()), f"{found.groups()} outside $1..${len(args)}"
    return tuple(args[int(index) - 1] for index in found.groups())


class _TableDb:
    """A table of ``rows`` matching rows in ``sessions`` sessions, whose reads return no page row."""

    def __init__(self, calls: list[tuple[str, tuple[object, ...]]], rows: int, sessions: int):
        self.calls, self.rows, self.sessions = calls, rows, sessions

    async def query_raw(self, query: str, *args: object):
        self.calls.append((query, args))
        budget = re.search(r"DESC LIMIT (\d+)", query)
        rows_read = min(int(budget.group(1)), self.rows) if budget else self.rows
        return [
            {
                "session_key": None,
                "api_key": None,
                "last_activity": None,
                "settled": None,
                "rows_read": rows_read,
                "session_count": self.sessions,
            }
        ]


def _page_bounds(sql: str) -> tuple[int, int]:
    """The LIMIT and OFFSET of the read's sessions."""
    found = re.search(r"LIMIT (\d+)(?: OFFSET (\d+))?\s*\) AS page", sql)
    assert found, sql
    return int(found.group(1)), int(found.group(2) or 0)


@pytest.mark.asyncio
async def test_a_read_that_fills_its_size_is_not_taken_for_every_row():
    """A read that took every row it asked for may have stopped short, so its 50 sessions are no total: the reads
    grow until one takes fewer rows than it asked for."""
    calls: list[tuple[str, tuple[object, ...]]] = []
    page = await read_session_page(
        _TableDb(calls, rows=9_000, sessions=50),
        where_clause="TRUE",
        params=(),
        page=1,
        page_size=10,
        status=None,
        sort_desc=True,
        session_cursor=None,
        cap=100,
    )
    sizes = [re.search(r"DESC LIMIT (\d+)", sql) for sql, _ in calls]
    assert [int(found.group(1)) if found else None for found in sizes] == [102, 258, 652, 1647, 4159, 10502]
    assert (page.total, page.total_is_capped) == (50, False)


@pytest.mark.parametrize("page_number", [pytest.param(5, id="at-the-cap"), pytest.param(6, id="past-the-cap")])
@pytest.mark.asyncio
async def test_a_page_starting_at_or_past_the_cap_reads_once_and_shows_no_session(page_number):
    calls: list[tuple[str, tuple[object, ...]]] = []
    page = await read_session_page(
        _TableDb(calls, rows=1_000, sessions=150),
        where_clause="TRUE",
        params=(),
        page=page_number,
        page_size=25,
        status=None,
        sort_desc=True,
        session_cursor=None,
        cap=100,
    )
    assert len(calls) == 1, "the first read already holds more sessions than the cap"
    assert _page_bounds(calls[0][0])[0] == 0
    assert (page.sessions, page.next_cursor) == ((), None)
    assert (page.total, page.total_is_capped) == (100, True)


@pytest.mark.asyncio
async def test_the_read_binds_the_filters_and_the_cursor_to_their_placeholders():
    calls: list[tuple[str, tuple[object, ...]]] = []
    db = _TableDb(calls, rows=50, sessions=40)
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
    (cursor_sql, cursor_args), (offset_sql, offset_args) = calls
    for sql, args in calls:
        assert {int(index) for index in re.findall(r"\$(\d+)", sql)} == set(range(1, len(args) + 1)), (
            "every argument needs a placeholder and every placeholder an argument"
        )
        assert _bound(sql, args, r"model = \$(\d+) AND api_key = \$(\d+)") == params
        assert "page.last_activity > summary.oldest AS settled" in sql, (
            "a session tied with the read's oldest row may rank after a session the LIMIT cut, so it is not settled"
        )
    assert _bound(
        cursor_sql, cursor_args, r"\(last_activity, session_key, api_key\) < \(\$(\d+)::timestamp, \$(\d+), \$(\d+)\)"
    ) == ("2026-10-28 09:00:00", "sess|9", "key-a")
    assert _page_bounds(cursor_sql) == (4, 0)
    assert _page_bounds(offset_sql) == (2, 2)


@pytest.mark.asyncio
async def test_a_status_filtered_read_binds_the_status_between_the_filters_and_the_cursor():
    calls: list[tuple[str, tuple[object, ...]]] = []
    await read_session_page(
        _TableDb(calls, rows=50, sessions=40),
        where_clause="model = $1",
        params=("gpt-4o",),
        status="failure",
        page=1,
        page_size=3,
        sort_desc=True,
        session_cursor="2026-10-28 09:00:00|key-a|sess-9",
        cap=100,
    )
    ((sql, args),) = calls
    assert {int(index) for index in re.findall(r"\$(\d+)", sql)} == set(range(1, len(args) + 1))
    assert _bound(sql, args, r"model = \$(\d+)") == ("gpt-4o",)
    assert _bound(sql, args, r"COALESCE\(status, 'success'\) = \$(\d+)") == ("failure",)
    assert _bound(sql, args, r"\(last_activity, session_key, api_key\) < \(\$(\d+)::timestamp, \$(\d+), \$(\d+)\)") == (
        "2026-10-28 09:00:00",
        "sess-9",
        "key-a",
    )
