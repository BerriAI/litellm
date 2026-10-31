"""Which sessions a logs page shows, and how many sessions its filters match.

A session is the matching ``LiteLLM_SpendLogs`` rows that share a session key and an api_key, ranked by
``(last_activity, session_key, api_key)``. A descending page reads only the newest matching rows. Every matching row
newer than the oldest row read is in the read, so a session last active after that row has its true last activity and
no session outside the read can rank before it. A read that settles both the page and the capped count is used. One
that does not is discarded for a larger one, so the result is the same as grouping every matching row.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, TypeAlias

from typing_extensions import ReadOnly, TypedDict

SESSION_KEY_EXPR: Final = "COALESCE(NULLIF(session_id, ''), request_id)"


class _ReadRow(TypedDict):
    session_key: ReadOnly[str | None]
    api_key: ReadOnly[str | None]
    last_activity: ReadOnly[str | None]
    settled: ReadOnly[bool | None]
    rows_read: ReadOnly[int]
    session_count: ReadOnly[int]
    read_seconds: ReadOnly[float | None]
    window_seconds: ReadOnly[float | None]


class _SupportsQueryRaw(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[_ReadRow]: ...


@dataclass(frozen=True, slots=True)
class Session:
    key: str
    api_key: str
    last_activity: str


@dataclass(frozen=True, slots=True)
class SessionPage:
    sessions: tuple[Session, ...]
    has_more: bool
    next_cursor: str | None
    total: int
    total_is_capped: bool


@dataclass(frozen=True, slots=True)
class _Request:
    cursor: tuple[str, str, str] | None
    offset: int
    page_limit: int
    sort_desc: bool
    cap: int


@dataclass(frozen=True, slots=True)
class _Read:
    candidates: tuple[Session, ...]
    last_settled: bool
    rows_read: int
    session_count: int
    read_everything: bool
    window_rows: float | None

    def next_budget(self, sessions_needed: int) -> int | None:
        """Rows for ``sessions_needed`` sessions at this read's density with a quarter to spare and at least twice this
        read, or ``None`` for every row once that would take half of the window's ``window_rows``."""
        rows: Final = max(2 * self.rows_read, -(-5 * sessions_needed * self.rows_read // (4 * self.session_count)))
        return None if self.window_rows is not None and 2 * rows >= self.window_rows else rows

    def decides(self, request: _Request) -> bool:
        page_settled: Final = (
            request.page_limit <= 0
            or self.read_everything
            or (len(self.candidates) > request.page_limit and self.last_settled)
        )
        return page_settled and (self.read_everything or self.session_count > request.cap)


_Reader: TypeAlias = Callable[[int | None], Awaitable[_Read]]


def _parse_session_cursor(session_cursor: str | None) -> tuple[str, str, str] | None:
    if session_cursor is None or session_cursor.count("|") < 2:
        return None
    last_activity, _, rest = session_cursor.partition("|")
    api_key, _, session_key = rest.partition("|")
    if not last_activity or not session_key:
        return None
    return (last_activity, session_key, api_key)


async def read_session_page(
    db: _SupportsQueryRaw,
    where_clause: str,
    params: Sequence[object],
    ctes: str,
    page: int,
    page_size: int,
    sort_desc: bool,
    session_cursor: str | None,
    cap: int,
    window_start: datetime | None = None,
) -> SessionPage:
    """The sessions of one page, resumed after ``session_cursor`` or, without one, after ``(page - 1) * page_size``
    sessions. ``where_clause`` reads ``params`` as ``$1`` onward and may name the CTEs that ``ctes`` declares, each
    followed by a comma. No page runs past the ``cap`` sessions that the capped total promises. ``window_start``, the
    UTC start of the date window, sizes reads against the window."""
    cursor: Final = _parse_session_cursor(session_cursor)
    offset: Final = (page - 1) * page_size if cursor is None else 0
    request: Final = _Request(cursor, offset, min(page_size, cap - offset), sort_desc, cap)
    return await _session_page(_sql_reader(db, where_clause, params, ctes, request, window_start), request)


async def _session_page(read: _Reader, request: _Request) -> SessionPage:
    last = await read(  # rebind-ok: each read replaces the last until one decides
        request.cap + request.offset + request.page_limit + 2 if request.sort_desc else None
    )
    while not last.decides(request):
        last = await read(last.next_budget(request.cap + 1))
    has_more: Final = len(last.candidates) > request.page_limit
    sessions: Final = last.candidates[: request.page_limit]
    final: Final = sessions[-1] if has_more and sessions else None
    return SessionPage(
        sessions=sessions,
        has_more=has_more,
        next_cursor=f"{final.last_activity}|{final.api_key}|{final.key}" if final else None,
        total=min(last.session_count, request.cap),
        total_is_capped=last.session_count > request.cap,
    )


def _sql_reader(
    db: _SupportsQueryRaw,
    where_clause: str,
    params: Sequence[object],
    ctes: str,
    request: _Request,
    window_start: datetime | None,
) -> _Reader:
    cursor_index: Final = len(params) + 1
    cursor_params: Final[tuple[object, ...]] = request.cursor or ()
    limit_index: Final = cursor_index + len(cursor_params)
    offset_params: Final[tuple[int, ...]] = (request.offset,) if request.offset and request.page_limit > 0 else ()
    window_params: Final[tuple[datetime, ...]] = (window_start,) if window_start is not None else ()
    window_index: Final = limit_index + 1 + len(offset_params)
    window_seconds_expr: Final = (
        f"EXTRACT(EPOCH FROM MAX(\"startTime\") - (${window_index}::timestamptz AT TIME ZONE 'UTC'))::float8"
        if window_params
        else "NULL::float8"
    )
    direction: Final = "DESC" if request.sort_desc else "ASC"
    after_cursor: Final = (
        f"WHERE (last_activity, session_key, api_key) {'<' if request.sort_desc else '>'} "
        f"(${cursor_index}::timestamp, ${cursor_index + 1}, ${cursor_index + 2})"
        if request.cursor
        else ""
    )
    page_query_args: Final = (
        *params,
        *cursor_params,
        request.page_limit + 1 if request.page_limit > 0 else 0,
        *offset_params,
        *window_params,
    )

    async def read(budget: int | None) -> _Read:
        # No MATERIALIZED: Postgres 10 and 11 cannot parse it.
        rows: Final = await db.query_raw(
            f"""
            WITH {ctes} scanned AS (
                SELECT {SESSION_KEY_EXPR} AS session_key, api_key, "startTime"
                FROM "LiteLLM_SpendLogs"
                WHERE {where_clause}
                {f'ORDER BY "startTime" DESC LIMIT {budget}' if budget is not None else ""}
            ),
            sessions AS (
                SELECT session_key, api_key, MAX("startTime") AS last_activity
                FROM scanned
                GROUP BY session_key, api_key
            ),
            summary AS (
                SELECT COUNT(*) AS rows_read,
                       MIN("startTime") AS oldest,
                       EXTRACT(EPOCH FROM MAX("startTime") - MIN("startTime"))::float8 AS read_seconds,
                       {window_seconds_expr} AS window_seconds,
                       (SELECT COUNT(*) FROM sessions) AS session_count
                FROM scanned
            )
            SELECT page.session_key,
                   page.api_key,
                   page.last_activity::text AS last_activity,
                   page.last_activity > summary.oldest AS settled,
                   summary.rows_read,
                   summary.session_count,
                   summary.read_seconds,
                   summary.window_seconds
            FROM summary
            LEFT JOIN LATERAL (
                SELECT session_key, api_key, last_activity
                FROM sessions
                {after_cursor}
                ORDER BY last_activity {direction}, session_key {direction}, api_key {direction}
                LIMIT ${limit_index} {f"OFFSET ${limit_index + 1}" if offset_params else ""}
            ) AS page ON TRUE
            """,
            *page_query_args,
        )
        rows_read: Final = int(rows[0]["rows_read"])
        read_seconds: Final = rows[0]["read_seconds"]
        window_seconds: Final = rows[0]["window_seconds"]
        return _Read(
            candidates=tuple(
                Session(str(row["session_key"]), str(row["api_key"]), str(row["last_activity"]))
                for row in rows
                if row["session_key"] is not None
            ),
            last_settled=bool(rows[-1]["settled"]),
            rows_read=rows_read,
            session_count=int(rows[0]["session_count"]),
            read_everything=budget is None or rows_read < budget,
            window_rows=rows_read * window_seconds / read_seconds
            if read_seconds and window_seconds is not None
            else None,
        )

    return read
