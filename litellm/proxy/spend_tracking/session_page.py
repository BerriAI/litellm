"""Which sessions a logs page shows and how many match, read from the newest rows instead of every row. Reads grow
until one settles both, so the result is what grouping every row would give."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

from typing_extensions import ReadOnly, TypedDict

MCP_CALL_TYPES_SQL: Final = "('call_mcp_tool', 'list_mcp_tools')"
AGENT_CALL_TYPE_SQL: Final = "'asend_message'"
SESSION_KEY_EXPR: Final = "COALESCE(NULLIF(session_id, ''), request_id)"
SESSION_REPRESENTATIVE_ORDER_SQL: Final = (
    f"(call_type = {AGENT_CALL_TYPE_SQL}) DESC, "
    f'CASE WHEN call_type = {AGENT_CALL_TYPE_SQL} THEN "endTime" END DESC NULLS LAST, '
    f'call_type IN {MCP_CALL_TYPES_SQL}, "startTime" DESC, request_id'
)


class _ReadRow(TypedDict):
    session_key: ReadOnly[str | None]
    api_key: ReadOnly[str | None]
    last_activity: ReadOnly[str | None]
    settled: ReadOnly[bool | None]
    rows_read: ReadOnly[int]
    session_count: ReadOnly[int]


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
    status: str | None
    cap: int

    def first_read_size(self) -> int | None:
        """Every row when oldest first, or when filtered by outcome, which any of a session's rows can decide. Otherwise
        the fewest that could settle a page: a session past the cap and an older row."""
        return self.cap + 2 if self.sort_desc and self.status is None else None


@dataclass(frozen=True, slots=True)
class _Read:
    candidates: tuple[Session, ...]
    last_settled: bool
    rows_read: int
    session_count: int
    saw_every_row: bool

    def settles(self, request: _Request) -> bool:
        page_settled: Final = (
            request.page_limit <= 0
            or self.saw_every_row
            or (len(self.candidates) > request.page_limit and self.last_settled)
        )
        return page_settled and (self.saw_every_row or self.session_count > request.cap)

    def next_size(self, request: _Request) -> int:
        rows_per_session: Final = self.rows_read / self.session_count
        return max(2 * self.rows_read, math.ceil(1.25 * (request.cap + 1) * rows_per_session))


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
    *,
    where_clause: str,
    params: Sequence[object],
    status: str | None,
    page: int,
    page_size: int,
    sort_desc: bool,
    session_cursor: str | None,
    cap: int,
) -> SessionPage:
    """One page of sessions after ``session_cursor``, or after ``(page - 1) * page_size`` sessions, keeping only those
    whose outcome is ``status`` when given. ``where_clause`` reads ``params`` as ``$1`` onward."""
    cursor: Final = _parse_session_cursor(session_cursor)
    offset: Final = (page - 1) * page_size if cursor is None else 0
    request: Final = _Request(cursor, offset, min(page_size, cap - offset), sort_desc, status, cap)
    return await _session_page(_sql_reader(db, where_clause, params, request), request)


async def _session_page(read: _Reader, request: _Request) -> SessionPage:
    last = await read(request.first_read_size())  # rebind-ok: each read replaces the last until one settles the page
    while not last.settles(request):
        last = await read(last.next_size(request))
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


def _sessions_sql(request: _Request, status_index: int) -> str:
    """Each session's last activity, from its rows in ``scanned``, keeping only sessions whose outcome is the status."""
    if request.status is None:
        return (
            'SELECT session_key, api_key, MAX("startTime") AS last_activity FROM scanned GROUP BY session_key, api_key'
        )
    return f"""
        SELECT session_key, api_key, last_activity
        FROM (
            SELECT DISTINCT ON (session_key, api_key)
                   session_key,
                   api_key,
                   status,
                   MAX("startTime") OVER (PARTITION BY session_key, api_key) AS last_activity
            FROM scanned
            ORDER BY session_key, api_key, {SESSION_REPRESENTATIVE_ORDER_SQL}
        ) AS outcomes
        WHERE COALESCE(status, 'success') = ${status_index}
    """


def _sql_reader(db: _SupportsQueryRaw, where_clause: str, params: Sequence[object], request: _Request) -> _Reader:
    status_index: Final = len(params) + 1
    status_args: Final = () if request.status is None else (request.status,)
    cursor_index: Final = status_index + len(status_args)
    after_cursor: Final = (
        f"WHERE (last_activity, session_key, api_key) {'<' if request.sort_desc else '>'} "
        f"(${cursor_index}::timestamp, ${cursor_index + 1}, ${cursor_index + 2})"
        if request.cursor
        else ""
    )
    outcome_columns: Final = ', status, call_type, "endTime", request_id' if request.status is not None else ""
    direction: Final = "DESC" if request.sort_desc else "ASC"
    page_rows: Final = request.page_limit + 1 if request.page_limit > 0 else 0
    skip: Final = f"OFFSET {request.offset}" if page_rows and request.offset else ""

    async def read(size: int | None) -> _Read:
        # LEFT JOIN keeps the summary row when no session is on the page.
        rows: Final = await db.query_raw(
            f"""
            WITH scanned AS (
                SELECT {SESSION_KEY_EXPR} AS session_key, api_key, "startTime"{outcome_columns}
                FROM "LiteLLM_SpendLogs"
                WHERE {where_clause}
                {f'ORDER BY "startTime" DESC LIMIT {size}' if size is not None else ""}
            ),
            sessions AS ({_sessions_sql(request, status_index)}),
            summary AS (
                SELECT COUNT(*) AS rows_read,
                       MIN("startTime") AS oldest,
                       (SELECT COUNT(*) FROM sessions) AS session_count
                FROM scanned
            )
            SELECT page.session_key,
                   page.api_key,
                   page.last_activity::text AS last_activity,
                   page.last_activity > summary.oldest AS settled,
                   summary.rows_read,
                   summary.session_count
            FROM summary
            LEFT JOIN LATERAL (
                SELECT session_key, api_key, last_activity
                FROM sessions
                {after_cursor}
                ORDER BY last_activity {direction}, session_key {direction}, api_key {direction}
                LIMIT {page_rows} {skip}
            ) AS page ON TRUE
            ORDER BY page.last_activity {direction}, page.session_key {direction}, page.api_key {direction}
            """,
            *params,
            *status_args,
            *(request.cursor or ()),
        )
        rows_read: Final = int(rows[0]["rows_read"])
        return _Read(
            candidates=tuple(
                Session(str(row["session_key"]), str(row["api_key"]), str(row["last_activity"]))
                for row in rows
                if row["session_key"] is not None
            ),
            last_settled=bool(rows[-1]["settled"]),
            rows_read=rows_read,
            session_count=int(rows[0]["session_count"]),
            saw_every_row=size is None or rows_read < size,
        )

    return read
