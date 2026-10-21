import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
import pytest_asyncio
from integration._support.client import Gateway
from prisma import Prisma
from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from litellm.proxy.auth.authorization import AllRows, OwnedRows, ReadScope
from litellm.proxy.spend_tracking.spend_management_endpoints import _spend_log_payload_query, read_scope_sql


@dataclass(frozen=True, slots=True)
class SpendRow:
    request_id: str
    user: str | None
    team_id: str | None
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class RequestId:
    request_id: str


REQUEST_IDS: Final = TypeAdapter(tuple[RequestId, ...])
ROWS: Final = (
    SpendRow("own", "caller", None, "foreign"),
    SpendRow("team-1", "other", "first"),
    SpendRow("team-2", "third", "second"),
    SpendRow("foreign", "other", "outside"),
    SpendRow("ownerless", None, None),
    SpendRow("team-ownerless", None, "first"),
)


def _seed_rows(
    connection: psycopg.Connection,
    schema: str,
    rows: tuple[SpendRow, ...],
    session_id: str,
    started: datetime,
) -> None:
    utc_timestamp: Final = started.astimezone(timezone.utc).replace(tzinfo=None)
    with connection.cursor() as cursor:
        cursor.executemany(
            sql.SQL(
                'INSERT INTO {} (request_id, "user", team_id, litellm_call_id, session_id, '
                '"startTime", "endTime", messages, response, call_type) '
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'acompletion')"
            ).format(sql.Identifier(schema, "LiteLLM_SpendLogs")),
            tuple(
                (
                    row.request_id,
                    row.user,
                    row.team_id,
                    row.call_id,
                    session_id,
                    utc_timestamp,
                    utc_timestamp,
                    Jsonb([{"role": "user", "content": row.request_id + " payload"}]),
                    Jsonb({"id": row.request_id}),
                )
                for row in rows
            ),
        )


@pytest_asyncio.fixture(loop_scope="function")
async def spend_database() -> AsyncIterator[Prisma]:
    schema: Final = f"integration_spend_scope_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema}))
    )
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL('CREATE TABLE {} (LIKE public."LiteLLM_SpendLogs" INCLUDING ALL)').format(
                    sql.Identifier(schema, "LiteLLM_SpendLogs")
                )
            )
            _seed_rows(setup, schema, ROWS, "scope-session", datetime(2026, 1, 1, tzinfo=timezone.utc))
            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                yield database
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.asyncio
@pytest.mark.parametrize("preceding_filters", [False, True])
@pytest.mark.parametrize(
    ("scope", "user_filter", "expected"),
    [
        (AllRows(), None, ("foreign", "own", "ownerless", "team-1", "team-2", "team-ownerless")),
        (OwnedRows("caller"), None, ("own",)),
        (OwnedRows(None), None, ()),
        (OwnedRows(None, ("first", "second")), None, ("team-1", "team-2", "team-ownerless")),
        (OwnedRows(None, ("first", "second")), "other", ("team-1",)),
        (OwnedRows("caller", ("first", "second")), None, ("own", "team-1", "team-2", "team-ownerless")),
        (OwnedRows("caller", ("first", "second")), "other", ("team-1",)),
        (OwnedRows("caller", ("first' OR TRUE --",)), None, ("own",)),
        (OwnedRows("caller' OR TRUE --", ("first",)), None, ("team-1", "team-ownerless")),
    ],
)
async def test_ownership_sql_selects_allowed_rows_and_intersects_filters(
    spend_database: Prisma,
    scope: ReadScope,
    user_filter: str | None,
    expected: tuple[str, ...],
    preceding_filters: bool,
) -> None:
    window_params: Final = ("scope-session", "2026-01-01", "2026-01-02") if preceding_filters else ()
    window_sql: Final = (
        'session_id = $1 AND "startTime" >= $2::timestamp AND "startTime" < $3::timestamp AND '
        if preceding_filters
        else ""
    )
    clause, scope_params = read_scope_sql(scope, len(window_params) + 1)
    filter_sql: Final = f' AND "user" = ${len(window_params) + len(scope_params) + 1}' if user_filter else ""
    params: Final = window_params + scope_params + ((user_filter,) if user_filter else ())
    result: Final = await spend_database.query_raw(
        f'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE {window_sql}{clause or "TRUE"}{filter_sql} '
        "ORDER BY request_id",
        *params,
    )
    assert tuple(row.request_id for row in REQUEST_IDS.validate_python(result)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "expected"),
    [(AllRows(), ("foreign",)), (OwnedRows("caller"), ("own",)), (OwnedRows(None), ())],
)
async def test_payload_sql_filters_foreign_collisions_and_prefers_exact_ids_for_admins(
    spend_database: Prisma, scope: ReadScope, expected: tuple[str, ...]
) -> None:
    query, params = _spend_log_payload_query("foreign", scope)
    result: Final = await spend_database.query_raw(query, *params)
    assert tuple(row.request_id for row in REQUEST_IDS.validate_python(result)) == expected


def _delete_session(session_id: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE session_id = %s', (session_id,))


@pytest.mark.parametrize(
    ("member_role", "permissions", "team_access"),
    [
        ("admin", [], True),
        ("user", ["/spend/logs"], True),
        ("user", ["/key/info"], False),
        ("user", [], False),
    ],
)
def test_spend_log_routes_preserve_user_and_permitted_team_access(
    gateway: Gateway, member_role: str, permissions: list[str], team_access: bool
) -> None:
    session_id: Final = f"scope-{uuid.uuid4().hex}"
    started: Final = datetime.now(timezone.utc) - timedelta(hours=1)
    with gateway.scenario() as scenario:
        caller: Final = scenario.user(user_role="internal_user")
        other: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(
            members_with_roles=[{"user_id": caller, "role": member_role}],
            team_member_permissions=list(permissions),
        )
        outside_team: Final = scenario.team(
            members_with_roles=[{"user_id": other, "role": "admin"}],
            team_member_permissions=["/spend/logs"],
        )
        key: Final = scenario.key(user_id=caller)
        other_key: Final = scenario.key(user_id=other)
        rows: Final = (
            SpendRow(session_id + "-own", caller, None, session_id + "-foreign"),
            SpendRow(session_id + "-team", other, team),
            SpendRow(session_id + "-foreign", other, outside_team),
            SpendRow(session_id + "-ownerless", None, None),
            SpendRow(session_id + "-outside", other, outside_team),
        )
        scenario.cleanups.callback(_delete_session, session_id)
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            _seed_rows(connection, "public", rows, session_id, started)
        expected: Final = (rows[0].request_id, rows[1].request_id) if team_access else (rows[0].request_id,)
        session: Final = gateway.request("GET", "/spend/logs/session/ui", key=key, params={"session_id": session_id})
        assert session.status_code == 200, session.text
        assert session.json()["total"] == len(expected), session.text
        assert sorted(row["request_id"] for row in session.json()["data"]) == list(expected), session.text
        filters: Final = {
            "session_id": session_id,
            "start_date": (started - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": (started + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        }
        listed: Final = gateway.request("GET", "/spend/logs/ui", key=key, params=filters)
        assert listed.status_code == 200, listed.text
        assert sorted(row["request_id"] for row in listed.json()["data"]) == list(expected), listed.text
        narrowed: Final = gateway.request("GET", "/spend/logs/ui", key=key, params={**filters, "user_id": other})
        assert narrowed.status_code == 200, narrowed.text
        assert [row["request_id"] for row in narrowed.json()["data"]] == (
            [rows[1].request_id] if team_access else []
        ), narrowed.text
        refused: Final = gateway.request("GET", f"/spend/logs/ui/{rows[4].request_id}", key=key)
        assert refused.status_code == 403, refused.text
        for caller_key, expected_id in ((key, rows[0].request_id), (other_key, rows[2].request_id)):
            payload: Final = gateway.request("GET", f"/spend/logs/ui/{rows[2].request_id}", key=caller_key)
            assert payload.status_code == 200, payload.text
            assert payload.json()["messages"] == [{"role": "user", "content": expected_id + " payload"}], payload.text
        admin: Final = gateway.request("GET", f"/spend/logs/ui/{rows[2].request_id}")
        assert admin.status_code == 200, admin.text
        assert admin.json()["messages"] == [{"role": "user", "content": rows[2].request_id + " payload"}], admin.text
