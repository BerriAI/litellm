import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Final

import httpx
import psycopg
import pytest
from integration._support.client import Gateway, Scenario, object_value
from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import JsonValue

USER_SPEND: Final = "LiteLLM_DailyUserSpend"
TEAM_SPEND: Final = "LiteLLM_DailyTeamSpend"
TAG_SPEND: Final = "LiteLLM_DailyTagSpend"
ORGANIZATION_SPEND: Final = "LiteLLM_DailyOrganizationSpend"
END_USER_SPEND: Final = "LiteLLM_DailyEndUserSpend"
AGENT_SPEND: Final = "LiteLLM_DailyAgentSpend"
DAY: Final = "2026-02-03"
AGGREGATED_USER_ACTIVITY: Final = "/user/daily/activity/aggregated"

INSERT_DAILY_ROW: Final = sql.SQL(
    "INSERT INTO {table} (id, {entity}, date, api_key, model, model_group, custom_llm_provider, prompt_tokens,"
    " completion_tokens, spend, api_requests, successful_requests, failed_requests, updated_at)"
    " VALUES (gen_random_uuid()::text, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())"
)
DELETE_DAILY_ROWS: Final = sql.SQL("DELETE FROM {table} WHERE api_key = ANY(%s)")
INSERT_SPEND_LOG: Final = (
    'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, "startTime", "endTime", metadata)'
    " VALUES (%s, 'acompletion', %s, %s::timestamp, %s::timestamp, %s)"
)
DELETE_SPEND_LOG: Final = 'DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
LOCK_TABLE: Final = sql.SQL("LOCK TABLE {table} IN ACCESS EXCLUSIVE MODE")


@dataclass(frozen=True, slots=True)
class Route:
    path: str
    table: str
    entity_column: str
    entity_filter: str | None


ROUTES: Final = (
    Route("/user/daily/activity", USER_SPEND, "user_id", None),
    Route(AGGREGATED_USER_ACTIVITY, USER_SPEND, "user_id", None),
    Route("/team/daily/activity", TEAM_SPEND, "team_id", "team_ids"),
    Route("/team/daily/activity/aggregated", TEAM_SPEND, "team_id", "team_ids"),
    Route("/tag/daily/activity", TAG_SPEND, "tag", "tags"),
    Route("/organization/daily/activity", ORGANIZATION_SPEND, "organization_id", "organization_ids"),
    Route("/customer/daily/activity", END_USER_SPEND, "end_user_id", "end_user_ids"),
    Route("/end_user/daily/activity", END_USER_SPEND, "end_user_id", "end_user_ids"),
    Route("/agent/daily/activity", AGENT_SPEND, "agent_id", "agent_ids"),
)


def user_with_an_email(scenario: Scenario) -> tuple[str, str]:
    email: Final = f"integration-{uuid.uuid4().hex}@example.com"
    return scenario.user(user_email=email), email


def key_no_key_table_holds() -> str:
    return f"integration-ownerless-{uuid.uuid4().hex}"


def activity_of_key(
    gateway: Gateway, path: str, api_key: str, *, reader: str | None = None, **filters: str
) -> httpx.Response:
    return gateway.request(
        "GET", path, params={"start_date": DAY, "end_date": DAY, "api_key": api_key, **filters}, key=reader
    )


@dataclass(frozen=True, slots=True)
class DailyRow:
    table: str
    entity_column: str
    entity: str | None
    api_key: str
    date: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    spend: float
    successful_requests: int
    failed_requests: int


def _insert(connection: psycopg.Connection[tuple[object, ...]], row: DailyRow) -> None:
    connection.execute(
        INSERT_DAILY_ROW.format(table=sql.Identifier(row.table), entity=sql.Identifier(row.entity_column)),
        (
            row.entity,
            row.date,
            row.api_key,
            row.model,
            row.model,
            row.provider,
            row.prompt_tokens,
            row.completion_tokens,
            row.spend,
            row.successful_requests + row.failed_requests,
            row.successful_requests,
            row.failed_requests,
        ),
    )


def insert_daily_rows(rows: Sequence[DailyRow], *, database_url: str | None = None) -> None:
    with psycopg.connect(database_url or os.environ["DATABASE_URL"]) as connection:
        for row in rows:
            _insert(connection, row)


def delete_daily_rows(rows: Sequence[DailyRow], *, database_url: str | None = None) -> None:
    with psycopg.connect(database_url or os.environ["DATABASE_URL"]) as connection:
        for table in sorted({row.table for row in rows}):
            connection.execute(
                DELETE_DAILY_ROWS.format(table=sql.Identifier(table)),
                (sorted({row.api_key for row in rows if row.table == table}),),
            )


@contextmanager
def daily_rows(rows: Sequence[DailyRow], *, database_url: str | None = None) -> Iterator[None]:
    insert_daily_rows(rows, database_url=database_url)
    try:
        yield
    finally:
        delete_daily_rows(rows, database_url=database_url)


@contextmanager
def spend_log_naming_only_an_alias(request_id: str, api_key: str, started: str, alias: str) -> Iterator[None]:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            INSERT_SPEND_LOG, (request_id, api_key, started, started, Jsonb({"user_api_key_alias": alias}))
        )
    try:
        yield
    finally:
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute(DELETE_SPEND_LOG, (request_id,))


@contextmanager
def locked_table(table: str, *, database_url: str | None = None) -> Iterator[None]:
    with psycopg.connect(database_url or os.environ["DATABASE_URL"]) as connection:
        connection.execute(LOCK_TABLE.format(table=sql.Identifier(table)))
        try:
            yield
        finally:
            connection.rollback()


def records_of_key(node: JsonValue, api_key: str) -> tuple[JsonValue, ...]:
    if isinstance(node, list):
        return tuple(chain.from_iterable(records_of_key(item, api_key) for item in node))
    if not isinstance(node, dict):
        return ()
    nested: Final = tuple(chain.from_iterable(records_of_key(value, api_key) for value in node.values()))
    return (node[api_key], *nested) if api_key in node else nested


def seeded_row(table: str, entity_column: str, entity: str | None, api_key: str, date: str) -> DailyRow:
    return DailyRow(table, entity_column, entity, api_key, date, "gpt-4o-mini", "openai", 10, 5, 0.25, 1, 0)


def user_row(user: str | None, api_key: str, date: str) -> DailyRow:
    return seeded_row(USER_SPEND, "user_id", user, api_key, date)


def seeded_metrics(rows: int) -> dict[str, float]:
    return {
        "spend": 0.25 * rows,
        "prompt_tokens": 10 * rows,
        "completion_tokens": 5 * rows,
        "total_tokens": 15 * rows,
        "api_requests": rows,
        "successful_requests": rows,
    }


def key_metadata(
    *,
    alias: str | None = None,
    team: str | None = None,
    user: str | None = None,
    email: str | None = None,
    exists: bool = False,
) -> dict[str, JsonValue]:
    return {"key_alias": alias, "team_id": team, "user_id": user, "user_email": email, "key_exists": exists}


def counted(metrics: JsonValue) -> dict[str, JsonValue]:
    return {name: value for name, value in object_value(metrics).items() if value}


def assert_key_reported(
    response: httpx.Response,
    api_key: str,
    date: str,
    metadata: Mapping[str, JsonValue],
    metrics: Mapping[str, float],
) -> None:
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    records: Final = tuple(object_value(record) for record in records_of_key(body, api_key))
    assert records, response.text
    assert all(record["metadata"] == metadata for record in records), response.text
    assert all(counted(record["metrics"]) == pytest.approx(metrics) for record in records), response.text
    days: Final = body["results"]
    assert isinstance(days, list) and len(days) == 1, response.text
    day: Final = object_value(days[0])
    assert day["date"] == date, response.text
    assert counted(day["metrics"]) == pytest.approx(metrics), response.text
    assert object_value(body["metadata"])["total_spend"] == pytest.approx(metrics["spend"]), response.text


def assert_key_owner_and_totals(
    response: httpx.Response,
    api_key: str,
    metadata: Mapping[str, JsonValue],
    totals: Mapping[str, float],
) -> None:
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    records: Final = tuple(object_value(record) for record in records_of_key(body, api_key))
    assert records, response.text
    assert all(record["metadata"] == metadata for record in records), response.text
    reported: Final = object_value(body["metadata"])
    assert {name: reported[name] for name in totals} == pytest.approx(totals), response.text
