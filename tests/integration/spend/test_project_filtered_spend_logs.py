from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final
from urllib.parse import urlencode

import httpx
import psycopg
import pytest
from psycopg.types.json import Jsonb

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.spend_tracking.spend_management_endpoints import SPEND_LOGS_PAGINATION_COUNT_CAP
from tests.integration._support.client import Gateway

START: Final = "2026-01-01 00:00:00"
END: Final = "2026-01-02 00:00:00"
BASE_COUNT: Final = 46349


def _query(
    gateway: Gateway,
    filters: Mapping[str, str],
    projects: tuple[str, ...] | None,
    *,
    path: str = "/spend/logs/v2",
    key: str | None = None,
) -> httpx.Response:
    pairs: Final = tuple(filters.items()) + tuple(("project_ids", project) for project in projects or ())
    return gateway.request("GET", path + "?" + urlencode(pairs), key=key)


def _delete_rows(prefix: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], options="-c timezone=UTC") as connection:
        connection.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (prefix + "%",))


def _seed_base(prefix: str, team: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], options="-c timezone=UTC") as connection:
        connection.execute(
            """INSERT INTO "LiteLLM_SpendLogs"
            (request_id, call_type, "startTime", "endTime", team_id, "user", metadata,
             prompt_tokens, completion_tokens, total_tokens, spend, model)
            SELECT %s || '-' || n, 'acompletion',
              timestamp '2026-01-01' + n * interval '1 second',
              timestamp '2026-01-01' + n * interval '1 second', %s,
              CASE WHEN (n/2100) %% 2 = 0 THEN 'fixture-member-a' ELSE 'fixture-member-b' END,
              jsonb_build_object('user_api_key_project_id',
                CASE WHEN n %% 2100=0 AND n/2100<22
                     THEN CASE WHEN (n/2100) %% 2=0 THEN 'fixture-p1' ELSE 'fixture-p2' END
                     ELSE 'fixture-unbound' END,
                'usage_object', CASE WHEN n=4200 THEN '{"prompt_tokens_details":{"cached_tokens":null}}'::jsonb
                     WHEN n=6300 THEN '{"prompt_tokens_details":{}}'::jsonb
                     ELSE jsonb_build_object('prompt_tokens_details',
                          jsonb_build_object('cached_tokens',CASE WHEN n=0 THEN 1024 ELSE 0 END)) END),
              CASE WHEN n=0 THEN 8196 WHEN n=2100 THEN 5518 ELSE 17 END, 3,
              CASE WHEN n=0 THEN 8199 WHEN n=2100 THEN 5521 ELSE 20 END,
              0.001, 'fixture-model'
            FROM generate_series(0,%s) n""",
            (prefix, team, BASE_COUNT - 1),
        )


def _seed_edges(prefix: str, team: str, member: str) -> None:
    rows: Final = (
        ("p1", {"user_api_key_project_id": "fixture-p1", "cache_read_input_tokens": 0}),
        ("p2", {"user_api_key_project_id": "fixture-p2", "cache_read_input_tokens": None}),
        ("missing", {}),
        ("null", {"user_api_key_project_id": None}),
        ("number", {"user_api_key_project_id": 42}),
        ("number-text", {"user_api_key_project_id": "42"}),
        ("spaces", {"user_api_key_project_id": " fixture-p1 "}),
        ("unbound", {"user_api_key_project_id": "fixture-unbound"}),
        ("quote", {"user_api_key_project_id": "x') OR TRUE --"}),
    )
    with psycopg.connect(os.environ["DATABASE_URL"], options="-c timezone=UTC") as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO "LiteLLM_SpendLogs"
                (request_id,call_type,"startTime","endTime",team_id,"user",metadata)
                VALUES (%s,'acompletion',%s,%s,%s,%s,%s)""",
                tuple(
                    (
                        prefix + "-" + name,
                        datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
                        datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
                        team,
                        member,
                        Jsonb(metadata),
                    )
                    for name, metadata in rows
                ),
            )


def test_project_filter_returns_the_same_complete_raw_records_with_far_fewer_pages(gateway: Gateway) -> None:
    prefix: Final = "project-filter-" + uuid.uuid4().hex
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        scenario.cleanups.callback(_delete_rows, prefix)
        _seed_base(prefix, team)
        filters: Final = {"team_id": team, "start_date": START, "end_date": END, "page_size": "1000"}
        first: Final = _query(gateway, filters, None)
        assert first.status_code == 200, first.text
        assert "project_filter" not in first.json(), first.text
        assert first.json()["total"] == min(BASE_COUNT, SPEND_LOGS_PAGINATION_COUNT_CAP), first.text
        assert first.json()["total_is_capped"] == (BASE_COUNT > SPEND_LOGS_PAGINATION_COUNT_CAP), first.text
        reference = []
        baseline_pages = 0
        for page in range(1, BASE_COUNT // 1000 + 3):
            response: Final = _query(gateway, {**filters, "page": str(page)}, None)
            assert response.status_code == 200, response.text
            data: Final = response.json()["data"]
            baseline_pages += 1
            reference.extend(
                row for row in data if row["metadata"].get("user_api_key_project_id") in ("fixture-p1", "fixture-p2")
            )
            if not data:
                break
        else:
            pytest.fail("baseline pagination did not reach an empty page")
        filtered: Final = _query(gateway, filters, ("fixture-p1", "fixture-p2"))
        assert filtered.status_code == 200, filtered.text
        body: Final = filtered.json()
        assert body["data"] == sorted(reference, key=lambda row: row["startTime"], reverse=True), filtered.text
        assert body["total"] == len(reference) == 22, filtered.text
        assert body["total_is_capped"] is False, filtered.text
        assert body["project_filter"] == {"version": 1, "project_ids": ["fixture-p1", "fixture-p2"]}, filtered.text
        empty: Final = _query(gateway, {**filters, "page": "2"}, ("fixture-p1", "fixture-p2"))
        assert empty.status_code == 200 and empty.json()["data"] == [], empty.text
        assert baseline_pages == (BASE_COUNT + 999) // 1000 + 1
        by_id: Final = {row["request_id"]: row for row in body["data"]}
        assert by_id[prefix + "-0"]["prompt_tokens"] == 8196
        assert by_id[prefix + "-0"]["metadata"]["usage_object"]["prompt_tokens_details"]["cached_tokens"] == 1024
        assert by_id[prefix + "-2100"]["prompt_tokens"] == 5518
        assert by_id[prefix + "-2100"]["metadata"]["usage_object"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert by_id[prefix + "-4200"]["metadata"]["usage_object"]["prompt_tokens_details"]["cached_tokens"] is None
        assert "cached_tokens" not in by_id[prefix + "-6300"]["metadata"]["usage_object"]["prompt_tokens_details"]


@pytest.mark.parametrize(
    ("projects", "expected"),
    (
        (("fixture-p1", "fixture-p2"), ("p1", "p2")),
        (("fixture-p1", "fixture-p1"), ("p1",)),
        (("42",), ("number-text",)),
        ((" fixture-p1 ",), ("spaces",)),
        (("x') OR TRUE --",), ("quote",)),
        (("fixture-absent",), ()),
    ),
)
def test_project_filter_preserves_exact_json_types_and_bound_parameters(
    gateway: Gateway, projects: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    prefix: Final = "project-filter-edge-" + uuid.uuid4().hex
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        scenario.cleanups.callback(_delete_rows, prefix)
        _seed_edges(prefix, team, "fixture-member")
        response: Final = _query(gateway, {"team_id": team, "start_date": START, "end_date": END}, projects)
        assert response.status_code == 200, response.text
        assert sorted(row["request_id"] for row in response.json()["data"]) == sorted(
            prefix + "-" + name for name in expected
        )
        assert response.json()["total"] == len(expected), response.text
        assert response.json()["project_filter"] == {"version": 1, "project_ids": list(projects)}, response.text


@pytest.mark.parametrize(
    ("path", "extra", "projects"),
    (
        ("/spend/logs/ui", {}, ("fixture-p1",)),
        ("/spend/logs/v2", {"group_by_session": "true"}, ("fixture-p1",)),
        ("/spend/logs/v2", {"session_cursor": "fixture-cursor"}, ("fixture-p1",)),
        ("/spend/logs/v2", {}, ("",)),
        ("/spend/logs/v2", {}, (" ",)),
    ),
)
def test_project_filter_rejects_unsupported_modes_and_blank_values(
    gateway: Gateway, path: str, extra: dict[str, str], projects: tuple[str, ...]
) -> None:
    response: Final = _query(gateway, {"start_date": START, "end_date": END, **extra}, projects, path=path)
    assert response.status_code == 400, response.text


def test_project_filter_does_not_grant_access_to_another_team(gateway: Gateway) -> None:
    prefix: Final = "project-filter-auth-" + uuid.uuid4().hex
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role=LitellmUserRoles.INTERNAL_USER.value)
        own_team: Final = scenario.team(members_with_roles=[{"user_id": user, "role": "admin"}])
        foreign_team: Final = scenario.team()
        key: Final = scenario.key(user_id=user, team_id=own_team)
        scenario.cleanups.callback(_delete_rows, prefix)
        _seed_edges(prefix, foreign_team, user)
        filters: Final = {"team_id": foreign_team, "start_date": START, "end_date": END}
        baseline: Final = _query(gateway, filters, None, key=key)
        filtered: Final = _query(gateway, filters, ("fixture-p1",), key=key)
        assert baseline.status_code == filtered.status_code == 403, (baseline.text, filtered.text)
        assert "data" not in filtered.json(), filtered.text


def test_project_filter_keeps_member_and_key_filters_and_inclusive_dates(gateway: Gateway) -> None:
    prefix: Final = "project-filter-scope-" + uuid.uuid4().hex
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        scenario.cleanups.callback(_delete_rows, prefix)
        _seed_edges(prefix, team, "fixture-member")
        with psycopg.connect(os.environ["DATABASE_URL"], options="-c timezone=UTC") as connection:
            connection.execute(
                """UPDATE "LiteLLM_SpendLogs" SET "startTime"=%s, "endTime"=%s, api_key=%s WHERE request_id=%s""",
                (
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                    "fixture-log-key",
                    prefix + "-p1",
                ),
            )
        filters: Final = {
            "team_id": team,
            "start_date": START,
            "end_date": END,
            "user_id": "fixture-member",
            "api_key": "fixture-log-key",
        }
        baseline: Final = _query(gateway, filters, None)
        response: Final = _query(gateway, filters, ("fixture-p1", "fixture-p2"))
        assert baseline.status_code == response.status_code == 200, (baseline.text, response.text)
        assert response.json()["data"] == baseline.json()["data"], response.text
        assert len(response.json()["data"]) == 1 and response.json()["data"][0]["request_id"] == prefix + "-p1"
        wrong_member: Final = _query(gateway, {**filters, "user_id": "fixture-other"}, ("fixture-p1",))
        assert wrong_member.status_code == 200 and wrong_member.json()["data"] == [], wrong_member.text
        wrong_key: Final = _query(gateway, {**filters, "api_key": "fixture-other"}, ("fixture-p1",))
        assert wrong_key.status_code == 200 and wrong_key.json()["data"] == [], wrong_key.text


def test_project_filter_requires_authentication(gateway: Gateway) -> None:
    response: Final = gateway.client.get(
        "/spend/logs/v2", params={"start_date": START, "end_date": END, "project_ids": "fixture-p1"}
    )
    assert response.status_code == 401, response.text


def test_project_filter_intersects_the_callers_implicit_team_scope_and_search(gateway: Gateway) -> None:
    prefix: Final = "project-filter-implicit-" + uuid.uuid4().hex
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role=LitellmUserRoles.INTERNAL_USER.value)
        own_team: Final = scenario.team(members_with_roles=[{"user_id": user, "role": "admin"}])
        foreign_team: Final = scenario.team()
        key: Final = scenario.key(user_id=user, team_id=own_team)
        scenario.cleanups.callback(_delete_rows, prefix)
        _seed_edges(prefix + "-own", own_team, user)
        _seed_edges(prefix + "-foreign", foreign_team, "fixture-foreign-member")
        filters: Final = {"start_date": START, "end_date": END}
        baseline: Final = _query(gateway, filters, None, key=key)
        filtered: Final = _query(gateway, filters, ("fixture-p1",), key=key)
        assert baseline.status_code == filtered.status_code == 200, (baseline.text, filtered.text)
        assert len(baseline.json()["data"]) == 9, baseline.text
        assert [row["request_id"] for row in filtered.json()["data"]] == [prefix + "-own-p1"], filtered.text
        for selector in ("search", "request_id"):
            response: Final = _query(gateway, {**filters, selector: prefix + "-own-unbound"}, ("fixture-p1",), key=key)
            assert response.status_code == 200 and response.json()["data"] == [], response.text
            foreign: Final = _query(gateway, {**filters, selector: prefix + "-foreign-p1"}, ("fixture-p1",), key=key)
            assert foreign.status_code == 200 and foreign.json()["data"] == [], foreign.text


def test_project_ids_are_documented_as_repeated_optional_query_values(gateway: Gateway) -> None:
    response: Final = gateway.request("GET", "/openapi.json")
    assert response.status_code == 200, response.text[:500]
    parameters: Final = response.json()["paths"]["/spend/logs/v2"]["get"]["parameters"]
    project: Final = next(parameter for parameter in parameters if parameter["name"] == "project_ids")
    assert project["in"] == "query" and project["required"] is False, project
    assert {"type": "array", "items": {"type": "string"}} in project["schema"]["anyOf"], project
