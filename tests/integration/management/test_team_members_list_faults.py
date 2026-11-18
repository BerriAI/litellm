"""`GET /management/v1/teams/{team_id}/members` when the request is malformed, the roster is empty,
the database drops out, or a worker dies while pages are being read.

The outage and worker-kill rows run on a two-worker proxy the test owns. Nothing sleeps, and only
processes the test started are signalled.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, list_value, object_value
from tests.integration._support.database import scratch_database, write_rows
from tests.integration._support.database_relay import database_relay, plaintext_database_url
from tests.integration._support.process import owned_proxy_process

PROBLEM_TYPE_BASE: Final = "urn:litellm:error:"
WORKERS: Final = 2
BURST: Final = 24
SEEDED_MEMBERS: Final = 250
PAGE_SIZE: Final = 100
OUTAGE_READS: Final = 24
AFTER_KILL_WAIT_SECONDS: Final = 30
REMOVE_FROM_ENVIRONMENT: Final = ("DATABASE_URL_READ_REPLICA",)

SEED_ROSTER_SQL: Final = """
UPDATE "LiteLLM_TeamTable"
SET members_with_roles = (
    SELECT jsonb_agg(jsonb_build_object('user_id', %s || '-' || lpad(n::text, 5, '0'), 'role', 'user') ORDER BY n)
    FROM generate_series(1, %s::int) AS n
)
WHERE team_id = %s
"""


def _path(team_id: str) -> str:
    return f"/management/v1/teams/{team_id}/members"


def _get(gateway: Gateway, team_id: str, query: Sequence[tuple[str, str]]) -> httpx.Response:
    return gateway.client.get(_path(team_id), params=tuple(query), headers={"Authorization": f"Bearer {gateway.key}"})


def _user_ids(response: httpx.Response) -> list[JsonValue]:
    assert response.status_code == 200, f"{response.request.url}: {response.status_code} {response.text}"
    return [object_value(row)["user_id"] for row in list_value(object_value(response.json())["data"])]


def _seeded_page(prefix: str, page: int) -> list[JsonValue]:
    first: Final = (page - 1) * PAGE_SIZE + 1
    return [f"{prefix}-{n:05d}" for n in range(first, min(first + PAGE_SIZE, SEEDED_MEMBERS + 1))]


@pytest.mark.parametrize(
    ("query", "problem"),
    [
        ((("sort", "favourite_colour"),), "invalid-sort-field"),
        ((("filter[favourite_colour][eq]", "blue"),), "unknown-query-parameter"),
        ((("filter[role][gt]", "admin"),), "unsupported-filter-operator"),
        ((("page", "0"),), "invalid-query-parameter"),
        ((("page", "one"),), "invalid-query-parameter"),
        ((("page_size", "0"),), "invalid-query-parameter"),
        ((("page_size", ""),), "invalid-query-parameter"),
        ((("q", "ann"), ("q", "ann")), "duplicate-query-parameter"),
        ((("filter[role][eq]", "user"), ("filter[role][eq]", "user")), "duplicate-query-parameter"),
    ],
)
def test_a_malformed_query_is_a_400_problem_and_the_next_read_still_answers(
    query: tuple[tuple[str, str], ...], problem: str, gateway: Gateway, scenario: Scenario
) -> None:
    team_id: Final = scenario.team()
    refused: Final = _get(gateway, team_id, query)
    assert refused.status_code == 400, f"{query}: {refused.status_code} {refused.text}"
    body: Final = object_value(refused.json())
    assert (body["type"], body["status"]) == (f"{PROBLEM_TYPE_BASE}{problem}", 400), refused.text
    assert _get(gateway, team_id, ()).status_code == 200


def test_a_5_kb_search_matches_nobody(gateway: Gateway, scenario: Scenario) -> None:
    team_id: Final = scenario.team()
    scenario.member(team_id)
    response: Final = _get(gateway, team_id, (("q", "x" * 5120),))
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    assert (body["data"], object_value(body["meta"])["total_count"]) == ([], 0), response.text


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer sk-not-a-key-of-this-proxy"}])
def test_a_caller_without_a_valid_key_is_refused_with_401(
    headers: Mapping[str, str], gateway: Gateway, scenario: Scenario
) -> None:
    team_id: Final = scenario.team()
    scenario.member(team_id)
    response: Final = gateway.client.get(_path(team_id), headers=headers)
    assert response.status_code == 401, response.text
    assert "data" not in object_value(response.json()), response.text


def _data_and_meta(response: httpx.Response) -> tuple[JsonValue, JsonValue]:
    assert response.status_code == 200, f"{response.request.url}: {response.status_code} {response.text}"
    body: Final = object_value(response.json())
    return body["data"], body["meta"]


def test_a_team_with_an_empty_roster_lists_no_members(gateway: Gateway, scenario: Scenario) -> None:
    team_id: Final = scenario.team()
    write_rows('UPDATE "LiteLLM_TeamTable" SET members_with_roles = %s::jsonb WHERE team_id = %s', ("[]", team_id))
    response: Final = _get(gateway, team_id, ())
    assert _data_and_meta(response) == (
        [],
        {"total_count": 0, "page": 1, "page_size": 50, "total_pages": 0},
    ), response.text


def test_a_page_past_the_end_is_empty_and_still_counts_the_roster(gateway: Gateway, scenario: Scenario) -> None:
    team_id: Final = scenario.team()
    prefix: Final = f"faults{uuid.uuid4().hex[:12]}"
    write_rows(SEED_ROSTER_SQL, (prefix, str(SEEDED_MEMBERS), team_id))
    response: Final = _get(gateway, team_id, (("page", "4"), ("page_size", str(PAGE_SIZE))))
    assert _data_and_meta(response) == (
        [],
        {"total_count": SEEDED_MEMBERS, "page": 4, "page_size": PAGE_SIZE, "total_pages": 3},
    ), response.text


@pytest.mark.timeout(240)
def test_members_answer_503_database_unavailable_during_an_outage_and_200_once_it_ends(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = f"outage{uuid.uuid4().hex[:12]}"
    with (
        scratch_database() as database_url,
        database_relay(database_url, marker.encode()) as (relay, relayed_url),
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": plaintext_database_url(relayed_url), "PRISMA_HEALTH_WATCHDOG_ENABLED": "false"},
            remove_environment=REMOVE_FROM_ENVIRONMENT,
            workers=WORKERS,
        ) as owned,
        ThreadPoolExecutor(OUTAGE_READS) as pool,
    ):
        team_id: Final = f"{marker}-team"
        owned.gateway.post("/team/new", {"team_id": team_id, "team_alias": marker})
        owned.gateway.post(
            "/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": f"{marker}-member"}}
        )
        before: Final = _get(owned.gateway, team_id, ())
        assert _user_ids(before)[-1:] == [f"{marker}-member"], before.text

        relay.arm()
        during: Final = tuple(pool.map(lambda _: _get(owned.gateway, team_id, ()), range(OUTAGE_READS)))
        assert relay.tripped.is_set(), "no members read reached the database relay"
        assert relay.refused > 0, "the proxy never tried to reconnect while the database refused connections"
        answers: Final = [
            (response.status_code, json.loads(response.text).get("type"), json.loads(response.text).get("title"))
            for response in during
        ]
        assert answers == [(503, f"{PROBLEM_TYPE_BASE}database-unavailable", "Database unavailable")] * OUTAGE_READS, [
            f"{response.status_code} {response.text[:200]}" for response in during
        ]

        after: Final = eventually(
            lambda: _get(owned.gateway, team_id, ()),
            lambda response: response.status_code == 200,
            seconds=90,
            return_last_on_timeout=True,
        )
        assert relay.reconnected.is_set(), "the proxy never reconnected to the database"
        assert after.status_code == 200 and after.json() == before.json(), f"{after.status_code} {after.text}"


@pytest.mark.timeout(240)
def test_pages_stay_correct_while_one_of_two_workers_is_killed_mid_burst(
    gateway: Gateway, scenario: Scenario, tmp_path: Path
) -> None:
    team_id: Final = scenario.team()
    prefix: Final = f"faults{uuid.uuid4().hex[:12]}"
    write_rows(SEED_ROSTER_SQL, (prefix, str(SEEDED_MEMBERS), team_id))
    stop: Final = threading.Event()
    killed: Final = threading.Event()
    served_after_kill: Final = threading.Semaphore(0)

    with (
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"]},
            remove_environment=REMOVE_FROM_ENVIRONMENT,
            workers=WORKERS,
        ) as owned,
        ThreadPoolExecutor(BURST) as pool,
    ):
        base_url: Final = str(owned.gateway.client.base_url)

        def read_pages(reader: int) -> tuple[tuple[bool, int, str], ...]:
            page: Final = reader % 3 + 1
            outcomes: list[tuple[bool, int, str]] = []  # mutable-ok: appended by one reader thread until told to stop
            while not stop.is_set():
                after_kill = killed.is_set()
                try:
                    with httpx.Client(base_url=base_url, timeout=15, trust_env=False) as fresh:
                        response = fresh.get(
                            _path(team_id),
                            params={"page": str(page), "page_size": str(PAGE_SIZE)},
                            headers={"Authorization": f"Bearer {owned.gateway.key}"},
                        )
                except httpx.TransportError as dropped:
                    outcomes.append((after_kill, 0, type(dropped).__name__))
                    continue
                correct = response.status_code == 200 and _user_ids(response) == _seeded_page(prefix, page)
                outcomes.append((after_kill, response.status_code, "" if correct else response.text[:300]))
                if after_kill and response.status_code == 200:
                    served_after_kill.release()
            return tuple(outcomes)

        before: Final = owned.worker_pids()
        assert len(before) == WORKERS, sorted(before)
        readers: Final = [pool.submit(read_pages, reader) for reader in range(BURST)]
        try:
            with owned.replacing_worker(min(before)) as kill:
                kill()
                killed.set()
            survivors: Final = owned.worker_pids()
            all(served_after_kill.acquire(timeout=AFTER_KILL_WAIT_SECONDS) for _ in range(BURST))
        finally:
            stop.set()
        outcomes: Final = tuple(outcome for reader in readers for outcome in reader.result(timeout=60))

    assert len(survivors) == WORKERS and min(before) not in survivors, (sorted(before), sorted(survivors))
    wrong: Final = [outcome for outcome in outcomes if outcome[1] != 0 and outcome != (outcome[0], 200, "")]
    assert wrong == [], f"{len(wrong)} of {len(outcomes)} answered pages were not the expected 200 page: {wrong[:5]}"
    answered_after_kill: Final = [outcome for outcome in outcomes if outcome[0] and outcome[1] == 200]
    assert len(answered_after_kill) >= BURST, (
        f"only {len(answered_after_kill)} pages answered after the kill, of {len(outcomes)} requests"
    )
