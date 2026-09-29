import os
import signal
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.daily_activity import (
    AGGREGATED_USER_ACTIVITY,
    DAY,
    TEAM_SPEND,
    USER_SPEND,
    activity_of_key,
    assert_key_reported,
    daily_rows,
    insert_daily_rows,
    key_metadata,
    key_no_key_table_holds,
    locked_table,
    records_of_key,
    seeded_metrics,
    seeded_row,
    user_row,
    user_with_an_email,
)
from integration._support.database import scratch_database
from integration._support.process import OwnedProxy, group_members, owned_proxy_process

USER_ACTIVITY: Final = "/user/daily/activity"
TEAM_ACTIVITY: Final = "/team/daily/activity"
AGGREGATED_TEAM_ACTIVITY: Final = "/team/daily/activity/aggregated"
KEYS_OF_ONE_TEAM: Final = 300
GIVES_UP_WITHIN_SECONDS: Final = 10
READS_AFTER_THE_WORKER_IS_REPLACED: Final = 6


@contextmanager
def _proxy_on(gateway: Gateway, directory: Path, database_url: str, *, workers: int = 1) -> Iterator[OwnedProxy]:
    with owned_proxy_process(
        gateway,
        directory,
        {"DATABASE_URL": database_url},
        remove_environment=("DATABASE_URL_READ_REPLICA",),
        workers=workers,
    ) as owned:
        yield owned


def _owner_on(candidate: Gateway) -> tuple[str, str]:
    owner: Final = f"integration-{uuid.uuid4().hex}"
    email: Final = f"{owner}@example.com"
    candidate.post("/user/new", {"user_id": owner, "user_email": email, "auto_create_key": False})
    return owner, email


def _read_on_a_new_connection(candidate: Gateway, api_key: str) -> httpx.Response:
    return candidate.request(
        "GET",
        AGGREGATED_USER_ACTIVITY,
        params={"start_date": DAY, "end_date": DAY, "api_key": api_key},
        headers={"Connection": "close"},
    )


def _running_children(owned: OwnedProxy) -> tuple[int, ...]:
    return tuple(
        member.pid
        for member in group_members(owned.process.pid)
        if member.pid != owned.process.pid and member.is_running() and member.status() != psutil.STATUS_ZOMBIE
    )


def test_user_reading_a_key_shared_with_another_user_is_shown_no_owner_and_nothing_of_the_other_user(
    gateway: Gateway,
) -> None:
    api_key: Final = key_no_key_table_holds()
    with gateway.scenario() as scenario:
        reader, _ = user_with_an_email(scenario)
        other, other_email = user_with_an_email(scenario)
        reader_key: Final = scenario.key(user_id=reader)
        with daily_rows((user_row(reader, api_key, DAY), user_row(other, api_key, DAY))):
            response: Final = activity_of_key(gateway, USER_ACTIVITY, api_key, reader=reader_key)
        assert_key_reported(response, api_key, DAY, key_metadata(), seeded_metrics(1))
        assert other not in response.text
        assert other_email not in response.text


def test_user_reading_a_key_only_they_spent_with_is_shown_themselves_as_its_owner(gateway: Gateway) -> None:
    api_key: Final = key_no_key_table_holds()
    with gateway.scenario() as scenario:
        reader, email = user_with_an_email(scenario)
        reader_key: Final = scenario.key(user_id=reader)
        with daily_rows((user_row(reader, api_key, DAY),)):
            response: Final = activity_of_key(gateway, USER_ACTIVITY, api_key, reader=reader_key)
        assert_key_reported(response, api_key, DAY, key_metadata(user=reader, email=email), seeded_metrics(1))


def test_user_reading_a_key_only_another_user_spent_with_is_shown_nothing_of_it(gateway: Gateway) -> None:
    api_key: Final = key_no_key_table_holds()
    with gateway.scenario() as scenario:
        reader, _ = user_with_an_email(scenario)
        other, other_email = user_with_an_email(scenario)
        reader_key: Final = scenario.key(user_id=reader)
        with daily_rows((user_row(other, api_key, DAY),)):
            response: Final = activity_of_key(gateway, USER_ACTIVITY, api_key, reader=reader_key)
        assert response.status_code == 200, response.text
        assert object_value(response.json())["results"] == [], response.text
        assert other not in response.text
        assert other_email not in response.text


def test_invalid_key_is_refused_without_naming_the_owner(gateway: Gateway) -> None:
    api_key: Final = key_no_key_table_holds()
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)):
            response: Final = activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key, reader="sk-not-a-key")
        assert response.status_code == 401, response.text
        assert owner not in response.text
        assert email not in response.text


def test_five_kilobyte_key_is_reported_with_the_one_user_its_daily_spend_names(gateway: Gateway) -> None:
    api_key: Final = f"integration-5kb-{uuid.uuid4().hex}-{'k' * 5000}"
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(user=owner, email=email),
                seeded_metrics(1),
            )


def test_key_with_no_daily_spend_is_reported_as_no_activity(gateway: Gateway) -> None:
    response: Final = activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, key_no_key_table_holds())
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    assert body["results"] == [], response.text
    totals: Final = object_value(body["metadata"])
    assert [totals["total_spend"], totals["total_api_requests"]] == [0.0, 0], response.text


def test_every_key_of_a_team_is_reported_with_its_own_user(gateway: Gateway) -> None:
    team: Final = f"integration-entity-{uuid.uuid4().hex}"
    owners: Final = {key_no_key_table_holds(): f"integration-owner-{uuid.uuid4().hex}" for _ in range(KEYS_OF_ONE_TEAM)}
    rows: Final = tuple(
        chain.from_iterable(
            (user_row(owner, api_key, DAY), seeded_row(TEAM_SPEND, "team_id", team, api_key, DAY))
            for api_key, owner in owners.items()
        )
    )
    with daily_rows(rows):
        response: Final = gateway.request(
            "GET", AGGREGATED_TEAM_ACTIVITY, params={"start_date": DAY, "end_date": DAY, "team_ids": team}
        )
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    days: Final = body["results"]
    assert isinstance(days, list) and len(days) == 1, response.text
    reported: Final = object_value(object_value(object_value(days[0])["breakdown"])["api_keys"])
    assert {api_key: object_value(record)["metadata"] for api_key, record in reported.items()} == {
        api_key: key_metadata(user=owner) for api_key, owner in owners.items()
    }, response.text
    totals: Final = object_value(body["metadata"])
    assert totals["total_api_requests"] == KEYS_OF_ONE_TEAM, response.text
    assert totals["total_spend"] == pytest.approx(0.25 * KEYS_OF_ONE_TEAM), response.text


def test_reading_the_same_activity_twice_gives_the_same_answer(gateway: Gateway) -> None:
    api_key: Final = key_no_key_table_holds()
    with gateway.scenario() as scenario:
        owner, _ = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)):
            first: Final = activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key)
            second: Final = activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key)
    assert [first.status_code, second.status_code] == [200, 200], [first.text, second.text]
    assert records_of_key(first.json(), api_key), first.text
    assert first.json() == second.json(), [first.text, second.text]


def test_key_stops_being_reported_with_an_owner_once_a_second_user_spends_with_it(gateway: Gateway) -> None:
    api_key: Final = key_no_key_table_holds()
    with gateway.scenario() as scenario:
        first, email = user_with_an_email(scenario)
        second, _ = user_with_an_email(scenario)
        with daily_rows((user_row(first, api_key, DAY),)):
            alone: Final = activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key)
            with daily_rows((user_row(second, api_key, DAY),)):
                shared: Final = activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key)
    assert_key_reported(alone, api_key, DAY, key_metadata(user=first, email=email), seeded_metrics(1))
    assert_key_reported(shared, api_key, DAY, key_metadata(), seeded_metrics(2))


@pytest.mark.timeout(300)
def test_owner_lookup_gives_up_while_daily_user_spend_is_locked_and_answers_once_it_is_not(
    gateway: Gateway, tmp_path: Path
) -> None:
    api_key: Final = key_no_key_table_holds()
    team: Final = f"integration-entity-{uuid.uuid4().hex}"
    with scratch_database() as database_url, _proxy_on(gateway, tmp_path, database_url) as owned:
        owner, email = _owner_on(owned.gateway)
        rows: Final = (user_row(owner, api_key, DAY), seeded_row(TEAM_SPEND, "team_id", team, api_key, DAY))
        with daily_rows(rows, database_url=database_url):
            with locked_table(USER_SPEND, database_url=database_url):
                started: Final = time.monotonic()
                locked: Final = activity_of_key(owned.gateway, TEAM_ACTIVITY, api_key, team_ids=team)
                waited: Final = time.monotonic() - started
            unlocked: Final = activity_of_key(owned.gateway, TEAM_ACTIVITY, api_key, team_ids=team)
    assert waited < GIVES_UP_WITHIN_SECONDS, waited
    assert_key_reported(locked, api_key, DAY, key_metadata(), seeded_metrics(1))
    assert_key_reported(unlocked, api_key, DAY, key_metadata(user=owner, email=email), seeded_metrics(1))


@pytest.mark.timeout(300)
def test_owner_is_reported_while_a_worker_is_killed_and_after_it_is_replaced(gateway: Gateway, tmp_path: Path) -> None:
    api_key: Final = key_no_key_table_holds()
    with scratch_database() as database_url, _proxy_on(gateway, tmp_path, database_url, workers=2) as owned:
        owner, email = _owner_on(owned.gateway)
        with daily_rows((user_row(owner, api_key, DAY),), database_url=database_url):
            before: Final = _read_on_a_new_connection(owned.gateway, api_key)
            members: Final = tuple(
                member for member in group_members(owned.process.pid) if member.pid != owned.process.pid
            )
            children: Final = tuple(member.pid for member in members)
            workers: Final = tuple(
                member.pid for member in members if any("spawn_main" in part for part in member.cmdline())
            )
            assert len(workers) >= 2, workers
            os.kill(workers[0], signal.SIGKILL)
            during: Final = _read_on_a_new_connection(owned.gateway, api_key)
            eventually(
                lambda: _running_children(owned),
                lambda pids: len(pids) >= len(children) and any(pid not in children for pid in pids),
                seconds=30,
            )
            after: Final = tuple(
                _read_on_a_new_connection(owned.gateway, api_key) for _ in range(READS_AFTER_THE_WORKER_IS_REPLACED)
            )
    for response in (before, during, *after):
        assert_key_reported(response, api_key, DAY, key_metadata(user=owner, email=email), seeded_metrics(1))


@pytest.mark.timeout(300)
def test_owner_is_reported_again_after_the_proxy_restarts(gateway: Gateway, tmp_path: Path) -> None:
    api_key: Final = key_no_key_table_holds()
    with scratch_database() as database_url:
        with _proxy_on(gateway, tmp_path, database_url) as first:
            owner, email = _owner_on(first.gateway)
            insert_daily_rows((user_row(owner, api_key, DAY),), database_url=database_url)
            before: Final = activity_of_key(first.gateway, AGGREGATED_USER_ACTIVITY, api_key)
        with _proxy_on(gateway, tmp_path, database_url) as second:
            after: Final = activity_of_key(second.gateway, AGGREGATED_USER_ACTIVITY, api_key)
    for response in (before, after):
        assert_key_reported(response, api_key, DAY, key_metadata(user=owner, email=email), seeded_metrics(1))
