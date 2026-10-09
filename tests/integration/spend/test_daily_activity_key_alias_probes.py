import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.daily_activity import (
    AGGREGATED_USER_ACTIVITY,
    DAY,
    ROUTES,
    SPEND_LOGS_TABLE,
    USER_SPEND,
    Route,
    SpendLogRow,
    activity_of_key,
    assert_key_reported,
    daily_rows,
    digest_no_key_table_holds,
    key_metadata,
    locked_table,
    named_row,
    nameless_rows,
    records_of_key,
    seeded_metrics,
    seeded_row,
    spend_logs_of_key,
    started_at,
    user_row,
    user_with_an_email,
)
from integration._support.database import read_rows, scratch_database
from integration._support.process import OwnedProxy, owned_proxy_process
from pydantic import JsonValue

DAY_OUTSIDE_THE_WINDOW: Final = "2026-02-10"
GIVES_UP_WITHIN_SECONDS: Final = 10
CONCURRENT_READS: Final = 20
CACHED_MISS_CLEARS_WITHIN_SECONDS: Final = 45
ALIAS_OF_ONE_SPEND_LOG: Final = (
    "SELECT metadata->>'user_api_key_alias' AS alias FROM \"LiteLLM_SpendLogs\" WHERE request_id = %s"
)


def _alias() -> str:
    return f"integration-alias-{uuid.uuid4().hex}"


def _named_between_fifty_and_fifty(alias: str) -> tuple[SpendLogRow, ...]:
    return (*nameless_rows(50), named_row(50, alias), *nameless_rows(50, 51))


def _oldest_named(alias: str) -> tuple[SpendLogRow, ...]:
    return (named_row(0, alias), *nameless_rows(150, 1))


def _newest_named(alias: str) -> tuple[SpendLogRow, ...]:
    return (*nameless_rows(150), named_row(150, alias))


def _both_edges_named(alias: str) -> tuple[SpendLogRow, ...]:
    return (named_row(0, alias), *nameless_rows(150, 1), named_row(151, alias))


def _named_after_one_hundred(alias: str) -> tuple[SpendLogRow, ...]:
    return (*nameless_rows(100), named_row(100, alias), *nameless_rows(99, 101))


def _named_after_ninety_nine(alias: str) -> tuple[SpendLogRow, ...]:
    return (*nameless_rows(99), named_row(99, alias), *nameless_rows(100, 100))


def _named_only_in_the_middle(alias: str) -> tuple[SpendLogRow, ...]:
    return (*nameless_rows(100), named_row(100, alias), *nameless_rows(100, 101))


def _renamed_and_renamed_back(alias: str, other: str) -> tuple[SpendLogRow, ...]:
    return (
        named_row(0, alias),
        *nameless_rows(100, 1),
        named_row(101, other),
        *nameless_rows(100, 102),
        named_row(202, alias),
    )


def _team_in_the_column(team: str) -> SpendLogRow:
    return SpendLogRow(started_at(0), {}, team_id=team)


def _team_in_the_metadata(team: str) -> SpendLogRow:
    return SpendLogRow(started_at(0), {"user_api_key_team_id": team})


def _user_in_the_column(user: str) -> SpendLogRow:
    return SpendLogRow(started_at(0), {}, user=user)


def _user_in_the_metadata(user: str) -> SpendLogRow:
    return SpendLogRow(started_at(0), {"user_api_key_user_id": user})


def _activity_on_route(gateway: Gateway, route: Route, api_key: str, entity: str) -> httpx.Response:
    filters: Final = {} if route.entity_filter is None else {route.entity_filter: entity}
    return activity_of_key(gateway, route.path, api_key, **filters)


def _reported_aliases(response: httpx.Response, api_key: str) -> tuple[JsonValue, ...]:
    if response.status_code != 200:
        return ()
    return tuple(
        object_value(object_value(record)["metadata"])["key_alias"]
        for record in records_of_key(object_value(response.json()), api_key)
    )


def _names_the_key(api_key: str, alias: str) -> Callable[[httpx.Response], bool]:
    def names(response: httpx.Response) -> bool:
        reported: Final = _reported_aliases(response, api_key)
        return bool(reported) and frozenset(reported) == frozenset((alias,))

    return names


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


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.path.strip("/").replace("/", "_"))
def test_alias_named_only_by_a_spend_log_is_reported_on_every_daily_activity_route(
    gateway: Gateway, route: Route
) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    entity: Final = f"integration-entity-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        entity_rows: Final = (
            () if route.table == USER_SPEND else (seeded_row(route.table, route.entity_column, entity, api_key, DAY),)
        )
        filters: Final = {} if route.entity_filter is None else {route.entity_filter: entity}
        with (
            daily_rows((user_row(owner, api_key, DAY), *entity_rows)),
            spend_logs_of_key(api_key, (named_row(0, alias),)),
        ):
            assert_key_reported(
                activity_of_key(gateway, route.path, api_key, **filters),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email),
                seeded_metrics(1),
            )


@pytest.mark.parametrize(
    "layout",
    (
        pytest.param(_named_between_fifty_and_fifty, id="named_between_50_and_50_nameless"),
        pytest.param(_oldest_named, id="oldest_named_150_nameless_newer"),
        pytest.param(_newest_named, id="newest_named_150_nameless_older"),
        pytest.param(_both_edges_named, id="both_edges_named_150_nameless_between"),
        pytest.param(_named_after_one_hundred, id="100_nameless_named_99_nameless"),
        pytest.param(_named_after_ninety_nine, id="99_nameless_named_100_nameless"),
    ),
)
def test_alias_on_an_edge_of_the_window_is_reported_whatever_surrounds_it(
    gateway: Gateway, layout: Callable[[str], tuple[SpendLogRow, ...]]
) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)), spend_logs_of_key(api_key, layout(alias)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email),
                seeded_metrics(1),
            )


def test_alias_named_only_in_the_middle_of_two_hundred_nameless_rows_is_not_picked_up(gateway: Gateway) -> None:
    api_key: Final = digest_no_key_table_holds()
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with (
            daily_rows((user_row(owner, api_key, DAY),)),
            spend_logs_of_key(api_key, _named_only_in_the_middle(_alias())),
        ):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(user=owner, email=email),
                seeded_metrics(1),
            )


def test_key_renamed_and_renamed_back_is_reported_with_the_alias_on_both_edges(gateway: Gateway) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        rows: Final = _renamed_and_renamed_back(alias, _alias())
        with daily_rows((user_row(owner, api_key, DAY),)), spend_logs_of_key(api_key, rows):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email),
                seeded_metrics(1),
            )


@pytest.mark.parametrize(
    "spend_log_of_team",
    (
        pytest.param(_team_in_the_column, id="team_id_column"),
        pytest.param(_team_in_the_metadata, id="team_id_in_metadata"),
    ),
)
def test_team_named_only_by_a_spend_log_is_reported_next_to_the_daily_owner(
    gateway: Gateway, spend_log_of_team: Callable[[str], SpendLogRow]
) -> None:
    api_key: Final = digest_no_key_table_holds()
    team: Final = f"integration-team-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)), spend_logs_of_key(api_key, (spend_log_of_team(team),)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(team=team, user=owner, email=email),
                seeded_metrics(1),
            )


@pytest.mark.parametrize(
    "spend_log_of_user",
    (
        pytest.param(_user_in_the_column, id="user_column"),
        pytest.param(_user_in_the_metadata, id="user_id_in_metadata"),
    ),
)
def test_user_named_by_a_spend_log_beats_the_owner_the_daily_rows_name(
    gateway: Gateway, spend_log_of_user: Callable[[str], SpendLogRow]
) -> None:
    api_key: Final = digest_no_key_table_holds()
    with gateway.scenario() as scenario:
        daily_owner, _ = user_with_an_email(scenario)
        log_user, log_email = user_with_an_email(scenario)
        with (
            daily_rows((user_row(daily_owner, api_key, DAY),)),
            spend_logs_of_key(api_key, (spend_log_of_user(log_user),)),
        ):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(user=log_user, email=log_email),
                seeded_metrics(1),
            )


def test_hashed_jwt_digest_is_named_by_its_spend_log(gateway: Gateway) -> None:
    api_key: Final = f"hashed-jwt-{sha256(uuid.uuid4().bytes).hexdigest()}"
    alias: Final = _alias()
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)), spend_logs_of_key(api_key, (named_row(0, alias),)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email),
                seeded_metrics(1),
            )


@pytest.mark.parametrize(
    ("started", "inside_the_window"),
    (
        pytest.param("2026-02-01 23:59:59", False, id="second_before_the_window"),
        pytest.param("2026-02-02 00:00:00", True, id="first_second_of_the_window"),
        pytest.param("2026-02-04 23:59:59", True, id="last_second_of_the_window"),
        pytest.param("2026-02-05 00:00:00", False, id="first_second_after_the_window"),
    ),
)
def test_spend_log_names_the_key_only_from_one_day_before_to_two_days_after_the_read(
    gateway: Gateway, started: str, inside_the_window: bool
) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    row: Final = SpendLogRow(started, {"user_api_key_alias": alias})
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)), spend_logs_of_key(api_key, (row,)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias if inside_the_window else None, user=owner, email=email),
                seeded_metrics(1),
            )


def test_two_aliases_on_the_two_edges_leave_the_key_unnamed(gateway: Gateway) -> None:
    api_key: Final = digest_no_key_table_holds()
    rows: Final = (named_row(0, _alias()), *nameless_rows(150, 1), named_row(151, _alias()))
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)), spend_logs_of_key(api_key, rows):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(user=owner, email=email),
                seeded_metrics(1),
            )


@pytest.mark.parametrize(
    "unnamed_rows",
    (
        pytest.param((SpendLogRow(started_at(0), {"user_api_key_alias": ""}),), id="empty_string_alias"),
        pytest.param(
            (SpendLogRow(started_at(0), ["x"]), SpendLogRow(started_at(1), "x")), id="array_then_string_metadata"
        ),
    ),
)
def test_rows_without_a_usable_alias_do_not_hide_the_named_row_after_them(
    gateway: Gateway, unnamed_rows: tuple[SpendLogRow, ...]
) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    rows: Final = (*unnamed_rows, named_row(len(unnamed_rows), alias))
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)), spend_logs_of_key(api_key, rows):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email),
                seeded_metrics(1),
            )


@pytest.mark.parametrize(
    "stored_alias",
    (
        pytest.param(123, id="json_int"),
        pytest.param(["a"], id="json_list"),
        pytest.param("a" * 5000, id="five_kb_string"),
    ),
)
def test_alias_of_an_unexpected_shape_is_reported_as_postgres_renders_it(
    gateway: Gateway, stored_alias: JsonValue
) -> None:
    api_key: Final = digest_no_key_table_holds()
    row: Final = SpendLogRow(started_at(0), {"user_api_key_alias": stored_alias})
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY),)), spend_logs_of_key(api_key, (row,)) as request_ids:
            rendered: Final = read_rows(ALIAS_OF_ONE_SPEND_LOG, (request_ids[0],))[0]["alias"]
            assert isinstance(rendered, str) and rendered, rendered
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=rendered, user=owner, email=email),
                seeded_metrics(1),
            )


@pytest.mark.timeout(300)
def test_alias_found_once_is_served_from_the_cache_for_the_same_window_only(gateway: Gateway, tmp_path: Path) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    with scratch_database() as database_url, _proxy_on(gateway, tmp_path, database_url) as owned:
        owner, email = _owner_on(owned.gateway)
        rows: Final = (user_row(owner, api_key, DAY), user_row(owner, api_key, DAY_OUTSIDE_THE_WINDOW))
        with daily_rows(rows, database_url=database_url):
            with spend_logs_of_key(api_key, (named_row(0, alias),), database_url=database_url):
                first: Final = activity_of_key(owned.gateway, AGGREGATED_USER_ACTIVITY, api_key)
            cached: Final = activity_of_key(owned.gateway, AGGREGATED_USER_ACTIVITY, api_key)
            other_window: Final = owned.gateway.request(
                "GET",
                AGGREGATED_USER_ACTIVITY,
                params={"start_date": DAY_OUTSIDE_THE_WINDOW, "end_date": DAY_OUTSIDE_THE_WINDOW, "api_key": api_key},
            )
    named: Final = key_metadata(alias=alias, user=owner, email=email)
    assert_key_reported(first, api_key, DAY, named, seeded_metrics(1))
    assert_key_reported(cached, api_key, DAY, named, seeded_metrics(1))
    assert_key_reported(
        other_window, api_key, DAY_OUTSIDE_THE_WINDOW, key_metadata(user=owner, email=email), seeded_metrics(1)
    )


@pytest.mark.timeout(300)
def test_alias_logged_after_a_cached_miss_shows_once_the_miss_expires(gateway: Gateway, tmp_path: Path) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    with scratch_database() as database_url, _proxy_on(gateway, tmp_path, database_url) as owned:
        owner, email = _owner_on(owned.gateway)
        with daily_rows((user_row(owner, api_key, DAY),), database_url=database_url):
            missed: Final = activity_of_key(owned.gateway, AGGREGATED_USER_ACTIVITY, api_key)
            with spend_logs_of_key(api_key, (named_row(0, alias),), database_url=database_url):
                named: Final = eventually(
                    lambda: activity_of_key(owned.gateway, AGGREGATED_USER_ACTIVITY, api_key),
                    _names_the_key(api_key, alias),
                    seconds=CACHED_MISS_CLEARS_WITHIN_SECONDS,
                )
    assert_key_reported(missed, api_key, DAY, key_metadata(user=owner, email=email), seeded_metrics(1))
    assert_key_reported(named, api_key, DAY, key_metadata(alias=alias, user=owner, email=email), seeded_metrics(1))


@pytest.mark.timeout(300)
def test_alias_lookup_gives_up_while_spend_logs_are_locked_and_answers_once_they_are_not(
    gateway: Gateway, tmp_path: Path
) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    with scratch_database() as database_url, _proxy_on(gateway, tmp_path, database_url, workers=2) as owned:
        owner, email = _owner_on(owned.gateway)
        with (
            daily_rows((user_row(owner, api_key, DAY),), database_url=database_url),
            spend_logs_of_key(api_key, (named_row(0, alias),), database_url=database_url),
        ):
            with locked_table(SPEND_LOGS_TABLE, database_url=database_url):
                started: Final = time.monotonic()
                locked: Final = activity_of_key(owned.gateway, AGGREGATED_USER_ACTIVITY, api_key)
                waited: Final = time.monotonic() - started
            unlocked: Final = eventually(
                lambda: activity_of_key(owned.gateway, AGGREGATED_USER_ACTIVITY, api_key),
                _names_the_key(api_key, alias),
                seconds=CACHED_MISS_CLEARS_WITHIN_SECONDS,
            )
    assert waited < GIVES_UP_WITHIN_SECONDS, waited
    assert_key_reported(locked, api_key, DAY, key_metadata(user=owner, email=email), seeded_metrics(1))
    assert_key_reported(unlocked, api_key, DAY, key_metadata(alias=alias, user=owner, email=email), seeded_metrics(1))


def test_concurrent_reads_over_every_route_all_name_a_fresh_key(gateway: Gateway) -> None:
    api_key: Final = digest_no_key_table_holds()
    alias: Final = _alias()
    entity: Final = f"integration-entity-{uuid.uuid4().hex}"
    entity_columns: Final = {route.table: route.entity_column for route in ROUTES if route.table != USER_SPEND}
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        rows: Final = (
            user_row(owner, api_key, DAY),
            *(seeded_row(table, column, entity, api_key, DAY) for table, column in entity_columns.items()),
        )
        with (
            daily_rows(rows),
            spend_logs_of_key(api_key, (named_row(0, alias),)),
            ThreadPoolExecutor(CONCURRENT_READS) as pool,
        ):
            reads: Final = tuple(
                pool.submit(_activity_on_route, gateway, ROUTES[index % len(ROUTES)], api_key, entity)
                for index in range(CONCURRENT_READS)
            )
            responses: Final = tuple(read.result() for read in reads)
    for response in responses:
        assert_key_reported(
            response, api_key, DAY, key_metadata(alias=alias, user=owner, email=email), seeded_metrics(1)
        )
