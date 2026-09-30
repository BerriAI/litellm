import uuid
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, string_value
from integration._support.daily_activity import (
    AGGREGATED_USER_ACTIVITY,
    DAY,
    ROUTES,
    USER_SPEND,
    Route,
    activity_of_key,
    assert_key_reported,
    daily_rows,
    key_metadata,
    key_no_key_table_holds,
    seeded_metrics,
    seeded_row,
    spend_log_naming_only_an_alias,
    user_row,
    user_with_an_email,
)


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.path.strip("/").replace("/", "_"))
def test_key_missing_from_the_key_tables_is_reported_with_the_one_user_its_daily_spend_names(
    gateway: Gateway, route: Route
) -> None:
    api_key: Final = key_no_key_table_holds()
    entity: Final = f"integration-entity-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        entity_rows: Final = (
            () if route.table == USER_SPEND else (seeded_row(route.table, route.entity_column, entity, api_key, DAY),)
        )
        filters: Final = {} if route.entity_filter is None else {route.entity_filter: entity}
        with daily_rows((user_row(owner, api_key, DAY), *entity_rows)):
            assert_key_reported(
                activity_of_key(gateway, route.path, api_key, **filters),
                api_key,
                DAY,
                key_metadata(user=owner, email=email),
                seeded_metrics(1),
            )


def test_key_whose_daily_spend_names_two_users_is_reported_with_no_owner(gateway: Gateway) -> None:
    api_key: Final = key_no_key_table_holds()
    with gateway.scenario() as scenario:
        first, _ = user_with_an_email(scenario)
        second, _ = user_with_an_email(scenario)
        with daily_rows((user_row(first, api_key, DAY), user_row(second, api_key, DAY))):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(),
                seeded_metrics(2),
            )


@pytest.mark.parametrize("unnamed", ["", None], ids=["blank_user", "null_user"])
def test_daily_spend_rows_naming_no_user_do_not_hide_the_one_user_the_others_name(
    gateway: Gateway, unnamed: str | None
) -> None:
    api_key: Final = key_no_key_table_holds()
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with daily_rows((user_row(owner, api_key, DAY), user_row(unnamed, api_key, DAY))):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(user=owner, email=email),
                seeded_metrics(2),
            )


def test_key_whose_daily_spend_names_no_user_at_all_is_reported_with_no_owner(gateway: Gateway) -> None:
    api_key: Final = key_no_key_table_holds()
    with daily_rows((user_row("", api_key, DAY), user_row(None, api_key, DAY))):
        assert_key_reported(
            activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
            api_key,
            DAY,
            key_metadata(),
            seeded_metrics(2),
        )


def test_owner_the_user_table_does_not_hold_is_reported_by_id_with_no_email(gateway: Gateway) -> None:
    api_key: Final = key_no_key_table_holds()
    owner: Final = f"integration-departed-{uuid.uuid4().hex}"
    with daily_rows((user_row(owner, api_key, DAY),)):
        assert_key_reported(
            activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
            api_key,
            DAY,
            key_metadata(user=owner),
            seeded_metrics(1),
        )


def _stored_form(token: str) -> str:
    return sha256(token.encode()).hexdigest()


def _deleted_key(gateway: Gateway, scenario: Scenario, alias: str, **fields: str) -> str:
    token: Final = string_value(gateway.post("/key/generate", {"key_alias": alias, **fields})["key"])
    scenario.delete_key(token)
    return _stored_form(token)


def test_live_key_keeps_its_own_user_when_its_daily_spend_names_another(gateway: Gateway) -> None:
    alias: Final = f"integration-alias-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        other, _ = user_with_an_email(scenario)
        api_key: Final = _stored_form(scenario.key(user_id=owner, key_alias=alias))
        with daily_rows((user_row(other, api_key, DAY),)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email, exists=True),
                seeded_metrics(1),
            )


def test_live_key_with_no_user_is_not_given_the_user_its_daily_spend_names(gateway: Gateway) -> None:
    alias: Final = f"integration-alias-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        spender, _ = user_with_an_email(scenario)
        api_key: Final = _stored_form(scenario.key(key_alias=alias))
        with daily_rows((user_row(spender, api_key, DAY),)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, exists=True),
                seeded_metrics(1),
            )


def test_deleted_key_keeps_its_own_user_when_its_daily_spend_names_another(gateway: Gateway) -> None:
    alias: Final = f"integration-alias-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        other, _ = user_with_an_email(scenario)
        api_key: Final = _deleted_key(gateway, scenario, alias, user_id=owner)
        with daily_rows((user_row(other, api_key, DAY),)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email),
                seeded_metrics(1),
            )


def test_deleted_key_with_no_user_keeps_its_alias_and_gains_the_one_user_its_daily_spend_names(
    gateway: Gateway,
) -> None:
    alias: Final = f"integration-alias-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        api_key: Final = _deleted_key(gateway, scenario, alias)
        with daily_rows((user_row(owner, api_key, DAY),)):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email),
                seeded_metrics(1),
            )


def test_key_named_only_by_a_spend_log_alias_keeps_that_alias_and_gains_the_one_user_its_daily_spend_names(
    gateway: Gateway,
) -> None:
    api_key: Final = sha256(uuid.uuid4().bytes).hexdigest()
    alias: Final = f"integration-alias-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        owner, email = user_with_an_email(scenario)
        with (
            spend_log_naming_only_an_alias(f"integration-{uuid.uuid4().hex}", api_key, f"{DAY} 12:00:00", alias),
            daily_rows((user_row(owner, api_key, DAY),)),
        ):
            assert_key_reported(
                activity_of_key(gateway, AGGREGATED_USER_ACTIVITY, api_key),
                api_key,
                DAY,
                key_metadata(alias=alias, user=owner, email=email),
                seeded_metrics(1),
            )
