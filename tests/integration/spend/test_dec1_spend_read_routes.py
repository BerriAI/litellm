"""Contract: the spend read routes, their filters and their response shapes.

Finance scripts and the admin UI read spend through these routes rather than SQL:

    GET  /spend/logs                 request rows, filtered by request_id, api_key, user_id, or a date range
    GET  /spend/tags                 per-tag request count and spend, optional date range
    GET  /spend/keys, /spend/users   key and user rows including their spend
    GET  /global/spend/tags          per-tag spend for the UI, needs both dates
    GET  /global/spend/all_tag_names every tag ever logged
    POST /global/spend/end_users     per-customer request count and spend
    GET  /global/spend/provider      spend per provider, needs both dates
    GET  /global/spend/teams         daily spend per team alias
    GET  /global/spend/keys, /global/spend/models   top keys and models, proxy admins only

Each test makes two chat calls of 0.02 with fresh ids (key, user, tag, customer), so the totals it expects are
0.04 and 2 requests regardless of what else is in the shared database. Date filters are built from the UTC day of
the request rows themselves, never from today's date.
"""

from __future__ import annotations

import uuid
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, list_value, object_value, string_value
from tests.integration._support.wire import Wire
from tests.integration.spend.dec1_spend_support import (
    ask,
    chat_model,
    hashed,
    new_customer,
    number,
    rows_of_key,
    scripted_upstream,
    served,
    spend_reaches,
)

_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _listed(response: httpx.Response) -> list[dict[str, JsonValue]]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return _ROWS.validate_json(response.content)


def _two_calls(
    gateway: Gateway, scenario: Scenario, wire: Wire, key: str | None = None, **body: JsonValue
) -> tuple[str, list[dict[str, JsonValue]]]:
    """Make two chat calls with `key` (a fresh key when omitted) and return it with its two landed rows."""
    model: Final = chat_model(scenario, wire)
    caller: Final = key or scenario.key()
    served(ask(gateway, model, caller, **body))
    served(ask(gateway, model, caller, **body))
    return caller, rows_of_key(gateway, caller, 2)


def _refused_to_internal_user(response: httpx.Response, route: str) -> None:
    assert response.status_code == 401, f"{response.status_code} {response.text}"
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    message: Final = string_value(error["message"])
    assert "Only proxy admin" in message and f"Route={route}" in message, response.text
    assert "Your role=internal_user" in message, response.text


def _internal_user_key(scenario: Scenario) -> tuple[str, str]:
    user: Final = scenario.user(user_role="internal_user")
    return user, scenario.key(user_id=user, key_alias=f"dec1-read-{uuid.uuid4().hex}")


def test_spend_logs_by_request_id_returns_that_one_row(gateway: Gateway) -> None:
    """`GET /spend/logs?request_id=<id>` as admin returns a list with exactly the row of that request: same
    `request_id`, hashed `api_key`, `call_type` `acompletion`, total_tokens 15 and spend 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        key, rows = _two_calls(gateway, scenario, wire)
        wanted: Final = string_value(rows[0]["request_id"])
        found: Final = _listed(gateway.request("GET", "/spend/logs", params={"request_id": wanted}))
        assert len(found) == 1, found
        assert (found[0]["request_id"], found[0]["api_key"], found[0]["call_type"]) == (
            wanted,
            hashed(key),
            "acompletion",
        ), found
        assert (found[0]["total_tokens"], number(found[0]["spend"])) == (15, pytest.approx(0.02)), found


def test_spend_logs_request_id_filter_also_matches_the_litellm_call_id(gateway: Gateway) -> None:
    """The `request_id` query parameter matches either the row's `request_id` or its `litellm_call_id`. Passing
    the call id (the value of the `x-litellm-call-id` response header) returns the same single row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        _, rows = _two_calls(gateway, scenario, wire)
        call_id: Final = string_value(rows[1]["litellm_call_id"])
        found: Final = _listed(gateway.request("GET", "/spend/logs", params={"request_id": call_id}))
        assert [row["request_id"] for row in found] == [rows[1]["request_id"]], found


def test_spend_logs_unknown_request_id_returns_an_empty_list(gateway: Gateway) -> None:
    """A `request_id` that matches nothing is answered 200 with `[]`, not 404."""
    found: Final = _listed(
        gateway.request("GET", "/spend/logs", params={"request_id": f"dec1-none-{uuid.uuid4().hex}"})
    )
    assert found == []


@pytest.mark.parametrize("form", ["raw", "hashed"])
def test_spend_logs_by_api_key_accepts_the_raw_key_and_its_hash(gateway: Gateway, form: str) -> None:
    """`GET /spend/logs?api_key=` takes the raw `sk-` key or its SHA-256 hash. Both return exactly the key's two
    rows, newest first, and every row carries the hash, never the raw key."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        key, rows = _two_calls(gateway, scenario, wire)
        value: Final = key if form == "raw" else hashed(key)
        found: Final = _listed(gateway.request("GET", "/spend/logs", params={"api_key": value}))
        assert [row["request_id"] for row in found] == [rows[1]["request_id"], rows[0]["request_id"]], found
        assert {row["api_key"] for row in found} == {hashed(key)}, found


def test_spend_logs_by_user_id_returns_only_that_users_rows(gateway: Gateway) -> None:
    """Two users each make two calls. `GET /spend/logs?user_id=<first user>` as admin returns the first user's
    two rows and none of the second user's."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        first_user: Final = scenario.user()
        second_user: Final = scenario.user()
        _, mine = _two_calls(gateway, scenario, wire, scenario.key(user_id=first_user))
        _two_calls(gateway, scenario, wire, scenario.key(user_id=second_user))
        found: Final = _listed(gateway.request("GET", "/spend/logs", params={"user_id": first_user}))
        assert {row["request_id"] for row in found} == {row["request_id"] for row in mine}, found
        assert {row["user"] for row in found} == {first_user}, found


def test_spend_logs_date_range_summary_groups_spend_by_day_key_user_and_model(gateway: Gateway) -> None:
    """With `start_date` and `end_date` (and the default `summarize=true`) the route returns one object per day
    instead of rows. For the key's day: `spend` 0.04, the hashed key as a field with 0.04, `users` mapping the
    owner to 0.04 and `models` mapping the deployment's model `openai/gpt-4o-mini` (as the row's `model` column
    stores it) to 0.04. The following day is padded with a zero entry."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        user: Final = scenario.user()
        key, rows = _two_calls(gateway, scenario, wire, scenario.key(user_id=user))
        day, next_day = string_value(rows[0]["start_day"]), string_value(rows[0]["next_day"])
        found: Final = _listed(
            gateway.request(
                "GET", "/spend/logs", params={"api_key": hashed(key), "start_date": day, "end_date": next_day}
            )
        )
        assert [entry["startTime"] for entry in found] == [day, next_day], found
        assert number(found[0]["spend"]) == pytest.approx(0.04), found
        assert number(found[0][hashed(key)]) == pytest.approx(0.04), found
        assert found[0]["users"] == {user: pytest.approx(0.04)}, found
        assert found[0]["models"] == {"openai/gpt-4o-mini": pytest.approx(0.04)}, found
        assert found[1] == {"startTime": next_day, "spend": 0, "users": {}, "models": {}}, found


def test_spend_logs_date_range_with_summarize_false_returns_the_rows(gateway: Gateway) -> None:
    """The same date range with `summarize=false` returns the individual rows inside the range, newest first,
    with the usual row fields."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        key, rows = _two_calls(gateway, scenario, wire)
        found: Final = _listed(
            gateway.request(
                "GET",
                "/spend/logs",
                params={
                    "api_key": hashed(key),
                    "start_date": string_value(rows[0]["start_day"]),
                    "end_date": string_value(rows[0]["next_day"]),
                    "summarize": "false",
                },
            )
        )
        assert [row["request_id"] for row in found] == [rows[1]["request_id"], rows[0]["request_id"]], found
        assert all(number(row["spend"]) == pytest.approx(0.02) for row in found), found


def test_spend_logs_requires_authentication(gateway: Gateway) -> None:
    """`GET /spend/logs` with a key that does not exist is rejected with 401 and returns no rows."""
    response: Final = gateway.request("GET", "/spend/logs", key=f"sk-dec1-unknown-{uuid.uuid4().hex}")
    assert response.status_code == 401, response.text
    assert "request_id" not in response.text, response.text


def test_spend_tags_reports_request_count_and_spend_per_tag(gateway: Gateway) -> None:
    """Two calls carry a fresh tag. `GET /spend/tags` as admin contains one entry for it:
    `individual_request_tag` the tag, `log_count` 2 and `total_spend` 0.04."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        tag: Final = f"dec1-read-tag-{uuid.uuid4().hex}"
        _two_calls(gateway, scenario, wire, metadata={"tags": [tag]})
        entries: Final = [
            entry for entry in _listed(gateway.request("GET", "/spend/tags")) if entry["individual_request_tag"] == tag
        ]
        assert len(entries) == 1, entries
        assert (entries[0]["log_count"], number(entries[0]["total_spend"])) == (2, pytest.approx(0.04)), entries


def test_spend_tags_date_range_includes_the_end_day_and_excludes_other_days(gateway: Gateway) -> None:
    """`start_date` and `end_date` are whole UTC days and the end day is inclusive: a window of just the request
    day contains the tag with `log_count` 2, a window that starts the day after does not contain it."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        tag: Final = f"dec1-window-tag-{uuid.uuid4().hex}"
        _, rows = _two_calls(gateway, scenario, wire, metadata={"tags": [tag]})
        day, next_day = string_value(rows[0]["start_day"]), string_value(rows[1]["next_day"])
        inside: Final = _listed(gateway.request("GET", "/spend/tags", params={"start_date": day, "end_date": day}))
        after: Final = _listed(gateway.request("GET", "/spend/tags", params={"start_date": next_day}))
        assert [entry["log_count"] for entry in inside if entry["individual_request_tag"] == tag] == [2], inside
        assert tag not in {entry["individual_request_tag"] for entry in after}


def test_spend_tags_rejects_a_malformed_date_with_400(gateway: Gateway) -> None:
    """`GET /spend/tags?start_date=yesterday` is answered 400 and the body says
    `start_date must be a YYYY-MM-DD date`, instead of failing inside the SQL query with a 500."""
    response: Final = gateway.request("GET", "/spend/tags", params={"start_date": "yesterday"})
    assert response.status_code == 400, response.text
    assert "start_date must be a YYYY-MM-DD date" in response.text, response.text


def test_spend_tags_for_an_internal_user_only_counts_their_own_requests(gateway: Gateway) -> None:
    """Two internal users each tag two calls with their own fresh tag. When the first user reads `/spend/tags`
    with their key, their tag is listed with `log_count` 2 and the other user's tag is absent."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        _, mine = _internal_user_key(scenario)
        _, theirs = _internal_user_key(scenario)
        my_tag, their_tag = f"dec1-mine-{uuid.uuid4().hex}", f"dec1-theirs-{uuid.uuid4().hex}"
        _two_calls(gateway, scenario, wire, mine, metadata={"tags": [my_tag]})
        _two_calls(gateway, scenario, wire, theirs, metadata={"tags": [their_tag]})
        counts: Final = {
            entry["individual_request_tag"]: entry["log_count"]
            for entry in _listed(gateway.request("GET", "/spend/tags", key=mine))
        }
        assert counts.get(my_tag) == 2, counts
        assert their_tag not in counts, counts


def test_global_spend_tags_requires_both_dates(gateway: Gateway) -> None:
    """`GET /global/spend/tags` without `start_date` and `end_date` is answered 400 with the message
    `Please provide start_date and end_date`."""
    response: Final = gateway.request("GET", "/global/spend/tags")
    assert response.status_code == 400, response.text
    assert "Please provide start_date and end_date" in response.text, response.text


def test_global_spend_tags_filters_by_tag_and_rounds_spend(gateway: Gateway) -> None:
    """`GET /global/spend/tags` with both dates and `tags=<tag>` returns
    `{"spend_per_tag": [{"name": <tag>, "spend": 0.04, "log_count": 2}]}`.
    UNVERIFIED: the route reads the `DailyTagSpend` SQL view, assumed to exist in the integration database."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        tag: Final = f"dec1-global-tag-{uuid.uuid4().hex}"
        _, rows = _two_calls(gateway, scenario, wire, metadata={"tags": [tag]})
        report: Final = gateway.get(
            "/global/spend/tags",
            {
                "start_date": string_value(rows[0]["start_day"]),
                "end_date": string_value(rows[0]["next_day"]),
                "tags": tag,
            },
        )
        assert report == {"spend_per_tag": [{"name": tag, "spend": pytest.approx(0.04), "log_count": 2}]}, report


def test_global_all_tag_names_lists_a_tag_once_it_has_been_logged(gateway: Gateway) -> None:
    """After a tagged request has landed, `GET /global/spend/all_tag_names` returns `{"tag_names": [...]}` and the
    list contains the tag exactly once."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        tag: Final = f"dec1-names-tag-{uuid.uuid4().hex}"
        _two_calls(gateway, scenario, wire, metadata={"tags": [tag]})
        names: Final = list_value(gateway.get("/global/spend/all_tag_names")["tag_names"])
        assert [name for name in names if name == tag] == [tag], tag


def test_global_spend_end_users_reports_count_and_spend_per_customer_for_a_key(gateway: Gateway) -> None:
    """Two calls by one key name a customer. `POST /global/spend/end_users` with `api_key` set to the hashed key
    returns exactly `[{"end_user": <customer>, "total_count": 2, "total_spend": 0.04}]`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        customer: Final = new_customer(gateway, scenario)
        key, _ = _two_calls(gateway, scenario, wire, user=customer)
        found: Final = _listed(gateway.request("POST", "/global/spend/end_users", {"api_key": hashed(key)}))
        assert found == [{"end_user": customer, "total_count": 2, "total_spend": pytest.approx(0.04)}], found


def test_global_spend_provider_requires_both_dates(gateway: Gateway) -> None:
    """`GET /global/spend/provider` without dates is answered 400 with
    `{"detail": {"error": "Please provide start_date and end_date"}}`."""
    response: Final = gateway.request("GET", "/global/spend/provider")
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": {"error": "Please provide start_date and end_date"}}, response.text


def test_global_spend_provider_attributes_deployment_spend_to_its_provider(gateway: Gateway) -> None:
    """After two calls to an `openai/` deployment, `GET /global/spend/provider` for the request day returns a list
    of `{"provider", "spend"}` objects with one entry for `openai` whose spend is at least 0.04. Other tests
    share the provider, so only a lower bound is asserted."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        _, rows = _two_calls(gateway, scenario, wire)
        day: Final = string_value(rows[0]["start_day"])
        found: Final = _listed(
            gateway.request("GET", "/global/spend/provider", params={"start_date": day, "end_date": day})
        )
        openai: Final = [entry for entry in found if entry["provider"] == "openai"]
        assert len(openai) == 1 and number(openai[0]["spend"]) >= 0.04 - 1e-9, found
        assert all(set(entry) == {"provider", "spend"} for entry in found), found


def test_global_spend_teams_reports_daily_spend_under_the_team_alias(gateway: Gateway) -> None:
    """Two calls by a team key. `GET /global/spend/teams` returns `daily_spend`, `teams` and
    `total_spend_per_team`. The team's alias is in `teams`, and the entry for the request day maps the alias to
    0.04 (spend is rounded to two decimals there)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        alias: Final = f"dec1-read-team-{uuid.uuid4().hex}"
        team: Final = scenario.team(team_alias=alias)
        _, rows = _two_calls(gateway, scenario, wire, scenario.key(team_id=team))
        spend_reaches(gateway, "team", team, 0.04)
        report: Final = gateway.get("/global/spend/teams")
        assert {"daily_spend", "teams", "total_spend_per_team"} <= set(report), report
        assert alias in list_value(report["teams"]), report["teams"]
        days: Final = [object_value(entry) for entry in list_value(report["daily_spend"])]
        mine: Final = [entry[alias] for entry in days if alias in entry]
        assert mine == [pytest.approx(0.04)], (mine, rows[0]["start_day"])


def test_global_spend_models_refuses_an_internal_users_key(gateway: Gateway) -> None:
    """An internal user makes two calls and then asks `GET /global/spend/models` with that user's key. The route is
    refused with 401 "Only proxy admin" and no spend is returned.
    Pinned current behaviour: the handler has a branch that scopes the answer to an internal user's own rows
    (`litellm/proxy/spend_tracking/spend_management_endpoints.py:4256`), but route authorization never lets an
    internal user reach it, because the global spend routes are deliberately kept off the non-admin allow-lists
    (`litellm/proxy/_types.py:827`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        _, key = _internal_user_key(scenario)
        _two_calls(gateway, scenario, wire, key)
        _refused_to_internal_user(gateway.request("GET", "/global/spend/models", key=key), "/global/spend/models")


def test_global_spend_keys_refuses_an_internal_users_key(gateway: Gateway) -> None:
    """An internal user makes two calls with their key, another key makes two more, and the internal user asks
    `GET /global/spend/keys`. The route is refused with 401 "Only proxy admin" and lists no key, neither theirs
    nor the other one.
    Pinned current behaviour: the handler's internal user branch
    (`litellm/proxy/spend_tracking/spend_management_endpoints.py:4005`) is unreachable, because the global spend
    routes are deliberately kept off the non-admin allow-lists (`litellm/proxy/_types.py:827`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        _, key = _internal_user_key(scenario)
        _two_calls(gateway, scenario, wire, key)
        other, _ = _two_calls(gateway, scenario, wire)
        response: Final = gateway.request("GET", "/global/spend/keys", key=key)
        _refused_to_internal_user(response, "/global/spend/keys")
        assert hashed(key) not in response.text and hashed(other) not in response.text, response.text


def test_global_spend_keys_rejects_a_limit_below_one(gateway: Gateway) -> None:
    """`GET /global/spend/keys?limit=0` as admin is answered 422 with
    `{"detail": {"error": "Invalid limit: 0, error: Limit must be greater than 0"}}`."""
    response: Final = gateway.request("GET", "/global/spend/keys", params={"limit": "0"})
    assert response.status_code == 422, response.text
    assert response.json() == {"detail": {"error": "Invalid limit: 0, error: Limit must be greater than 0"}}


def test_global_spend_reports_a_total_and_the_proxy_budget(gateway: Gateway) -> None:
    """`GET /global/spend` returns an object with exactly the keys `spend` and `max_budget`. After this test's own
    two calls the total is at least 0.04.
    UNVERIFIED: the route sums the `MonthlyGlobalSpend` SQL view, assumed to exist in the integration database."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        _two_calls(gateway, scenario, wire)
        report: Final = gateway.get("/global/spend")
        assert set(report) == {"spend", "max_budget"}, report
        assert number(report["spend"]) >= 0.04 - 1e-9, report


def test_spend_keys_as_admin_includes_the_key_with_its_spend(gateway: Gateway) -> None:
    """After two calls have been applied, `GET /spend/keys` as admin lists the key's row (matched by hashed
    `token`) with `spend` 0.04."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        key, _ = _two_calls(gateway, scenario, wire)
        spend_reaches(gateway, "key", hashed(key), 0.04)
        mine: Final = [row for row in _listed(gateway.request("GET", "/spend/keys")) if row["token"] == hashed(key)]
        assert [number(row["spend"]) for row in mine] == [pytest.approx(0.04)], mine


def test_spend_keys_for_an_internal_user_lists_only_keys_they_own(gateway: Gateway) -> None:
    """An internal user owns one key and another user owns another. `GET /spend/keys` with the first user's key
    returns only rows whose `user_id` is the first user."""
    with gateway.scenario() as scenario:
        user, key = _internal_user_key(scenario)
        _internal_user_key(scenario)
        found: Final = _listed(gateway.request("GET", "/spend/keys", key=key))
        assert [row["token"] for row in found] == [hashed(key)], found
        assert {row["user_id"] for row in found} == {user}, found


def test_spend_users_by_user_id_returns_the_user_row_with_spend_and_no_password(gateway: Gateway) -> None:
    """After two calls by a user's key, `GET /spend/users?user_id=<id>` as admin returns a one-element list with
    that user's row, `spend` 0.04, and no `password` field."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        user: Final = scenario.user()
        _two_calls(gateway, scenario, wire, scenario.key(user_id=user))
        spend_reaches(gateway, "user", user, 0.04)
        found: Final = _listed(gateway.request("GET", "/spend/users", params={"user_id": user}))
        assert [(row["user_id"], number(row["spend"])) for row in found] == [(user, pytest.approx(0.04))], found
        assert "password" not in found[0], found


def test_spend_users_refuses_an_internal_user_asking_for_someone_else(gateway: Gateway) -> None:
    """An internal user calls `GET /spend/users?user_id=<another user>`. The answer is 403 with
    `{"detail": {"error": "Not authorized to view spend for another user."}}`, and asking without `user_id`
    returns only their own row."""
    with gateway.scenario() as scenario:
        user, key = _internal_user_key(scenario)
        other, _ = _internal_user_key(scenario)
        refused: Final = gateway.request("GET", "/spend/users", params={"user_id": other}, key=key)
        assert refused.status_code == 403, refused.text
        assert JSON_OBJECT.validate_json(refused.content) == {
            "detail": {"error": "Not authorized to view spend for another user."}
        }
        own: Final = _listed(gateway.request("GET", "/spend/users", key=key))
        assert [row["user_id"] for row in own] == [user], own
