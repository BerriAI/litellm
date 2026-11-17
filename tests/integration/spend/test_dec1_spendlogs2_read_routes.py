"""Contract: the legacy (v1) spend log read routes and the entity daily activity routes.

The V2 plan keeps the read side in Python, but every route below reads the columns the Rust writer will produce,
and `/spend/logs` v1 returns the raw rows, so any column the port renames, retypes or drops shows up here.
Read from `litellm/proxy/spend_tracking/spend_management_endpoints.py`:

1. `GET /spend/logs` (`view_spend_logs`, `:3490`). Without a full date pair it returns every matching row with
   all columns (heavy ones included), newest first, capped at 10,000 rows with `x-litellm-spend-logs-truncated:
   true` on a capped answer (`_find_spend_logs`, `:439`). With both `start_date` and `end_date` it parses them as
   `%Y-%m-%d` at UTC midnight and filters `startTime` between them inclusive, so the end day itself is excluded
   after its first instant; `summarize` (default true) returns one item per day plus zero padding up to the end
   date (`_daily_summary_item`, `:428`). A date that does not parse is a 500 whose message starts
   `/spend/logs Error`.
2. `GET /spend/logs/ui/{request_id}` returns JSON `null` for an id it cannot find.
3. `GET /spend/logs/session/ui` (`:4582`) pages one session, newest first, `page_size` 1 to 100, envelope
   `data`, `total`, `page`, `page_size`, `total_pages`, without the heavy columns.
4. `GET /global/spend/logs` (`:3893`) reads the `MonthlyGlobalSpendPerKey` view with the `api_key` exactly as
   given, so it needs the hashed token.
5. `/team`, `/tag`, `/organization` and `/customer/daily/activity` answer `SpendAnalyticsPaginatedResponse`
   (`results`, `metadata` with `total_spend`, `total_api_requests`, `page`, `total_pages`, `has_more`).

Round 1 and `test_dec1_spend_read_routes.py` cover the request id, api key and user filters of v1, the v2 and ui
list routes; these tests cover the shapes and edges left.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, list_value, object_value
from tests.integration.spend.dec1_spend_support import (
    ask,
    chat_model,
    hashed,
    new_customer,
    new_tag,
    number,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served_id,
)
from tests.integration.spend.dec1_spendlogs2_support import get_json, utc_day

pytestmark: Final = pytest.mark.shares_worker_state

HEAVY_COLUMNS: Final = ("messages", "response", "proxy_server_request")


def _v1(gateway: Gateway, **params: str) -> list[JsonValue]:
    return list(list_value(get_json(gateway, "/spend/logs", params)))


def _served_key(gateway: Gateway, scenario: Scenario, model: str, calls: int = 1, **fields: JsonValue) -> str:
    key: Final = scenario.key(models=[model], **fields)
    for _ in range(calls):
        served_id(ask(gateway, model, key))
    rows_of_key(gateway, key, calls)
    return key


def test_v1_rows_include_the_heavy_payload_columns(gateway: Gateway) -> None:
    """`GET /spend/logs?api_key=<raw key>` returns the key's row with every column, including `messages`,
    `response` and `proxy_server_request` that the v2 and ui list routes leave out, plus `request_id`,
    `call_type`, `spend` and `startTime`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _served_key(gateway, scenario, model)
        rows: Final = _v1(gateway, api_key=key)
        assert len(rows) == 1, rows
        row: Final = object_value(rows[0])
        assert set(HEAVY_COLUMNS) | {"request_id", "call_type", "spend", "startTime", "api_key"} <= set(row), row


def test_v1_rows_are_newest_first(gateway: Gateway) -> None:
    """Two calls on one key: `GET /spend/logs?api_key=<hash>` lists the second call's `request_id` first
    (`order={"startTime": "desc"}`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        first: Final = served_id(ask(gateway, model, key))
        second: Final = served_id(ask(gateway, model, key))
        rows_of_key(gateway, key, 2)
        listed: Final = [object_value(row)["request_id"] for row in _v1(gateway, api_key=hashed(key))]
        assert listed == [second, first], listed


def test_v1_metadata_and_tags_are_json_values_not_strings(gateway: Gateway) -> None:
    """The v1 row's `metadata` is a JSON object and `request_tags` a JSON array (Prisma `Json` columns), not
    strings holding JSON: a writer that stores text would change what every v1 client parses."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _served_key(gateway, scenario, model)
        row: Final = object_value(_v1(gateway, api_key=key)[0])
        assert isinstance(row["metadata"], dict), row
        assert isinstance(row["request_tags"], list), row


def test_v1_small_answer_has_no_truncation_header(gateway: Gateway) -> None:
    """A v1 answer far below the 10,000 row cap carries no `x-litellm-spend-logs-truncated` header."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _served_key(gateway, scenario, model)
        response: Final = gateway.request("GET", "/spend/logs", params={"api_key": key})
        assert response.status_code == 200, response.text
        assert "x-litellm-spend-logs-truncated" not in response.headers


def test_v1_date_range_ending_today_excludes_a_row_written_today(gateway: Gateway) -> None:
    """`start_date` yesterday and `end_date` today with `summarize=false` returns no row written today: both
    dates are UTC midnight and the filter is `startTime <= end`, so the end day is effectively exclusive.
    A port that treats `end_date` as the whole end day changes every v1 report."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _served_key(gateway, scenario, model)
        rows: Final = _v1(gateway, api_key=key, start_date=utc_day(-1), end_date=utc_day(0), summarize="false")
        assert rows == [], rows


def test_v1_date_range_ending_tomorrow_includes_a_row_written_today(gateway: Gateway) -> None:
    """`start_date` today and `end_date` tomorrow with `summarize=false` returns today's row for the key."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _served_key(gateway, scenario, model)
        rows: Final = _v1(gateway, api_key=key, start_date=utc_day(0), end_date=utc_day(1), summarize="false")
        assert len(rows) == 1, rows


def test_v1_start_date_alone_is_ignored(gateway: Gateway) -> None:
    """Only `start_date` (a day far in the future) is given: the date branch needs both, so the filter is
    dropped and the key's row is still listed."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _served_key(gateway, scenario, model)
        assert len(_v1(gateway, api_key=key, start_date="2999-01-01")) == 1


def test_v1_malformed_date_is_a_500_with_the_route_prefix(gateway: Gateway) -> None:
    """`start_date=not-a-date&end_date=2026-01-01` fails `strptime` and is answered 500 with an error message that
    starts `/spend/logs Error` (the generic `ProxyException` wrapper, `:3700`). UNVERIFIED: the message is under
    `error.message` in the response body."""
    response: Final = gateway.request(
        "GET", "/spend/logs", params={"start_date": "not-a-date", "end_date": "2026-01-01"}
    )
    assert response.status_code == 500, response.text
    assert "/spend/logs Error" in response.text, response.text


def test_v1_summary_pads_every_day_up_to_the_end_date(gateway: Gateway) -> None:
    """Summary for one key from today to today + 2: three items, today's with `spend` 0.02, then two padding
    items with `spend` 0, `users` {} and `models` {} for today + 1 and today + 2."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _served_key(gateway, scenario, model)
        items: Final = [
            object_value(item) for item in _v1(gateway, api_key=key, start_date=utc_day(0), end_date=utc_day(2))
        ]
        assert len(items) == 3, items
        assert number(items[0]["spend"]) == pytest.approx(0.02), items
        for padding in items[1:]:
            assert (padding["spend"], padding["users"], padding["models"]) == (0, {}, {}), padding
        assert [str(item["startTime"])[:10] for item in items] == [utc_day(0), utc_day(1), utc_day(2)], items


def test_v1_summary_item_names_the_hashed_key_and_the_model(gateway: Gateway) -> None:
    """Today's summary item has a top-level field named by the key's sha256 hash whose value is 0.02, and
    `models` maps the row's `model` to 0.02 (`_sum_spend_by`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        item: Final = object_value(_v1(gateway, api_key=key, start_date=utc_day(0), end_date=utc_day(1))[0])
        assert number(item[hashed(key)]) == pytest.approx(0.02), item
        assert number(object_value(item["models"])[str(row["model"])]) == pytest.approx(0.02), item


def test_ui_detail_of_an_unknown_request_id_is_null(gateway: Gateway) -> None:
    """`GET /spend/logs/ui/<random id>` as the admin answers 200 with the JSON body `null`: no logger and no row
    had it. A port answering 404 breaks the dashboard's empty state."""
    response: Final = gateway.request("GET", f"/spend/logs/ui/dec1-sl2-missing-{uuid.uuid4().hex}")
    assert response.status_code == 200, response.text
    assert response.json() is None


def _session_calls(gateway: Gateway, scenario: Scenario, model: str, count: int) -> tuple[str, list[str]]:
    session: Final = f"dec1-sl2-session-{uuid.uuid4().hex}"
    key: Final = scenario.key(models=[model])
    ids: Final = [served_id(ask(gateway, model, key, headers={"x-litellm-session-id": session})) for _ in range(count)]
    rows_of_key(gateway, key, count)
    return session, ids


def test_session_route_returns_the_paging_envelope(gateway: Gateway) -> None:
    """Three calls share `x-litellm-session-id`. `GET /spend/logs/session/ui?session_id=<s>&page_size=2` answers
    `total` 3, `page` 1, `page_size` 2, `total_pages` 2 and two rows in `data`, newest first."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        session, ids = _session_calls(gateway, scenario, model, 3)
        body: Final = object_value(
            get_json(gateway, "/spend/logs/session/ui", {"session_id": session, "page_size": "2"})
        )
        assert (body["total"], body["page"], body["page_size"], body["total_pages"]) == (3, 1, 2, 2), body
        listed: Final = [object_value(row)["request_id"] for row in list_value(body["data"])]
        assert listed == [ids[2], ids[1]], listed


def test_session_route_second_page_holds_the_oldest_call(gateway: Gateway) -> None:
    """Same session, `page=2&page_size=2`: `data` holds only the first call's row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        session, ids = _session_calls(gateway, scenario, model, 3)
        body: Final = object_value(
            get_json(gateway, "/spend/logs/session/ui", {"session_id": session, "page": "2", "page_size": "2"})
        )
        assert [object_value(row)["request_id"] for row in list_value(body["data"])] == [ids[0]], body


def test_session_route_rows_leave_out_the_heavy_columns(gateway: Gateway) -> None:
    """Rows returned by the session route have none of `messages`, `response`, `proxy_server_request` (the raw
    SELECT lists the light columns only) but do have `session_id` and `metadata`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        session, _ = _session_calls(gateway, scenario, model, 1)
        body: Final = object_value(get_json(gateway, "/spend/logs/session/ui", {"session_id": session}))
        row: Final = object_value(list_value(body["data"])[0])
        assert not set(HEAVY_COLUMNS) & set(row), row
        assert row["session_id"] == session and isinstance(row["metadata"], dict), row


@pytest.mark.parametrize(
    "params",
    [{"session_id": "dec1-sl2-any", "page_size": "101"}, {"session_id": "dec1-sl2-any", "page": "0"}, {}],
    ids=["page_size_over_100", "page_zero", "no_session_id"],
)
def test_session_route_rejects_bad_paging_with_422(gateway: Gateway, params: dict[str, str]) -> None:
    """`page_size` above 100, `page` below 1 and a missing `session_id` are refused 422 by FastAPI query
    validation (`ge=1`, `le=100`, required)."""
    response: Final = gateway.request("GET", "/spend/logs/session/ui", params=params)
    assert response.status_code == 422, response.text


def test_global_spend_logs_needs_the_hashed_key(gateway: Gateway) -> None:
    """`GET /global/spend/logs?api_key=<raw key>` answers `[]` while `?api_key=<sha256>` answers one entry with
    `spend` 0.02: the route passes the value straight into the `MonthlyGlobalSpendPerKey` view.
    UNVERIFIED: the integration database has that view (created by `litellm/scripts/db/create_views.py`) and
    Prometheus is not connected."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _served_key(gateway, scenario, model)
        assert get_json(gateway, "/global/spend/logs", {"api_key": key}) == []
        entries: Final = list_value(get_json(gateway, "/global/spend/logs", {"api_key": hashed(key)}))
        assert len(entries) == 1, entries
        assert number(object_value(entries[0])["spend"]) == pytest.approx(0.02), entries


def _activity(gateway: Gateway, path: str, params: dict[str, str]) -> dict[str, JsonValue]:
    def read() -> dict[str, JsonValue]:
        return object_value(get_json(gateway, path, {**params, "start_date": utc_day(0), "end_date": utc_day(0)}))

    return eventually(read, lambda body: number(object_value(body["metadata"])["total_api_requests"]) >= 1, seconds=70)


def _assert_one_day_one_call(body: dict[str, JsonValue]) -> None:
    metadata: Final = object_value(body["metadata"])
    assert number(metadata["total_spend"]) == pytest.approx(0.02), body
    assert (metadata["total_api_requests"], metadata["page"], metadata["total_pages"], metadata["has_more"]) == (
        1,
        1,
        1,
        False,
    ), body
    results: Final = list_value(body["results"])
    assert len(results) == 1 and str(object_value(results[0])["date"]) == utc_day(0), body
    assert number(object_value(object_value(results[0])["metrics"])["spend"]) == pytest.approx(0.02), body


def test_team_daily_activity_reports_the_teams_one_call(gateway: Gateway) -> None:
    """One call on a team key. `GET /team/daily/activity?team_ids=<team>` for today answers one result dated today
    with `metrics.spend` 0.02, and `metadata` with `total_spend` 0.02, `total_api_requests` 1, `page` 1,
    `total_pages` 1, `has_more` false."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team()
        _served_key(gateway, scenario, model, team_id=team)
        _assert_one_day_one_call(_activity(gateway, "/team/daily/activity", {"team_ids": team}))


def test_tag_daily_activity_reports_the_tags_one_call(gateway: Gateway) -> None:
    """One call tagged with a fresh tag through `metadata.tags`. `GET /tag/daily/activity?tags=<tag>` answers the
    same one-day, one-call envelope."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        tag: Final = new_tag(gateway, scenario)
        key: Final = scenario.key(models=[model])
        served_id(ask(gateway, model, key, metadata={"tags": [tag]}))
        _assert_one_day_one_call(_activity(gateway, "/tag/daily/activity", {"tags": tag}))


def test_organization_daily_activity_reports_the_organizations_one_call(gateway: Gateway) -> None:
    """One call on a key of a team inside a fresh organization. `GET /organization/daily/activity?organization_ids=
    <org>` answers the one-day, one-call envelope. UNVERIFIED: a team created with `organization_id` stamps the
    org on the row (`organization_id` column) without extra model grants."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        org: Final = scenario.organization()
        team: Final = scenario.team(organization_id=org)
        _served_key(gateway, scenario, model, team_id=team)
        _assert_one_day_one_call(_activity(gateway, "/organization/daily/activity", {"organization_ids": org}))


def test_customer_daily_activity_reports_the_customers_one_call(gateway: Gateway) -> None:
    """One call with `user: <customer>` for a customer created with `/customer/new`.
    `GET /customer/daily/activity?end_user_ids=<customer>` answers the one-day, one-call envelope."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        customer: Final = new_customer(gateway, scenario)
        key: Final = scenario.key(models=[model])
        served_id(ask(gateway, model, key, user=customer))
        _assert_one_day_one_call(_activity(gateway, "/customer/daily/activity", {"end_user_ids": customer}))
