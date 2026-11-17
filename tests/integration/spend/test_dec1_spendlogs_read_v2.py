"""Contract: the paginated spend log list routes `/spend/logs/v2` (public) and `/spend/logs/ui` (dashboard).

Both are served by `ui_view_spend_logs` (`litellm/proxy/spend_tracking/spend_management_endpoints.py:2569`). They
read `LiteLLM_SpendLogs` with raw SQL, so whatever the Rust writer puts in each column is exactly what these
routes filter and sort on: `status`, `cache_hit` (`'True'`), `spend`, `model_group`, `end_user`,
`metadata->>'user_api_key_alias'` and `metadata->'error_information'->>'error_code'`. The response is
`{data, total, page, page_size, total_pages, total_is_capped}`, and list rows never carry the heavy `messages`,
`response` or `proxy_server_request` columns (`_SPEND_LOG_LIST_COLUMNS`, :114). Validation errors are 400 with
exact messages. Rewriting either the writer or these routes must keep all of it.

Each test makes its own key and reads only that key's rows, so other traffic on the shared database cannot
interfere.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, list_value, object_value
from tests.integration.spend.dec1_spend_support import (
    ask,
    chat_model,
    hashed,
    number,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served_id,
)
from tests.integration.spend.dec1_spendlogs_support import (
    error_text,
    failing_upstream,
    list_body,
    list_logs,
    today_window,
)

V2: Final = "/spend/logs/v2"
UI: Final = "/spend/logs/ui"


def _page(gateway: Gateway, path: str, key: str, **params: str) -> dict[str, object]:
    return list_body(list_logs(gateway, path, {**today_window(), "api_key": hashed(key), **params}))


def _ids(page: dict[str, object]) -> list[str]:
    return [str(object_value(row)["request_id"]) for row in list_value(page["data"])]


@pytest.mark.parametrize("path", [V2, UI])
def test_list_route_returns_the_pagination_envelope(gateway: Gateway, path: str) -> None:
    """One key makes three calls. `GET <path>?api_key=<hash>&page_size=2` answers 200 with `total` 3,
    `page` 1, `page_size` 2, `total_pages` 2, `total_is_capped` false and two rows in `data`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        for _ in range(3):
            served_id(ask(gateway, model, key))
        rows_of_key(gateway, key, 3)
        page: Final = _page(gateway, path, key, page_size="2")
        assert (page["total"], page["page"], page["page_size"], page["total_pages"]) == (3, 1, 2, 2), page
        assert page["total_is_capped"] is False, page
        assert len(list_value(page["data"])) == 2, page


def test_list_rows_leave_out_the_heavy_payload_columns(gateway: Gateway) -> None:
    """A row returned by `/spend/logs/v2` has `request_id`, `spend`, `model_group` and `request_duration_ms` but
    none of `messages`, `response` or `proxy_server_request`; those are only served by
    `/spend/logs/ui/{request_id}`. A port that adds them back makes every list page megabytes large."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served_id(ask(gateway, model, key))
        rows_of_key(gateway, key, 1)
        row: Final = object_value(list_value(_page(gateway, V2, key)["data"])[0])
        assert {"request_id", "spend", "model_group", "request_duration_ms"} <= set(row), row
        assert not {"messages", "response", "proxy_server_request"} & set(row), row


def test_list_rows_are_newest_first_by_default(gateway: Gateway) -> None:
    """Three calls in sequence: `/spend/logs/v2` lists them in reverse call order (`sort_by` defaults to
    `startTime`, `sort_order` to `desc`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        ids: Final = [served_id(ask(gateway, model, key)) for _ in range(3)]
        rows_of_key(gateway, key, 3)
        assert _ids(_page(gateway, V2, key)) == list(reversed(ids))


def test_list_sort_order_asc_reverses_the_order(gateway: Gateway) -> None:
    """With `sort_order=asc` the same three rows come back in call order."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        ids: Final = [served_id(ask(gateway, model, key)) for _ in range(3)]
        rows_of_key(gateway, key, 3)
        assert _ids(_page(gateway, V2, key, sort_order="asc")) == ids


def test_list_rejects_an_unknown_sort_field_with_400(gateway: Gateway) -> None:
    """`sort_by=user` is refused with 400 and the message `Invalid sort_by: user. Must be one of: endTime, model,
    request_duration_ms, spend, startTime, total_tokens, ttft_ms` (sorted list, :2733)."""
    response: Final = list_logs(gateway, V2, {**today_window(), "sort_by": "user"})
    assert response.status_code == 400, response.text
    assert (
        "Invalid sort_by: user. Must be one of: endTime, model, request_duration_ms, spend, startTime, "
        "total_tokens, ttft_ms"
    ) in error_text(response)


def test_list_rejects_an_unknown_sort_order_with_400(gateway: Gateway) -> None:
    """`sort_order=up` is refused with 400 and `Invalid sort_order: up. Must be one of: asc, desc`."""
    response: Final = list_logs(gateway, V2, {**today_window(), "sort_order": "up"})
    assert response.status_code == 400, response.text
    assert "Invalid sort_order: up. Must be one of: asc, desc" in error_text(response)


def test_list_rejects_an_unknown_cache_filter_with_400(gateway: Gateway) -> None:
    """`cache_hit_filter=maybe` is refused with 400 and `Invalid cache_hit_filter: maybe. Must be one of: hit,
    miss`."""
    response: Final = list_logs(gateway, V2, {**today_window(), "cache_hit_filter": "maybe"})
    assert response.status_code == 400, response.text
    assert "Invalid cache_hit_filter: maybe. Must be one of: hit, miss" in error_text(response)


def test_list_rejects_an_unknown_span_type_with_400(gateway: Gateway) -> None:
    """`span_type=tool` is refused with 400 and `Invalid span_type: tool. Must be one of: llm, agent, mcp,
    batch`."""
    response: Final = list_logs(gateway, V2, {**today_window(), "span_type": "tool"})
    assert response.status_code == 400, response.text
    assert "Invalid span_type: tool. Must be one of: llm, agent, mcp, batch" in error_text(response)


def test_v2_requires_a_date_window_even_with_a_request_id(gateway: Gateway) -> None:
    """`/spend/logs/v2?request_id=x` without dates is refused with 400 `Start date and end date are required`;
    only the UI route drops the window for a request id lookup (:2764)."""
    response: Final = list_logs(gateway, V2, {"request_id": f"dec1-{uuid.uuid4().hex}"})
    assert response.status_code == 400, response.text
    assert "Start date and end date are required" in error_text(response)


def test_ui_finds_a_row_by_request_id_without_a_date_window(gateway: Gateway) -> None:
    """`/spend/logs/ui?request_id=<id>` with no dates returns exactly that row: the id lookup ignores the window
    because `request_id` is the primary key (LIT-3981)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        identity: Final = served_id(ask(gateway, model, key))
        row_by_id(gateway, identity)
        page: Final = list_body(list_logs(gateway, UI, {"request_id": identity}))
        assert _ids(page) == [identity], page


def test_v2_accepts_plain_dates_while_ui_requires_a_time(gateway: Gateway) -> None:
    """`start_date=2026-01-01&end_date=2026-01-02` is accepted by `/spend/logs/v2` (200) but refused by
    `/spend/logs/ui` with 400 `Invalid date format: 2026-01-01. Expected: 'YYYY-MM-DD HH:MM:SS'`."""
    params: Final = {"start_date": "2026-01-01", "end_date": "2026-01-02"}
    assert list_logs(gateway, V2, params).status_code == 200
    refused: Final = list_logs(gateway, UI, params)
    assert refused.status_code == 400, refused.text
    assert "Invalid date format: 2026-01-01. Expected: 'YYYY-MM-DD HH:MM:SS'" in error_text(refused)


def test_status_filter_failure_keeps_only_failure_rows(gateway: Gateway) -> None:
    """A key has one served call and one provider failure. `status_filter=failure` lists only the failure row,
    and `status_filter=success` only the served one: the filter compares the `status` column."""
    with scripted_upstream() as ok, failing_upstream(500) as broken, gateway.scenario() as scenario:
        good: Final = chat_model(scenario, ok)
        bad: Final = chat_model(scenario, broken)
        key: Final = scenario.key(models=[good, bad])
        served_identity: Final = served_id(ask(gateway, good, key))
        failed: Final = ask(gateway, bad, key)
        assert failed.status_code == 500, failed.text
        rows_of_key(gateway, key, 2)
        assert _ids(_page(gateway, V2, key, status_filter="failure")) == [failed.headers["x-litellm-call-id"]]
        assert _ids(_page(gateway, V2, key, status_filter="success")) == [served_identity]


def test_error_code_filter_matches_the_stored_error_information(gateway: Gateway) -> None:
    """The failure row of a provider 500 is found with `error_code=500` and not with `error_code=429`: the
    filter reads `metadata->'error_information'->>'error_code'`, which must stay a string."""
    with failing_upstream(500) as broken, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, broken)
        key: Final = scenario.key(models=[model])
        failed: Final = ask(gateway, model, key)
        rows_of_key(gateway, key, 1)
        assert _ids(_page(gateway, V2, key, error_code="500")) == [failed.headers["x-litellm-call-id"]]
        assert _ids(_page(gateway, V2, key, error_code="429")) == []


def test_error_message_filter_is_a_substring_match(gateway: Gateway) -> None:
    """The provider fails with message `dec1 spendlogs upstream failure`. `error_message=spendlogs upstream`
    finds the row; `error_message=nothing-like-this` finds none."""
    with failing_upstream(500) as broken, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, broken)
        key: Final = scenario.key(models=[model])
        ask(gateway, model, key)
        rows_of_key(gateway, key, 1)
        assert len(_ids(_page(gateway, V2, key, error_message="spendlogs upstream"))) == 1
        assert _ids(_page(gateway, V2, key, error_message="nothing-like-this")) == []


def test_key_alias_filter_is_a_substring_match_on_the_stored_alias(gateway: Gateway) -> None:
    """A key with alias `dec1-sl-<hex>-alias` makes one call. `key_alias=<hex>` (no `api_key` filter) finds that
    row, through `metadata->>'user_api_key_alias' LIKE %...%`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        marker: Final = uuid.uuid4().hex
        key: Final = scenario.key(models=[model], key_alias=f"dec1-sl-{marker}-alias")
        identity: Final = served_id(ask(gateway, model, key))
        rows_of_key(gateway, key, 1)
        page: Final = list_body(list_logs(gateway, V2, {**today_window(), "key_alias": marker}))
        assert _ids(page) == [identity], page


def test_cache_hit_filter_separates_hits_from_misses(gateway: Gateway) -> None:
    """The same prompt is sent twice, so the second call is a response cache hit. `cache_hit_filter=hit` lists
    only the hit row (whose `cache_hit` column is `True`), `cache_hit_filter=miss` only the first one."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"dec1 cache filter {uuid.uuid4().hex}"
        first: Final = served_id(ask(gateway, model, key, text=prompt))
        served_id(ask(gateway, model, key, text=prompt))
        rows_of_key(gateway, key, 2)
        hits: Final = _ids(_page(gateway, V2, key, cache_hit_filter="hit"))
        assert len(hits) == 1 and hits[0].startswith(f"{first}_cache_hit"), hits
        assert _ids(_page(gateway, V2, key, cache_hit_filter="miss")) == [first]


def test_min_spend_filter_hides_free_rows(gateway: Gateway) -> None:
    """After one billed call (0.02) and its free cache hit, `min_spend=0.01` lists only the billed row and
    `max_spend=0` only the free one."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"dec1 spend filter {uuid.uuid4().hex}"
        billed: Final = served_id(ask(gateway, model, key, text=prompt))
        served_id(ask(gateway, model, key, text=prompt))
        rows_of_key(gateway, key, 2)
        assert _ids(_page(gateway, V2, key, min_spend="0.01")) == [billed]
        free: Final = _ids(_page(gateway, V2, key, max_spend="0"))
        assert len(free) == 1 and free[0] != billed, free


def test_end_user_filter_lists_the_rows_of_that_customer(gateway: Gateway) -> None:
    """Two calls on one key, one naming `user: <customer>` in the body. `end_user=<customer>` lists exactly that
    one row, through the `end_user` column."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        customer: Final = f"dec1-sl-customer-{uuid.uuid4().hex}"
        named: Final = served_id(ask(gateway, model, key, user=customer))
        served_id(ask(gateway, model, key))
        rows_of_key(gateway, key, 2)
        assert _ids(_page(gateway, V2, key, end_user=customer)) == [named]


def test_sort_by_spend_puts_the_most_expensive_row_first(gateway: Gateway) -> None:
    """A key calls a 0.02 deployment and a 0.04 deployment (prices doubled). `sort_by=spend` lists the 0.04 row
    first and the returned `spend` values are 0.04 then 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        cheap: Final = chat_model(scenario, wire)
        dear: Final = chat_model(scenario, wire, input_cost_per_token=0.002, output_cost_per_token=0.004)
        key: Final = scenario.key(models=[cheap, dear])
        served_id(ask(gateway, cheap, key))
        served_id(ask(gateway, dear, key))
        rows_of_key(gateway, key, 2)
        rows: Final = list_value(_page(gateway, V2, key, sort_by="spend")["data"])
        spends: Final = [number(object_value(row)["spend"]) for row in rows]
        assert spends == pytest.approx([0.04, 0.02]), spends


def test_detail_route_returns_the_stored_request_for_a_request_id(gateway: Gateway) -> None:
    """`GET /spend/logs/ui/<request_id>` as admin answers 200 with a JSON object for a row that exists, and JSON
    null for an id that was never written.
    UNVERIFIED: the payload's field names (`_resolve_request_response_payload`, :3476); only presence is pinned."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        identity: Final = served_id(ask(gateway, model, key))
        row_by_id(gateway, identity)
        found: Final = gateway.request("GET", f"/spend/logs/ui/{identity}")
        missing: Final = gateway.request("GET", f"/spend/logs/ui/dec1-missing-{uuid.uuid4().hex}")
        assert found.status_code == 200 and isinstance(found.json(), dict), found.text
        assert missing.status_code == 200 and missing.json() is None, missing.text


def test_list_route_requires_authentication(gateway: Gateway) -> None:
    """`/spend/logs/v2` without an Authorization header is refused with 401 and never lists rows."""
    response: Final = gateway.client.get(V2, params=today_window())
    assert response.status_code == 401, response.text
