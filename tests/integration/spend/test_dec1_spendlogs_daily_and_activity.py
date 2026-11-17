"""Contract: what the daily aggregate tables and `/user/daily/activity` derive from each request row.

In Spend Logs V2 phase 1 the Rust port stops at the committed request row and its delivery; the Python ledger
worker keeps projecting the daily tables (`litellm/accounting/projection.py:141-231`). That makes the request
row's columns the only input: `api_requests`, `successful_requests` and `failed_requests` come from
`metadata.status`; `timed_requests` and `total_response_time_ms` only from successful rows with a
`request_duration_ms`; the conflict key is `(entity, date, api_key, model, custom_llm_provider,
mcp_namespaced_tool_name, endpoint)` with NULLs normalized to `''`
(`litellm/proxy/db/daily_spend_bulk_upsert.py:100`). This file pins the counters the earlier daily contract file
(`test_dec1_spend_daily_aggregates.py`) did not, the response cache row id, and the totals the
`/user/daily/activity` route reports from those tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, list_value, object_value, string_value
from tests.integration.spend.dec1_spend_support import (
    LANDING_SECONDS,
    ask,
    chat_model,
    daily_row,
    daily_rows,
    number,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served,
    served_id,
)
from tests.integration.spend.dec1_spendlogs_support import failing_upstream


def test_failed_request_adds_no_response_time_to_the_daily_row(gateway: Gateway) -> None:
    """A key makes one served call and one call that the provider fails with 500. Its daily user row (both calls
    use one deployment model) has `api_requests` 2, `failed_requests` 1, `timed_requests` 1 and
    `total_response_time_ms` equal to the served row's `request_duration_ms` only."""
    with scripted_upstream() as ok, failing_upstream(500) as broken, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, ok)
        bad: Final = chat_model(scenario, broken)
        key: Final = scenario.key(models=[model, bad])
        served_row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert ask(gateway, bad, key).status_code == 500
        daily: Final = eventually(
            lambda: daily_row(gateway, "LiteLLM_DailyUserSpend", key),
            lambda found: found["api_requests"] == 2,
            seconds=LANDING_SECONDS,
        )
        assert (daily["failed_requests"], daily["timed_requests"]) == (1, 1), daily
        assert number(daily["total_response_time_ms"]) == number(served_row["request_duration_ms"]), daily


def test_daily_row_of_a_call_without_tools_has_an_empty_tool_name(gateway: Gateway) -> None:
    """A plain chat call's daily user row has `mcp_namespaced_tool_name` `''` (not NULL) and `custom_llm_provider`
    `openai`: NULLs in the conflict key would insert a new row on every flush."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served_id(ask(gateway, model, key))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert daily["mcp_namespaced_tool_name"] == "", daily
        assert daily["custom_llm_provider"] == "openai", daily


def test_daily_end_user_row_is_keyed_by_the_customer_and_the_key(gateway: Gateway) -> None:
    """Two keys call on behalf of the same customer (`user` in the body). `LiteLLM_DailyEndUserSpend` has one row
    per key for that `end_user_id`, each with `api_requests` 1 and `spend` 0.02, because `api_key` is part of
    the conflict key."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        first: Final = scenario.key(models=[model])
        second: Final = scenario.key(models=[model])
        customer: Final = f"dec1-sl-eu-{uuid.uuid4().hex}"
        served_id(ask(gateway, model, first, user=customer))
        served_id(ask(gateway, model, second, user=customer))
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT api_key, api_requests, spend FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id = %s',
                (customer,),
            ),
            lambda found: len(found) == 2,
            seconds=LANDING_SECONDS,
        )
        assert {row["api_requests"] for row in rows} == {1}, rows
        assert [number(row["spend"]) for row in rows] == pytest.approx([0.02, 0.02]), rows


def test_daily_tag_row_carries_the_request_id_of_its_latest_request(gateway: Gateway) -> None:
    """Only `LiteLLM_DailyTagSpend` carries a `request_id` (`carries_request_id`,
    daily_spend_bulk_upsert.py:57). After one tagged call its tag row's `request_id` is that call's id."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tag: Final = f"dec1-sl-tag-{uuid.uuid4().hex}"
        identity: Final = served_id(ask(gateway, model, key, metadata={"tags": [tag]}))
        rows: Final = eventually(
            lambda: gateway.rows('SELECT request_id FROM "LiteLLM_DailyTagSpend" WHERE tag = %s', (tag,)),
            lambda found: len(found) == 1,
            seconds=LANDING_SECONDS,
        )
        assert rows[0]["request_id"] == identity, rows


def test_daily_rows_keep_the_model_group_of_the_request(gateway: Gateway) -> None:
    """The daily user row stores the public `model_group` next to the deployment `model`
    `openai/gpt-4o-mini`; usage pages group by the public name."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served_id(ask(gateway, model, key))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert daily["model_group"] == model, daily
        assert daily["model"] == "openai/gpt-4o-mini", daily


def test_cache_hit_rows_get_distinct_suffixed_request_ids(gateway: Gateway) -> None:
    """The same prompt is sent three times. The first row's `request_id` is the response id; the two cache hit
    rows have ids that start with `<that id>_cache_hit` and differ from each other, because `request_id` is the
    primary key and the suffix carries a timestamp (spend_tracking_utils.py:1027)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"dec1 repeat {uuid.uuid4().hex}"
        first: Final = served_id(ask(gateway, model, key, text=prompt))
        served(ask(gateway, model, key, text=prompt))
        served(ask(gateway, model, key, text=prompt))
        ids: Final = [string_value(row["request_id"]) for row in rows_of_key(gateway, key, 3)]
        hits: Final = [identity for identity in ids if identity != first]
        assert len(hits) == 2 and len(set(hits)) == 2, ids
        assert all(identity.startswith(f"{first}_cache_hit") for identity in hits), ids


def test_cache_hit_row_keeps_the_cache_key_reported_to_the_caller(gateway: Gateway) -> None:
    """The cache hit response carries `x-litellm-cache-key`; the hit row's `cache_key` column equals it, so a
    caller can tie a free answer to the entry that served it.
    UNVERIFIED: that the header is set on the hit response (common_request_processing.py:2906 sets it from the
    hidden params)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"dec1 repeat {uuid.uuid4().hex}"
        first: Final = served_id(ask(gateway, model, key, text=prompt))
        hit: Final = ask(gateway, model, key, text=prompt)
        rows: Final = rows_of_key(gateway, key, 2)
        hit_row: Final = next(row for row in rows if row["request_id"] != first)
        assert hit_row["cache_key"] == hit.headers["x-litellm-cache-key"], (dict(hit.headers), hit_row)


def _activity(gateway: Gateway, user: str) -> dict[str, object]:
    today: Final = datetime.now(timezone.utc)
    return gateway.get(
        "/user/daily/activity",
        {
            "user_id": user,
            "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        },
    )


def test_user_daily_activity_totals_match_the_request_rows(gateway: Gateway) -> None:
    """A user's key makes two served calls (0.02 each). `/user/daily/activity?user_id=<user>` reports
    `metadata.total_spend` 0.04, `total_api_requests` 2, `total_successful_requests` 2, `total_failed_requests`
    0, `total_prompt_tokens` 20 and `total_completion_tokens` 10."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(models=[model])
        key: Final = scenario.key(user_id=user, models=[model])
        served_id(ask(gateway, model, key))
        served_id(ask(gateway, model, key))
        daily_rows(gateway, "LiteLLM_DailyUserSpend", key, 1)
        body: Final = eventually(
            lambda: _activity(gateway, user),
            lambda found: object_value(found["metadata"])["total_api_requests"] == 2,
            seconds=LANDING_SECONDS,
        )
        totals: Final = object_value(body["metadata"])
        assert number(totals["total_spend"]) == pytest.approx(0.04), totals
        assert (totals["total_successful_requests"], totals["total_failed_requests"]) == (2, 0), totals
        assert (totals["total_prompt_tokens"], totals["total_completion_tokens"]) == (20, 10), totals


def test_user_daily_activity_breaks_spend_down_by_model_group_and_endpoint(gateway: Gateway) -> None:
    """The same activity response has one result day whose `breakdown.model_groups` has the public model name
    with `metrics.spend` 0.02 after one call, and whose `breakdown.endpoints` has `/chat/completions`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(models=[model])
        key: Final = scenario.key(user_id=user, models=[model])
        served_id(ask(gateway, model, key))
        daily_rows(gateway, "LiteLLM_DailyUserSpend", key, 1)
        body: Final = eventually(
            lambda: _activity(gateway, user),
            lambda found: len(list_value(found["results"])) == 1,
            seconds=LANDING_SECONDS,
        )
        breakdown: Final = object_value(object_value(list_value(body["results"])[0])["breakdown"])
        group: Final = object_value(object_value(breakdown["model_groups"])[model])
        assert number(object_value(group["metrics"])["spend"]) == pytest.approx(0.02), group
        assert "/chat/completions" in object_value(breakdown["endpoints"]), breakdown
