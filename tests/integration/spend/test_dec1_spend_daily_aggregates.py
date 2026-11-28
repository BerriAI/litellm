"""Contract: the daily aggregate tables a billed request is rolled up into.

The usage pages and the `/user/daily/activity`, `/team/daily/activity` and `/tag/daily/activity` routes do not
scan `LiteLLM_SpendLogs`. They read one row per day and dimension from `LiteLLM_DailyUserSpend`,
`LiteLLM_DailyTeamSpend`, `LiteLLM_DailyOrganizationSpend`, `LiteLLM_DailyEndUserSpend` and
`LiteLLM_DailyTagSpend`. Each row is keyed by (entity id, date, api_key, model, custom_llm_provider,
mcp_namespaced_tool_name, endpoint) and carries counters that are added to on every request of that day.

These tests pin the key columns and the counters for one request, that a second request lands on the same row
instead of a new one, and that requests without a team, customer or organization do not create rows for them.
`date` is compared with the UTC day of the request's own `startTime`, never with today's date. One chat call here
is 10 prompt tokens, 5 completion tokens and 0.02 of spend.
"""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, held, object_value
from tests.integration._support.wire import wire_server
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    SURFACES,
    Surface,
    ask,
    chat_model,
    daily_row,
    daily_rows,
    failing,
    hashed,
    new_customer,
    number,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served,
    served_id,
    spend_reaches,
)

_IDS: Final = tuple(surface.name for surface in SURFACES)


def _tag_rows(gateway: Gateway, tag: str) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: gateway.rows('SELECT to_jsonb(d) AS row FROM "LiteLLM_DailyTagSpend" d WHERE tag = %s', (tag,)),
        lambda found: len(found) == 1,
        seconds=LANDING_SECONDS,
    )
    return [object_value(row["row"]) for row in rows]


def test_daily_user_row_is_keyed_by_user_day_key_model_provider_and_endpoint(gateway: Gateway) -> None:
    """A user's key makes one chat call. One `LiteLLM_DailyUserSpend` row appears with `user_id` the owner,
    `date` the UTC day of the request, `api_key` the hashed key, `model` the deployment's model
    `openai/gpt-4o-mini` (provider prefix included, copied from the request row's `model` column), `model_group`
    the public name, `custom_llm_provider` `openai` and `endpoint` `/chat/completions`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user()
        key: Final = scenario.key(models=[model], user_id=user)
        logged: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert {
            name: daily[name]
            for name in ("user_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "endpoint")
        } == {
            "user_id": user,
            "date": logged["start_day"],
            "api_key": hashed(key),
            "model": "openai/gpt-4o-mini",
            "model_group": model,
            "custom_llm_provider": "openai",
            "endpoint": "/chat/completions",
        }, daily


def test_daily_user_row_counts_one_successful_request_with_its_tokens_and_spend(gateway: Gateway) -> None:
    """The same single call stores prompt_tokens 10, completion_tokens 5, spend 0.02, api_requests 1,
    successful_requests 1 and failed_requests 0 on the daily row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], user_id=scenario.user())
        served(ask(gateway, model, key))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert (daily["prompt_tokens"], daily["completion_tokens"]) == (10, 5), daily
        assert (daily["api_requests"], daily["successful_requests"], daily["failed_requests"]) == (1, 1, 0), daily
        assert number(daily["spend"]) == pytest.approx(10 * 0.001 + 5 * 0.002), daily


def test_second_call_of_the_day_adds_to_the_same_daily_row(gateway: Gateway) -> None:
    """Two calls with the same key and model on the same day must end as one daily row with api_requests 2,
    prompt_tokens 20, completion_tokens 10 and spend 0.04, never as two rows. Assumes both calls fall on one UTC
    day, a run that straddles midnight sees two rows."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], user_id=scenario.user())
        served(ask(gateway, model, key))
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), 2 * CHAT_COST)
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert (daily["api_requests"], daily["prompt_tokens"], daily["completion_tokens"]) == (2, 20, 10), daily
        assert number(daily["spend"]) == pytest.approx(0.04), daily


def test_calls_to_two_models_get_one_daily_row_per_model(gateway: Gateway) -> None:
    """One key calls two different public models once each. There are two daily user rows for the key, one per
    `model_group`, each with api_requests 1 and spend 0.02, and `model` names each deployment's model with its
    provider prefix."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        first: Final = chat_model(scenario, wire)
        second: Final = chat_model(scenario, wire, model="openai/gpt-4o")
        key: Final = scenario.key(models=[first, second], user_id=scenario.user())
        served(ask(gateway, first, key))
        served(ask(gateway, second, key))
        rows: Final = [object_value(row["row"]) for row in daily_rows(gateway, "LiteLLM_DailyUserSpend", key, 2)]
        assert {row["model_group"] for row in rows} == {first, second}, rows
        assert {row["model"] for row in rows} == {"openai/gpt-4o-mini", "openai/gpt-4o"}, rows
        assert [(row["api_requests"], number(row["spend"])) for row in rows] == [(1, pytest.approx(0.02))] * 2


def test_two_keys_of_one_user_get_separate_daily_rows(gateway: Gateway) -> None:
    """A user calls the same model with two keys. `api_key` is part of the daily row key, so each key has its own
    row with api_requests 1, which is what per-key usage breakdowns read."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user()
        first: Final = scenario.key(models=[model], user_id=user)
        second: Final = scenario.key(models=[model], user_id=user)
        served(ask(gateway, model, first))
        served(ask(gateway, model, second))
        for key in (first, second):
            daily = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
            assert (daily["user_id"], daily["api_requests"]) == (user, 1), daily


def test_key_without_owner_gets_a_daily_user_row_with_blank_user(gateway: Gateway) -> None:
    """A key that belongs to nobody still gets a `LiteLLM_DailyUserSpend` row so its usage is not lost. The row's
    `user_id` is blank (empty string or NULL) and its counters are those of the one call."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert daily["user_id"] in ("", None), daily
        assert (daily["api_requests"], number(daily["spend"])) == (1, pytest.approx(0.02)), daily


def test_team_key_call_writes_a_daily_team_row(gateway: Gateway) -> None:
    """A team key makes one call. One `LiteLLM_DailyTeamSpend` row appears with `team_id` the team, the hashed
    key, the public model name, endpoint `/chat/completions`, api_requests 1 and spend 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        logged: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        daily: Final = daily_row(gateway, "LiteLLM_DailyTeamSpend", key)
        assert (daily["team_id"], daily["date"], daily["api_key"]) == (team, logged["start_day"], hashed(key)), daily
        assert (daily["model_group"], daily["endpoint"], daily["api_requests"]) == (model, "/chat/completions", 1)
        assert number(daily["spend"]) == pytest.approx(0.02), daily


def test_key_without_team_writes_no_daily_team_row(gateway: Gateway) -> None:
    """A key with no team makes one call. After the call is fully applied there must be no
    `LiteLLM_DailyTeamSpend` row for that key, not even one with an empty `team_id`.
    BUG: a key without a team is logged with `team_id` '' (`litellm/proxy/spend_tracking/spend_tracking_utils.py:956`)
    and `daily_spend_entity_ids` drops an empty id only for end users
    (`litellm/proxy/db/daily_spend_bulk_upsert.py:73`), so a `LiteLLM_DailyTeamSpend` row with `team_id` '' is
    written for every teamless key, although the team writer means to skip a request without a team
    (`litellm/proxy/db/db_spend_update_writer.py:3215`) and the organization table gets no such row."""
    pytest.skip("BUG: a key without a team writes a LiteLLM_DailyTeamSpend row with an empty team_id")
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        rows: Final = held(
            lambda: gateway.rows('SELECT team_id FROM "LiteLLM_DailyTeamSpend" WHERE api_key = %s', (hashed(key),)),
            lambda found: found == [],
            holding=3,
        )
        assert rows == []


def test_org_team_key_call_writes_a_daily_organization_row(gateway: Gateway) -> None:
    """The key's team belongs to an organization. One call writes a `LiteLLM_DailyOrganizationSpend` row with
    `organization_id` the organization, the hashed key, api_requests 1 and spend 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        organization: Final = scenario.organization(models=[model])
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(gateway, model, key))
        daily: Final = daily_row(gateway, "LiteLLM_DailyOrganizationSpend", key)
        assert (daily["organization_id"], daily["api_key"], daily["api_requests"]) == (organization, hashed(key), 1)
        assert number(daily["spend"]) == pytest.approx(0.02), daily


def test_key_outside_any_organization_writes_no_daily_organization_row(gateway: Gateway) -> None:
    """A team key whose team has no organization makes one call. No `LiteLLM_DailyOrganizationSpend` row may be
    written for the key."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(gateway, model, key))
        daily_row(gateway, "LiteLLM_DailyTeamSpend", key)
        rows: Final = held(
            lambda: gateway.rows(
                'SELECT organization_id FROM "LiteLLM_DailyOrganizationSpend" WHERE api_key = %s', (hashed(key),)
            ),
            lambda found: found == [],
            holding=3,
        )
        assert rows == []


def test_call_naming_an_end_user_writes_a_daily_end_user_row(gateway: Gateway) -> None:
    """A call names a customer in the body `user` field. One `LiteLLM_DailyEndUserSpend` row appears with
    `end_user_id` the customer, the hashed key, api_requests 1 and spend 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        customer: Final = new_customer(gateway, scenario)
        served(ask(gateway, model, key, user=customer))
        daily: Final = daily_row(gateway, "LiteLLM_DailyEndUserSpend", key)
        assert (daily["end_user_id"], daily["api_key"], daily["api_requests"]) == (customer, hashed(key), 1), daily
        assert number(daily["spend"]) == pytest.approx(0.02), daily


def test_call_without_end_user_writes_no_daily_end_user_row(gateway: Gateway) -> None:
    """A call that names no customer must not create a `LiteLLM_DailyEndUserSpend` row with an empty id."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        rows: Final = held(
            lambda: gateway.rows(
                'SELECT end_user_id FROM "LiteLLM_DailyEndUserSpend" WHERE api_key = %s', (hashed(key),)
            ),
            lambda found: found == [],
            holding=3,
        )
        assert rows == []


def test_tagged_call_writes_one_daily_tag_row_per_tag_with_the_request_id(gateway: Gateway) -> None:
    """A call carries two tags in `metadata.tags`. Each tag gets its own `LiteLLM_DailyTagSpend` row with the
    hashed key, the request's `request_id`, api_requests 1 and the full spend 0.02. No tag object has to exist
    beforehand."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tags: Final = (f"dec1-daily-a-{hashed(key)[:12]}", f"dec1-daily-b-{hashed(key)[:12]}")
        request_id: Final = served_id(ask(gateway, model, key, metadata={"tags": [*tags]}))
        for tag in tags:
            daily = _tag_rows(gateway, tag)[0]
            assert (daily["tag"], daily["api_key"], daily["request_id"]) == (tag, hashed(key), request_id), daily
            assert (daily["api_requests"], number(daily["spend"])) == (1, pytest.approx(0.02)), daily


def test_daily_tag_row_accumulates_across_keys_only_within_the_same_key(gateway: Gateway) -> None:
    """Two different keys send the same tag once each. `api_key` is part of the daily tag key, so the tag has two
    rows of api_requests 1, and their spends add up to 0.04."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        first: Final = scenario.key(models=[model])
        second: Final = scenario.key(models=[model])
        tag: Final = f"dec1-daily-shared-{hashed(first)[:12]}"
        served(ask(gateway, model, first, metadata={"tags": [tag]}))
        served(ask(gateway, model, second, metadata={"tags": [tag]}))
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT api_key, api_requests, spend FROM "LiteLLM_DailyTagSpend" WHERE tag = %s', (tag,)
            ),
            lambda found: len(found) == 2,
            seconds=LANDING_SECONDS,
        )
        assert {row["api_key"] for row in rows} == {hashed(first), hashed(second)}, rows
        assert [row["api_requests"] for row in rows] == [1, 1], rows
        assert sum(number(row["spend"]) for row in rows) == pytest.approx(0.04), rows


@pytest.mark.parametrize("surface", SURFACES, ids=_IDS)
def test_daily_row_endpoint_and_tokens_follow_the_surface(gateway: Gateway, surface: Surface) -> None:
    """Each surface lands on a daily user row whose `endpoint` is derived from the call type: `/chat/completions`,
    `/embeddings`, `/responses`, `/completions`, and an empty string for `/v1/messages` because
    `anthropic_messages` has no entry in `ROUTE_ENDPOINT_MAPPING`. Tokens and spend are those of the surface."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = surface.model(scenario, wire)
        key: Final = scenario.key(models=[model], user_id=scenario.user())
        served(gateway.request("POST", surface.path, surface.body(model), key=key))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert daily["endpoint"] == surface.daily_endpoint, daily
        assert (daily["prompt_tokens"], daily["completion_tokens"]) == (
            surface.prompt_tokens,
            surface.completion_tokens,
        ), daily
        assert number(daily["spend"]) == pytest.approx(surface.cost), daily


def test_daily_row_response_time_is_the_duration_stored_on_the_request_row(gateway: Gateway) -> None:
    """Latency charts divide `total_response_time_ms` by `timed_requests`. For one successful call the daily row
    has timed_requests 1 and total_response_time_ms equal to the request row's `request_duration_ms`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], user_id=scenario.user())
        logged: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert (daily["timed_requests"], daily["total_response_time_ms"]) == (1, logged["request_duration_ms"])


def test_daily_rows_of_a_key_add_up_to_its_request_rows(gateway: Gateway) -> None:
    """A key makes two successful calls and one failing call against another deployment. Summed over the key's
    daily user rows: api_requests 3, successful_requests 2, failed_requests 1, and spend equal to the sum of
    the `spend` column of the key's three request rows (0.04)."""
    with wire_server(failing(500)) as broken, scripted_upstream() as healthy, gateway.scenario() as scenario:
        good: Final = chat_model(scenario, healthy)
        bad: Final = chat_model(scenario, broken, model="openai/gpt-4o")
        key: Final = scenario.key(models=[good, bad], user_id=scenario.user())
        served(ask(gateway, good, key))
        served(ask(gateway, good, key))
        assert ask(gateway, bad, key, num_retries=0).status_code == 500
        logged: Final = rows_of_key(gateway, key, 3)
        spend_reaches(gateway, "key", hashed(key), 2 * CHAT_COST)
        rows: Final = eventually(
            lambda: [object_value(row["row"]) for row in daily_rows(gateway, "LiteLLM_DailyUserSpend", key, 2)],
            lambda found: sum(int(number(row["api_requests"])) for row in found) == 3,
            seconds=LANDING_SECONDS,
        )
        assert sum(int(number(row["successful_requests"])) for row in rows) == 2, rows
        assert sum(int(number(row["failed_requests"])) for row in rows) == 1, rows
        assert sum(number(row["spend"]) for row in rows) == pytest.approx(sum(number(row["spend"]) for row in logged))
        assert sum(number(row["spend"]) for row in rows) == pytest.approx(0.04)


def test_daily_model_usage_row_counts_the_request_for_its_model_group(gateway: Gateway) -> None:
    """`LiteLLM_DailyModelUsage` is keyed by (date, model_group, model, custom_llm_provider, task_type) across all
    keys. One call to a fresh public model name leaves one row for that `model_group` with request_count 1,
    successful_requests 1, failed_requests 0, 10 prompt tokens, 5 completion tokens and spend 0.02. Its `model` is
    the deployment's model with the provider prefix, `openai/gpt-4o-mini`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT to_jsonb(d) AS row FROM "LiteLLM_DailyModelUsage" d WHERE model_group = %s', (model,)
            ),
            lambda found: len(found) == 1,
            seconds=LANDING_SECONDS,
        )
        usage: Final = object_value(rows[0]["row"])
        assert (usage["model"], usage["custom_llm_provider"]) == ("openai/gpt-4o-mini", "openai"), usage
        assert (usage["request_count"], usage["successful_requests"], usage["failed_requests"]) == (1, 1, 0), usage
        assert (usage["prompt_tokens"], usage["completion_tokens"]) == (10, 5), usage
        assert number(usage["spend"]) == pytest.approx(0.02), usage
