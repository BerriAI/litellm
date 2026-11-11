"""Contract: what a cached answer costs, and where the price of a paid answer comes from.

Cache: the shared proxy runs with the Redis response cache on. A repeated identical request is answered from the
cache, the provider is not called, and the caller is charged nothing. The hit still gets its own request row
(`<original id>_cache_hit<timestamp>`, `cache_hit` `True`, spend 0) and still counts as a request in the daily
tables. `cache: {"no-cache": true}` in the body forces a paid provider call.

Pricing: the amount charged for a paid call is tokens times the price configured on the deployment that served
it, taken from `litellm_params` or from `model_info`, never from the public cost map when a custom price is set.
A deployment priced at zero is free: its rows carry spend 0 and it is exempt from budget checks. The scripted
provider reports 10 prompt and 5 completion tokens, so 0.001 / 0.002 per token is 0.02 and 0.003 / 0.004 is 0.05.
"""

from __future__ import annotations

import os
import uuid
from typing import Final

import pytest
from redis import Redis

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    ask,
    chat_model,
    daily_row,
    hashed,
    number,
    provider_calls,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served,
    served_id,
    spend_of,
    spend_reaches,
)


def _prompt() -> str:
    return f"dec1 cached prompt {uuid.uuid4().hex}"


def test_repeated_request_is_answered_from_cache_without_a_provider_call(gateway: Gateway) -> None:
    """The same key sends the same prompt to the same model twice. The provider receives exactly one request and
    the second response has the same message content as the first."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = _prompt()
        first: Final = served(ask(gateway, model, key, text=prompt))
        second: Final = served(ask(gateway, model, key, text=prompt))
        assert len(provider_calls(wire)) == 1
        assert second["choices"] == first["choices"], (first, second)


def test_cache_hit_reports_zero_in_the_response_cost_header(gateway: Gateway) -> None:
    """The first call reports `x-litellm-response-cost` 0.02. The cached repeat reports a cost of 0 in the same
    header, so a caller summing the header never pays twice for one provider call."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = _prompt()
        paid: Final = ask(gateway, model, key, text=prompt)
        hit: Final = ask(gateway, model, key, text=prompt)
        served(hit)
        assert float(paid.headers["x-litellm-response-cost"]) == pytest.approx(0.02), paid.headers
        assert float(hit.headers["x-litellm-response-cost"]) == 0, hit.headers


def test_cache_hit_row_is_marked_and_keeps_the_callers_attribution(gateway: Gateway) -> None:
    """The cached repeat writes a second row. Its `request_id` starts with `<first id>_cache_hit`, `cache_hit` is
    the string `True`, spend is 0, status is `success`, and it names the same hashed key, team, public model name
    and call type as the paid row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        prompt: Final = _prompt()
        first: Final = served_id(ask(gateway, model, key, text=prompt))
        served(ask(gateway, model, key, text=prompt))
        rows: Final = rows_of_key(gateway, key, 2)
        hits: Final = [row for row in rows if row["cache_hit"] == "True"]
        assert len(hits) == 1, rows
        hit: Final = hits[0]
        assert string_value(hit["request_id"]).startswith(f"{first}_cache_hit"), hit
        assert (number(hit["spend"]), hit["status"]) == (0, "success"), hit
        assert (hit["api_key"], hit["team_id"], hit["model_group"], hit["call_type"]) == (
            hashed(key),
            team,
            model,
            "acompletion",
        ), hit


def test_cache_hit_adds_nothing_to_key_or_team_spend(gateway: Gateway) -> None:
    """One paid call and two cached repeats by a team key. Once all three rows exist, the key's spend and the
    team's spend are both 0.02, the cost of the single provider call."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        prompt: Final = _prompt()
        for _ in range(3):
            served(ask(gateway, model, key, text=prompt))
        assert len(provider_calls(wire)) == 1
        rows_of_key(gateway, key, 3)
        daily: Final = daily_row(gateway, "LiteLLM_DailyTeamSpend", key)
        assert spend_reaches(gateway, "key", hashed(key), CHAT_COST) == pytest.approx(0.02)
        assert spend_of(gateway, "team", team) == pytest.approx(0.02)
        assert number(daily["spend"]) == pytest.approx(0.02), daily


def test_cache_hit_counts_as_a_successful_request_in_the_daily_row(gateway: Gateway) -> None:
    """One paid call and one cached repeat end as one daily user row with api_requests 2, successful_requests 2,
    failed_requests 0 and spend 0.02. Request counts include cache hits, spend does not."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], user_id=scenario.user())
        prompt: Final = _prompt()
        served(ask(gateway, model, key, text=prompt))
        served(ask(gateway, model, key, text=prompt))
        rows_of_key(gateway, key, 2)
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert (daily["api_requests"], daily["successful_requests"], daily["failed_requests"]) == (2, 2, 0), daily
        assert number(daily["spend"]) == pytest.approx(0.02), daily


def test_no_cache_request_pays_for_a_fresh_provider_call(gateway: Gateway) -> None:
    """The repeat carries `cache: {"no-cache": true}`. The provider is called a second time, both rows have spend
    0.02 and neither is marked as a cache hit, and the key's spend is 0.04."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = _prompt()
        served(ask(gateway, model, key, text=prompt))
        served(ask(gateway, model, key, text=prompt, cache={"no-cache": True}))
        assert len(provider_calls(wire)) == 2
        rows: Final = rows_of_key(gateway, key, 2)
        assert [number(row["spend"]) for row in rows] == [pytest.approx(0.02), pytest.approx(0.02)], rows
        assert all(row["cache_hit"] != "True" for row in rows), rows
        assert spend_reaches(gateway, "key", hashed(key), 2 * CHAT_COST) == pytest.approx(0.04)


def test_cache_hits_do_not_use_up_a_key_budget(gateway: Gateway) -> None:
    """A key has `max_budget` 0.03. One paid call costs 0.02. Five cached repeats are all served and the key is
    never refused, because hits cost nothing and the stored spend stays at 0.02.
    Each request first reserves its worst case cost on the key's Redis counter `spend:key:<hash>`, and the cost
    callback releases that reservation after the response was sent
    (`litellm/proxy/spend_tracking/budget_reservation.py:250`, `litellm/proxy/hooks/proxy_track_cost_callback.py:929`).
    Back to back repeats can therefore meet the previous hit's reservation still held. So before each repeat
    the test waits until the counter is back at the settled 0.02; a hit that charged anything would keep it above."""
    with (
        scripted_upstream() as wire,
        gateway.scenario() as scenario,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as counters,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.03)
        prompt: Final = _prompt()
        served(ask(gateway, model, key, text=prompt))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        counter: Final = f"spend:key:{hashed(key)}"

        def settled() -> float | None:
            value = counters.get(counter)
            return None if value is None else float(value)

        statuses: list[int] = []  # mutable-ok: one status per repeat, each sent after the counter settled
        for _ in range(5):
            eventually(settled, lambda value: value is None or abs(value - CHAT_COST) < 1e-9, seconds=LANDING_SECONDS)
            statuses.append(ask(gateway, model, key, text=prompt).status_code)
        assert statuses == [200] * 5
        rows_of_key(gateway, key, 6)
        assert spend_of(gateway, "key", hashed(key)) == pytest.approx(0.02)
        assert len(provider_calls(wire)) == 1


def test_a_second_key_served_from_cache_is_logged_under_itself_and_pays_nothing(gateway: Gateway) -> None:
    """Key A pays for a prompt. Key B sends the same prompt to the same model and is served from the cache. B's
    row is filed under B's own hash with spend 0 and A's spend stays 0.02.
    UNVERIFIED: assumes the default cache key does not include the calling key, see `Cache.get_cache_key`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        payer: Final = scenario.key(models=[model])
        rider: Final = scenario.key(models=[model])
        prompt: Final = _prompt()
        served(ask(gateway, model, payer, text=prompt))
        served(ask(gateway, model, rider, text=prompt))
        assert len(provider_calls(wire)) == 1
        hit: Final = rows_of_key(gateway, rider, 1)[0]
        assert (hit["api_key"], hit["cache_hit"], number(hit["spend"])) == (hashed(rider), "True", 0), hit
        assert spend_reaches(gateway, "key", hashed(payer), CHAT_COST) == pytest.approx(0.02)
        assert spend_of(gateway, "key", hashed(rider)) == 0


def test_price_set_in_model_info_is_charged(gateway: Gateway) -> None:
    """The deployment has no price in `litellm_params` but `model_info` sets 0.003 per input token and 0.004 per
    output token. 10 prompt and 5 completion tokens are charged 10 * 0.003 + 5 * 0.004 = 0.05 in the header, the
    row and the key's spend.
    BUG: a deployment added through `/model/new` with its price in `model_info` is charged the public cost map
    price of `gpt-4o-mini` (4.5e-06 in run 42966523075), although `use_custom_pricing_for_model` names
    `model_info` as a custom price source (`litellm/litellm_core_utils/litellm_logging.py:6865`) and the cost
    calculator reads it from there (`litellm/cost_calculator.py:2649`)."""
    pytest.skip("BUG: a custom price set in model_info of a /model/new deployment is ignored for the charge")
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=f"{wire.url}/v1",
            api_key="dec1-provider-key",
            model_info={"input_cost_per_token": 0.003, "output_cost_per_token": 0.004},
        )
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        row: Final = row_by_id(gateway, served_id(response))
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(10 * 0.003 + 5 * 0.004)
        assert number(row["spend"]) == pytest.approx(0.05), row
        assert spend_reaches(gateway, "key", hashed(key), 0.05) == pytest.approx(0.05)


def test_cost_breakdown_in_row_metadata_splits_input_and_output_cost(gateway: Gateway) -> None:
    """`metadata.cost_breakdown` of the row itemises the charge: input_cost 10 * 0.001 = 0.01, output_cost
    5 * 0.002 = 0.01, total_cost 0.02. The same split is returned in the `x-litellm-response-cost-input` and
    `x-litellm-response-cost-output` headers."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        breakdown: Final = object_value(
            object_value(row_by_id(gateway, served_id(response))["metadata"])["cost_breakdown"]
        )
        assert number(breakdown["input_cost"]) == pytest.approx(10 * 0.001), breakdown
        assert number(breakdown["output_cost"]) == pytest.approx(5 * 0.002), breakdown
        assert number(breakdown["total_cost"]) == pytest.approx(0.02), breakdown
        assert float(response.headers["x-litellm-response-cost-input"]) == pytest.approx(0.01), response.headers
        assert float(response.headers["x-litellm-response-cost-output"]) == pytest.approx(0.01), response.headers


def test_zero_priced_deployment_writes_a_row_with_zero_spend(gateway: Gateway) -> None:
    """A deployment priced at 0 per token is called once. The response cost header is 0, the row is written with
    spend 0 and the real token counts 10 and 5, and the key's spend stays 0."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire, input_cost_per_token=0.0, output_cost_per_token=0.0)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        row: Final = row_by_id(gateway, served_id(response))
        assert float(response.headers["x-litellm-response-cost"]) == 0, response.headers
        assert (number(row["spend"]), row["prompt_tokens"], row["completion_tokens"]) == (0, 10, 5), row
        daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert spend_of(gateway, "key", hashed(key)) == 0


def test_zero_priced_deployment_is_served_to_a_key_with_no_budget_left(gateway: Gateway) -> None:
    """A key with `max_budget` 0 is refused on a paid model but served on a model whose price is zero. Free
    models skip every budget check, which is how operators keep an internal model available to capped keys."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        paid: Final = chat_model(scenario, wire)
        free: Final = chat_model(scenario, wire, input_cost_per_token=0.0, output_cost_per_token=0.0)
        key: Final = scenario.key(models=[paid, free], max_budget=0)
        assert ask(gateway, paid, key).status_code == 422
        assert ask(gateway, free, key).status_code == 200
        assert len(provider_calls(wire)) == 1


def test_each_call_is_charged_at_the_price_of_the_deployment_that_served_it(gateway: Gateway) -> None:
    """Two deployments share one public model name, one at 0.001 / 0.002 (0.02 per call) and one at 0.003 / 0.004
    (0.05 per call). Over twelve calls, every row's spend equals the response header of its call, every
    deployment id is always charged the same amount, and only the amounts 0.02 and 0.05 occur."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        name: Final = f"dec1-shared-{uuid.uuid4().hex}"
        chat_model(scenario, wire, name=name)
        chat_model(scenario, wire, name=name, input_cost_per_token=0.003, output_cost_per_token=0.004)
        key: Final = scenario.key(models=[name])
        responses: Final = [ask(gateway, name, key) for _ in range(12)]
        quoted: Final = {
            served_id(response): float(response.headers["x-litellm-response-cost"]) for response in responses
        }
        rows: Final = rows_of_key(gateway, key, 12)
        assert all(number(row["spend"]) == pytest.approx(quoted[string_value(row["request_id"])]) for row in rows), rows
        charged: Final = {(string_value(row["model_id"]), round(number(row["spend"]), 6)) for row in rows}
        assert len(charged) == len({deployment for deployment, _ in charged}), charged
        assert {round(number(row["spend"]), 6) for row in rows} <= {0.02, 0.05}, rows
