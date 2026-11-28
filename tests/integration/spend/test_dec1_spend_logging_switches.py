"""Contract: the settings that change what is written per request, and what keeps working when they are on.

`general_settings.disable_spend_logs: true` stops writing `LiteLLM_SpendLogs` rows. It must not stop billing: the
key, team and user totals, the daily tables, the response cost header and budget enforcement all keep working,
because operators turn request logs off for volume or privacy and still bill on totals.

`general_settings.disable_spend_updates: true` is the bigger switch: no request row and no spend totals at all.

`general_settings.store_prompts_in_spend_logs: true` (or the `STORE_PROMPTS_IN_SPEND_LOGS` environment variable)
makes the row carry the request body in `proxy_server_request` and the provider answer in `response`. A key or a
team can opt out with `metadata.store_prompts_in_spend_logs: false`. Stored strings longer than
`MAX_STRING_LENGTH_PROMPT_IN_DB` characters are truncated.

Every test boots its own proxy with one setting changed. One served chat call costs 0.02.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, held, list_value, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Wire
from tests.integration.spend.dec1_spend_support import (
    CHAT,
    CHAT_COST,
    LANDING_SECONDS,
    REPLY_TEXT,
    TINY_BUDGET,
    ask,
    chat_model,
    daily_row,
    hashed,
    number,
    proxy_config,
    refusal,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served,
    served_id,
    spend_of,
    spend_reaches,
    stream,
)

pytestmark: Final = pytest.mark.shares_worker_state


@contextmanager
def _proxy(
    gateway: Gateway,
    directory: Path,
    settings: Mapping[str, JsonValue],
    environment: Mapping[str, str] | None = None,
) -> Iterator[tuple[Gateway, Scenario, Wire]]:
    """An owned proxy with `settings` merged into `general_settings`, a scenario on it and a scripted provider."""
    config: Final = proxy_config(directory, general_settings=settings)
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, directory, dict(environment or {}), config=config) as owned,
        owned.scenario() as scenario,
    ):
        yield owned, scenario, wire


def _no_rows(owned: Gateway, key: str) -> list[dict[str, JsonValue]]:
    return held(
        lambda: owned.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(key),)),
        lambda found: found == [],
        holding=5,
        seconds=LANDING_SECONDS,
    )


def _stored_prompt(row: Mapping[str, JsonValue]) -> str:
    messages: Final = list_value(object_value(row["proxy_server_request"])["messages"])
    return string_value(object_value(messages[0])["content"])


def test_disable_spend_logs_writes_no_request_row_but_charges_the_key(gateway: Gateway, tmp_path: Path) -> None:
    """With `disable_spend_logs: true` one chat call is served. The key's spend reaches 0.02, and for five seconds
    after that no `LiteLLM_SpendLogs` row exists for the key."""
    with _proxy(gateway, tmp_path, {"disable_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(owned, model, key))
        assert spend_reaches(owned, "key", hashed(key), CHAT_COST) == pytest.approx(0.02)
        assert _no_rows(owned, key) == []


def test_disable_spend_logs_still_charges_team_and_user(gateway: Gateway, tmp_path: Path) -> None:
    """With request logs off, a team key owned by a user still adds 0.02 to the team's spend and 0.02 to the
    user's spend."""
    with _proxy(gateway, tmp_path, {"disable_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        served(ask(owned, model, key))
        assert spend_reaches(owned, "team", team, CHAT_COST) == pytest.approx(0.02)
        assert spend_reaches(owned, "user", user, CHAT_COST) == pytest.approx(0.02)


def test_disable_spend_logs_still_writes_the_daily_user_row(gateway: Gateway, tmp_path: Path) -> None:
    """With request logs off, the `LiteLLM_DailyUserSpend` row for the key is still written with api_requests 1,
    10 prompt tokens, 5 completion tokens and spend 0.02. The usage pages do not depend on request logs."""
    with _proxy(gateway, tmp_path, {"disable_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], user_id=scenario.user())
        served(ask(owned, model, key))
        daily: Final = daily_row(owned, "LiteLLM_DailyUserSpend", key)
        assert (daily["api_requests"], daily["prompt_tokens"], daily["completion_tokens"]) == (1, 10, 5), daily
        assert number(daily["spend"]) == pytest.approx(0.02), daily


def test_disable_spend_logs_still_reports_the_cost_header(gateway: Gateway, tmp_path: Path) -> None:
    """With request logs off, the response still carries `x-litellm-response-cost: 0.02` and the key's stored
    spend ends up equal to that header."""
    with _proxy(gateway, tmp_path, {"disable_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(owned, model, key)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.02), response.headers
        assert spend_reaches(owned, "key", hashed(key), CHAT_COST) == pytest.approx(
            float(response.headers["x-litellm-response-cost"])
        )


def test_disable_spend_logs_still_enforces_the_key_budget(gateway: Gateway, tmp_path: Path) -> None:
    """With request logs off, a key with `max_budget` 0.0000001 is served once and then refused with 422
    `budget_exceeded`. Budgets read the totals, not the request log."""
    with _proxy(gateway, tmp_path, {"disable_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        assert refusal(ask(owned, model, key)).startswith("Budget has been exceeded! Key="), key


def test_disable_spend_logs_keeps_an_accounting_delivery_for_the_request(gateway: Gateway, tmp_path: Path) -> None:
    """With request logs off, the charge still travels through the ledger: a `LiteLLM_SpendLogDelivery` row for
    the response id exists and reaches state `applied`. Assumes the delivery keeps the response id as
    `request_id` when the log row itself is suppressed."""
    with _proxy(gateway, tmp_path, {"disable_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        deliveries: Final = owned.rows(
            'SELECT state::text AS state FROM "LiteLLM_SpendLogDelivery" WHERE request_id = %s', (request_id,)
        )
        assert deliveries == [{"state": "applied"}], deliveries


def test_disable_spend_updates_writes_no_row_and_leaves_every_total_at_zero(gateway: Gateway, tmp_path: Path) -> None:
    """With `disable_spend_updates: true` a team key is served twice. The response still reports its cost, but no
    request row is written and the key's and the team's spend stay 0 for the whole observation window."""
    with _proxy(gateway, tmp_path, {"disable_spend_updates": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        response: Final = ask(owned, model, key)
        served(response)
        served(ask(owned, model, key))
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.02), response.headers
        assert _no_rows(owned, key) == []
        assert spend_of(owned, "key", hashed(key)) == 0
        assert spend_of(owned, "team", team) == 0


def test_store_prompts_puts_the_request_body_into_proxy_server_request(gateway: Gateway, tmp_path: Path) -> None:
    """With `store_prompts_in_spend_logs: true` the row's `proxy_server_request` holds the caller's body: the
    public model name and the messages with the exact prompt text. The `messages` column stays an empty object
    for chat calls, it is only used by realtime sessions."""
    with _proxy(gateway, tmp_path, {"store_prompts_in_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"dec1 stored prompt {uuid.uuid4().hex}"
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt)))
        assert object_value(row["proxy_server_request"])["model"] == model, row
        assert _stored_prompt(row) == prompt, row
        assert row["messages"] == {}, row


def test_store_prompts_puts_the_provider_answer_into_response(gateway: Gateway, tmp_path: Path) -> None:
    """With prompt storage on, the row's `response` column holds the completion that was returned: the same `id`
    as the request row and the assistant text the provider sent."""
    with _proxy(gateway, tmp_path, {"store_prompts_in_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(owned, model, key))
        stored: Final = object_value(row_by_id(owned, request_id)["response"])
        choice: Final = object_value(list_value(stored["choices"])[0])
        assert stored["id"] == request_id, stored
        assert object_value(choice["message"])["content"] == REPLY_TEXT, stored


def test_store_prompts_stores_the_assembled_text_of_a_streamed_answer(gateway: Gateway, tmp_path: Path) -> None:
    """With prompt storage on, a streamed chat completion stores one assembled response, not chunks: the
    `response` column contains the full assistant text after the stream has ended."""
    with _proxy(gateway, tmp_path, {"store_prompts_in_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        stream(owned, CHAT.path, CHAT.body(model, streaming=True), key)
        stored: Final = object_value(rows_of_key(owned, key, 1)[0]["response"])
        choice: Final = object_value(list_value(stored["choices"])[0])
        assert object_value(choice["message"])["content"] == REPLY_TEXT, stored


def test_store_prompts_never_stores_the_callers_key_or_the_provider_key(gateway: Gateway, tmp_path: Path) -> None:
    """With prompt storage on, the whole stored row is searched for secrets. Neither the caller's raw `sk-` key
    nor the deployment's provider api key may appear in any column."""
    with _proxy(gateway, tmp_path, {"store_prompts_in_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        text: Final = json.dumps(row)
        assert row["proxy_server_request"] != {}, row
        assert key not in text and "dec1-provider-key" not in text, "a secret was stored in the spend log row"


def test_store_prompts_can_be_enabled_by_environment_variable(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy without the config setting but with `STORE_PROMPTS_IN_SPEND_LOGS=true` in its environment stores
    the prompt in `proxy_server_request` just the same."""
    with _proxy(gateway, tmp_path, {}, {"STORE_PROMPTS_IN_SPEND_LOGS": "true"}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"dec1 env stored prompt {uuid.uuid4().hex}"
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt)))
        assert _stored_prompt(row) == prompt, row


@pytest.mark.parametrize("owner", ["key", "team"])
def test_key_or_team_metadata_can_opt_out_of_prompt_storage(gateway: Gateway, tmp_path: Path, owner: str) -> None:
    """Prompt storage is on proxy-wide, but the key (or the key's team) has
    `metadata.store_prompts_in_spend_logs: false`. Its row keeps `proxy_server_request` and `response` empty,
    while another key on the same proxy has its prompt stored."""
    with _proxy(gateway, tmp_path, {"store_prompts_in_spend_logs": True}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        opt_out: Final[dict[str, JsonValue]] = {"store_prompts_in_spend_logs": False}
        private: Final = (
            scenario.key(models=[model], metadata=opt_out)
            if owner == "key"
            else scenario.key(team_id=scenario.team(models=[model], metadata=opt_out), models=[model])
        )
        public: Final = scenario.key(models=[model])
        hidden: Final = row_by_id(owned, served_id(ask(owned, model, private)))
        shown: Final = row_by_id(owned, served_id(ask(owned, model, public)))
        assert (hidden["proxy_server_request"], hidden["response"]) == ({}, {}), hidden
        assert shown["proxy_server_request"] != {} and shown["response"] != {}, shown


def test_stored_prompt_is_truncated_to_the_configured_length(gateway: Gateway, tmp_path: Path) -> None:
    """With `MAX_STRING_LENGTH_PROMPT_IN_DB=200` a 5000 character prompt is sent. The provider still receives all
    5000 characters, but the copy stored in `proxy_server_request` is shorter than the original."""
    with _proxy(
        gateway, tmp_path, {"store_prompts_in_spend_logs": True}, {"MAX_STRING_LENGTH_PROMPT_IN_DB": "200"}
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"dec1-long-{uuid.uuid4().hex}-".ljust(5000, "x")
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt)))
        sent: Final = [request for request in wire.drain() if request.method == "POST"]
        assert prompt in sent[0].body.decode(), "the provider must receive the untruncated prompt"
        assert len(_stored_prompt(row)) < len(prompt), row
        assert prompt not in json.dumps(row), row
