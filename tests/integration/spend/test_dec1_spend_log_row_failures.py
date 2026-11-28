"""Contract: what a failed request leaves in `LiteLLM_SpendLogs`, and what it must not charge.

A request can fail at the provider (4xx or 5xx from upstream) or be refused by the proxy before any provider is
called (budget exceeded). Both leave one row with `status` `failure`, zero spend and an `error_information`
object, so operators can see error rates per key and model. Neither may move any spend counter. Two settings
change this: `general_settings.disable_error_logs` drops every failure row, and
`general_settings.disable_pre_routing_error_logs` drops only the rows of requests the proxy itself refused.

Every request here sends `num_retries: 0` so one caller request is one provider attempt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, held, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import wire_server
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    TINY_BUDGET,
    ask,
    chat_model,
    daily_row,
    failing,
    hashed,
    metadata_of,
    number,
    provider_calls,
    proxy_config,
    refusal,
    rows_of_key,
    scripted_upstream,
    served,
    spend_of,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.mark.parametrize("status", [400, 429, 500])
def test_provider_error_writes_a_failure_row_with_the_provider_status(gateway: Gateway, status: int) -> None:
    """The provider answers every call with 400, 429 or 500. The caller gets a non-200 response, and one row is
    written with `status` `failure`, `metadata.status` `failure`, `metadata.error_information.error_code` equal
    to the provider status as a string, and the provider's message in `error_message`."""
    with wire_server(failing(status)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key, num_retries=0)
        assert response.status_code == status, response.text
        row: Final = rows_of_key(gateway, key, 1)[0]
        error: Final = object_value(metadata_of(row)["error_information"])
        assert (row["status"], metadata_of(row)["status"]) == ("failure", "failure"), row
        assert error["error_code"] == str(status), error
        assert "dec1 scripted upstream failure" in string_value(error["error_message"]), error


def test_failure_row_still_names_the_key_model_group_and_call_type(gateway: Gateway) -> None:
    """A failed chat call is attributed exactly like a successful one: hashed key in `api_key`, owner in `user`,
    team in `team_id`, public model name in `model_group`, `call_type` `acompletion`. Error dashboards group
    on these columns."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["api_key"], row["user"], row["team_id"]) == (hashed(key), user, team), row
        assert (row["model_group"], row["call_type"]) == (model, "acompletion"), row


def test_failure_row_has_zero_spend_and_only_the_estimated_prompt_tokens(gateway: Gateway) -> None:
    """The provider returned an error and no usage, so the failure row must store spend 0 and no completion
    tokens. A failed request is never billed. The prompt the proxy dispatched is still counted: the row carries
    the locally estimated prompt tokens and a total equal to them, which the existing contract
    `tests/integration/spend/test_failed_dispatch_tokens.py` pins for provider 500s after dispatch."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert number(row["spend"]) == 0, row
        assert number(row["prompt_tokens"]) > 0, row
        assert (row["completion_tokens"], row["total_tokens"]) == (0, row["prompt_tokens"]), row


def test_failure_row_request_id_is_the_litellm_call_id(gateway: Gateway) -> None:
    """A failed call has no provider response id, so the row is keyed by the proxy's own call id: `request_id`
    equals the `litellm_call_id` column and is not empty. UNVERIFIED: read from `get_spend_logs_id`, which
    falls back to the standard logging payload id and then `litellm_call_id`."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert row["request_id"], row
        assert row["request_id"] == row["litellm_call_id"], row


def test_failed_request_does_not_change_key_team_or_user_spend(gateway: Gateway) -> None:
    """A team key owned by a user gets a provider 500. After the failure row has landed, the key, the team and
    the user still have spend 0."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        rows_of_key(gateway, key, 1)
        daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert spend_of(gateway, "key", hashed(key)) == 0
        assert spend_of(gateway, "team", team) == 0
        assert spend_of(gateway, "user", user) == 0


def test_failed_request_counts_as_a_failed_request_in_daily_user_spend(gateway: Gateway) -> None:
    """One provider failure produces a `LiteLLM_DailyUserSpend` row for the key with api_requests 1,
    failed_requests 1, successful_requests 0 and spend 0. Error-rate charts read these counters."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], user_id=scenario.user())
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert (daily["api_requests"], daily["failed_requests"], daily["successful_requests"]) == (1, 1, 0), daily
        assert number(daily["spend"]) == 0, daily


def test_success_after_failure_is_billed_once(gateway: Gateway) -> None:
    """The same key fails once against a broken deployment and then succeeds once against a healthy one. Two
    rows exist, one `failure` with spend 0 and one `success` with spend 0.02, and the key's spend is 0.02."""
    with wire_server(failing(500)) as broken, scripted_upstream() as healthy, gateway.scenario() as scenario:
        bad_model: Final = chat_model(scenario, broken)
        good_model: Final = chat_model(scenario, healthy)
        key: Final = scenario.key(models=[bad_model, good_model])
        assert ask(gateway, bad_model, key, num_retries=0).status_code == 500
        served(ask(gateway, good_model, key))
        rows: Final = rows_of_key(gateway, key, 2)
        assert sorted((string_value(row["status"]), number(row["spend"])) for row in rows) == [
            ("failure", 0.0),
            ("success", pytest.approx(CHAT_COST)),
        ]
        assert spend_reaches(gateway, "key", hashed(key), CHAT_COST) == pytest.approx(CHAT_COST)


def test_budget_refusal_writes_a_failure_row_with_code_422_and_no_spend(gateway: Gateway) -> None:
    """A key over its budget is refused with 422 before the provider is called. By default that refusal is
    logged: a second row appears with `status` `failure`, error_code `422`, error_class `BudgetExceededError`
    and spend 0, and the provider saw only the first, served call."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        refusal(ask(gateway, model, key))
        rows: Final = rows_of_key(gateway, key, 2)
        refused: Final = next(row for row in rows if row["status"] == "failure")
        error: Final = object_value(metadata_of(refused)["error_information"])
        assert (error["error_code"], error["error_class"]) == ("422", "BudgetExceededError"), error
        assert number(refused["spend"]) == 0, refused
        assert len(provider_calls(wire)) == 1


def test_budget_refusal_does_not_add_to_the_key_spend(gateway: Gateway) -> None:
    """After one served call (0.02) the key is refused three times. Its stored spend must stay at 0.02 while
    the refusals are logged, a refusal is never charged."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        for _ in range(3):
            refusal(ask(gateway, model, key))
        rows_of_key(gateway, key, 4)
        assert spend_of(gateway, "key", hashed(key)) == pytest.approx(CHAT_COST)


def test_disable_error_logs_drops_failure_rows_but_keeps_success_rows(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy started with `general_settings.disable_error_logs: true` serves one failing and one successful
    call for the same key. Only the success row is ever written, the failure leaves no row."""
    config: Final = proxy_config(tmp_path, general_settings={"disable_error_logs": True})
    with (
        wire_server(failing(500)) as broken,
        scripted_upstream() as healthy,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
    ):
        bad_model: Final = chat_model(scenario, broken)
        good_model: Final = chat_model(scenario, healthy)
        key: Final = scenario.key(models=[bad_model, good_model])
        assert ask(owned, bad_model, key, num_retries=0).status_code == 500
        served(ask(owned, good_model, key))
        rows: Final = held(
            lambda: owned.rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(key),)),
            lambda found: len(found) == 1,
            holding=5,
            seconds=LANDING_SECONDS,
        )
        assert rows == [{"status": "success"}], rows


def test_disable_pre_routing_error_logs_drops_budget_refusal_rows_only(gateway: Gateway, tmp_path: Path) -> None:
    """With `general_settings.disable_pre_routing_error_logs: true` a budget refusal writes no row, so a client
    retrying an exhausted key cannot flood the table, while a provider failure on another key is still logged."""
    config: Final = proxy_config(tmp_path, general_settings={"disable_pre_routing_error_logs": True})
    with (
        wire_server(failing(500)) as broken,
        scripted_upstream() as healthy,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
    ):
        bad_model: Final = chat_model(scenario, broken)
        good_model: Final = chat_model(scenario, healthy)
        exhausted: Final = scenario.key(models=[good_model], max_budget=TINY_BUDGET)
        failing_key: Final = scenario.key(models=[bad_model])
        served(ask(owned, good_model, exhausted))
        spend_reaches(owned, "key", hashed(exhausted), CHAT_COST)
        refusal(ask(owned, good_model, exhausted))
        assert ask(owned, bad_model, failing_key, num_retries=0).status_code == 500
        assert rows_of_key(owned, failing_key, 1)[0]["status"] == "failure"
        kept: Final = eventually(
            lambda: owned.rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(exhausted),)),
            lambda found: len(found) >= 1,
            seconds=LANDING_SECONDS,
        )
        assert kept == [{"status": "success"}], kept
