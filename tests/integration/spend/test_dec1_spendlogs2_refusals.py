"""Contract: which refused requests leave a `LiteLLM_SpendLogs` row, and what that row says.

Spend Logs V2 subscribes the Rust writer to `post_call_failure` precisely because a request the proxy refuses
before any provider attempt (auth, pre-routing, budget, rate limit, guardrail) never reaches `attempt_ended`
(plan, "V2 hooks and fields"). Today those rows come from `ProxyLogging._handle_logging_proxy_only_error`
(`litellm/proxy/utils.py:4550`) and `_ProxyDBLogger.async_post_call_failure_hook`
(`litellm/proxy/hooks/proxy_track_cost_callback.py:469`):

1. A failure on an inference or info route with a presented key writes one row with `status = 'failure'`,
   `spend = 0`, `api_key` = sha256 of the presented key, and `metadata.error_information` built by
   `StandardLoggingPayloadSetup.get_error_information` (`error_code`, `error_class`, `llm_provider`,
   `error_message`, `traceback`).
2. Auth failures are enriched from the key object when the key exists (`_enrich_failure_metadata_with_key_info`,
   `:1042`): alias, user, team, org and project are filled in even though auth refused the request.
3. Nothing is written for a request with no credential at all, nor for management routes
   (`_is_proxy_only_llm_api_error`, `utils.py:4514`).
4. None of these rows charges anything: key, team and daily spend do not move, while the daily row counts a
   failed request.

What these files do not repeat: the existence of the 422 budget row and the two `disable_*_error_logs` switches
(`test_dec1_spend_log_row_failures.py`), the guardrail entry inside the row (`test_dec1_guardrails_spend.py`),
the requester IP on an auth failure (`test_requester_ip_address_logging.py`).
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.spend.dec1_spend_support import (
    TINY_BUDGET,
    ask,
    chat_model,
    hashed,
    number,
    provider_calls,
    scripted_upstream,
    served_id,
    spend_reaches,
)
from tests.integration.spend.dec1_spendlogs2_support import (
    BLOCKING_CODE,
    custom_code_guardrail,
    daily_user_counters,
    error_information,
    failure_rows,
    first_refusal,
    fresh_call_id,
    guarded_ask,
    metadata_dict,
    no_rows_with_call_id,
    rows_with_call_id,
)

pytestmark: Final = pytest.mark.shares_worker_state


def test_an_unknown_virtual_key_writes_one_failure_row_hashed_under_that_key(gateway: Gateway) -> None:
    """A chat call with a well formed but unknown `sk-` key is refused 401 before any provider call. Exactly one
    row lands for it with `status = 'failure'`, `spend = 0`, `api_key` equal to sha256 of the presented key and
    `metadata.error_information.error_code` "401". A port that only writes rows on `attempt_ended` loses it."""
    stranger: Final = f"sk-dec1-sl2-unknown-{uuid.uuid4().hex}"
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        response: Final = ask(gateway, model, stranger)
        assert response.status_code == 401, response.text
        assert provider_calls(wire) == ()
        row: Final = failure_rows(gateway, stranger, 1)[0]
        assert row["api_key"] == hashed(stranger), row
        assert number(row["spend"]) == 0.0, row
        assert error_information(row)["error_code"] == "401", row


def test_an_auth_failure_row_carries_no_tokens_and_no_provider(gateway: Gateway) -> None:
    """The same unknown key refusal stores `prompt_tokens`, `completion_tokens` and `total_tokens` 0, because no
    provider call was made. Pinned current: the 401 carries no `x-litellm-call-id` header (CI run 44161684479),
    because auth refuses before the endpoint that sets the call id header runs, so the caller cannot match the
    refusal to its row by call id."""
    stranger: Final = f"sk-dec1-sl2-unknown-{uuid.uuid4().hex}"
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        response: Final = ask(gateway, model, stranger)
        assert response.status_code == 401, response.text
        row: Final = failure_rows(gateway, stranger, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (0, 0, 0), row
        assert "x-litellm-call-id" not in response.headers, dict(response.headers)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/embeddings"])
def test_a_request_without_any_credential_writes_no_row(gateway: Gateway, path: str) -> None:
    """A call with no `Authorization` header at all (and a client chosen `x-litellm-call-id`) is refused 401 and
    leaves no row under that call id: `_enrich_failure_metadata_with_key_info` has no key hash to attribute it
    to and the proxy-only error path writes nothing for an anonymous caller. Pinned for chat and embeddings
    (`/v1/messages` is pinned in `test_spend_log_tool_payload_content.py`)."""
    call_id: Final = fresh_call_id()
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        body: Final = (
            {"model": model, "input": "dec1 anonymous"}
            if path.endswith("embeddings")
            else {"model": model, "messages": [{"role": "user", "content": "dec1 anonymous"}]}
        )
        response: Final = gateway.client.post(path, json=body, headers={"x-litellm-call-id": call_id})
        assert response.status_code == 401, response.text
        no_rows_with_call_id(gateway, call_id)


def test_a_blocked_key_failure_row_is_enriched_with_the_keys_alias_and_team(gateway: Gateway) -> None:
    """A team key with alias `dec1-sl2-blocked-<id>` is blocked through `POST /key/block`; its next call is
    refused 401. The failure row's metadata still names `user_api_key_alias` and `user_api_key_team_id`, filled
    from the key object by `_enrich_failure_metadata_with_key_info` (proxy_track_cost_callback.py:1042), so a
    dashboard can attribute refused traffic."""
    alias: Final = f"dec1-sl2-blocked-{uuid.uuid4().hex[:8]}"
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team()
        key: Final = scenario.key(models=[model], team_id=team, key_alias=alias)
        gateway.post("/key/block", {"key": key})
        refused: Final = eventually(lambda: ask(gateway, model, key), lambda r: r.status_code == 401, seconds=30)
        assert refused.status_code == 401, refused.text
        metadata: Final = metadata_dict(failure_rows(gateway, key, 1)[0])
        assert metadata["user_api_key_alias"] == alias, metadata
        assert metadata["user_api_key_team_id"] == team, metadata


def test_an_expired_key_failure_row_keeps_the_owner_user_id(gateway: Gateway) -> None:
    """A key generated for a user with `duration: "1s"` is refused 401 once it expires (polled, no sleep). Its
    failure row has `metadata.user_api_key_user_id` equal to that user and `error_information.error_code` "401".
    UNVERIFIED: the expired-key 401 reaches the enrichment with the hash set; read from `auth_exception_handler.py`
    which copies the resolved identity before logging."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user()
        key: Final = scenario.key(models=[model], user_id=user, duration="1s")
        refused: Final = eventually(lambda: ask(gateway, model, key), lambda r: r.status_code == 401, seconds=30)
        assert refused.status_code == 401, refused.text
        row: Final = failure_rows(gateway, key, 1)[0]
        assert metadata_dict(row)["user_api_key_user_id"] == user, row
        assert error_information(row)["error_code"] == "401", row


def test_a_model_the_key_may_not_use_writes_a_failure_row_naming_the_requested_model(gateway: Gateway) -> None:
    """A key limited to model A asks for model B and is refused (403, `key_model_access_denied`, with the client
    message of litellm/proxy/auth/model_access_denied.py:5; pinned current, it used to be 401). One failure row
    lands with `model_group` B, so per-model refusal reports keep working.
    UNVERIFIED: `model_group` on a refusal before routing is taken from the request body model
    (`_routed_metadata_fallback`, proxy_track_cost_callback.py:216); it may be empty."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        allowed: Final = chat_model(scenario, wire)
        other: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[allowed])
        response: Final = ask(gateway, other, key)
        assert response.status_code == 403, response.text
        assert "key_model_access_denied" in response.text, response.text
        assert provider_calls(wire) == ()
        row: Final = failure_rows(gateway, key, 1)[0]
        assert row["model_group"] == other, row


def test_an_unknown_model_writes_a_400_failure_row(gateway: Gateway) -> None:
    """A valid key asking for a model name nobody configured is refused 400 before routing. One failure row
    lands with `error_information.error_code` "400": `is_pre_routing_rejection` treats `ProxyModelNotFoundError`
    as pre-routing (`proxy_track_cost_callback.py:1147`), which only suppresses the row under
    `disable_pre_routing_error_logs`. UNVERIFIED: the status of an unknown model is 400 on the shared proxy."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = ask(gateway, f"dec1-sl2-missing-{uuid.uuid4().hex}", key)
        assert response.status_code == 400, response.text
        row: Final = failure_rows(gateway, key, 1)[0]
        assert error_information(row)["error_code"] == "400", row
        assert number(row["spend"]) == 0.0, row


def test_a_budget_refusal_row_names_the_deployments_provider(gateway: Gateway) -> None:
    """A key with `max_budget` 1e-7 is served once and then refused 422. The refusal row's
    `error_information.llm_provider` is "openai": the auth handler resolves the provider of the requested model
    for a `BudgetExceededError` that carries none (`auth_exception_handler.py`, `resolve_llm_provider_for_rate_limit`)
    and `error_class` is "BudgetExceededError". UNVERIFIED: exact `error_class` spelling."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        served_id(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), 0.02)
        refused: Final = eventually(lambda: ask(gateway, model, key), lambda r: r.status_code == 422, seconds=70)
        assert refused.status_code == 422, refused.text
        information: Final = error_information(failure_rows(gateway, key, 1)[0])
        assert information["llm_provider"] == "openai", information
        assert information["error_class"] == "BudgetExceededError", information


def test_a_rate_limited_call_writes_a_429_failure_row_with_no_spend(gateway: Gateway) -> None:
    """A key with `rpm_limit` 1 is served once; a following call is refused 429 without reaching the provider.
    One failure row lands with `error_information.error_code` "429" and `spend` 0, and the provider saw exactly
    one call. Over-limit calls on a cached key are refused at auth before the body is read (#51428), so the row
    may lack the model; only code and spend are pinned."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        refused: Final = first_refusal(lambda: ask(gateway, model, key), 429)
        assert refused.status_code == 429
        row: Final = failure_rows(gateway, key, 1)[0]
        assert error_information(row)["error_code"] == "429", row
        assert number(row["spend"]) == 0.0, row
        assert len(provider_calls(wire)) >= 1


def test_a_rate_limited_call_does_not_add_to_the_key_spend(gateway: Gateway) -> None:
    """After one served call (0.02) and one 429 refusal on a key with `rpm_limit` 1, the key's stored spend is
    exactly 0.02 once the refusal row has landed: a refusal is never billed, whichever process writes it."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        first_refusal(lambda: ask(gateway, model, key), 429)
        failure_rows(gateway, key, 1)
        served: Final = len(provider_calls(wire))
        assert spend_reaches(gateway, "key", hashed(key), 0.02 * served) == pytest.approx(0.02 * served)


def test_a_rate_limited_call_counts_as_a_failed_request_in_the_daily_row(gateway: Gateway) -> None:
    """The same key's `LiteLLM_DailyUserSpend` counters, once projected, show `failed_requests` 1 and
    `successful_requests` equal to the served calls: the ledger projects failure rows into the daily tables
    (`accounting/projection.py:141`). UNVERIFIED: a refusal row without a model still gets a daily row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        first_refusal(lambda: ask(gateway, model, key), 429)
        served: Final = len(provider_calls(wire))
        counters: Final = eventually(
            lambda: daily_user_counters(gateway, key),
            lambda found: (
                found is not None and found["failed_requests"] == 1 and found["successful_requests"] == served
            ),
            seconds=70,
        )
        assert counters is not None and counters["api_requests"] == served + 1, counters


def test_a_guardrail_block_row_has_error_code_400_and_the_guardrail_message(gateway: Gateway) -> None:
    """A `custom_code` pre_call guardrail created through `POST /guardrails` returns `block(...)`; a chat naming
    it is refused 400 (`custom_code_guardrail.py:426`). Its failure row has `error_information.error_code` "400"
    and an `error_message` containing `dec1 spendlogs2 blocked input`, and the provider is never called."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        guardrail: Final = custom_code_guardrail(gateway, scenario, BLOCKING_CODE)
        assert guarded_ask(gateway, model, key, guardrail).status_code == 400
        provider_calls(wire)
        response: Final = ask(gateway, model, key, metadata={"guardrails": [guardrail]})
        assert response.status_code == 400, response.text
        information: Final = error_information(failure_rows(gateway, key, 2)[1])
        assert information["error_code"] == "400", information
        assert "dec1 spendlogs2 blocked input" in str(information["error_message"]), information
        assert provider_calls(wire) == ()


def test_a_guardrail_block_row_keeps_the_requested_model_group_and_call_type(gateway: Gateway) -> None:
    """The guardrail refusal happens after the logging object exists (guardrails run after request setup), so
    the row is not a pre-routing rejection: it carries `call_type` "acompletion" and `model_group` equal to the
    requested model, unlike an auth refusal."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        guardrail: Final = custom_code_guardrail(gateway, scenario, BLOCKING_CODE)
        assert guarded_ask(gateway, model, key, guardrail).status_code == 400
        row: Final = failure_rows(gateway, key, 1)[0]
        assert (row["call_type"], row["model_group"]) == ("acompletion", model), row


def test_a_refusal_row_is_filed_under_the_client_supplied_call_id(gateway: Gateway) -> None:
    """A guardrail refusal sent with `x-litellm-call-id: <id>` writes exactly one row whose `litellm_call_id`
    and `request_id` are both that id, since there is no provider response id to prefer (`get_spend_logs_id`
    falls back to `litellm_call_id`, spend_tracking_utils.py:408)."""
    call_id: Final = fresh_call_id()
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        guardrail: Final = custom_code_guardrail(gateway, scenario, BLOCKING_CODE)
        guarded_ask(gateway, model, key, guardrail)
        response: Final = ask(
            gateway, model, key, headers={"x-litellm-call-id": call_id}, metadata={"guardrails": [guardrail]}
        )
        assert response.status_code == 400, response.text
        row: Final = rows_with_call_id(gateway, call_id, 1)[0]
        assert row["request_id"] == call_id, row


def test_a_refused_request_never_increments_team_spend(gateway: Gateway) -> None:
    """A team key is refused by a guardrail three times; after the three failure rows land the team's stored
    spend equals 0.02 times the calls that were served while the new guardrail was still syncing (usually 0). Failure rows with zero cost are skipped by the counter
    step (`counted_rows` filters `row[0] > 0`, proxy_track_cost_callback.py:722)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team()
        key: Final = scenario.key(models=[model], team_id=team)
        guardrail: Final = custom_code_guardrail(gateway, scenario, BLOCKING_CODE)
        guarded_ask(gateway, model, key, guardrail)
        for _ in range(2):
            assert ask(gateway, model, key, metadata={"guardrails": [guardrail]}).status_code == 400
        failure_rows(gateway, key, 3)
        served: Final = len(provider_calls(wire))
        expected: Final = 0.02 * served
        stored: Final = eventually(
            lambda: number(
                gateway.rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))[0]["spend"]
            ),
            lambda spend: spend >= expected - 1e-9,
            seconds=70,
        )
        assert stored == pytest.approx(expected), (stored, served)
