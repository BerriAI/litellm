"""SpendLogs rows, entity spend and budgets for `/v1/messages` on every native Messages provider.

The Rust Messages path (anthropic, Azure AI Foundry Claude, Bedrock InvokeModel) and the next port (Vertex AI
Claude, draft #51294) must produce the same bookkeeping as Python did: one `anthropic_messages` SpendLogs row per
call whose `custom_llm_provider` names the provider (`bedrock` for InvokeModel), whose `request_id` is the
message `id`, whose tokens and spend come from the provider usage and the deployment prices, and an
`x-litellm-response-cost` header equal to that spend. The cost then lands on the key, its team, its owner and the
end user named by `metadata.user_id`. Budgets are enforced at auth by the Rust budget engine (#51535) before the
provider is called, so an over-budget key or team is refused with HTTP 422 in the Anthropic envelope
(`error.type: "invalid_request_error"`, `error.litellm_error_type: "budget_exceeded"`) whichever provider is
behind the model, and a failed provider call costs nothing.

Prices come from the deployment (`input_cost_per_token` 0.001, `output_cost_per_token` 0.002), and every
provider echo reports 11 input and 7 output tokens, so one call costs `11 * 0.001 + 7 * 0.002 = 0.025`.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, anthropic_error
from integration.messages_endpoint.dec1_msgprov2_support import (
    END_USER_SPEND,
    KEY_SPEND,
    NATIVE_PROVIDERS,
    SPEND_PROVIDER_NAME,
    TEAM_SPEND,
    USER_SPEND,
    hashed,
    native_deployment,
    native_error,
    spend_settles,
    stored_spend,
)
from integration.messages_endpoint.dec1_msgprov_support import (
    INPUT_COST,
    OUTPUT_COST,
    PRICES,
    only_spend_row,
    spend_of,
    unique_text,
)

ONE_CALL: Final = 11 * INPUT_COST + 7 * OUTPUT_COST


def _budget_refusal(response_status: int, error: dict[str, object], text: str) -> None:
    assert response_status == 422, text
    assert error["type"] == "invalid_request_error", text
    assert error["litellm_error_type"] == "budget_exceeded", text


# ---------------------------------------------------------------- SpendLogs rows


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_one_call_writes_one_row_named_after_the_provider(gateway: Gateway, provider: str) -> None:
    """One unary call writes exactly one SpendLogs row for the model group with `call_type: anthropic_messages`,
    `status: success`, `custom_llm_provider` equal to `anthropic`, `azure_ai`, `bedrock` or `vertex_ai`,
    `prompt_tokens: 11`, `completion_tokens: 7` and `spend: 0.025`."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["status"], row["custom_llm_provider"]) == (
            "anthropic_messages",
            "success",
            SPEND_PROVIDER_NAME[provider],
        )
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert spend_of(row) == pytest.approx(ONE_CALL)


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_the_row_request_id_is_the_message_id(gateway: Gateway, provider: str) -> None:
    """The row's `request_id` equals the `id` of the message the caller received, so a client can find the row of
    a response. On the Rust path the id comes from the provider's message; a port that mints its own id for the
    row breaks that join. UNVERIFIED for Bedrock and Vertex, where only Anthropic was confirmed by an existing
    test."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert only_spend_row(rig)["request_id"] == JSON_OBJECT.validate_json(response.content)["id"]


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_the_response_cost_header_equals_the_row_spend(gateway: Gateway, provider: str) -> None:
    """The `x-litellm-response-cost` header of a unary answer reads 0.025 and the row's spend is 0.025: the
    header and the ledger are computed from the same usage."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(ONE_CALL)
        assert spend_of(only_spend_row(rig)) == pytest.approx(ONE_CALL)


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_a_streamed_call_writes_one_row_with_the_streamed_usage(gateway: Gateway, provider: str) -> None:
    """A `stream: true` call read to the end writes exactly one row with `completion_tokens: 7` and spend 0.025,
    from `message_start` input and `message_delta` output usage (or Bedrock's chunk frames)."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        response, text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200, text
        row: Final = only_spend_row(rig)
        assert row["completion_tokens"] == 7
        assert spend_of(row) == pytest.approx(ONE_CALL)


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_a_provider_error_writes_a_failure_row_with_zero_spend(gateway: Gateway, provider: str) -> None:
    """A provider 429 is answered with HTTP 429 and writes one row with `status: failure`, the provider name and
    spend 0, so throttling is visible per provider without charging anyone."""
    reply: Final = native_error(provider, 429, "rate_limit_error", "synthetic slow down")
    with native_deployment(gateway, provider, lambda _r: reply, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 429
        row: Final = only_spend_row(rig)
        assert (row["status"], row["custom_llm_provider"]) == ("failure", SPEND_PROVIDER_NAME[provider])
        assert spend_of(row) == 0


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_the_row_carries_the_calling_key_hash_and_team(gateway: Gateway, provider: str) -> None:
    """A call made with a virtual key of a team writes a row whose `api_key` is the SHA-256 hex of the key and
    whose `team_id` is the team, on every native provider."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        team: Final = rig.scenario.team()
        key: Final = rig.scenario.key(team_id=team, models=[rig.model])
        assert rig.post(rig.body(), key=key).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["api_key"], row["team_id"]) == (hashed(key), team)


# ---------------------------------------------------------------- Entity spend


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_the_call_cost_lands_on_the_key_team_and_owner(gateway: Gateway, provider: str) -> None:
    """A key owned by a user and in a team, with no prior spend, makes one call. The key's, the team's and the
    user's stored `spend` each reach exactly 0.025."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        team: Final = rig.scenario.team()
        owner: Final = rig.scenario.member(team)
        key: Final = rig.scenario.key(team_id=team, user_id=owner, models=[rig.model])
        assert rig.post(rig.body(), key=key).status_code == 200
        assert spend_settles(gateway, KEY_SPEND, hashed(key), ONE_CALL) == pytest.approx(ONE_CALL)
        assert spend_settles(gateway, TEAM_SPEND, team, ONE_CALL) == pytest.approx(ONE_CALL)
        assert spend_settles(gateway, USER_SPEND, owner, ONE_CALL) == pytest.approx(ONE_CALL)


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_the_anthropic_metadata_user_id_is_charged_as_the_end_user(gateway: Gateway, provider: str) -> None:
    """A call with `metadata.user_id: <fresh id>` charges that end user: the `LiteLLM_EndUserTable` row for the id
    reaches spend 0.025. Anthropic clients name their end user this way, and the Rust path only receives
    `metadata.user_id` (#51355). UNVERIFIED: that end user spend is upserted without a prior `/customer/new`."""
    end_user: Final = unique_text("enduser").replace(" ", "-")
    with native_deployment(gateway, provider, **PRICES) as rig:
        assert rig.post(rig.body(metadata={"user_id": end_user})).status_code == 200
        assert spend_settles(gateway, END_USER_SPEND, end_user, ONE_CALL) == pytest.approx(ONE_CALL)


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_a_failed_call_leaves_the_key_spend_at_zero(gateway: Gateway, provider: str) -> None:
    """A provider 500 on a fresh key is answered with an error and the key's stored spend is still 0 once the
    failure row has landed: the budget hold is released and nothing is charged."""
    reply: Final = native_error(provider, 500, "api_error", "synthetic failure")
    with native_deployment(gateway, provider, lambda _r: reply, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=1.0)
        assert rig.post(rig.body(), key=key).status_code >= 500
        only_spend_row(rig)
        assert (stored_spend(gateway, KEY_SPEND, hashed(key)) or 0.0) == 0.0


# ---------------------------------------------------------------- Budgets


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_a_zero_budget_key_is_refused_before_the_provider(gateway: Gateway, provider: str) -> None:
    """A key with `max_budget: 0` is refused with HTTP 422 in the Anthropic envelope, `error.type:
    invalid_request_error` and `error.litellm_error_type: budget_exceeded`, and the provider peer receives no
    POST, whichever native provider serves the model."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=0.0)
        response: Final = rig.post(rig.body(), key=key)
        _budget_refusal(response.status_code, anthropic_error(response), response.text)
        assert rig.calls() == ()


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_a_zero_budget_team_with_no_spend_is_still_served(gateway: Gateway, provider: str) -> None:
    """A key of a team with `max_budget: 0` and no spend yet calls the provider. Pinned from CI run 44176682961:
    the team check refuses only when spend is strictly over the budget (`over`, `spend > max_budget`,
    litellm/rust/crates/callbacks-budgets/src/checks.rs:164-166 and :231-245 on main), unlike the key check, which
    refuses at or over (`at_or_over`, checks.rs:160-162). So 0 spend against a 0 budget is served with 200 and
    the provider is called once; a migration that switches the team check to `>=` would change this."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        team: Final = rig.scenario.team(max_budget=0.0)
        key: Final = rig.scenario.key(team_id=team, models=[rig.model])
        response: Final = rig.post(rig.body(), key=key)
        assert response.status_code == 200, response.text
        assert len(rig.calls()) == 1


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_a_key_is_refused_once_its_spend_passes_the_budget(gateway: Gateway, provider: str) -> None:
    """A key with `max_budget: 0.02` makes one call costing 0.025, which is served (the hold shrinks to what is
    left rather than refusing). Once the key's stored spend reads 0.025, the next call is refused with 422
    `budget_exceeded` and the provider sees only the first call."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=0.02)
        assert rig.post(rig.body(), key=key).status_code == 200
        spend_settles(gateway, KEY_SPEND, hashed(key), ONE_CALL)
        refused: Final = rig.post(rig.body(), key=key)
        _budget_refusal(refused.status_code, anthropic_error(refused), refused.text)
        assert len(rig.calls()) == 1


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_a_streamed_call_over_budget_is_a_json_error_not_an_event_stream(gateway: Gateway, provider: str) -> None:
    """A zero budget key asking for `stream: true` gets a plain HTTP 422 with `content-type: application/json` and
    the Anthropic envelope, not a 200 SSE stream with an error event, because the refusal happens at auth."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=0.0)
        response: Final = rig.post(rig.body(stream=True), key=key)
        assert response.status_code == 422, response.text
        assert response.headers["content-type"].startswith("application/json")
        assert (
            object_value(JSON_OBJECT.validate_json(response.content)["error"])["litellm_error_type"]
            == "budget_exceeded"
        )
        assert rig.calls() == ()


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_count_tokens_is_not_charged_to_the_key(gateway: Gateway, provider: str) -> None:
    """A `/v1/messages/count_tokens` call followed by one `/v1/messages` call leaves the key's stored spend at
    exactly 0.025: counting is never billed, whether the provider counts remotely or locally. UNVERIFIED for
    Bedrock and Azure AI, whose count calls reach the peer echo, which answers a message; the count answer itself
    is not asserted here."""
    with native_deployment(gateway, provider, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=1.0)
        count_body: Final = {"model": rig.model, "messages": [{"role": "user", "content": unique_text()}]}
        rig.post(count_body, key=key, path="/v1/messages/count_tokens")
        assert rig.post(rig.body(), key=key).status_code == 200
        assert spend_settles(gateway, KEY_SPEND, hashed(key), ONE_CALL) == pytest.approx(ONE_CALL)
