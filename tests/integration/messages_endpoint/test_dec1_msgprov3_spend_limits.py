"""Spend, budgets and rate limits on `/v1/messages` for bridged providers, and rate limits for every provider.

`test_dec1_msgprov2_native_spend_budgets.py` pins SpendLogs rows, entity spend and budget refusals for the native
Messages providers (anthropic, azure_ai, bedrock_invoke, vertex_ai). The bridged providers (Bedrock Converse,
Gemini, hosted vLLM, OpenAI through the Responses API) reach their upstream through the chat bridge
(`litellm/llms/anthropic/messages/adapters/handler.py`), so their usage is translated twice before it is billed;
these tests pin that the bookkeeping is still one `anthropic_messages` row per call named after the real provider,
that the cost lands on the key, team and end user, and that budgets refuse before the provider is called (the Rust
budget engine admits at auth, #51535: HTTP 422, `litellm_error_type: budget_exceeded`).

Rate limits are provider independent in the limiter (`litellm/proxy/hooks/parallel_request_limiter_v3.py`), but the
token window is settled from the usage each provider reports, so the settled count is the place a per-provider
usage translation bug would show. A key over its `rpm_limit` is refused at auth before the body is read (#51428),
so the provider is never called.

Every echo reports 11 input and 7 output tokens; with the deployment prices (0.001 in, 0.002 out) one call costs
0.025 and settles 18 tokens.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import Rig, anthropic_error
from integration.messages_endpoint.dec1_msgprov2_support import (
    END_USER_SPEND,
    KEY_SPEND,
    NATIVE_PROVIDERS,
    TEAM_SPEND,
    hashed,
    native_deployment,
    rows_for_group,
    spend_settles,
    stored_spend,
)
from integration.messages_endpoint.dec1_msgprov3_support import (
    BRIDGED_PROVIDERS,
    BRIDGED_SPEND_PROVIDER,
    bridged_deployment,
)
from integration.messages_endpoint.dec1_msgprov_support import (
    INPUT_COST,
    OUTPUT_COST,
    PRICES,
    aws_error,
    only_spend_row,
    openai_error_reply,
    spend_of,
    unique_text,
)
from integration.providers.dec1_gemini_support import reply_json
from redis import Redis

ONE_CALL: Final = 11 * INPUT_COST + 7 * OUTPUT_COST
ALL_PROVIDERS: Final = (*NATIVE_PROVIDERS, *BRIDGED_PROVIDERS)
STREAMING_BRIDGED: Final = ("bedrock_converse", "gemini", "hosted_vllm")
"""Bridged providers whose round-one echo answers a stream; the OpenAI Responses echo here is unary only."""


def _deployment(
    gateway: Gateway, provider: str, respond: Callable[[Request], Reply] | None = None, **params: object
) -> AbstractContextManager[Rig]:
    if provider in NATIVE_PROVIDERS:
        return native_deployment(gateway, provider, respond, **params)  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough
    return bridged_deployment(gateway, provider, respond, **params)  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def _server_error(provider: str) -> Reply:
    """A 500 in the provider's own wire format."""
    match provider:
        case "bedrock_converse":
            return aws_error(500, "InternalServerException", "synthetic failure")
        case "gemini":
            return reply_json(
                {"error": {"code": 500, "message": "synthetic failure", "status": "INTERNAL"}}, status=500
            )
        case _:
            return openai_error_reply(500, "synthetic failure", "server_error")


def _token_counter(key: str) -> int | None:
    """The key's settled token count, read from the shared Redis at `{api_key:<hash>}:tokens` (the layout of
    `litellm/rust/crates/redis-store/src/rate_limits/keys.rs`, as `authorization/dec1_limits_support.py` reads it)."""
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as redis:
        value: Final = redis.get(f"{{api_key:{hashed(key)}}}:tokens")
    return None if value is None else int(float(str(value)))


def _budget_refusal(status: int, error: dict[str, object], text: str) -> None:
    assert status == 422, text
    assert error["type"] == "invalid_request_error", text
    assert error["litellm_error_type"] == "budget_exceeded", text


# ---------------------------------------------------------------- bridged SpendLogs rows and entity spend


@pytest.mark.parametrize("provider", BRIDGED_PROVIDERS)
def test_one_bridged_call_writes_one_messages_row_named_after_the_real_provider(
    gateway: Gateway, provider: str
) -> None:
    """One unary bridged call writes exactly one SpendLogs row with `call_type: anthropic_messages`, `status:
    success` and `custom_llm_provider` naming the provider behind the bridge (`bedrock`, `gemini`, `hosted_vllm`,
    `openai`), not `anthropic`: the bridge logs under the Messages call type but keeps the deployment's provider."""
    with _deployment(gateway, provider, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["status"], row["custom_llm_provider"]) == (
            "anthropic_messages",
            "success",
            BRIDGED_SPEND_PROVIDER[provider],
        )


@pytest.mark.parametrize("provider", BRIDGED_PROVIDERS)
def test_the_bridged_cost_header_equals_the_row_spend(gateway: Gateway, provider: str) -> None:
    """The bridged answer's `x-litellm-response-cost` is 0.025 (11 * 0.001 + 7 * 0.002) and the row's `spend`,
    `prompt_tokens` and `completion_tokens` are 0.025, 11 and 7: the chat usage translated back to Anthropic usage is
    the usage billed."""
    with _deployment(gateway, provider, **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(ONE_CALL)
        row: Final = only_spend_row(rig)
        assert (spend_of(row), row["prompt_tokens"], row["completion_tokens"]) == (pytest.approx(ONE_CALL), 11, 7)


@pytest.mark.parametrize("provider", STREAMING_BRIDGED)
def test_a_streamed_bridged_call_is_billed_once_from_the_streamed_usage(gateway: Gateway, provider: str) -> None:
    """A `stream: true` bridged call writes one row whose spend is 0.025 and whose tokens are 11 and 7, the usage of
    the provider's last stream frame (Converse `metadata`, Gemini `usageMetadata`, the OpenAI usage chunk)."""
    with _deployment(gateway, provider, **PRICES) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        row: Final = only_spend_row(rig)
        assert (spend_of(row), row["prompt_tokens"], row["completion_tokens"]) == (pytest.approx(ONE_CALL), 11, 7)


@pytest.mark.parametrize("provider", BRIDGED_PROVIDERS)
def test_the_bridged_call_cost_lands_on_the_key_and_its_team(gateway: Gateway, provider: str) -> None:
    """A team key with no prior spend makes one bridged call; the key's and the team's stored `spend` each settle
    at exactly 0.025."""
    with _deployment(gateway, provider, **PRICES) as rig:
        team: Final = rig.scenario.team()
        key: Final = rig.scenario.key(team_id=team, models=[rig.model])
        assert rig.post(rig.body(), key=key).status_code == 200
        assert spend_settles(gateway, KEY_SPEND, hashed(key), ONE_CALL) == pytest.approx(ONE_CALL)
        assert spend_settles(gateway, TEAM_SPEND, team, ONE_CALL) == pytest.approx(ONE_CALL)


@pytest.mark.parametrize("provider", BRIDGED_PROVIDERS)
def test_the_anthropic_metadata_user_id_is_charged_as_the_end_user_on_a_bridged_call(
    gateway: Gateway, provider: str
) -> None:
    """`metadata: {"user_id": <fresh id>}` on a bridged call charges that end user 0.025 in `LiteLLM_EndUserTable`,
    the same as on native providers. The bridge maps `metadata.user_id` to the chat `user`, so a port that drops
    `metadata` before the bridge would lose end-user spend. UNVERIFIED: the end user row is upserted without a prior
    `/customer/new` (same assumption as the native test)."""
    end_user: Final = unique_text("enduser").replace(" ", "-")
    with _deployment(gateway, provider, **PRICES) as rig:
        assert rig.post(rig.body(metadata={"user_id": end_user})).status_code == 200
        assert spend_settles(gateway, END_USER_SPEND, end_user, ONE_CALL) == pytest.approx(ONE_CALL)


@pytest.mark.parametrize("provider", BRIDGED_PROVIDERS)
def test_a_failed_bridged_call_writes_a_failure_row_and_charges_nothing(gateway: Gateway, provider: str) -> None:
    """A provider 500 on a fresh key with `max_budget: 1.0` is answered with an error, at least one `failure` row
    lands for the model group with spend 0, and the key's stored spend is still 0 (or unset): the budget hold is
    released and nothing is charged. UNVERIFIED: how many attempts the shared proxy's retry policy makes, so only
    the status of every row is checked."""
    reply: Final = _server_error(provider)
    with _deployment(gateway, provider, lambda _request: reply, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=1.0)
        assert rig.post(rig.body(), key=key).status_code >= 500
        rows: Final = rows_for_group(rig.model, 1)
        assert {row["status"] for row in rows} == {"failure"}, rows
        assert all(spend_of(row) == 0 for row in rows), rows
        assert (stored_spend(gateway, KEY_SPEND, hashed(key)) or 0.0) == 0.0


# ---------------------------------------------------------------- bridged budgets


@pytest.mark.parametrize("provider", BRIDGED_PROVIDERS)
def test_a_zero_budget_key_is_refused_before_a_bridged_provider(gateway: Gateway, provider: str) -> None:
    """A key with `max_budget: 0` is refused with HTTP 422 in the Anthropic envelope (`invalid_request_error`,
    `litellm_error_type: budget_exceeded`) and the bridged provider is never called."""
    with _deployment(gateway, provider, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=0.0)
        response: Final = rig.post(rig.body(), key=key)
        _budget_refusal(response.status_code, anthropic_error(response), response.text)
        assert rig.calls() == ()


@pytest.mark.parametrize("provider", BRIDGED_PROVIDERS)
def test_a_zero_budget_team_with_no_spend_is_still_served_on_a_bridged_provider(
    gateway: Gateway, provider: str
) -> None:
    """A key of a team with `max_budget: 0` and no spend yet calls the bridged provider. Pinned from CI run 44176682961:
    the team check refuses only when spend is strictly over the budget (`over`, `spend > max_budget`,
    litellm/rust/crates/callbacks-budgets/src/checks.rs:164-166 and :231-245 on main), unlike the key check, which
    refuses at or over (`at_or_over`, checks.rs:160-162). So 0 spend against a 0 budget is served with 200 and
    the bridged provider is called once; a migration that switches the team check to `>=` would change this."""
    with _deployment(gateway, provider, **PRICES) as rig:
        team: Final = rig.scenario.team(max_budget=0.0)
        key: Final = rig.scenario.key(team_id=team, models=[rig.model])
        response: Final = rig.post(rig.body(), key=key)
        assert response.status_code == 200, response.text
        assert len(rig.calls()) == 1


@pytest.mark.parametrize("provider", BRIDGED_PROVIDERS)
def test_a_key_is_refused_on_a_bridged_provider_once_its_spend_passes_the_budget(
    gateway: Gateway, provider: str
) -> None:
    """A key with `max_budget: 0.02` makes one bridged call costing 0.025, which is served; once its stored spend
    reads 0.025 the next call is refused with 422 `budget_exceeded` and the provider saw only the first call."""
    with _deployment(gateway, provider, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=0.02)
        assert rig.post(rig.body(), key=key).status_code == 200
        spend_settles(gateway, KEY_SPEND, hashed(key), ONE_CALL)
        refused: Final = rig.post(rig.body(), key=key)
        _budget_refusal(refused.status_code, anthropic_error(refused), refused.text)
        assert len(rig.calls()) == 1


# ---------------------------------------------------------------- rate limits on every provider


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_a_key_over_its_rpm_limit_is_refused_before_the_provider(gateway: Gateway, provider: str) -> None:
    """A key with `rpm_limit: 1` makes two calls in one minute. The first is served, the second is HTTP 429 and the
    provider saw exactly one call, whichever provider serves the model (native Rust, native Python or bridged). The
    second is refused at auth before the body is read (#51428), so the body shape of the 429 is not pinned here
    (`authorization/test_dec1_limits_endpoints.py` covers it for Anthropic)."""
    with _deployment(gateway, provider, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], rpm_limit=1)
        assert rig.post(rig.body(), key=key).status_code == 200
        refused: Final = rig.post(rig.body(), key=key)
        assert refused.status_code == 429, refused.text
        assert len(rig.calls()) == 1


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_the_key_token_window_settles_at_the_provider_usage(gateway: Gateway, provider: str) -> None:
    """A key with `tpm_limit: 100000` makes one call answered with 11 input and 7 output tokens; its Redis token
    counter `{api_key:<hash>}:tokens` settles at 18 (`litellm/rust/crates/redis-store/src/rate_limits/keys.rs`; settled from the reported usage, `parallel_request_limiter_v3.py`). A provider whose usage is lost in
    translation would settle at the reservation (input estimate plus `max_tokens`) instead."""
    with _deployment(gateway, provider, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], tpm_limit=100000)
        assert rig.post(rig.body(max_tokens=64), key=key).status_code == 200
        assert eventually(lambda: _token_counter(key), lambda value: value == 18, seconds=20) == 18
