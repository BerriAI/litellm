"""What one provider pass-through call writes to `LiteLLM_SpendLogs`, and what the caller receives byte for byte.

The proxy forwards the caller's body untouched, relays the provider's answer untouched (a stream frame for frame),
and afterwards parses the provider's native response to price it (`success_handler.py:263`, one handler per
provider). Each served call must leave exactly one row with `call_type = pass_through_endpoint`, attributed to
the caller's key hash and team, with the provider, the model and the token counts read from the native usage
block: Anthropic `usage.input_tokens`/`output_tokens` (split across `message_start` and `message_delta` when
streamed), Cohere v2 `usage.tokens` (not `billed_units`, `v2_transformation.py:248`) and OpenAI `usage`.
Spend is the cost-map price of those tokens, or the deployment's own price when the Anthropic body names a model
served by exactly one router deployment (`anthropic_deployment_pricing_model_info`).

These rows are what the Rust spend-logging rewrite must keep producing for pass-through traffic, which the
sibling spend agents left out.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.wire import Wire, wire_server
from integration.providers.dec1_passthru_support import (
    ANTHROPIC_INPUT_RATE,
    ANTHROPIC_INPUT_TOKENS,
    ANTHROPIC_MODEL,
    ANTHROPIC_OUTPUT_RATE,
    ANTHROPIC_OUTPUT_TOKENS,
    COHERE_INPUT_RATE,
    COHERE_MODEL,
    COHERE_OUTPUT_RATE,
    COHERE_TOKENS,
    FAIL_MARKER,
    OPENAI_MODEL,
    OPENAI_USAGE,
    PASS_THROUGH_CALL_TYPE,
    PROVIDER_FAILURE,
    anthropic_body,
    anthropic_frames,
    bearer,
    cohere_body,
    digest,
    marker,
    number,
    openai_body,
    openai_frames,
    provider_proxy,
    scripted_provider,
    send,
    spend_row,
    spend_rows,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(300)

_GPT_4O_MINI_RATES: Final = (1.5e-07, 6e-07)


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(scripted_provider) as served:
        yield served


@pytest.fixture(scope="module")
def proxy(wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with (
        gateway_from_environment() as shared,
        provider_proxy(shared, tmp_path_factory.mktemp("dec1-passthru-spend"), wire) as candidate,
    ):
        yield candidate


@pytest.fixture(autouse=True)
def _drained(wire: Wire) -> None:
    wire.drain()


def _columns(row: dict[str, JsonValue], *names: str) -> dict[str, JsonValue]:
    return {name: row[name] for name in names}


def test_an_anthropic_call_writes_one_row_with_the_native_token_counts_and_price(proxy: Gateway) -> None:
    """One `/anthropic/v1/messages` call whose response reports `input_tokens: 7` and `output_tokens: 3` on
    `claude-sonnet-4-5` must write one row with `call_type` `pass_through_endpoint`, `custom_llm_provider`
    `anthropic`, `model` `claude-sonnet-4-5`, `prompt_tokens` 7, `completion_tokens` 3, `status` `success` and
    `spend` 7 * 3e-06 + 3 * 1.5e-05."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        row: Final = spend_row(proxy, key)
        assert _columns(
            row, "call_type", "custom_llm_provider", "model", "prompt_tokens", "completion_tokens", "status"
        ) == {
            "call_type": PASS_THROUGH_CALL_TYPE,
            "custom_llm_provider": "anthropic",
            "model": ANTHROPIC_MODEL,
            "prompt_tokens": ANTHROPIC_INPUT_TOKENS,
            "completion_tokens": ANTHROPIC_OUTPUT_TOKENS,
            "status": "success",
        }, row
        assert number(row["spend"]) == pytest.approx(
            ANTHROPIC_INPUT_TOKENS * ANTHROPIC_INPUT_RATE + ANTHROPIC_OUTPUT_TOKENS * ANTHROPIC_OUTPUT_RATE
        )


def test_a_streamed_anthropic_call_is_relayed_byte_for_byte(proxy: Gateway) -> None:
    """A `stream: true` call to `/anthropic/v1/messages` must answer 200 `text/event-stream` whose body is exactly
    the six SSE frames the provider sent, concatenated, with nothing added (no `[DONE]`, no cost injection unless
    asked) and nothing re-serialised."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        with proxy.client.stream(
            "POST", "/anthropic/v1/messages", json=anthropic_body(marker(), stream=True), headers=bearer(key)
        ) as response:
            body: Final = response.read()
            assert response.status_code == 200, body
            assert response.headers["content-type"].startswith("text/event-stream"), dict(response.headers)
        assert body == b"".join(anthropic_frames("scripted anthropic")), body


def test_a_streamed_anthropic_call_bills_input_from_message_start_and_output_from_message_delta(
    proxy: Gateway,
) -> None:
    """The same streamed call reports `input_tokens: 7` in `message_start` and the final `output_tokens: 3` in
    `message_delta` (with a provisional `output_tokens: 1` in `message_start`). The row must record 7 and 3, not
    the provisional 1, and the same spend as the non-streamed call."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        with proxy.client.stream(
            "POST", "/anthropic/v1/messages", json=anthropic_body(marker(), stream=True), headers=bearer(key)
        ) as response:
            response.read()
            assert response.status_code == 200
        row: Final = spend_row(proxy, key)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (ANTHROPIC_INPUT_TOKENS, ANTHROPIC_OUTPUT_TOKENS)
        assert number(row["spend"]) == pytest.approx(
            ANTHROPIC_INPUT_TOKENS * ANTHROPIC_INPUT_RATE + ANTHROPIC_OUTPUT_TOKENS * ANTHROPIC_OUTPUT_RATE
        )


def test_a_cohere_call_bills_usage_tokens_not_billed_units(proxy: Gateway) -> None:
    """The Cohere v2 reply carries `usage.tokens` {13, 5} and different `usage.billed_units` {2, 1}. The row must
    record `prompt_tokens` 13 and `completion_tokens` 5, priced at `command-a-03-2025` rates (2.5e-06 in, 1e-05
    out), with `model` `command-a-03-2025` and `call_type` `pass_through_endpoint`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key)).status_code == 200
        row: Final = spend_row(proxy, key)
        assert _columns(row, "call_type", "model", "prompt_tokens", "completion_tokens") == {
            "call_type": PASS_THROUGH_CALL_TYPE,
            "model": COHERE_MODEL,
            "prompt_tokens": COHERE_TOKENS[0],
            "completion_tokens": COHERE_TOKENS[1],
        }, row
        assert number(row["spend"]) == pytest.approx(
            COHERE_TOKENS[0] * COHERE_INPUT_RATE + COHERE_TOKENS[1] * COHERE_OUTPUT_RATE
        )


def test_a_cohere_row_names_cohere_as_its_provider(proxy: Gateway) -> None:
    """The same Cohere call's row must carry a Cohere provider name in `custom_llm_provider`. UNVERIFIED: the
    handler's `llm_provider_name` is `LlmProviders.COHERE` (`cohere`), while the cost map files the model under
    `cohere_chat`; this pins `cohere`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key)).status_code == 200
        assert spend_row(proxy, key)["custom_llm_provider"] == "cohere"


def test_an_openai_call_writes_a_row_priced_from_the_openai_usage_block(proxy: Gateway) -> None:
    """One `/openai/v1/chat/completions` call on `gpt-4o-mini` whose reply reports 5 prompt and 3 completion
    tokens must write one row with `custom_llm_provider` `openai`, `model` `gpt-4o-mini`, tokens 5 and 3 and spend
    5 * 1.5e-07 + 3 * 6e-07."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert send(proxy, "/openai/v1/chat/completions", openai_body(marker()), headers=bearer(key)).status_code == 200
        row: Final = spend_row(proxy, key)
        assert _columns(row, "custom_llm_provider", "model", "prompt_tokens", "completion_tokens") == {
            "custom_llm_provider": "openai",
            "model": OPENAI_MODEL,
            "prompt_tokens": OPENAI_USAGE[0],
            "completion_tokens": OPENAI_USAGE[1],
        }, row
        assert number(row["spend"]) == pytest.approx(
            OPENAI_USAGE[0] * _GPT_4O_MINI_RATES[0] + OPENAI_USAGE[1] * _GPT_4O_MINI_RATES[1]
        )


@pytest.mark.parametrize("route", ["/openai/v1/chat/completions", "/mistral/v1/chat/completions"])
def test_a_streamed_openai_shaped_call_is_relayed_byte_for_byte_with_its_done_sentinel(
    proxy: Gateway, route: str
) -> None:
    """A `stream: true` call to the OpenAI or Mistral pass-through route must answer with exactly the provider's
    three frames, the last being `data: [DONE]`. Pass-through never rewrites chunks or adds the proxy's own
    `[DONE]`."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        with proxy.client.stream(
            "POST", route, json=openai_body(marker(), stream=True), headers=bearer(key)
        ) as response:
            body: Final = response.read()
            assert response.status_code == 200, body
        assert body == b"".join(openai_frames("scripted openai", OPENAI_MODEL)), body


def test_a_mistral_call_writes_one_success_row_for_the_key(proxy: Gateway) -> None:
    """The success handler has no Mistral parser, so a `/mistral/v1/chat/completions` call is logged as a generic
    pass-through. It must still write exactly one row for the key with `call_type` `pass_through_endpoint` and
    `status` `success`. UNVERIFIED: the generic row's `model`, tokens and spend; this pins spend 0."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert (
            send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(key)).status_code == 200
        )
        row: Final = spend_row(proxy, key)
        assert (row["call_type"], row["status"]) == (PASS_THROUGH_CALL_TYPE, "success"), row
        assert number(row["spend"]) == 0.0, row


def test_the_row_is_attributed_to_the_keys_team(proxy: Gateway) -> None:
    """A key of a team calls `/anthropic/v1/messages`. The row's `team_id` must be that team, so team spend reports
    include pass-through traffic."""
    with proxy.scenario() as scenario:
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        assert spend_row(proxy, key)["team_id"] == team


def test_the_rows_request_id_is_the_call_id_the_response_announced(proxy: Gateway) -> None:
    """A `/cohere/v2/chat` call answers with an `x-litellm-call-id` header. The single row for the key must have
    that value as `request_id`, so a caller can find its own row. UNVERIFIED: assumes pass-through keys the row by
    the call id rather than by the provider's response id."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        assert response.status_code == 200, response.text
        assert spend_row(proxy, key)["request_id"] == response.headers["x-litellm-call-id"]


def test_the_x_litellm_tags_header_lands_in_the_rows_request_tags(proxy: Gateway) -> None:
    """A `/cohere/v2/chat` call carries `x-litellm-tags: dec1-a,dec1-b`. The row's `request_tags` must contain both
    tags; the header is read on pass-through routes even though the body is never touched."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        headers: Final = {**bearer(key), "x-litellm-tags": "dec1-a,dec1-b"}
        assert send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=headers).status_code == 200
        tags: Final = spend_row(proxy, key)["request_tags"]
        parsed: Final = json.loads(tags) if isinstance(tags, str) else tags
        assert isinstance(parsed, list) and {"dec1-a", "dec1-b"} <= set(parsed), tags


def test_a_provider_error_is_relayed_unchanged_and_logged_as_a_failure(proxy: Gateway) -> None:
    """The provider answers a `/cohere/v2/chat` call with 500 and a JSON error body. The caller must receive 500
    and exactly that body, and the key's row must have `status` `failure` with spend 0. UNVERIFIED: the failure
    row's columns for provider routes; `spend/test_passthrough_failure.py` pins the same for configured
    endpoints."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker(FAIL_MARKER)), headers=bearer(key))
        assert (response.status_code, response.content) == (500, PROVIDER_FAILURE), response.text
        row: Final = spend_row(proxy, key)
        assert row["status"] == "failure", row
        assert number(row["spend"]) == 0.0, row


def test_a_body_model_served_by_one_priced_anthropic_deployment_bills_that_deployments_price(proxy: Gateway) -> None:
    """A router group `dec1-claude-<hex>` has one deployment `anthropic/dec1-claude-<hex>` with
    `input_cost_per_token: 0.001` and `output_cost_per_token: 0.002`. An `/anthropic/v1/messages` call whose body
    names `dec1-claude-<hex>` reports 7 and 3 tokens. Its row must be priced 7 * 0.001 + 3 * 0.002 = 0.013: the
    pass-through selects no deployment, but the only deployment serving that exact Anthropic model supplies the
    price (`anthropic_deployment_pricing_model_info`, `llm_passthrough_endpoints.py:296`)."""
    name: Final = f"dec1-claude-{uuid.uuid4().hex[:10]}"
    with proxy.scenario() as scenario:
        scenario.model(name=name, model=f"anthropic/{name}", input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key()
        response: Final = send(
            proxy, "/anthropic/v1/messages", anthropic_body(marker(), model=name), headers=bearer(key)
        )
        assert response.status_code == 200, response.text
        assert number(spend_row(proxy, key)["spend"]) == pytest.approx(7 * 0.001 + 3 * 0.002)


def test_a_body_model_served_by_two_anthropic_deployments_does_not_borrow_either_price(proxy: Gateway) -> None:
    """The same unique model is served by two deployments priced 0.001 and 0.005 per input token. Neither price may
    be used, because any pick could bill another deployment's negotiated price; the model is unknown to the cost
    map, so the row's spend must be 0. UNVERIFIED: whether an unpriced model writes spend 0 or a failure row."""
    name: Final = f"dec1-claude-{uuid.uuid4().hex[:10]}"
    with proxy.scenario() as scenario:
        for rate in (0.001, 0.005):
            scenario.model(name=name, model=f"anthropic/{name}", input_cost_per_token=rate, output_cost_per_token=rate)
        key: Final = scenario.key()
        response: Final = send(
            proxy, "/anthropic/v1/messages", anthropic_body(marker(), model=name), headers=bearer(key)
        )
        assert response.status_code == 200, response.text
        assert number(spend_row(proxy, key)["spend"]) == 0.0


def test_two_calls_write_two_rows_with_distinct_request_ids(proxy: Gateway) -> None:
    """Two `/anthropic/v1/messages` calls by one key must write two rows with different `request_id`s, both hashed
    to the same `api_key`. Pass-through rows are never merged or deduplicated by body."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        body: Final = anthropic_body("dec1 identical body")
        for _ in range(2):
            assert send(proxy, "/anthropic/v1/messages", body, headers=bearer(key)).status_code == 200
        rows: Final = eventually(lambda: spend_rows(proxy, key), lambda found: len(found) == 2, seconds=70)
        assert len({row["request_id"] for row in rows}) == 2, rows
        assert digest(key) not in {str(row["request_id"]) for row in rows}
