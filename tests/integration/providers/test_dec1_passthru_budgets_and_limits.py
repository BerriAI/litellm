"""Budgets and rate limits on the built-in provider pass-through routes.

A pass-through call is authenticated like any LLM call, so the key, team and model budget checks and the v3
rate limiter (`parallel_request_limiter_v3.py`) apply to `/anthropic/*`, `/cohere/*`, `/mistral/*`, `/gemini/*`
and `/openai/*` exactly as to `/v1/chat/completions`. A refused call never reaches the provider. What differs is
the envelope and how the spend that closes a budget is learned: the proxy forwards the caller's body untouched
and prices the provider's native response afterwards (`PassThroughEndpointLogging`), so spend lands
asynchronously and a budget closes one settled call later.

Refusal shapes pinned here (both are moving to Rust: budgets in #51535, the limiter as a CustomLoggerV2
callback): a budget refusal is 422 with `type` `budget_exceeded` and `Budget has been exceeded! <Entity>=...`;
a rate-limit refusal is 429 with `type` `throttling_error` and `Rate limit exceeded for <scope>: <value>. Limit
type: <kind>. Current limit: <n>, Remaining: 0. Limit resets at: ...`. On `/anthropic/*` both arrive in the
Anthropic envelope (`{"type": "error", "error": {...}}`), where 422 becomes `invalid_request_error` and 429
becomes `rate_limit_error`, with the LiteLLM type kept in `litellm_error_type` (`anthropic_error_payload.py:84`).

Lesson from earlier runs: over-limit requests on a cached key are refused at auth before the body is read
(#51428), so the second refusal of a burst may carry no `x-litellm-call-id`; nothing here depends on it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.wire import Wire, wire_server
from integration.authorization.dec1_limits_support import RESET_AT, hashed, limit_message
from integration.providers.dec1_passthru_support import (
    ANTHROPIC_INPUT_RATE,
    ANTHROPIC_INPUT_TOKENS,
    ANTHROPIC_OUTPUT_RATE,
    ANTHROPIC_OUTPUT_TOKENS,
    HOLD_MARKER,
    SPEND_SECONDS,
    anthropic_body,
    anthropic_error,
    bearer,
    cohere_body,
    gemini_body,
    gemini_path,
    hold_in_background,
    key_spend,
    marker,
    message_of,
    openai_body,
    openai_error,
    posts,
    provider_proxy,
    refused_until,
    scripted_provider,
    send,
    wait_for_posts,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(300)

_ANTHROPIC_COST: Final = ANTHROPIC_INPUT_TOKENS * ANTHROPIC_INPUT_RATE + ANTHROPIC_OUTPUT_TOKENS * ANTHROPIC_OUTPUT_RATE
_TINY_BUDGET: Final = 1e-05
"""Smaller than one scripted Anthropic call (6.6e-05), so a single settled call closes the budget."""


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(scripted_provider) as served:
        yield served


@pytest.fixture(scope="module")
def proxy(wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with (
        gateway_from_environment() as shared,
        provider_proxy(shared, tmp_path_factory.mktemp("dec1-passthru-limits"), wire) as candidate,
    ):
        yield candidate


@pytest.fixture(autouse=True)
def _drained(wire: Wire) -> None:
    wire.drain()


def _assert_limited_message(message: str, scope: str, value: str, kind: str, limit: int) -> None:
    prefix: Final = limit_message(scope, value, kind, limit)
    assert message.startswith(prefix), message
    assert RESET_AT.match(message.removeprefix(prefix)), message


def test_a_zero_budget_key_is_refused_on_the_anthropic_route_in_the_anthropic_envelope(
    proxy: Gateway, wire: Wire
) -> None:
    """A key with `max_budget: 0` posts to `/anthropic/v1/messages`. The answer must be 422 with `{"type":
    "error", "error": {"type": "invalid_request_error", "litellm_error_type": "budget_exceeded", "message":
    "Budget has been exceeded! Key=key (sk-...<last 4>) ..."}}` and the provider must receive nothing."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(max_budget=0)
        response: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        error: Final = anthropic_error(response, 422)
        assert (error["type"], error.get("litellm_error_type")) == ("invalid_request_error", "budget_exceeded")
        assert message_of(error).startswith(f"Budget has been exceeded! Key=key (sk-...{key[-4:]})"), response.text
        assert posts(wire) == ()


@pytest.mark.parametrize(
    ("route", "body", "headers_for"),
    [
        pytest.param("/cohere/v2/chat", cohere_body("dec1"), "bearer", id="cohere"),
        pytest.param("/mistral/v1/chat/completions", openai_body("dec1"), "bearer", id="mistral"),
        pytest.param("/openai/v1/chat/completions", openai_body("dec1"), "bearer", id="openai"),
        pytest.param(gemini_path(), gemini_body("dec1"), "goog", id="gemini"),
    ],
)
def test_a_zero_budget_key_is_refused_on_every_other_provider_route_in_the_openai_envelope(
    proxy: Gateway, wire: Wire, route: str, body: dict[str, JsonValue], headers_for: str
) -> None:
    """A key with `max_budget: 0` calls a Cohere, Mistral, OpenAI or Gemini pass-through route. Each must answer
    422 with `error.type` `budget_exceeded`, `error.code` `"422"` and a message starting `Budget has been
    exceeded!`, and the provider must receive nothing. Gemini authenticates through `x-goog-api-key`, so the
    budget check runs inside the route's own auth call and must refuse the same way."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(max_budget=0)
        headers: Final = bearer(key) if headers_for == "bearer" else {"x-goog-api-key": key}
        response: Final = send(proxy, route, body, headers=headers)
        error: Final = openai_error(response, 422)
        assert (error["type"], error["code"]) == ("budget_exceeded", "422"), response.text
        assert message_of(error).startswith("Budget has been exceeded!"), response.text
        assert posts(wire) == ()


def test_anthropic_pass_through_spend_reaches_the_keys_spend(proxy: Gateway) -> None:
    """One `/anthropic/v1/messages` call whose native response reports 7 input and 3 output tokens on
    `claude-sonnet-4-5` must raise the key's `/key/info` spend to 7 * 3e-06 + 3 * 1.5e-05 = 6.6e-05. This is the
    figure a key budget is compared against."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        spent: Final = eventually(lambda: key_spend(proxy, key), lambda value: value > 0, seconds=SPEND_SECONDS)
        assert spent == pytest.approx(_ANTHROPIC_COST)


def test_a_key_budget_closes_after_pass_through_spend_settles(proxy: Gateway, wire: Wire) -> None:
    """A key has `max_budget: 1e-05`. Its first Anthropic pass-through call is served. Once that call's 6.6e-05
    has settled, the next call must be refused with 422 `budget_exceeded`, and the provider must have received only
    the calls that were served before the refusal."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(max_budget=_TINY_BUDGET)
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        refused: Final = refused_until(
            lambda: send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key)), 422
        )
        assert openai_error(refused, 422)["type"] == "budget_exceeded", refused.text
        assert len(posts(wire)) >= 1
        assert send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key)).status_code == 422
        assert posts(wire) == ()


def test_a_team_budget_closes_for_every_team_key_after_pass_through_spend(proxy: Gateway, wire: Wire) -> None:
    """A team has `max_budget: 1e-05` and two keys. The first key spends 6.6e-05 through `/anthropic/v1/messages`.
    The second key must then be refused on `/mistral/v1/chat/completions` with 422 `budget_exceeded` whose message
    names `Team=<team_id>`, and the provider must receive nothing for that refusal."""
    with proxy.scenario() as scenario:
        team: Final = scenario.team(max_budget=_TINY_BUDGET)
        spender: Final = scenario.key(team_id=team)
        other: Final = scenario.key(team_id=team)
        assert (
            send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(spender)).status_code == 200
        )
        refused: Final = refused_until(
            lambda: send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(other)), 422
        )
        assert f"Team={team}" in message_of(openai_error(refused, 422)), refused.text
        wire.drain()
        assert (
            send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(other)).status_code == 422
        )
        assert posts(wire) == ()


def test_a_keys_rpm_limit_refuses_the_second_anthropic_call_in_the_anthropic_envelope(
    proxy: Gateway, wire: Wire
) -> None:
    """A key with `rpm_limit: 1` calls `/anthropic/v1/messages` twice. The first is served; the second must be
    429 with `error.type` `rate_limit_error`, `litellm_error_type` `throttling_error` and the message `Rate limit
    exceeded for api_key: <key hash>. Limit type: requests. Current limit: 1, Remaining: 0. Limit resets at:
    <time> UTC`, and the provider must have received one call."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=1)
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        refused: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        error: Final = anthropic_error(refused, 429)
        assert (error["type"], error.get("litellm_error_type")) == ("rate_limit_error", "throttling_error")
        _assert_limited_message(message_of(error), "api_key", hashed(key), "requests", 1)
        assert len(posts(wire)) == 1


def test_a_keys_rpm_limit_refuses_the_second_cohere_call_in_the_openai_envelope(proxy: Gateway, wire: Wire) -> None:
    """A key with `rpm_limit: 1` calls `/cohere/v2/chat` twice. The second must be 429 with `error.type`
    `throttling_error`, `error.code` `"429"`, `error.param` null and the `api_key` requests message."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=1)
        assert send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key)).status_code == 200
        refused: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        error: Final = openai_error(refused, 429)
        assert (error["type"], error["code"], error["param"]) == ("throttling_error", "429", None), refused.text
        _assert_limited_message(message_of(error), "api_key", hashed(key), "requests", 1)
        assert len(posts(wire)) == 1


def test_one_key_request_window_is_shared_across_provider_routes_and_chat(proxy: Gateway, wire: Wire) -> None:
    """A key with `rpm_limit: 2` calls `/anthropic/v1/messages`, then a Gemini `generateContent`, then
    `/cohere/v2/chat`. The first two are served and the third must be refused with 429: the key has one request
    window whatever route or provider spends it."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=2)
        first: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        second: Final = send(proxy, gemini_path(), gemini_body(marker()), headers={"x-goog-api-key": key})
        third: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        assert (first.status_code, second.status_code, third.status_code) == (200, 200, 429), third.text
        assert len(posts(wire)) == 2


def test_a_team_rpm_limit_applies_across_the_teams_keys_on_pass_through(proxy: Gateway, wire: Wire) -> None:
    """A team has `rpm_limit: 1` and two keys. One key calls `/mistral/v1/chat/completions`; the other key's call
    to `/openai/v1/chat/completions` must then be 429 with `Rate limit exceeded for team: <team_id>. Limit type:
    requests. Current limit: 1`, and only the first call may reach a provider."""
    with proxy.scenario() as scenario:
        team: Final = scenario.team(rpm_limit=1)
        first: Final = scenario.key(team_id=team)
        second: Final = scenario.key(team_id=team)
        assert (
            send(proxy, "/mistral/v1/chat/completions", openai_body(marker()), headers=bearer(first)).status_code == 200
        )
        refused: Final = send(proxy, "/openai/v1/chat/completions", openai_body(marker()), headers=bearer(second))
        _assert_limited_message(message_of(openai_error(refused, 429)), "team", team, "requests", 1)
        assert len(posts(wire)) == 1


def test_a_keys_tpm_limit_settles_from_the_native_usage_of_a_pass_through_response(proxy: Gateway, wire: Wire) -> None:
    """A key has `tpm_limit: 15`. Each Anthropic pass-through response reports 7 + 3 = 10 tokens. The first call
    is served (0 used), the second is served (10 used, under 15) and, once that settles to 20, a third must be
    refused with 429 `Limit type: tokens`. UNVERIFIED: assumes the limiter's success hook settles token counters
    from the pass-through logging payload's usage, as it does for chat."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(tpm_limit=15)
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        refused: Final = refused_until(
            lambda: send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)),
            429,
            seconds=30,
        )
        _assert_limited_message(message_of(anthropic_error(refused, 429)), "api_key", hashed(key), "tokens", 15)


def test_a_keys_parallel_limit_holds_while_a_pass_through_call_is_in_flight(proxy: Gateway, wire: Wire) -> None:
    """A key has `max_parallel_requests: 1`. One `/anthropic/v1/messages` call is parked at the provider. A second
    call on `/cohere/v2/chat` must be refused with 429 `Limit type: max_parallel_requests` while it is parked; once
    released, the parked call must complete with 200 and a new call must be served, so the slot was given back."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(max_parallel_requests=1)
        held: Final = hold_in_background(
            lambda: send(proxy, "/anthropic/v1/messages", anthropic_body(marker(HOLD_MARKER)), headers=bearer(key))
        )
        wait_for_posts(wire, 1)
        refused: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        _assert_limited_message(
            message_of(openai_error(refused, 429)), "api_key", hashed(key), "max_parallel_requests", 1
        )
        assert held.release().status_code == 200
        eventually(
            lambda: send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key)).status_code,
            lambda status: status == 200,
            seconds=30,
        )


def test_a_keys_per_model_rpm_limit_applies_to_the_body_model_on_the_anthropic_route(
    proxy: Gateway, wire: Wire
) -> None:
    """A key has `model_rpm_limit: {"claude-sonnet-4-5": 1}`. Two `/anthropic/v1/messages` calls name that model.
    The second must be 429 for `model_per_key: <key hash>:claude-sonnet-4-5`, and a call naming
    `claude-haiku-4-5` must still be served. UNVERIFIED: `_rate_limited_model` builds the descriptor from the body
    model whether or not the router serves it; `test_dec1_ratelim_request_lifecycle.py` asserts the opposite for
    operator-configured pass-through endpoints, so one of the two is wrong."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={"claude-sonnet-4-5": 1})
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        refused: Final = send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key))
        _assert_limited_message(
            message_of(anthropic_error(refused, 429)),
            "model_per_key",
            f"{hashed(key)}:claude-sonnet-4-5",
            "requests",
            1,
        )
        other: Final = send(
            proxy, "/anthropic/v1/messages", anthropic_body(marker(), model="claude-haiku-4-5"), headers=bearer(key)
        )
        assert other.status_code == 200, other.text


def test_a_served_pass_through_response_reports_the_keys_remaining_requests(proxy: Gateway) -> None:
    """A key with `rpm_limit: 3` makes one `/cohere/v2/chat` call. The 200 must carry
    `x-ratelimit-api_key-limit-requests: 3` and `x-ratelimit-api_key-remaining-requests: 2`, as a chat response
    does. UNVERIFIED: pass-through copies `fastapi_response.headers` into its reply
    (`HttpPassThroughEndpointHelpers.get_response_headers`); confirm the limiter writes them there for this route."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=3)
        response: Final = send(proxy, "/cohere/v2/chat", cohere_body(marker()), headers=bearer(key))
        assert response.status_code == 200, response.text
        assert response.headers.get("x-ratelimit-api_key-limit-requests") == "3", dict(response.headers)
        assert response.headers.get("x-ratelimit-api_key-remaining-requests") == "2", dict(response.headers)


def test_a_rate_limited_pass_through_call_writes_no_success_spend(proxy: Gateway, wire: Wire) -> None:
    """A key with `rpm_limit: 1` is served once on `/anthropic/v1/messages` and refused once. Its spend must settle
    at exactly one call's cost (6.6e-05) and stay there: a refused call is never billed."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=1)
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 200
        assert send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code == 429
        spent: Final = eventually(lambda: key_spend(proxy, key), lambda value: value > 0, seconds=SPEND_SECONDS)
        assert spent == pytest.approx(_ANTHROPIC_COST)
        assert len(posts(wire)) == 1
