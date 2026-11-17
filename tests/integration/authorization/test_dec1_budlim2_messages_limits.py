"""Contract: rate limits on ``POST /v1/messages`` and the token count route next to it.

The v3 limiter (litellm/proxy/hooks/parallel_request_limiter_v3.py) runs in two places for a Messages call:

- before the body is read, for a key already in the auth cache (``raise_if_over_limit_before_body``, #51428):
  only windows that need no body (the key's requests and parallel slots). That refusal is raised from auth, so
  the central handler answers it with ``anthropic_error_content``: Anthropic type ``rate_limit_error``, the
  limiter's own type kept as ``litellm_error_type: "throttling_error"``, and the limiter's headers
  (``retry-after``, ``rate_limit_type``, ``rate_limited_by``) on the response.
- in the pre-call hook, after the body is read: token windows, team and model scopes. That refusal is answered by
  the Messages handler (litellm/proxy/anthropic_endpoints/endpoints.py), with Anthropic type ``rate_limit_error``
  and the ``x-litellm-call-id`` of the request.

Without ``max_tokens`` the limiter reserves ``max(input estimate, floor)`` output tokens, where the floor is
``min(1024, int(0.25 * smallest tpm_limit))``, and when that floor is below 1024 it writes the floor into the
request as ``max_tokens`` (``_apply_implicit_output_cap``). On ``/v1/messages``, where Anthropic requires
``max_tokens``, that means a key with a small ``tpm_limit`` is served a request that any other key is refused
with 400. Both sides are pinned here.

``/v1/messages/count_tokens`` is an unbilled route (route_checks.py ``_TOKEN_COUNTING_SEGMENTS``): the
pre-body check skips it (#51675) and its handler never runs the limiter, so counting must never use up or be
blocked by a key's limits.

Prompts are 45 characters, which the limiter estimates as 11 input tokens (``len(text) // 4``); held prompts are
50 characters (12 tokens).
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.authorization.dec1_budlim2_support import (
    COUNTED_INPUT_TOKENS,
    PROMPT_TOKENS,
    WINDOW_SECONDS,
    anthropic_error,
    count_tokens,
    message_of,
    messages,
    messages_model,
    surface_provider,
)
from tests.integration.authorization.dec1_limits_support import (
    RESET_AT,
    counter_key,
    hashed,
    limit_message,
    redis_integer,
    scope_headers,
    shared_redis,
)
from tests.integration.spend.dec1_budgets_support import background

pytestmark: Final = pytest.mark.shares_worker_state

MISSING_MAX_TOKENS: Final = "anthropic_messages: Missing required parameter: 'max_tokens'."


def _assert_limit_text(message: str, scope: str, value: str, kind: str, limit: int, remaining: int = 0) -> None:
    prefix: Final = limit_message(scope, value, kind, limit, remaining)
    assert message.startswith(prefix), message
    assert RESET_AT.match(message.removeprefix(prefix)), message


def test_a_messages_call_without_max_tokens_under_a_small_tpm_limit_is_served_with_an_implicit_cap(
    gateway: Gateway,
) -> None:
    """A key with ``tpm_limit: 1000`` sends ``/v1/messages`` without ``max_tokens``. The limiter's floor is
    min(1024, int(1000 * 0.25)) = 250, below 1024, so it writes ``max_tokens: 250`` into the request before the
    handler checks required fields. The call is answered 200, the provider receives ``max_tokens: 250``, and the
    reservation reported is 11 input + 250 output (``x-ratelimit-api_key-remaining-tokens: 739``). A port that
    validates the body before the limiter, or stops writing the cap, turns this into a 400."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = messages(gateway, model, key, max_tokens=None)
        assert served.status_code == 200, served.text
        sent: Final = [request.body for request in provider.posts()]
        assert len(sent) == 1, sent
        assert b'"max_tokens": 250' in sent[0] or b'"max_tokens":250' in sent[0], sent[0]
        assert scope_headers(served, "api_key")["remaining-tokens"] == str(1000 - PROMPT_TOKENS - 250)


def test_a_messages_call_without_max_tokens_under_a_large_tpm_limit_is_refused_for_the_missing_parameter(
    gateway: Gateway,
) -> None:
    """A key with ``tpm_limit: 100000`` sends ``/v1/messages`` without ``max_tokens``. Its floor is
    min(1024, 25000) = 1024, the baseline, so no cap is written and the handler refuses with HTTP 400, Anthropic
    type ``invalid_request_error`` and exactly ``anthropic_messages: Missing required parameter: 'max_tokens'.``.
    The provider is not called."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], tpm_limit=100000)
        error: Final = anthropic_error(messages(gateway, model, key, max_tokens=None), 400)
        assert error["type"] == "invalid_request_error", error
        assert message_of(error) == MISSING_MAX_TOKENS
        assert provider.posts() == ()


def test_a_messages_tpm_refusal_is_a_rate_limit_error_with_the_call_id_and_no_litellm_error_type(
    gateway: Gateway,
) -> None:
    """A key with ``tpm_limit: 20`` sends ``/v1/messages`` with ``max_tokens: 500``. The token window needs the
    body, so the refusal comes from the pre-call hook through the Messages handler: HTTP 429, Anthropic type
    ``rate_limit_error``, the ``api_key`` tokens text (``Current limit: 20, Remaining: 20``), an
    ``x-litellm-call-id`` response header, and no ``litellm_error_type`` (``anthropic_handler_error_content``
    does not add it). The provider is not called."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], tpm_limit=20)
        refused: Final = messages(gateway, model, key, max_tokens=500)
        error: Final = anthropic_error(refused, 429)
        assert error["type"] == "rate_limit_error", error
        assert "litellm_error_type" not in error, error
        _assert_limit_text(message_of(error), "api_key", hashed(key), "tokens", 20, 20)
        assert refused.headers.get("x-litellm-call-id"), dict(refused.headers)
        assert provider.posts() == ()


def test_a_cached_key_over_its_rpm_on_messages_is_refused_with_the_limiter_headers_and_throttling_type(
    gateway: Gateway,
) -> None:
    """A key with ``rpm_limit: 1`` makes one ``/v1/messages`` call, which also puts it in the auth cache. The
    second call is refused before its body is read: HTTP 429, Anthropic type ``rate_limit_error``,
    ``litellm_error_type: "throttling_error"``, the ``api_key`` requests text, and the headers
    ``retry-after: 60``, ``rate_limited_by: api_key`` and ``rate_limit_type: requests``, which the central handler
    copies from the exception (``anthropic_error_json_response`` passes ``exc.headers``).
    UNVERIFIED: ``test_dec1_limits_endpoints.py::test_messages_requests_over_the_key_rpm_limit_get_an_anthropic_
    shaped_429`` skips the same ``retry-after`` as a bug of the handler path; this test pins the pre-body path,
    which by reading the code keeps the headers."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert messages(gateway, model, key).status_code == 200
        refused: Final = messages(gateway, model, key)
        error: Final = anthropic_error(refused, 429)
        assert error["type"] == "rate_limit_error", error
        assert error["litellm_error_type"] == "throttling_error", error
        _assert_limit_text(message_of(error), "api_key", hashed(key), "requests", 1)
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS), dict(refused.headers)
        assert refused.headers["rate_limited_by"] == "api_key", dict(refused.headers)
        assert refused.headers["rate_limit_type"] == "requests", dict(refused.headers)
        assert len(provider.posts()) == 1


@pytest.mark.timeout(180)
def test_a_messages_parallel_refusal_is_an_anthropic_rate_limit_error(gateway: Gateway) -> None:
    """A key with ``max_parallel_requests: 1`` has one held ``/v1/messages`` call in flight. A second Messages
    call is refused with HTTP 429, Anthropic type ``rate_limit_error`` and the text ``Rate limit exceeded for
    api_key: <hash>. Limit type: max_parallel_requests. Current limit: 1, Remaining: 0. Limit resets at: ...``.
    The first call is still served once released."""
    with surface_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        in_flight: Final = run(lambda: messages(gateway, model, key, hold=True))
        provider.wait_for(1)
        try:
            refused: Final = messages(gateway, model, key)
        finally:
            provider.release()
        error: Final = anthropic_error(refused, 429)
        assert error["type"] == "rate_limit_error", error
        _assert_limit_text(message_of(error), "api_key", hashed(key), "max_parallel_requests", 1)
        assert in_flight.result(timeout=60).status_code == 200


def test_a_team_tpm_refusal_on_messages_names_the_team(gateway: Gateway) -> None:
    """A key of a team with ``tpm_limit: 20`` (the key has none) sends ``/v1/messages`` with ``max_tokens: 500``.
    It is refused with HTTP 429, Anthropic type ``rate_limit_error`` and the ``team`` scope text
    ``Rate limit exceeded for team: <team_id>. Limit type: tokens. Current limit: 20, Remaining: 20``."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        team: Final = scenario.team(tpm_limit=20, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        error: Final = anthropic_error(messages(gateway, model, key, max_tokens=500), 429)
        _assert_limit_text(message_of(error), "team", team, "tokens", 20, 20)
        assert provider.posts() == ()


def test_a_served_messages_call_reports_the_key_request_and_token_windows(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 5`` and ``tpm_limit: 1000`` makes one ``/v1/messages`` call with
    ``max_tokens: 16``. The 200 carries ``x-ratelimit-api_key-limit-requests: 5``, ``-remaining-requests: 4``,
    ``-limit-tokens: 1000`` and ``-remaining-tokens: 973`` (11 + 16 reserved), the same names chat uses, so
    Anthropic clients can read the proxy's budget the same way."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=5, tpm_limit=1000)
        served: Final = messages(gateway, model, key, max_tokens=16)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {
            "limit-requests": "5",
            "remaining-requests": "4",
            "limit-tokens": "1000",
            "remaining-tokens": str(1000 - PROMPT_TOKENS - 16),
        }


def test_a_streamed_messages_call_over_the_tpm_limit_gets_a_json_429_not_an_event_stream(gateway: Gateway) -> None:
    """A key with ``tpm_limit: 20`` sends ``stream: true`` to ``/v1/messages`` with ``max_tokens: 500``. The
    refusal is decided before the stream opens, so the answer is HTTP 429 with ``content-type:
    application/json`` and the Anthropic envelope, not a 200 SSE stream carrying an ``error`` event."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], tpm_limit=20)
        refused: Final = messages(gateway, model, key, max_tokens=500, stream=True)
        assert anthropic_error(refused, 429)["type"] == "rate_limit_error"
        assert "event:" not in refused.text
        assert provider.posts() == ()


def test_messages_tokens_settle_to_the_usage_the_anthropic_provider_reported(gateway: Gateway) -> None:
    """A key with ``tpm_limit: 1000`` makes one streamed ``/v1/messages`` call; the provider's ``message_start``
    reports 12 input tokens and ``message_delta`` 3 output tokens. The key's Redis token window
    ``{api_key:<hash>}:tokens`` settles to 15, not to the 11 + 16 reserved. A port that settles streams from the
    reservation instead of the final usage would leave 27."""
    with surface_provider() as provider, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        streamed: Final = messages(gateway, model, key, max_tokens=16, stream=True)
        assert streamed.status_code == 200, streamed.text
        eventually(
            lambda: redis_integer(redis, counter_key("api_key", hashed(key), "tokens")),
            lambda value: value == 15,
            seconds=30,
        )


def test_count_tokens_is_served_for_a_key_that_used_up_its_rpm(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` uses its one request on ``/v1/messages``. ``POST /v1/messages/count_tokens``
    for the same key is still answered 200 with ``input_tokens: 12`` from the provider's count endpoint: the
    pre-body check skips unbilled routes (#51675) and the count handler never runs the limiter. Claude Code
    counts before every turn, so a 429 here would stall a client that is merely at its limit."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert messages(gateway, model, key).status_code == 200
        counted: Final = count_tokens(gateway, model, key)
        assert counted.status_code == 200, counted.text
        assert counted.json()["input_tokens"] == COUNTED_INPUT_TOKENS


def test_count_tokens_does_not_use_up_the_key_rpm_window(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` calls ``/v1/messages/count_tokens`` three times and then ``/v1/messages``
    once. The Messages call is served and the Redis window ``{api_key:<hash>}:requests`` reads 1 afterwards, so
    counting never counts as a request."""
    with surface_provider() as provider, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        for _ in range(3):
            assert count_tokens(gateway, model, key).status_code == 200
        assert messages(gateway, model, key).status_code == 200
        assert redis_integer(redis, counter_key("api_key", hashed(key), "requests")) == 1


def test_count_tokens_reserves_nothing_from_the_key_tpm_window(gateway: Gateway) -> None:
    """A key with ``tpm_limit: 1000`` calls ``/v1/messages/count_tokens``. Its token window
    ``{api_key:<hash>}:tokens`` stays absent, and the next ``/v1/messages`` call with ``max_tokens: 16`` reports
    ``x-ratelimit-api_key-remaining-tokens: 973`` exactly as on a fresh key."""
    with surface_provider() as provider, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert count_tokens(gateway, model, key).status_code == 200
        assert redis_integer(redis, counter_key("api_key", hashed(key), "tokens")) is None
        served: Final = messages(gateway, model, key, max_tokens=16)
        assert scope_headers(served, "api_key")["remaining-tokens"] == str(1000 - PROMPT_TOKENS - 16)


@pytest.mark.timeout(180)
def test_count_tokens_is_answered_while_the_keys_only_parallel_slot_is_taken(gateway: Gateway) -> None:
    """A key with ``max_parallel_requests: 1`` has one held ``/v1/messages`` call in flight. A
    ``/v1/messages/count_tokens`` call for the same key is answered 200 meanwhile: counting takes no parallel
    slot, so a client that counts while it waits for an answer is never refused."""
    with surface_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        in_flight: Final = run(lambda: messages(gateway, model, key, hold=True))
        provider.wait_for(1)
        try:
            counted: Final = count_tokens(gateway, model, key)
        finally:
            provider.release()
        assert counted.status_code == 200, counted.text
        assert in_flight.result(timeout=60).status_code == 200


def test_count_tokens_is_still_refused_for_a_key_over_its_budget(gateway: Gateway) -> None:
    """A key with ``max_budget: 0`` calls ``/v1/messages/count_tokens``. The auth budget check runs for every
    route that runs inference checks (``RouteChecks.runs_inference_checks``, user_api_key_auth.py:3400), and the
    count route is one of ``anthropic_routes``, so it is refused with HTTP 422 in the Anthropic envelope with
    ``litellm_error_type: "budget_exceeded"`` although counting itself is free. UNVERIFIED: #51675 exempted
    token counting only from rate limits; if budgets are meant to exempt it too, this test pins the opposite."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        error: Final = anthropic_error(count_tokens(gateway, model, key), 422)
        assert error["litellm_error_type"] == "budget_exceeded", error
        assert provider.posts() == ()
