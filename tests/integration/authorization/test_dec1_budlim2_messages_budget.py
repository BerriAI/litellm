"""Contract: budgets on ``POST /v1/messages``, the Anthropic Messages surface.

The budget hold ("reservation") is taken at auth for every route that runs inference checks, so it is taken for
``/v1/messages`` exactly as for chat: ``reserve_budget_for_request`` (litellm/proxy/spend_tracking/
budget_reservation.py:240) estimates the worst case cost, and the Rust engine (``callbacks-budgets``, #51535)
adds it to every budgeted entity's Redis counter (``spend:key:<sha256>``, ``spend:team:<id>``) before the provider
is called. The estimate reads the output cap from ``max_completion_tokens``, ``max_tokens`` or
``max_output_tokens`` and clamps it to the model's ``max_output_tokens`` (64000 for claude-sonnet-4-5) or 16384
when absent (budget_reservation.py:1468). After the call the hold settles to the real cost; a failed call or a
response that never claimed the hold releases it (budget_reservation_release_middleware.py).

``/v1/messages`` is special in two ways that a Rust port of the front door must keep:

1. Anthropic requires ``max_tokens``, but the hold is taken at auth before the handler validates the body, so a
   request without it is first held at the 16384 token default and then refused with 400; the hold must be
   released, or the key is locked out until the counter TTL.
2. Refusals are answered in the Anthropic envelope ``{"type": "error", "error": {"type": ..., "message": ...}}``
   (litellm/proxy/common_utils/anthropic_error_payload.py). A budget refusal is HTTP 422 with Anthropic type
   ``invalid_request_error`` and ``litellm_error_type: "budget_exceeded"``, because 422 has no Anthropic type of
   its own (``proxy_error_type``) and the proxy's type differs from both mapped types.

Price: input free, output 0.004 a token (``dec1_budlim2_support``). The scripted provider bills 3 output tokens
per Messages call, so one served call costs 0.012 and a hold is ``max_tokens * 0.004``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value
from tests.integration._support.process import OwnedState, owned_proxy
from tests.integration.authorization.dec1_budlim2_support import (
    MESSAGES_REAL_COST,
    NO_CAP_HOLD,
    OUTPUT_RATE,
    anthropic_error,
    message_of,
    messages,
    messages_model,
    surface_provider,
)
from tests.integration.spend.dec1_budgets_support import (
    background,
    budget_message,
    counter,
    counter_settles,
    hashed,
    key_counter,
    masked,
    parse_costs,
    redis_client,
)
from tests.integration.spend.dec1_spend_support import proxy_config, spend_reaches

pytestmark: Final = pytest.mark.shares_worker_state

MISSING_MAX_TOKENS: Final = "anthropic_messages: Missing required parameter: 'max_tokens'."


def _served(response_status: int, text: str) -> None:
    assert response_status == 200, text


@pytest.mark.timeout(180)
def test_an_in_flight_messages_call_holds_max_tokens_at_the_output_price(gateway: Gateway) -> None:
    """A key with ``max_budget`` 1.0 sends a ``/v1/messages`` call with ``max_tokens: 10`` that the provider keeps
    waiting. While it is in flight ``spend:key:<hash>`` reads exactly 0.04 (10 * 0.004), and after the answer it
    settles to the real cost 0.012 (3 output tokens). A port that skips the hold on the Anthropic route would read
    0 here and let a burst of Messages calls overspend."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: messages(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(10 * OUTPUT_RATE)
        provider.release()
        answered: Final = in_flight.result(timeout=60)
        _served(answered.status_code, answered.text)
        counter_settles(cache, key_counter(key), MESSAGES_REAL_COST)


@pytest.mark.timeout(180)
def test_a_messages_max_tokens_above_the_model_ceiling_holds_only_the_ceiling(gateway: Gateway) -> None:
    """A key with ``max_budget`` 1000 sends a held ``/v1/messages`` call with ``max_tokens: 100000`` to
    claude-sonnet-4-5, whose cost map ``max_output_tokens`` is 64000. The hold is clamped to the ceiling: the key
    counter reads 64000 * 0.004 = 256.0 while in flight, not 400.0. The clamp stops a caller from pinning a whole
    team budget with an absurd ``max_tokens`` (budget_reservation.py:1481)."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1000.0)
        in_flight: Final = run(lambda: messages(gateway, model, key, hold=True, max_tokens=100000))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(64000 * OUTPUT_RATE)
        provider.release()
        in_flight.result(timeout=60)
        counter_settles(cache, key_counter(key), MESSAGES_REAL_COST)


@pytest.mark.timeout(180)
def test_a_streamed_messages_call_holds_until_the_stream_ends_then_settles_to_its_cost(gateway: Gateway) -> None:
    """A key with ``max_budget`` 1.0 opens a ``stream: true`` ``/v1/messages`` call with ``max_tokens: 10`` that
    the provider holds before the first event. The counter reads 0.04 while held; once every SSE event has been
    read it settles to 0.012, the cost of the 12 input (free) and 3 output tokens in ``message_delta``. A port that
    settles at stream open would release the hold before the cost is known."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        in_flight: Final = run(lambda: messages(gateway, model, key, hold=True, max_tokens=10, stream=True))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(10 * OUTPUT_RATE)
        provider.release()
        streamed: Final = in_flight.result(timeout=60)
        assert streamed.status_code == 200, streamed.text
        assert "event: message_stop" in streamed.text, streamed.text
        counter_settles(cache, key_counter(key), MESSAGES_REAL_COST)


@pytest.mark.timeout(180)
def test_a_concurrent_messages_call_is_refused_in_the_anthropic_envelope_while_a_hold_pins_the_budget(
    gateway: Gateway,
) -> None:
    """A key with an alias and ``max_budget`` 1.0 has one held ``/v1/messages`` call with ``max_tokens: 1000``
    (estimate 4.0, shrunk to the 1.0 left). A second Messages call sent meanwhile is refused with HTTP 422, an
    Anthropic envelope of type ``invalid_request_error``, ``litellm_error_type: "budget_exceeded"`` and exactly
    ``Budget has been exceeded! Key=<alias> (sk-...<last 4>) Current cost: 1.0, Max budget: 1.0``. The provider
    sees only the first call."""
    with surface_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = messages_model(scenario, provider)
        alias: Final = f"dec1-budlim2-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], max_budget=1.0, key_alias=alias)
        in_flight: Final = run(lambda: messages(gateway, model, key, hold=True, max_tokens=1000))
        provider.wait_for(1)
        try:
            second: Final = messages(gateway, model, key, max_tokens=5)
        finally:
            provider.release()
        error: Final = anthropic_error(second, 422)
        assert error["type"] == "invalid_request_error", second.text
        assert error["litellm_error_type"] == "budget_exceeded", second.text
        assert message_of(error) == budget_message(f"Key={alias} ({masked(key)})", "1.0", "1.0")
        assert in_flight.result(timeout=60).status_code == 200
        assert provider.arrivals == 1


@pytest.mark.timeout(180)
def test_a_messages_call_the_provider_fails_releases_its_whole_hold(gateway: Gateway) -> None:
    """The provider answers 500 to a ``/v1/messages`` call with ``max_tokens: 10`` on a key with ``max_budget``
    1.0. The caller gets an Anthropic ``api_error``, the key counter goes back to 0 and the key's stored spend
    stays 0, so a failing provider never eats the budget."""
    with surface_provider(failing_status=500) as provider, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        failed: Final = messages(gateway, model, key, max_tokens=10)
        assert anthropic_error(failed, 500)["type"] == "api_error", failed.text
        counter_settles(cache, key_counter(key), 0.0)
        info: Final = gateway.get("/key/info", {"key": key})
        assert float(str(object_value(info["info"])["spend"])) == 0.0


@pytest.mark.timeout(180)
def test_a_messages_call_without_max_tokens_is_refused_with_400_and_its_hold_is_released(gateway: Gateway) -> None:
    """A key with ``max_budget`` 1.0 and no rate limits sends ``/v1/messages`` without ``max_tokens``. Auth has
    already held min(16384, 64000) * 0.004 = 65.536, shrunk to 1.0, before the handler sees the body. The handler
    refuses with HTTP 400 ``invalid_request_error`` and exactly ``anthropic_messages: Missing required parameter:
    'max_tokens'.`` (route_llm_request.py:308), the counter is released to 0 and a correct call right after is
    served. Without the release middleware the invalid call would lock the key until the counter's TTL."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        assert NO_CAP_HOLD > 1.0
        refused: Final = messages(gateway, model, key, max_tokens=None)
        error: Final = anthropic_error(refused, 400)
        assert error["type"] == "invalid_request_error", refused.text
        assert message_of(error) == MISSING_MAX_TOKENS
        assert provider.posts() == ()
        counter_settles(cache, key_counter(key), 0.0)
        assert messages(gateway, model, key, max_tokens=5).status_code == 200


@pytest.mark.timeout(180)
def test_a_messages_team_hold_refuses_another_team_key_with_the_admission_wording(gateway: Gateway) -> None:
    """A team with ``max_budget`` 1.0 has two keys. Key A holds a ``/v1/messages`` call with ``max_tokens: 1000``
    (hold shrunk to 1.0). Key B's Messages call with ``max_tokens: 16`` passes the auth check (the team's stored
    spend is 0) but the admission engine refuses it with HTTP 422 in the Anthropic envelope and the engine's
    wording ``Budget has been exceeded! Team=<team_id> Current cost: <1.0 + 0.064>, Max budget: 1.0``
    (engine.rs ``exceeded``), so the team budget binds across keys on this route too."""
    with surface_provider() as provider, gateway.scenario() as scenario, background() as run:
        model: Final = messages_model(scenario, provider)
        team: Final = scenario.team(max_budget=1.0, models=[model])
        key_a: Final = scenario.key(team_id=team, models=[model])
        key_b: Final = scenario.key(team_id=team, models=[model])
        in_flight: Final = run(lambda: messages(gateway, model, key_a, hold=True, max_tokens=1000))
        provider.wait_for(1)
        try:
            refused: Final = messages(gateway, model, key_b, max_tokens=16)
        finally:
            provider.release()
        error: Final = anthropic_error(refused, 422)
        assert error["litellm_error_type"] == "budget_exceeded", refused.text
        costs: Final = parse_costs(message_of(error))
        assert costs["label"] == f"Team={team}", costs
        assert float(str(costs["current"])) == pytest.approx(1.0 + 16 * OUTPUT_RATE)
        assert costs["max"] == "1.0", costs
        assert in_flight.result(timeout=60).status_code == 200


@pytest.mark.timeout(180)
def test_a_messages_call_settles_the_key_and_team_counters_to_the_same_real_cost(gateway: Gateway) -> None:
    """A key of a team, both with ``max_budget`` 5.0, makes one ``/v1/messages`` call with ``max_tokens: 50``.
    Both ``spend:key:<hash>`` and ``spend:team:<team_id>`` settle to 0.012 and the team's stored spend reaches
    0.012. A settlement that reconciled only the key's entry would leave the team holding 0.2."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = messages_model(scenario, provider)
        team: Final = scenario.team(max_budget=5.0, models=[model])
        key: Final = scenario.key(team_id=team, models=[model], max_budget=5.0)
        assert messages(gateway, model, key, max_tokens=50).status_code == 200
        counter_settles(cache, key_counter(key), MESSAGES_REAL_COST)
        counter_settles(cache, f"spend:team:{team}", MESSAGES_REAL_COST)
        assert spend_reaches(gateway, "team", team, MESSAGES_REAL_COST) == pytest.approx(MESSAGES_REAL_COST)


def test_a_streamed_messages_call_over_budget_gets_a_json_anthropic_error_not_an_event_stream(
    gateway: Gateway,
) -> None:
    """A key with ``max_budget`` 0 sends ``stream: true`` to ``/v1/messages``. Because the refusal happens at auth,
    before any stream is opened, the answer is a plain HTTP 422 with ``content-type: application/json`` and the
    Anthropic envelope, not an SSE ``event: error`` frame on a 200. Anthropic SDKs raise on the status; an SSE
    error would be read as a successful but broken stream."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        refused: Final = messages(gateway, model, key, max_tokens=10, stream=True)
        error: Final = anthropic_error(refused, 422)
        assert error["litellm_error_type"] == "budget_exceeded", refused.text
        assert "event:" not in refused.text
        assert provider.posts() == ()


def test_a_messages_budget_refusal_echoes_the_callers_x_request_id(gateway: Gateway) -> None:
    """A zero budget key sends ``/v1/messages`` with ``x-request-id: req-dec1-<hex>``. The 422 body carries
    ``request_id`` equal to that header next to ``type`` and ``error``, which is how Anthropic clients correlate
    errors (anthropic_error_payload.py ``_anthropic_content`` passes the header to
    ``transform_to_anthropic_error``). UNVERIFIED: assumes ``request_id`` is placed at the top level of the body."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        request_id: Final = f"req-dec1-{uuid.uuid4().hex}"
        refused: Final = messages(gateway, model, key, max_tokens=10, headers={"x-request-id": request_id})
        anthropic_error(refused, 422)
        assert JSON_OBJECT.validate_json(refused.content)["request_id"] == request_id, refused.text


def test_a_messages_budget_refusal_has_no_call_id_in_the_body_by_default(gateway: Gateway) -> None:
    """On the shared proxy (``include_litellm_call_id_in_error_body`` not set), a zero budget key's
    ``/v1/messages`` 422 has no ``litellm_call_id`` inside ``error`` (error_body_call_id.py:11). The id may only
    be added when the operator opts in; a port that always adds it changes the body clients compare against."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        error: Final = anthropic_error(messages(gateway, model, key, max_tokens=10), 422)
        assert "litellm_call_id" not in error, error


@pytest.mark.timeout(240)
def test_fail_closed_refuses_a_messages_estimate_that_does_not_fit_in_the_anthropic_envelope(
    gateway: Gateway, tmp_path: Path
) -> None:
    """On a proxy with ``general_settings.fail_closed_budget_enforcement: true``, a fresh key with ``max_budget``
    0.03 sends ``/v1/messages`` with ``max_tokens: 10`` (estimate 0.04). It is refused with HTTP 422, Anthropic
    type ``invalid_request_error``, ``litellm_error_type: "budget_exceeded"`` and exactly ``Budget has been
    exceeded! Key=<sha256> Current cost: 0.0, Estimated request cost: 0.04, Max budget: 0.03``. The provider is not
    called and the counter stays at 0."""
    config: Final = proxy_config(tmp_path, general_settings={"fail_closed_budget_enforcement": True})
    with (
        surface_provider() as provider,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.03)
        error: Final = anthropic_error(messages(owned, model, key, max_tokens=10), 422)
        assert error["type"] == "invalid_request_error", error
        assert error["litellm_error_type"] == "budget_exceeded", error
        assert message_of(error) == budget_message(f"Key={hashed(key)}", "0.0", "0.03", estimate="0.04")
        assert provider.posts() == ()
        assert counter(cache, key_counter(key)) in (None, 0.0)


@pytest.mark.timeout(240)
def test_fail_closed_refuses_a_messages_call_without_max_tokens_for_budget_before_the_missing_parameter(
    gateway: Gateway, tmp_path: Path
) -> None:
    """With fail closed on, a key with ``max_budget`` 1.0 sends ``/v1/messages`` without ``max_tokens``. Auth
    estimates 16384 * 0.004 = 65.536, which does not fit, so the caller gets the budget 422 (message containing
    ``Estimated request cost: 65.536``) and never the 400 for the missing parameter: the budget check runs at auth,
    before the handler validates the body. A port that validates the body first would answer 400 instead."""
    config: Final = proxy_config(tmp_path, general_settings={"fail_closed_budget_enforcement": True})
    with (
        surface_provider() as provider,
        owned_proxy(gateway, tmp_path, {}, config=config) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        error: Final = anthropic_error(messages(owned, model, key, max_tokens=None), 422)
        costs: Final = parse_costs(message_of(error))
        assert float(str(costs["estimate"])) == pytest.approx(NO_CAP_HOLD)
        assert provider.posts() == ()


@pytest.mark.timeout(240)
def test_budget_exceeded_status_code_429_makes_the_messages_refusal_an_anthropic_rate_limit_error(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy started with ``litellm_settings.budget_exceeded_status_code: 429`` refuses a zero budget key's
    ``/v1/messages`` call with HTTP 429. In the Anthropic envelope that status maps to ``error.type:
    "rate_limit_error"`` and the proxy's own type survives as ``litellm_error_type: "budget_exceeded"``
    (anthropic_error_payload.py ``_with_litellm_error_type``), so Anthropic SDKs back off and retry.
    UNVERIFIED: assumes ``litellm_settings`` sets the module value read by ``BudgetExceededError``."""
    config: Final = proxy_config(tmp_path, litellm_settings={"budget_exceeded_status_code": 429})
    with (
        surface_provider() as provider,
        owned_proxy(gateway, tmp_path, {}, config=config, state=OwnedState.SHARED_WITH_WORKER) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = messages_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        error: Final = anthropic_error(messages(owned, model, key, max_tokens=10), 429)
        assert error["type"] == "rate_limit_error", error
        assert error["litellm_error_type"] == "budget_exceeded", error
        assert provider.posts() == ()
