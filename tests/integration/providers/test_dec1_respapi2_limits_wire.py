"""Budgets, model access, rate limits and spend on `/v1/responses` for bridged providers.

`/v1/responses` is an LLM route, so it passes the same auth checks as chat completions before any provider is
called: a key whose stored spend has reached its `max_budget` is refused with the budget envelope (status
`litellm.budget_exceeded_status_code`, 422 by default, `litellm/__init__.py:393`, type `budget_exceeded`), a key
whose `models` list does not name the requested model is refused with 403 and the client message
`MODEL_ACCESS_DENIED_CLIENT_MESSAGE` (`litellm/proxy/auth/auth_checks.py:5215`,
`litellm/proxy/auth/model_access_denied.py:5`), and an `rpm_limit` key is refused with 429 once its window is
used. A bridged call is priced from the chat usage the provider reported, so a key's stored spend grows by the
deployment price of that usage. These are the checks the Rust auth and budget ports must keep for the Responses
route, not only for `/v1/chat/completions`. Every deployment here is bridged (Anthropic, Gemini, Groq) and every
canned answer costs 0.025 (11 * 0.001 + 7 * 0.002).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from integration._support.wire import Reply
from integration.providers.dec1_respapi2_support import every_respond, gemini, key_spend
from integration.providers.dec1_respapi_support import (
    CALL_COST,
    Call,
    anthropic,
    chat_bridge,
    json_reply,
    peer,
    responses,
    served,
    stream_events,
)


def _error(response_content: bytes) -> dict[str, object]:
    return dict(object_value(JSON_OBJECT.validate_json(response_content)["error"]))


def test_a_key_at_its_budget_is_refused_on_responses_without_calling_the_provider(gateway: Gateway) -> None:
    """A key with `max_budget: 0.025` makes one bridged Anthropic call (cost 0.025) and its spend lands. The next
    `/v1/responses` call must be refused with 422 and `error.type: budget_exceeded`, and the Anthropic peer must
    not receive a second POST (spend at the budget is already over it, the same boundary the chat suite pins in
    `spend/test_cache_and_quota.py`)."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=CALL_COST)
        served(responses(gateway, model, key=key))
        key_spend(gateway, key, CALL_COST)
        denied: Final = eventually(
            lambda: responses(gateway, model, key=key),
            lambda response: response.status_code == 422,
            seconds=30,
        )
        posts_after_denial: Final = len(provider.posts())
        again: Final = responses(gateway, model, key=key)
        assert again.status_code == 422, again.text
        assert len(provider.posts()) == posts_after_denial
    assert _error(denied.content)["type"] == "budget_exceeded"


def test_a_streamed_request_over_budget_is_refused_before_any_event(gateway: Gateway) -> None:
    """A key whose stored spend was set to 1.0 with `max_budget: 0.5` asks for a streamed bridged response. The
    proxy must answer 422 `budget_exceeded` as a plain JSON error (no `response.created` event) and the peer
    receives nothing, because the budget check runs at auth before the stream starts."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.5)
        gateway.post("/key/update", {"key": key, "spend": 1.0})
        response: Final = eventually(
            lambda: gateway.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": "over budget", "stream": True, "cache": {"no-cache": True}},
                key=key,
            ),
            lambda answered: answered.status_code == 422,
            seconds=30,
        )
        assert provider.posts() == ()
    assert _error(response.content)["type"] == "budget_exceeded"
    assert "response.created" not in response.text


def test_a_key_without_the_model_is_refused_403_on_responses(gateway: Gateway) -> None:
    """A key whose `models` list names only another model asks `/v1/responses` for a bridged Gemini deployment.
    The proxy must answer 403 with the client message `The requested model '<model>' is not available for this
    API key, or the model name is invalid. ...` (`model_access_denied.py:5`) and type
    `key_model_access_denied`, and the Gemini peer receives nothing."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        model: Final = gemini(scenario, provider)
        key: Final = scenario.key(models=["dec1-respapi2-some-other-model"])
        response: Final = responses(gateway, model, key=key)
        assert provider.posts() == ()
    assert response.status_code == 403, response.text
    error: Final = _error(response.content)
    assert str(error["message"]).startswith(f"The requested model '{model}' is not available for this API key")
    assert error["type"] == "key_model_access_denied"


def test_a_key_rpm_limit_of_one_refuses_the_second_responses_call_with_429(gateway: Gateway) -> None:
    """A key with `rpm_limit: 1` makes two bridged Gemini calls in a row. The first is served; the second must be
    refused with 429, and the peer must have received exactly one POST, so the Responses route counts against the
    same per key request window as chat (`parallel_request_limiter_v3`). Gemini, not Groq: a Groq reply without
    `service_tier` fails (groq/chat/transformation.py:329) and the first llama-3 call loads a HuggingFace tokenizer
    for about 23 seconds where the network is closed, which timed this test out in run 44176682961."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        model: Final = gemini(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        served(responses(gateway, model, key=key))
        second: Final = responses(gateway, model, key=key)
        assert second.status_code == 429, second.text
        assert len(provider.posts()) == 1


def test_a_bridged_call_adds_its_price_to_the_key_spend(gateway: Gateway) -> None:
    """One non streamed bridged Gemini call (11 and 7 tokens at 0.001 and 0.002) must raise the key's stored spend
    in `LiteLLM_VerificationToken` from 0 to exactly 0.025, priced from the Gemini `usageMetadata` the bridge
    mapped to chat usage."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        model: Final = gemini(scenario, provider)
        key: Final = scenario.key(models=[model])
        served(responses(gateway, model, key=key))
        assert key_spend(gateway, key, CALL_COST) == pytest.approx(CALL_COST)


def test_a_streamed_bridged_call_adds_its_price_to_the_key_spend(gateway: Gateway) -> None:
    """One streamed bridged call on the chat carrier, whose usage arrives only in the last chat chunk, must raise
    the key's stored spend to exactly 0.025, so streaming on `/v1/responses` is billed to the key as well as to the
    spend log row (round 1 pins the row)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        key: Final = scenario.key(models=[model])
        stream_events(gateway, model, key=key)
        assert key_spend(gateway, key, CALL_COST) == pytest.approx(CALL_COST)


def test_a_team_at_its_budget_is_refused_on_responses(gateway: Gateway) -> None:
    """A team with `max_budget: 0.5` whose stored spend is set to 1.0 owns a key that asks `/v1/responses` for a
    bridged Anthropic deployment. Once the stored spend is seen (the key's cached team object may serve a few calls
    first, which run 44176682961 showed reaching the peer), the proxy refuses with 422 `budget_exceeded` and a
    refused call adds no provider call. UNVERIFIED: that `/team/update` accepts `spend` the way `/key/update`
    does; if not, the team spend must be reached with a priced call first."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        team: Final = scenario.team(max_budget=0.5, models=[model])
        gateway.post("/team/update", {"team_id": team, "spend": 1.0})
        key: Final = scenario.key(team_id=team, models=[model])
        eventually(
            lambda: responses(gateway, model, key=key),
            lambda answered: answered.status_code == 422,
            seconds=30,
        )
        before: Final = len(provider.posts())
        response: Final = responses(gateway, model, key=key)
        assert response.status_code == 422, response.text
        assert len(provider.posts()) == before, provider.posts()
    assert _error(response.content)["type"] == "budget_exceeded"


def test_a_provider_failure_on_responses_adds_nothing_to_the_key_spend(gateway: Gateway) -> None:
    """The Anthropic peer answers 400 to a bridged call. After a second, served call on the same key lands, the
    key's spend must be exactly 0.025 (only the served call), so a failed turn is never billed to the key."""
    answered: Final[list[int]] = []  # mutable-ok: counts the peer's POSTs from its handler thread

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return every_respond(call)
        answered.append(1)
        if len(answered) == 1:
            return json_reply(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "dec1 refused"}}, 400
            )
        return every_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        key: Final = scenario.key(models=[model])
        assert responses(gateway, model, key=key).status_code == 400
        served(responses(gateway, model, key=key))
        assert key_spend(gateway, key, CALL_COST) == pytest.approx(CALL_COST)
