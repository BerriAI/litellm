from __future__ import annotations

import itertools
import uuid
from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from _pytest.mark.structures import ParameterSet
from integration._support.client import JSON_OBJECT, Gateway, Scenario
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Wire, wire_server
from integration.providers._gemini_long_decode_support import (
    API_KEY,
    BACKEND,
    CAP,
    CUT_TOKEN,
    FIRST_HALF,
    HOP_TOKEN,
    INPUT_COST,
    NO_CACHE,
    OUTPUT_COST,
    PLAIN,
    PROVIDERS,
    RESUMED,
    SECOND_HALF,
    Endpoint,
    Provider,
    answered,
    answers,
    at,
    complete,
    post,
    received,
    register,
    request_body,
    upstream,
)
from pydantic import JsonValue

from litellm._catalog import CATALOG

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_BUDGET_ENV: Final = "LITELLM_GEMINI_DEFAULT_MAX_OUTPUT_TOKENS"
_SMALL_BACKEND: Final = "gemini-2.5-flash"
_LIMIT_KEYS: Final = ("max_output_tokens", "maxOutputTokens")
_JOINED: Final = FIRST_HALF.text + SECOND_HALF.text


def _row_limit(key: str) -> int:
    limit: Final = JSON_OBJECT.validate_python(CATALOG.bundled().served()[key]).get("max_output_tokens")
    assert isinstance(limit, int), (key, limit)
    return limit


_SMALL_LIMIT: Final = _row_limit(f"gemini/{_SMALL_BACKEND}")
_LARGE_LIMIT: Final = min(_row_limit(f"{provider}/{BACKEND}") for provider in PROVIDERS)
_BUDGET: Final = (_SMALL_LIMIT + _LARGE_LIMIT) // 2


@pytest.fixture(scope="module")
def budgeted(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    assert _SMALL_LIMIT < _BUDGET < _LARGE_LIMIT, (_SMALL_LIMIT, _BUDGET, _LARGE_LIMIT)
    directory: Final = tmp_path_factory.mktemp("gemini-default-output-budget")
    with owned_proxy(session_gateway, directory, {_BUDGET_ENV: str(_BUDGET)}, workers=2) as owned:
        yield owned


def _limits_sent(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    config: Final = body.get("generationConfig")
    assert isinstance(config, dict), body
    return {key: config[key] for key in _LIMIT_KEYS if key in config}


def _mode(stream: bool) -> str:
    return "stream" if stream else "sync"


_REACHED: Final[tuple[ParameterSet, ...]] = tuple(
    pytest.param(provider, endpoint, stream, id=f"{provider}-{endpoint}-{_mode(stream)}")
    for provider, endpoint, stream in (
        ("gemini", "chat", False),
        ("gemini", "chat", True),
        ("vertex_ai", "chat", False),
        ("vertex_ai", "chat", True),
        ("gemini", "responses", False),
        ("vertex_ai", "responses", True),
    )
)


@pytest.mark.parametrize(("provider", "endpoint", "stream"), _REACHED)
def test_request_with_no_output_limit_reaches_google_with_the_env_budget(
    budgeted: Gateway, provider: Provider, endpoint: Endpoint, stream: bool
) -> None:
    with wire_server(upstream(answers(f"plain-{uuid.uuid4().hex}", PLAIN))) as wire, budgeted.scenario() as scenario:
        model: Final = register(budgeted, scenario, provider, wire)
        response: Final = post(budgeted, endpoint, request_body(endpoint, model, stream, cap=None))
        observed: Final = answered(endpoint, stream, response)
        assert (observed.answer, observed.finishes) == (PLAIN.text, (complete(endpoint),)), observed
        (only,) = received(wire, provider, stream, (False,))
        assert _limits_sent(only) == {"max_output_tokens": _BUDGET}, only


@pytest.mark.parametrize(
    ("provider", "stream"),
    [
        pytest.param(provider, stream, id=f"{provider}-{_mode(stream)}")
        for provider, stream in itertools.product(PROVIDERS, (False, True))
    ],
)
def test_request_with_no_output_limit_is_continued_into_one_joined_answer(
    budgeted: Gateway, provider: Provider, stream: bool
) -> None:
    respond: Final = upstream(
        answers(f"first-{uuid.uuid4().hex}", FIRST_HALF), answers(f"second-{uuid.uuid4().hex}", SECOND_HALF)
    )
    with wire_server(respond) as wire, budgeted.scenario() as scenario:
        model: Final = register(budgeted, scenario, provider, wire)
        response: Final = post(budgeted, "chat", request_body("chat", model, stream, cap=None))
        observed: Final = answered("chat", stream, response)
        assert (observed.answer, observed.streamed, observed.finishes) == (_JOINED, _JOINED, ("stop",)), observed
        assert observed.tokens == () and HOP_TOKEN not in observed.raw, observed.raw
        opening, continuing = received(wire, provider, stream, (False, True))
        assert _limits_sent(opening) == {"max_output_tokens": _BUDGET}, opening
        assert continuing == {**opening, "continuationToken": HOP_TOKEN}, (opening, continuing)


@pytest.mark.parametrize(
    ("endpoint", "limit", "sent"),
    [
        pytest.param("chat", {"max_tokens": CAP}, {"max_output_tokens": CAP}, id="chat-max_tokens"),
        pytest.param(
            "chat", {"max_completion_tokens": CAP}, {"max_output_tokens": CAP}, id="chat-max_completion_tokens"
        ),
        pytest.param(
            "chat",
            {"extra_body": {"generationConfig": {"maxOutputTokens": CAP}}},
            {"maxOutputTokens": CAP},
            id="chat-extra_body-generationConfig",
        ),
        pytest.param(
            "responses", {"max_output_tokens": CAP}, {"max_output_tokens": CAP}, id="responses-max_output_tokens"
        ),
    ],
)
def test_callers_own_output_limit_wins_over_the_env_budget(
    budgeted: Gateway, endpoint: Endpoint, limit: Mapping[str, JsonValue], sent: Mapping[str, JsonValue]
) -> None:
    with wire_server(upstream(answers(f"plain-{uuid.uuid4().hex}", PLAIN))) as wire, budgeted.scenario() as scenario:
        model: Final = register(budgeted, scenario, "gemini", wire)
        body: Final = request_body(endpoint, model, False, {**NO_CACHE, **limit}, cap=None)
        observed: Final = answered(endpoint, False, post(budgeted, endpoint, body))
        assert observed.answer == PLAIN.text, observed
        (only,) = received(wire, "gemini", False, (False,))
        assert _limits_sent(only) == sent, only


def _deploy(scenario: Scenario, wire: Wire, backend: str, model_info: Mapping[str, JsonValue] | None) -> str:
    return scenario.model(
        model=f"gemini/{backend}",
        api_base=wire.url,
        api_key=API_KEY,
        input_cost_per_token=INPUT_COST,
        output_cost_per_token=OUTPUT_COST,
        model_info=model_info,
    )


def test_env_budget_is_capped_per_deployment_and_never_at_a_siblings_limit(budgeted: Gateway) -> None:
    lowered: Final = _BUDGET // 4
    with wire_server(upstream(answers(f"plain-{uuid.uuid4().hex}", PLAIN))) as wire, budgeted.scenario() as scenario:
        deployments: Final = {
            "row-below-budget": _deploy(scenario, wire, _SMALL_BACKEND, None),
            "model-info-above-row": _deploy(scenario, wire, _SMALL_BACKEND, {"max_output_tokens": 2 * _SMALL_LIMIT}),
            "model-info-below-budget": _deploy(scenario, wire, BACKEND, {"max_output_tokens": lowered}),
            "sibling-without-model-info": _deploy(scenario, wire, BACKEND, None),
        }

        def limit_sent(model: str) -> dict[str, JsonValue]:
            _ = answered("chat", False, post(budgeted, "chat", request_body("chat", model, False, cap=None)))
            (only,) = wire.drain()
            return _limits_sent(JSON_OBJECT.validate_json(only.body))

        assert {name: limit_sent(model) for name, model in deployments.items()} == {
            "row-below-budget": {"max_output_tokens": _SMALL_LIMIT},
            "model-info-above-row": {"max_output_tokens": _SMALL_LIMIT},
            "model-info-below-budget": {"max_output_tokens": lowered},
            "sibling-without-model-info": {"max_output_tokens": _BUDGET},
        }


def test_resume_with_a_continuation_token_gets_no_env_budget(budgeted: Gateway) -> None:
    respond: Final = upstream(
        answers(f"restarted-{uuid.uuid4().hex}", PLAIN), answers(f"resumed-{uuid.uuid4().hex}", RESUMED)
    )
    with wire_server(respond) as wire, budgeted.scenario() as scenario:
        model: Final = register(budgeted, scenario, "gemini", wire)
        body: Final = request_body("chat", model, False, {**NO_CACHE, "continuation_token": CUT_TOKEN}, cap=None)
        observed: Final = answered("chat", False, post(budgeted, "chat", body))
        assert (observed.answer, observed.finishes) == (RESUMED.text, ("stop",)), observed
        (only,) = received(wire, "gemini", False, (True,))
        assert only["continuationToken"] == CUT_TOKEN, only
        assert [at(only, "generationConfig", key) for key in _LIMIT_KEYS] == [None, None], only
