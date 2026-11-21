"""A chat completions request bridged to OpenAI's Responses API keeps a system prompt longer than the Responses
`instructions` cap out of `instructions`, so the caller is not answered with OpenAI's string_above_max_length 400"""

from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import wire_server
from integration.providers._responses_bridge_long_instructions import (
    ANSWER,
    DEPLOYMENT_SHAPES,
    OPENAI_INSTRUCTIONS_MAX_CHARS,
    USER_TEXT,
    DeploymentShape,
    assert_system_text_moved_into_input,
    assert_system_text_stays_instructions,
    bridged_model,
    openai_responses_peer,
    posted_bodies,
    wire_shape,
)
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_LONG_SYSTEM: Final = "a" * (OPENAI_INSTRUCTIONS_MAX_CHARS + 1)
_BOUNDARY_SYSTEM: Final = "a" * OPENAI_INSTRUCTIONS_MAX_CHARS
_SHORT_SYSTEM: Final = "You are a careful policy reviewer."


def _chat(gateway: Gateway, model: str, system: str, **fields: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": USER_TEXT}],
            **fields,
        },
    )


def _assert_answered(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    choices: Final = _JSON_OBJECT.validate_json(response.content)["choices"]
    assert isinstance(choices, list) and choices, response.text
    first: Final = _JSON_OBJECT.validate_python(choices[0])
    assert _JSON_OBJECT.validate_python(first["message"])["content"] == ANSWER, response.text


@pytest.mark.parametrize("shape", DEPLOYMENT_SHAPES)
def test_regression_long_system_prompt_with_auto_truncation_reaches_openai_as_input(
    gateway: Gateway, shape: DeploymentShape
) -> None:
    with wire_server(openai_responses_peer()) as wire, gateway.scenario() as scenario:
        model: Final = bridged_model(scenario, shape, wire.url)
        response: Final = _chat(gateway, model, _LONG_SYSTEM, truncation="auto")
        bodies: Final = posted_bodies(wire.drain())
    for body in bodies:
        assert body.get("truncation") == "auto", body.get("truncation")
        assert_system_text_moved_into_input(body, _LONG_SYSTEM)
    assert len(bodies) == 1, [wire_shape(body) for body in bodies]
    _assert_answered(response)


@pytest.mark.parametrize("shape", DEPLOYMENT_SHAPES)
def test_guard_short_system_prompt_with_auto_truncation_stays_instructions(
    gateway: Gateway, shape: DeploymentShape
) -> None:
    with wire_server(openai_responses_peer()) as wire, gateway.scenario() as scenario:
        model: Final = bridged_model(scenario, shape, wire.url)
        response: Final = _chat(gateway, model, _SHORT_SYSTEM, truncation="auto")
        (body,) = posted_bodies(wire.drain())
    assert body.get("truncation") == "auto", body.get("truncation")
    assert_system_text_stays_instructions(body, _SHORT_SYSTEM)
    _assert_answered(response)


@pytest.mark.parametrize("shape", DEPLOYMENT_SHAPES)
def test_guard_system_prompt_exactly_at_the_instructions_cap_stays_instructions(
    gateway: Gateway, shape: DeploymentShape
) -> None:
    with wire_server(openai_responses_peer()) as wire, gateway.scenario() as scenario:
        model: Final = bridged_model(scenario, shape, wire.url)
        response: Final = _chat(gateway, model, _BOUNDARY_SYSTEM, truncation="auto")
        (body,) = posted_bodies(wire.drain())
    assert body.get("truncation") == "auto", body.get("truncation")
    assert_system_text_stays_instructions(body, _BOUNDARY_SYSTEM)
    _assert_answered(response)
