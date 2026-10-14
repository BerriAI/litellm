import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, assert_never

import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ASK,
    ASK_LABEL,
    CITIES,
    CLAUDE_TARGETS,
    EPHEMERAL,
    SYSTEM,
    SYSTEM_LABEL,
    ClaudeRoute,
    Mark,
    anthropic_labels,
    anthropic_marks,
    anthropic_peer,
    bedrock_labels,
    bedrock_peer,
    call_id,
    claude_litellm_params,
    final_label,
    final_text,
    new_marker,
)
from pydantic import JsonValue

Provider = ClaudeRoute
Site = Literal[
    "developer_part", "developer_string", "user_part", "detail_part", "function_call_output", "string_message"
]
MarkKey = Literal["prompt_cache_breakpoint", "cache_control"]

_PROVIDERS: Final[tuple[Provider, ...]] = ("anthropic", "bedrock_converse", "bedrock_invoke", "vertex_ai", "azure_ai")
_SITES: Final[tuple[Site, ...]] = ("developer_part", "user_part", "function_call_output", "string_message")
_FIVE_SITES: Final[tuple[Site, ...]] = (
    "developer_part",
    "user_part",
    "detail_part",
    "function_call_output",
    "string_message",
)
_EXPLICIT_30M: Final[dict[str, JsonValue]] = {"mode": "explicit", "ttl": "30m"}
_MARKS: Final[Mapping[MarkKey, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "prompt_cache_breakpoint": MappingProxyType({"prompt_cache_breakpoint": _EXPLICIT_30M}),
        "cache_control": MappingProxyType({"cache_control": EPHEMERAL}),
    }
)
_CITY: Final = CITIES[0]
_DETAIL: Final = "Answer in Celsius."
_TOOL_RESULT_TEXT: Final = "sunny."
_CAP_HINT: Final = "Claude accepts at most 4 cache breakpoints, found 5"


@dataclass(frozen=True, slots=True)
class _Outcome:
    status: int
    text: str


def _respond(provider: Provider, request: Request) -> Reply:
    return bedrock_peer(request) if provider == "bedrock_converse" else anthropic_peer(request)


def _labels(provider: Provider, request: Request) -> list[str]:
    return bedrock_labels(request) if provider == "bedrock_converse" else anthropic_labels(request)


def _deployment(gateway: Gateway, scenario: Scenario, wire: Wire, provider: Provider, **fields: JsonValue) -> str:
    return scenario.model(model_info=None, **claude_litellm_params(gateway, wire, provider), **fields)


def _input(marker: str, marks: Mapping[Site, Mapping[str, JsonValue]]) -> list[JsonValue]:
    def at(site: Site) -> Mapping[str, JsonValue]:
        return marks.get(site, {})

    developer: Final[JsonValue] = (
        {"role": "developer", "content": SYSTEM, **at("developer_string")}
        if "developer_string" in marks
        else {"role": "developer", "content": [{"type": "input_text", "text": SYSTEM, **at("developer_part")}]}
    )
    return [
        developer,
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": ASK, **at("user_part")},
                {"type": "input_text", "text": _DETAIL, **at("detail_part")},
            ],
        },
        {
            "type": "function_call",
            "call_id": call_id(_CITY),
            "name": "lookup_weather",
            "arguments": json.dumps({"city": _CITY}),
        },
        {
            "type": "function_call_output",
            "call_id": call_id(_CITY),
            "output": _TOOL_RESULT_TEXT,
            **at("function_call_output"),
        },
        {"role": "user", "content": final_text(marker), **at("string_message")},
    ]


def _body(marker: str, marks: Mapping[Site, Mapping[str, JsonValue]], **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "input": _input(marker, marks),
        "tools": [
            {
                "type": "function",
                "name": "lookup_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }
        ],
        **fields,
    }


def _marked(
    sites: tuple[Site, ...], key: MarkKey = "prompt_cache_breakpoint"
) -> Mapping[Site, Mapping[str, JsonValue]]:
    return MappingProxyType({site: _MARKS[key] for site in sites})


def _expected(provider: Provider, site: Site, marker: str) -> str:
    converse: Final = provider == "bedrock_converse"
    match site:
        case "developer_part" | "developer_string":
            return f"system:text:{SYSTEM}" if converse else SYSTEM_LABEL
        case "user_part":
            return ASK_LABEL
        case "detail_part":
            return f"user:text:{_DETAIL}"
        case "function_call_output":
            return f"user:{'toolResult' if converse else 'tool_result'}:{call_id(_CITY)}"
        case "string_message":
            return final_label(marker)
        case _:
            assert_never(site)


def _create(gateway: Gateway, model: str, body: dict[str, JsonValue]) -> _Outcome:
    with (
        httpx.Client(timeout=30, trust_env=False) as transport,
        openai.OpenAI(
            api_key=gateway.key,
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            max_retries=0,
            http_client=transport,
        ) as client,
    ):
        try:
            response: Final = client.responses.create(model=model, max_output_tokens=64, extra_body=body)
        except openai.APIStatusError as error:
            return _Outcome(error.status_code, error.response.text)
    return _Outcome(200, response.output_text)


def _only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, [request.target for request in received]
    return received[0]


@pytest.mark.parametrize("site", _SITES)
@pytest.mark.parametrize("provider", _PROVIDERS)
def test_breakpoint_becomes_a_cache_mark_on_the_same_block(gateway: Gateway, provider: Provider, site: Site) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        outcome: Final = _create(gateway, model, _body(marker, _marked((site,))))
        assert outcome.status == 200, outcome.text
        request: Final = _only_request(wire)
    assert outcome.text == "sunny"
    assert request.target == CLAUDE_TARGETS[provider], request.target
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert b"30m" not in request.body, request.body
    assert _labels(provider, request) == [_expected(provider, site, marker)]


@pytest.mark.parametrize("site", ("developer_string", "function_call_output", "string_message"))
@pytest.mark.parametrize("key", ("prompt_cache_breakpoint", "cache_control"))
@pytest.mark.parametrize("provider", ("anthropic", "bedrock_converse"))
def test_an_item_level_mark_becomes_a_message_level_cache_mark(
    gateway: Gateway, provider: Provider, key: MarkKey, site: Site
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        outcome: Final = _create(gateway, model, _body(marker, _marked((site,), key)))
        assert outcome.status == 200, outcome.text
        request: Final = _only_request(wire)
    assert _labels(provider, request) == [_expected(provider, site, marker)]


def test_a_client_cache_control_on_the_same_part_wins_over_the_breakpoint(gateway: Gateway) -> None:
    marker: Final = new_marker()
    hour: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
    both: Final = MappingProxyType({"cache_control": hour, "prompt_cache_breakpoint": _EXPLICIT_30M})
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "anthropic")
        outcome: Final = _create(gateway, model, _body(marker, MappingProxyType({"user_part": both})))
        assert outcome.status == 200, outcome.text
        request: Final = _only_request(wire)
    assert anthropic_marks(request) == (Mark(ASK_LABEL, "1h"),)
    assert b"prompt_cache_breakpoint" not in request.body, request.body


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_an_unmarked_request_reaches_claude_without_cache_marks(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        outcome: Final = _create(gateway, model, _body(marker, MappingProxyType({})))
        assert outcome.status == 200, outcome.text
        request: Final = _only_request(wire)
    assert _labels(provider, request) == []
    assert b"cache_control" not in request.body, request.body
    assert b"cachePoint" not in request.body, request.body


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_five_breakpoints_keep_the_latest_four_under_drop_params(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider, drop_params=True)
        outcome: Final = _create(gateway, model, _body(marker, _marked(_FIVE_SITES)))
        assert outcome.status == 200, outcome.text
        request: Final = _only_request(wire)
    assert _labels(provider, request) == [_expected(provider, site, marker) for site in _FIVE_SITES[1:]]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_five_breakpoints_are_a_400_before_any_provider_call_without_drop_params(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        outcome: Final = _create(gateway, model, _body(marker, _marked(_FIVE_SITES)))
        received: Final = wire.drain()
    assert outcome.status == 400, outcome.text
    assert _CAP_HINT in outcome.text, outcome.text
    assert "drop_params" in outcome.text, outcome.text
    assert received == (), [request.target for request in received]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_prompt_cache_options_are_a_400_before_any_provider_call_without_drop_params(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        outcome: Final = _create(
            gateway, model, _body(marker, _marked(("user_part",)), prompt_cache_options=_EXPLICIT_30M)
        )
        received: Final = wire.drain()
    assert outcome.status == 400, outcome.text
    assert "prompt_cache_options" in outcome.text, outcome.text
    assert "drop_params" in outcome.text, outcome.text
    assert received == (), [request.target for request in received]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_prompt_cache_options_are_dropped_under_drop_params(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider, drop_params=True)
        outcome: Final = _create(
            gateway, model, _body(marker, _marked(("user_part",)), prompt_cache_options=_EXPLICIT_30M)
        )
        assert outcome.status == 200, outcome.text
        request: Final = _only_request(wire)
    assert b"prompt_cache_options" not in request.body, request.body
    assert _labels(provider, request) == [ASK_LABEL]
