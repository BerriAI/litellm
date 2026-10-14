from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, assert_never

import pytest
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    ASK,
    ASK_LABEL,
    BEDROCK_MODEL,
    CITIES,
    EPHEMERAL,
    POINTS,
    PROVIDER_KEY,
    SYSTEM,
    SYSTEM_LABEL,
    TOOL,
    Mark,
    anthropic_labels,
    anthropic_marks,
    anthropic_peer,
    bedrock_labels,
    bedrock_peer,
    call_id,
    chat_body,
    final_label,
    final_text,
    gateway_injected,
    new_marker,
    owned_config,
    post_chat,
    tool_call,
    tool_use_label,
)
from pydantic import BaseModel, ConfigDict, JsonValue

Provider = Literal[
    "anthropic", "bedrock_converse", "bedrock_invoke", "bedrock_claude_platform", "vertex_ai", "azure_ai"
]
Site = Literal["tool", "system", "user_part", "tool_call", "tool_result_part", "string_message"]

_PROVIDERS: Final[tuple[Provider, ...]] = (
    "anthropic",
    "bedrock_converse",
    "bedrock_invoke",
    "bedrock_claude_platform",
    "vertex_ai",
    "azure_ai",
)
_EXTRA_BODY_PROVIDERS: Final[tuple[Provider, ...]] = ("bedrock_invoke", "bedrock_claude_platform", "azure_ai")
_SITES: Final[tuple[Site, ...]] = ("tool", "system", "user_part", "tool_call", "tool_result_part", "string_message")
_FIVE_SITES: Final[tuple[Site, ...]] = ("tool", "system", "user_part", "tool_call", "string_message")
_EXPLICIT: Final[dict[str, JsonValue]] = {"mode": "explicit"}
_EXPLICIT_30M: Final[dict[str, JsonValue]] = {"mode": "explicit", "ttl": "30m"}
_BREAKPOINT: Final[Mapping[str, JsonValue]] = MappingProxyType({"prompt_cache_breakpoint": _EXPLICIT})
_NATIVE: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache_control": EPHEMERAL})
_CITY: Final = CITIES[0]
_TOOL_RESULT_TEXT: Final = "sunny."
_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"
_TARGETS: Final[Mapping[Provider, str]] = MappingProxyType(
    {
        "anthropic": "/v1/messages",
        "bedrock_converse": f"/model/{BEDROCK_MODEL}/converse",
        "bedrock_invoke": f"/model/{BEDROCK_MODEL}/invoke",
        "bedrock_claude_platform": "/v1/messages",
        "vertex_ai": (
            f"/v1/projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}"
            f"/publishers/anthropic/models/{ANTHROPIC_MODEL}:rawPredict"
        ),
        "azure_ai": "/anthropic/v1/messages",
    }
)
_CAP_HINT: Final = "Claude accepts at most 4 cache breakpoints, found 5"


class _Nested(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    type: str
    text: str | None = None
    cache_control: JsonValue = None


class _Block(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    type: str
    tool_use_id: str | None = None
    content: str | tuple[_Nested, ...] | None = None


class _Message(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    role: str
    content: str | tuple[_Block, ...]


class _Body(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    messages: tuple[_Message, ...] = ()


def _nested_blocks(body: _Body) -> Iterator[tuple[_Block, _Nested]]:
    for message in body.messages:
        for block in message.content if isinstance(message.content, tuple) else ():
            if isinstance(block.content, tuple):
                yield from ((block, inner) for inner in block.content)


def _nested_tool_result_labels(request: Request) -> list[str]:
    return [
        f"user:tool_result:{block.tool_use_id}:{inner.type}:{inner.text}"
        for block, inner in _nested_blocks(_Body.model_validate_json(request.body))
        if inner.cache_control is not None
    ]


def _labels(provider: Provider, request: Request) -> list[str]:
    if provider == "bedrock_converse":
        return bedrock_labels(request)
    return [*anthropic_labels(request), *_nested_tool_result_labels(request)]


def _respond(provider: Provider, request: Request) -> Reply:
    return bedrock_peer(request) if provider == "bedrock_converse" else anthropic_peer(request)


def _litellm_params(gateway: Gateway, wire: Wire, provider: Provider) -> dict[str, JsonValue]:
    match provider:
        case "anthropic":
            return {"model": f"anthropic/{ANTHROPIC_MODEL}", "api_base": wire.url, "api_key": PROVIDER_KEY}
        case "bedrock_converse":
            return {
                "model": f"bedrock/converse/{BEDROCK_MODEL}",
                "api_key": PROVIDER_KEY,
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire.url,
            }
        case "bedrock_invoke":
            return {
                "model": f"bedrock/invoke/{BEDROCK_MODEL}",
                "api_key": _BEDROCK_TOKEN,
                "aws_region_name": "us-east-1",
                "api_base": wire.url,
                "aws_bedrock_runtime_endpoint": wire.url,
            }
        case "bedrock_claude_platform":
            return {
                "model": f"bedrock/claude_platform/{ANTHROPIC_MODEL}",
                "api_base": wire.url,
                "api_key": PROVIDER_KEY,
                "workspace_id": "wrkspc_scripted",
                "aws_region_name": "us-east-1",
            }
        case "vertex_ai":
            return {
                "model": f"vertex_ai/{ANTHROPIC_MODEL}",
                "api_base": wire.url,
                "api_key": None,
                "vertex_project": _VERTEX_PROJECT,
                "vertex_location": _VERTEX_LOCATION,
                "vertex_credentials": service_account_json(_VERTEX_PROJECT, gateway.upstream_url.rstrip("/")),
            }
        case "azure_ai":
            return {"model": f"azure_ai/{ANTHROPIC_MODEL}", "api_base": wire.url, "api_key": PROVIDER_KEY}
        case _:
            assert_never(provider)


def _deployment(gateway: Gateway, scenario: Scenario, wire: Wire, provider: Provider, **fields: JsonValue) -> str:
    return scenario.model(model_info=None, **_litellm_params(gateway, wire, provider), **fields)


def _messages(marker: str, marks: Mapping[Site, Mapping[str, JsonValue]]) -> list[JsonValue]:
    def at(site: Site) -> Mapping[str, JsonValue]:
        return marks.get(site, {})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM, **at("system")}]},
        {"role": "user", "content": [{"type": "text", "text": ASK, **at("user_part")}]},
        {"role": "assistant", "content": "", "tool_calls": [tool_call(_CITY, **at("tool_call"))]},
        {
            "role": "tool",
            "tool_call_id": call_id(_CITY),
            "content": [{"type": "text", "text": _TOOL_RESULT_TEXT, **at("tool_result_part")}],
        },
        {"role": "user", "content": final_text(marker), **at("string_message")},
    ]


def _body(
    model: str, marker: str, marks: Mapping[Site, Mapping[str, JsonValue]], **fields: JsonValue
) -> dict[str, JsonValue]:
    return {**chat_body(model, _messages(marker, marks), **fields), "tools": [{**TOOL, **marks.get("tool", {})}]}


def _marked(
    sites: tuple[Site, ...], mark: Mapping[str, JsonValue] = _BREAKPOINT
) -> Mapping[Site, Mapping[str, JsonValue]]:
    return MappingProxyType({site: mark for site in sites})


def _expected(provider: Provider, site: Site, marker: str) -> str:
    converse: Final = provider == "bedrock_converse"
    match site:
        case "tool":
            return "tool:lookup_weather"
        case "system":
            return f"system:text:{SYSTEM}" if converse else SYSTEM_LABEL
        case "user_part":
            return ASK_LABEL
        case "tool_call":
            return f"assistant:toolUse:{call_id(_CITY)}" if converse else tool_use_label(_CITY)
        case "tool_result_part":
            if converse:
                return f"user:toolResult:{call_id(_CITY)}"
            return f"user:tool_result:{call_id(_CITY)}:text:{_TOOL_RESULT_TEXT}"
        case "string_message":
            return final_label(marker)
        case _:
            assert_never(site)


def _latest_four(provider: Provider, marker: str) -> list[str]:
    return [_expected(provider, site, marker) for site in _FIVE_SITES[1:]]


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
        status, _, text = post_chat(gateway, _body(model, marker, _marked((site,))))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert request.target == _TARGETS[provider], request.target
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert _labels(provider, request) == [_expected(provider, site, marker)]


def test_breakpoint_ttl_is_not_forwarded(gateway: Gateway) -> None:
    marker: Final = new_marker()
    marks: Final = _marked(("user_part",), {"prompt_cache_breakpoint": _EXPLICIT_30M})
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "anthropic")
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert anthropic_marks(request) == (Mark(ASK_LABEL, None),)
    assert b"30m" not in request.body, request.body


def test_a_client_cache_control_on_the_same_block_wins_over_the_breakpoint(gateway: Gateway) -> None:
    marker: Final = new_marker()
    hour: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
    marks: Final = _marked(("user_part",), {"cache_control": hour, "prompt_cache_breakpoint": _EXPLICIT_30M})
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "anthropic")
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert anthropic_marks(request) == (Mark(ASK_LABEL, "1h"),)
    assert b"prompt_cache_breakpoint" not in request.body, request.body


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_a_block_carrying_both_keys_keeps_one_cache_mark_and_loses_the_breakpoint(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    marks: Final = _marked(("user_part",), {"cache_control": EPHEMERAL, "prompt_cache_breakpoint": _EXPLICIT_30M})
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert _labels(provider, request) == [_expected(provider, "user_part", marker)]
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert b"30m" not in request.body, request.body


@pytest.mark.parametrize("source", ("deployment", "request"))
@pytest.mark.parametrize("provider", _PROVIDERS)
def test_five_breakpoints_keep_the_latest_four_under_drop_params(
    gateway: Gateway, provider: Provider, source: str
) -> None:
    marker: Final = new_marker()
    deployment: Final[dict[str, JsonValue]] = {"drop_params": True} if source == "deployment" else {}
    request_fields: Final[dict[str, JsonValue]] = {"drop_params": True} if source == "request" else {}
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider, **deployment)
        status, _, text = post_chat(gateway, _body(model, marker, _marked(_FIVE_SITES), **request_fields))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert _labels(provider, request) == _latest_four(provider, marker)


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_five_breakpoints_are_a_400_before_any_provider_call_without_drop_params(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, _body(model, marker, _marked(_FIVE_SITES)))
        received: Final = wire.drain()
    assert status == 400, text
    assert _CAP_HINT in text, text
    assert "drop_params" in text, text
    assert received == (), [request.target for request in received]


def test_a_tool_result_with_two_marked_parts_is_one_converse_cache_point_within_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    marked_tool_result: Final[dict[str, JsonValue]] = {
        "role": "tool",
        "tool_call_id": call_id(_CITY),
        "content": [{"type": "text", "text": text, **_BREAKPOINT} for text in (_TOOL_RESULT_TEXT, "warm.")],
    }
    messages: Final = _messages(marker, _marked(("system", "user_part", "string_message")))
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "bedrock_converse")
        status, _, text = post_chat(gateway, chat_body(model, [*messages[:3], marked_tool_result, *messages[4:]]))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert bedrock_labels(request) == [
        _expected("bedrock_converse", site, marker)
        for site in ("system", "user_part", "tool_result_part", "string_message")
    ]


def test_injection_points_stand_down_behind_the_latest_four_breakpoints(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(
            gateway, scenario, wire, "anthropic", cache_control_injection_points=POINTS, drop_params=True
        )
        status, response_id, text = post_chat(gateway, _body(model, marker, _marked(_FIVE_SITES)))
        assert status == 200, text
        assert anthropic_labels(_only_request(wire)) == _latest_four("anthropic", marker)
    assert gateway_injected(response_id) is False


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_prompt_cache_options_are_a_400_before_any_provider_call_without_drop_params(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(
            gateway, _body(model, marker, _marked(("user_part",)), prompt_cache_options=_EXPLICIT_30M)
        )
        received: Final = wire.drain()
    assert status == 400, text
    assert "prompt_cache_options" in text, text
    assert "drop_params" in text, text
    assert received == (), [request.target for request in received]


@pytest.mark.parametrize("provider", _EXTRA_BODY_PROVIDERS)
def test_prompt_cache_options_in_extra_body_are_a_400_before_any_provider_call_without_drop_params(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    extra_body: Final[dict[str, JsonValue]] = {"prompt_cache_options": _EXPLICIT}
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, _body(model, marker, _marked(("user_part",)), extra_body=extra_body))
        received: Final = wire.drain()
    assert status == 400, text
    assert "prompt_cache_options" in text, text
    assert "drop_params" in text, text
    assert received == (), [request.target for request in received]


@pytest.mark.parametrize("provider", _EXTRA_BODY_PROVIDERS)
def test_prompt_cache_options_in_extra_body_are_dropped_under_drop_params(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    extra_body: Final[dict[str, JsonValue]] = {"prompt_cache_options": _EXPLICIT}
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider, drop_params=True)
        status, _, text = post_chat(gateway, _body(model, marker, _marked(("user_part",)), extra_body=extra_body))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert b"prompt_cache_options" not in request.body, request.body
    assert _labels(provider, request) == [ASK_LABEL]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_prompt_cache_options_are_dropped_under_drop_params(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider, drop_params=True)
        status, _, text = post_chat(
            gateway, _body(model, marker, _marked(("user_part",)), prompt_cache_options=_EXPLICIT_30M)
        )
        assert status == 200, text
        request: Final = _only_request(wire)
    assert b"prompt_cache_options" not in request.body, request.body
    assert _labels(provider, request) == [ASK_LABEL]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_five_native_cache_control_marks_are_forwarded_and_the_provider_error_reaches_the_caller(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, _body(model, marker, _marked(_FIVE_SITES, _NATIVE)))
        received: Final = wire.drain()
    assert status == 400, text
    assert "Found 5" in text, text
    assert _CAP_HINT not in text, text
    assert [len(_labels(provider, request)) for request in received] == [5], [request.target for request in received]


def _drop_params_config(directory: Path) -> Path:
    return owned_config(directory, [], litellm_settings={"drop_params": True})


@pytest.fixture(scope="module")
def drop_params_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """A proxy running with `litellm_settings: drop_params: true`, the setting the cap error names"""
    with gateway_from_environment() as lane:
        directory: Final = tmp_path_factory.mktemp("drop_params")
        with owned_proxy(lane, directory, {}, config=_drop_params_config(directory)) as owned:
            yield owned


@pytest.mark.timeout(240)
def test_proxy_level_drop_params_keeps_the_latest_four_breakpoints(drop_params_gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, drop_params_gateway.scenario() as scenario:
        model: Final = _deployment(drop_params_gateway, scenario, wire, "anthropic")
        status, _, text = post_chat(drop_params_gateway, _body(model, marker, _marked(_FIVE_SITES)))
        assert status == 200, text
        assert anthropic_labels(_only_request(wire)) == _latest_four("anthropic", marker)
