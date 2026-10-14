from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

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
from pydantic import JsonValue
from typing_extensions import assert_never

Provider = Literal[
    "anthropic", "bedrock_converse", "bedrock_invoke", "bedrock_claude_platform", "vertex_ai", "azure_ai"
]
Site = Literal["tool", "system", "user_part", "tool_call", "tool_result", "string_message"]

_PROVIDERS: Final[tuple[Provider, ...]] = (
    "anthropic",
    "bedrock_converse",
    "bedrock_invoke",
    "bedrock_claude_platform",
    "vertex_ai",
    "azure_ai",
)
_ANTHROPIC_BODY_PROVIDERS: Final = tuple(provider for provider in _PROVIDERS if provider != "bedrock_converse")
_SITES: Final[tuple[Site, ...]] = ("tool", "system", "user_part", "tool_call", "tool_result", "string_message")
_FIVE_SITES: Final[tuple[Site, ...]] = ("tool", "system", "user_part", "tool_call", "string_message")
_EXPLICIT: Final[dict[str, JsonValue]] = {"mode": "explicit"}
_BREAKPOINT: Final[Mapping[str, JsonValue]] = MappingProxyType({"prompt_cache_breakpoint": _EXPLICIT})
_NATIVE: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache_control": EPHEMERAL})
_ONE_HOUR: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
_CITY: Final = CITIES[0]
_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"


def _respond(provider: Provider, request: Request) -> Reply:
    return bedrock_peer(request) if provider == "bedrock_converse" else anthropic_peer(request)


def _labels(provider: Provider, request: Request) -> list[str]:
    return bedrock_labels(request) if provider == "bedrock_converse" else anthropic_labels(request)


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
        {"role": "tool", "tool_call_id": call_id(_CITY), "content": "sunny.", **at("tool_result")},
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
        case "tool_result":
            return f"user:toolResult:{call_id(_CITY)}" if converse else f"user:tool_result:{call_id(_CITY)}"
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
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert _labels(provider, request) == [_expected(provider, site, marker)]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_breakpoint_under_a_tool_function_marks_the_tool(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    function: Final = TOOL["function"]
    assert isinstance(function, dict), TOOL
    tool: Final[dict[str, JsonValue]] = {**TOOL, "function": {**function, "prompt_cache_breakpoint": _EXPLICIT}}
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, {**_body(model, marker, _marked(())), "tools": [tool]})
        assert status == 200, text
        request: Final = _only_request(wire)
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert _labels(provider, request) == [_expected(provider, "tool", marker)]


def test_breakpoint_ttl_claude_supports_is_kept(gateway: Gateway) -> None:
    marker: Final = new_marker()
    marks: Final = _marked(("user_part",), {"prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}})
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "anthropic")
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert anthropic_marks(request) == (Mark(ASK_LABEL, "1h"),)


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_breakpoint_ttl_claude_lacks_is_a_400_before_any_provider_call(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    marks: Final = _marked(("user_part",), {"prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"}})
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        received: Final = wire.drain()
    assert status == 400, text
    assert "30m" in text and "drop_params" in text, text
    assert received == (), [request.target for request in received]


def test_breakpoint_ttl_claude_lacks_keeps_the_mark_under_drop_params(gateway: Gateway) -> None:
    marker: Final = new_marker()
    marks: Final = _marked(("user_part",), {"prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"}})
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "anthropic", drop_params=True)
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert anthropic_marks(request) == (Mark(ASK_LABEL, None),)


def test_malformed_breakpoint_is_a_400_before_any_provider_call(gateway: Gateway) -> None:
    marker: Final = new_marker()
    marks: Final = _marked(("user_part",), {"prompt_cache_breakpoint": {"mode": "sometimes"}})
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "anthropic")
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        received: Final = wire.drain()
    assert status == 400, text
    assert "prompt_cache_breakpoint" in text and "drop_params" in text, text
    assert received == (), [request.target for request in received]


def test_a_client_cache_control_on_the_same_block_wins_over_the_breakpoint(gateway: Gateway) -> None:
    marker: Final = new_marker()
    hour: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
    marks: Final = _marked(("user_part",), {"cache_control": hour, "prompt_cache_breakpoint": _EXPLICIT})
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "anthropic")
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert anthropic_marks(request) == (Mark(ASK_LABEL, "1h"),)


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


@pytest.mark.parametrize("mark", (_BREAKPOINT, _NATIVE), ids=("breakpoint", "cache_control"))
@pytest.mark.parametrize("provider", _PROVIDERS)
def test_five_marks_without_drop_params_reach_the_provider_and_its_error_reaches_the_caller(
    gateway: Gateway, provider: Provider, mark: Mapping[str, JsonValue]
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, _body(model, marker, _marked(_FIVE_SITES, mark)))
        received: Final = wire.drain()
    assert status == 400, text
    assert "Found 5" in text, text
    assert [len(_labels(provider, request)) for request in received] == [5], [request.target for request in received]


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_drop_params_drops_the_breakpoint_past_the_cap_and_keeps_every_mark_the_client_wrote(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    written: Final = _FIVE_SITES[:-1]
    marks: Final = MappingProxyType({**_marked(written, _NATIVE), _FIVE_SITES[-1]: _BREAKPOINT})
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider, drop_params=True)
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert _labels(provider, request) == [_expected(provider, site, marker) for site in written]


@pytest.mark.parametrize("provider", _ANTHROPIC_BODY_PROVIDERS)
def test_a_breakpoint_ahead_of_a_one_hour_mark_reaches_claude_as_one_hour(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    marks: Final = MappingProxyType(
        {"system": _BREAKPOINT, "user_part": MappingProxyType({"cache_control": _ONE_HOUR})}
    )
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert anthropic_marks(request) == (Mark(SYSTEM_LABEL, "1h"), Mark(ASK_LABEL, "1h"))


def test_a_one_hour_breakpoint_after_a_five_minute_mark_loses_its_ttl_under_drop_params(gateway: Gateway) -> None:
    marker: Final = new_marker()
    marks: Final = MappingProxyType(
        {"system": _NATIVE, "user_part": MappingProxyType({"prompt_cache_breakpoint": {**_EXPLICIT, "ttl": "1h"}})}
    )
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "anthropic", drop_params=True)
        status, _, text = post_chat(gateway, _body(model, marker, marks))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert anthropic_marks(request) == (Mark(SYSTEM_LABEL, None), Mark(ASK_LABEL, None))


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


_PROFILE: Final = "arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/scripted"


@pytest.mark.parametrize(
    ("provider", "model", "base_model", "marked"),
    (
        ("bedrock_converse", f"bedrock/converse/{_PROFILE}", None, False),
        ("bedrock_converse", f"bedrock/converse/{_PROFILE}", "bedrock/mistral.mistral-large-2407-v1:0", False),
        ("bedrock_converse", f"bedrock/converse/{_PROFILE}", BEDROCK_MODEL, True),
        ("anthropic", "anthropic/glm-4.6", None, False),
        ("anthropic", "anthropic/glm-4.6", f"anthropic/{ANTHROPIC_MODEL}", True),
    ),
)
def test_a_model_id_without_claude_is_marked_only_when_its_base_model_is_claude(
    gateway: Gateway, provider: Provider, model: str, base_model: str | None, marked: bool
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: _respond(provider, request)) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(
            model_info=None if base_model is None else {"base_model": base_model},
            **{**_litellm_params(gateway, wire, provider), "model": model},
        )
        status, _, text = post_chat(gateway, _body(deployment, marker, _marked(("system",))))
        assert status == 200, text
        request: Final = _only_request(wire)
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert _labels(provider, request) == ([_expected(provider, "system", marker)] if marked else [])


def _drop_params_config(directory: Path) -> Path:
    return owned_config(directory, [], litellm_settings={"drop_params": True})


@pytest.fixture(scope="module")
def drop_params_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """A proxy running with `litellm_settings: drop_params: true`"""
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
