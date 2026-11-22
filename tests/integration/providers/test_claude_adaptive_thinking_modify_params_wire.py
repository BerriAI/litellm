import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.anthropic_thinking import (
    MODEL,
    SIGNATURE,
    THINKING,
    answer,
    marker_of,
    prompt,
    standard_peer,
    thinking_block,
)
from integration._support.client import Gateway, Scenario, gateway_from_environment, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

Route = Literal["anthropic", "bedrock_converse", "bedrock_invoke", "vertex_ai"]

_LEGACY_MODEL: Final = "claude-sonnet-4-6"
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"
_LANE_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_ADAPTIVE: Final[dict[str, JsonValue]] = {"type": "adaptive"}
_LEGACY: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 1024}
_EFFORT: Final[dict[str, JsonValue]] = {"reasoning_effort": "high"}
_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _modify_params_config(directory: Path) -> Path:
    lane: Final = object_value(yaml.safe_load(_LANE_CONFIG.read_text()))
    settings: Final = object_value(lane["litellm_settings"])
    path: Final = directory / "proxy_config_modify_params.yaml"
    path.write_text(yaml.safe_dump({**lane, "litellm_settings": {**settings, "modify_params": True}}, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def modify_params_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with gateway_from_environment() as lane:
        directory: Final = tmp_path_factory.mktemp("adaptive_thinking_modify_params")
        with owned_proxy(lane, directory, {}, config=_modify_params_config(directory)) as owned:
            yield owned


def _peer(request: Request) -> Reply:
    if not request.target.endswith("/converse"):
        return standard_peer(request)
    converse: Final[dict[str, JsonValue]] = {
        "output": {"message": {"role": "assistant", "content": [{"text": answer(marker_of(request))}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 9, "totalTokens": 21},
        "metrics": {"latencyMs": 1},
    }
    return Reply(body=json.dumps(converse).encode())


def _bedrock(scenario: Scenario, model: str, wire_url: str, runtime_endpoint: str | None = None) -> str:
    return scenario.model(
        model=model,
        api_base=wire_url,
        aws_access_key_id="AKIASCRIPTEDPROVIDER",
        aws_secret_access_key="scripted-secret",
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=runtime_endpoint,
    )


def _deployment(scenario: Scenario, route: Route, model: str, wire_url: str, upstream_url: str) -> str:
    match route:
        case "anthropic":
            return scenario.model(model=f"anthropic/{model}", api_base=wire_url, api_key="scripted-key")
        case "bedrock_converse":
            return _bedrock(scenario, f"bedrock/converse/anthropic.{model}", wire_url)
        case "bedrock_invoke":
            return _bedrock(scenario, f"bedrock/invoke/anthropic.{model}", wire_url, runtime_endpoint=wire_url)
        case "vertex_ai":
            return scenario.model(
                model=f"vertex_ai/{model}",
                api_base=(
                    f"{wire_url}/v1/projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}"
                    f"/publishers/anthropic/models/{model}"
                ),
                api_key=None,
                vertex_project=_VERTEX_PROJECT,
                vertex_location=_VERTEX_LOCATION,
                vertex_credentials=service_account_json(_VERTEX_PROJECT, upstream_url.rstrip("/")),
            )


def _tool_turn(city: str, thinking_blocks: list[JsonValue] | None) -> tuple[dict[str, JsonValue], ...]:
    call_id: Final = f"toolu_{uuid.uuid4().hex[:24]}"
    call: Final[dict[str, JsonValue]] = {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
    }
    carried: Final[dict[str, JsonValue]] = {} if thinking_blocks is None else {"thinking_blocks": thinking_blocks}
    return (
        {"role": "assistant", "content": None, "tool_calls": [call], **carried},
        {"role": "tool", "tool_call_id": call_id, "content": f"{city}: 18 degrees and sunny"},
    )


def _stripped_history(marker: str) -> list[JsonValue]:
    return [{"role": "user", "content": prompt(marker)}, *_tool_turn("Paris", None)]


def _history_with_earlier_thinking(marker: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": prompt(marker)},
        *_tool_turn("Paris", [thinking_block(THINKING, SIGNATURE)]),
        *_tool_turn("Tokyo", None),
    ]


def _replay(
    gateway: Gateway, wire: Wire, model: str, messages: list[JsonValue], stream: bool, thinking: dict[str, JsonValue]
) -> tuple[httpx.Response, Request]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": 2048,
        "messages": messages,
        "tools": [_TOOL],
        "stream": stream,
        "num_retries": 0,
        "cache": {"no-cache": True},
        **thinking,
    }
    response: Final = gateway.request("POST", "/v1/chat/completions", body)
    received: Final = wire.drain()
    assert len(received) == 1, [request.body for request in received]
    return response, received[0]


def _sent_thinking_type(route: Route, request: Request) -> JsonValue:
    body: Final = _JSON.validate_json(request.body)
    fields: Final = (
        object_value(body.get("additionalModelRequestFields") or {}) if route == "bedrock_converse" else body
    )
    thinking: Final = fields.get("thinking")
    return thinking.get("type") if isinstance(thinking, dict) else thinking


def _round_trip(
    gateway: Gateway,
    route: Route,
    model: str,
    *,
    thinking: dict[str, JsonValue],
    earlier_thinking: bool = False,
    stream: bool = False,
) -> JsonValue:
    marker: Final = uuid.uuid4().hex
    messages: Final = _history_with_earlier_thinking(marker) if earlier_thinking else _stripped_history(marker)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        deployed: Final = _deployment(scenario, route, model, wire.url, gateway.upstream_url)
        response, sent = _replay(gateway, wire, deployed, messages, stream, thinking)
    assert response.status_code == 200, response.text
    assert answer(marker) in response.text, response.text
    assert marker_of(sent) == marker, sent.body
    return _sent_thinking_type(route, sent)


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_reasoning_effort_keeps_adaptive_thinking_on_anthropic(modify_params_gateway: Gateway, stream: bool) -> None:
    assert _round_trip(modify_params_gateway, "anthropic", MODEL, thinking=_EFFORT, stream=stream) == "adaptive"


@pytest.mark.parametrize("route", ["bedrock_converse", "bedrock_invoke", "vertex_ai"])
def test_reasoning_effort_keeps_adaptive_thinking_on_sibling_routes(
    modify_params_gateway: Gateway, route: Route
) -> None:
    assert _round_trip(modify_params_gateway, route, MODEL, thinking=_EFFORT) == "adaptive"


@pytest.mark.parametrize("route", ["anthropic", "bedrock_converse"])
def test_explicit_adaptive_thinking_is_kept(modify_params_gateway: Gateway, route: Route) -> None:
    assert _round_trip(modify_params_gateway, route, MODEL, thinking={"thinking": _ADAPTIVE}) == "adaptive"


@pytest.mark.parametrize("route", ["anthropic", "bedrock_converse"])
def test_legacy_enabled_thinking_is_still_dropped(modify_params_gateway: Gateway, route: Route) -> None:
    assert _round_trip(modify_params_gateway, route, _LEGACY_MODEL, thinking={"thinking": _LEGACY}) is None


@pytest.mark.parametrize(
    ("model", "thinking", "expected"),
    [(MODEL, _EFFORT, "adaptive"), (_LEGACY_MODEL, {"thinking": _LEGACY}, "enabled")],
    ids=["adaptive", "legacy-enabled"],
)
def test_thinking_is_kept_when_an_earlier_tool_turn_carries_thinking_blocks(
    modify_params_gateway: Gateway, model: str, thinking: dict[str, JsonValue], expected: str
) -> None:
    kept: Final = _round_trip(modify_params_gateway, "anthropic", model, thinking=thinking, earlier_thinking=True)
    assert kept == expected


@pytest.mark.parametrize("thinking", [_EFFORT, {"thinking": _ADAPTIVE}], ids=["reasoning-effort", "explicit-adaptive"])
def test_adaptive_thinking_is_forwarded_unchanged_without_modify_params(
    gateway: Gateway, thinking: dict[str, JsonValue]
) -> None:
    assert _round_trip(gateway, "anthropic", MODEL, thinking=thinking) == "adaptive"
