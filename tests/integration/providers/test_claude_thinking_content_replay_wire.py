import uuid
from collections.abc import Mapping
from typing import Final, Literal

import pytest
from integration._support.anthropic_thinking import (
    BEDROCK_MODEL,
    MODEL,
    NO_CACHE,
    SIGNATURE,
    THINKING,
    answer,
    marker_of,
    prompt,
    standard_peer,
    thinking_block,
)
from integration._support.client import Gateway, Scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

Route = Literal["anthropic", "bedrock_invoke", "claude_platform", "vertex_ai", "azure_ai"]
Endpoint = Literal["chat", "messages"]

_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"
_ENABLED: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 1024}
_EPHEMERAL: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
_SIGNED: Final = thinking_block(THINKING, SIGNATURE)
_REDACTED: Final[dict[str, JsonValue]] = {"type": "redacted_thinking", "data": "scripted-redacted-" + "r" * 32}
_BLANK_REDACTED: Final[dict[str, JsonValue]] = {"type": "redacted_thinking", "data": " "}
_ROUTES: Final[tuple[Route, ...]] = ("anthropic", "bedrock_invoke", "claude_platform", "vertex_ai", "azure_ai")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _SentMessage(_Frozen):
    role: str
    content: str | tuple[dict[str, JsonValue], ...]


class _SentRequest(_Frozen):
    messages: tuple[_SentMessage, ...]


def _deployment(scenario: Scenario, route: Route, wire_url: str, upstream_url: str) -> str:
    match route:
        case "anthropic":
            return scenario.model(model=f"anthropic/{MODEL}", api_base=wire_url, api_key="scripted-key")
        case "bedrock_invoke":
            return scenario.model(
                model=f"bedrock/invoke/{BEDROCK_MODEL}",
                api_base=wire_url,
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire_url,
            )
        case "claude_platform":
            return scenario.model(
                model=f"bedrock/claude_platform/{MODEL}",
                api_base=wire_url,
                api_key="scripted-platform-key",
                aws_region_name="us-east-1",
                workspace_id="scripted-workspace",
            )
        case "vertex_ai":
            return scenario.model(
                model=f"vertex_ai/{MODEL}",
                api_base=(
                    f"{wire_url}/v1/projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}"
                    f"/publishers/anthropic/models/{MODEL}"
                ),
                api_key=None,
                vertex_project=_VERTEX_PROJECT,
                vertex_location=_VERTEX_LOCATION,
                vertex_credentials=service_account_json(_VERTEX_PROJECT, upstream_url.rstrip("/")),
            )
        case "azure_ai":
            return scenario.model(model=f"azure_ai/{MODEL}", api_base=wire_url, api_key="scripted-azure-key")


def _text(text: str) -> dict[str, JsonValue]:
    return {"type": "text", "text": text}


def _replayed_turn(
    gateway: Gateway,
    wire: Wire,
    *,
    endpoint: Endpoint = "chat",
    model: str,
    marker: str,
    assistant: Mapping[str, JsonValue],
    **fields: JsonValue,
) -> tuple[dict[str, JsonValue], ...]:
    follow_up: Final = uuid.uuid4().hex
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": 2048,
        "thinking": _ENABLED,
        "messages": [
            {"role": "user", "content": prompt(marker)},
            {"role": "assistant", **assistant},
            {"role": "user", "content": prompt(follow_up)},
        ],
        **NO_CACHE,
        **fields,
    }
    path: Final = "/v1/chat/completions" if endpoint == "chat" else "/v1/messages"
    response: Final = gateway.request("POST", path, body)
    assert response.status_code == 200, response.text
    assert answer(follow_up) in response.text, response.text
    received: Final = wire.drain()
    assert len(received) == 1, [request.body for request in received]
    assert marker_of(received[0]) == follow_up, received[0].body
    sent: Final = _SentRequest.model_validate_json(received[0].body).messages[1]
    assert sent.role == "assistant", received[0].body
    assert not isinstance(sent.content, str), received[0].body
    return sent.content


@pytest.mark.parametrize("route", _ROUTES)
def test_redacted_thinking_replayed_in_assistant_content_reaches_claude_once(gateway: Gateway, route: Route) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, route, wire.url, gateway.upstream_url)
        sent: Final = _replayed_turn(
            gateway, wire, model=model, marker=marker, assistant={"content": [_REDACTED, _text(answer(marker))]}
        )
    assert sent == (_REDACTED, _text(answer(marker)))


def test_block_in_content_and_thinking_blocks_reaches_claude_once_in_order(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        sent: Final = _replayed_turn(
            gateway,
            wire,
            model=model,
            marker=marker,
            assistant={"content": [_REDACTED, _text(answer(marker))], "thinking_blocks": [_SIGNED, _REDACTED]},
        )
    assert sent == (_SIGNED, _REDACTED, _text(answer(marker)))


def test_streamed_thinking_replayed_in_content_reaches_claude_without_its_index(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        sent: Final = _replayed_turn(
            gateway,
            wire,
            model=model,
            marker=marker,
            assistant={"content": [{**_SIGNED, "index": 0}, _text(answer(marker))]},
        )
    assert sent == (_SIGNED, _text(answer(marker)))


@pytest.mark.parametrize("drop", [True, False], ids=["drop-params", "no-drop-params"])
def test_replayed_thinking_cache_control_and_blank_data_follow_drop_params(gateway: Gateway, drop: bool) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final = ({**_SIGNED, "cache_control": _EPHEMERAL}, _BLANK_REDACTED, _REDACTED, _text(answer(marker)))
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        sent: Final = _replayed_turn(
            gateway, wire, model=model, marker=marker, assistant={"content": list(content)}, drop_params=drop
        )
    assert sent == ((_SIGNED, _REDACTED, _text(answer(marker))) if drop else content)


@pytest.mark.parametrize("endpoint", ["chat", "messages"])
def test_assistant_cache_injection_point_never_marks_a_thinking_part(gateway: Gateway, endpoint: Endpoint) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{MODEL}",
            api_base=wire.url,
            api_key="scripted-key",
            cache_control_injection_points=[{"location": "message", "role": "assistant"}],
        )
        sent: Final = _replayed_turn(
            gateway,
            wire,
            endpoint=endpoint,
            model=model,
            marker=marker,
            assistant={"content": [_text(answer(marker)), _SIGNED]},
        )
    assert sent == ({**_text(answer(marker)), "cache_control": _EPHEMERAL}, _SIGNED)
