import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    anthropic_deployment,
    anthropic_peer,
    final_text,
    new_marker,
)
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MODEL: Final = "empty-user-turn"
_PLACEHOLDER: Final = "[System: Empty message content sanitised to satisfy protocol]"
_REPLY: Final = "Noted."
_EMPTY_ROWS: Final[Mapping[str, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "list": MappingProxyType({"role": "user", "content": []}),
        "null": MappingProxyType({"role": "user", "content": None}),
        "missing": MappingProxyType({"role": "user"}),
        "empty-string": MappingProxyType({"role": "user", "content": ""}),
    }
)
_CHAT_EMPTY_KINDS: Final = ("list", "null", "missing")

Turn = tuple[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    provider: Wire


def _empty_row(kind: str) -> dict[str, JsonValue]:
    return dict(_EMPTY_ROWS[kind])


def _guardrail_peer(request: Request) -> Reply:
    kind: Final = request.target.split("/")[1]
    rows: Final = _JSON_OBJECT.validate_json(request.body)["structured_messages"]
    assert isinstance(rows, list), rows
    verdict: Final = {"action": "GUARDRAIL_INTERVENED", "structured_messages": [*rows, _empty_row(kind)]}
    return Reply(body=json.dumps(verdict).encode())


def _guardrail(kind: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": f"append-{kind}",
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": False,
            "api_base": f"{api_base}/{kind}",
            "api_key": "synthetic-guardrail-key",
        },
    }


def _config(directory: Path, provider_url: str, guardrail_url: str) -> Path:
    stock: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final[dict[str, JsonValue]] = {
        **stock,
        "model_list": [anthropic_deployment(_MODEL, provider_url)],
        "router_settings": {**object_value(stock["router_settings"]), "num_retries": 0},
        "guardrails": [_guardrail(kind, guardrail_url) for kind in _EMPTY_ROWS],
    }
    path: Final = directory / "empty-user-turn.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("empty-user-turn")
    with (
        wire_server(anthropic_peer) as provider,
        wire_server(_guardrail_peer) as guardrail,
        gateway_from_environment() as gateway,
    ):
        config: Final = _config(directory, provider.url, guardrail.url)
        with owned_proxy(gateway, directory, {}, config=config, workers=2) as owned:
            yield _Rig(owned, provider)


def _block(block: JsonValue) -> str:
    part: Final = object_value(block)
    return str(part["text"]) if part.get("type") == "text" else str(part.get("type"))


def _turn(message: JsonValue) -> Turn:
    row: Final = object_value(message)
    content: Final = row["content"]
    if isinstance(content, str):
        return (str(row["role"]), (content,))
    assert isinstance(content, list), row
    return (str(row["role"]), tuple(_block(block) for block in content))


def _send(rig: _Rig, path: str, body: Mapping[str, JsonValue]) -> None:
    response: Final = rig.gateway.client.post(
        path, json=dict(body), headers={"Authorization": f"Bearer {rig.gateway.key}"}
    )
    assert response.status_code == 200, response.text


def _provider_turns(rig: _Rig, marker: str) -> tuple[Turn, ...]:
    needle: Final = marker.encode()
    requests: Final = [request for request in rig.provider.drain() if needle in request.body]
    assert len(requests) == 1, requests
    messages: Final = _JSON_OBJECT.validate_json(requests[0].body)["messages"]
    assert isinstance(messages, list), messages
    return tuple(_turn(message) for message in messages)


def _ask_row(ask: str) -> dict[str, JsonValue]:
    return {"role": "user", "content": ask}


def _chat(messages: Sequence[JsonValue]) -> dict[str, JsonValue]:
    return {"model": _MODEL, "max_tokens": 64, "messages": list(messages)}


@pytest.mark.parametrize("kind", _CHAT_EMPTY_KINDS)
def test_chat_trailing_empty_user_message_keeps_a_user_turn(rig: _Rig, kind: str) -> None:
    marker: Final = new_marker()
    ask: Final = final_text(marker)
    _send(
        rig,
        "/v1/chat/completions",
        _chat([_ask_row(ask), {"role": "assistant", "content": _REPLY}, _empty_row(kind)]),
    )
    assert _provider_turns(rig, marker) == (("user", (ask,)), ("assistant", (_REPLY,)), ("user", (_PLACEHOLDER,)))


@pytest.mark.parametrize("kind", _CHAT_EMPTY_KINDS)
def test_chat_empty_user_message_beside_text_sends_only_the_text(rig: _Rig, kind: str) -> None:
    marker: Final = new_marker()
    ask: Final = final_text(marker)
    _send(rig, "/v1/chat/completions", _chat([_ask_row(ask), _empty_row(kind)]))
    assert _provider_turns(rig, marker) == (("user", (ask,)),)


def _system_then_empty_turns(rig: _Rig, kind: str) -> tuple[Turn, ...]:
    marker: Final = new_marker()
    ask: Final = final_text(marker)
    _send(
        rig,
        "/v1/chat/completions",
        _chat(
            [
                _ask_row(ask),
                {"role": "assistant", "content": _REPLY},
                {"role": "system", "content": "Answer in French."},
                _empty_row(kind),
            ]
        ),
    )
    turns: Final = _provider_turns(rig, marker)
    assert turns[0] == ("user", (ask,)), turns
    return turns[1:]


@pytest.mark.parametrize("kind", _CHAT_EMPTY_KINDS)
def test_chat_system_message_before_an_empty_user_message_is_placed_like_an_empty_string(rig: _Rig, kind: str) -> None:
    assert _system_then_empty_turns(rig, kind) == _system_then_empty_turns(rig, "empty-string")


def test_responses_trailing_empty_input_message_keeps_a_user_turn(rig: _Rig) -> None:
    marker: Final = new_marker()
    ask: Final = final_text(marker)
    body: Final[dict[str, JsonValue]] = {
        "model": _MODEL,
        "max_output_tokens": 64,
        "input": [
            _ask_row(ask),
            {"role": "assistant", "content": _REPLY},
            {"role": "user", "content": []},
        ],
    }
    _send(rig, "/v1/responses", body)
    assert _provider_turns(rig, marker) == (("user", (ask,)), ("assistant", (_REPLY,)), ("user", (_PLACEHOLDER,)))


@pytest.mark.parametrize("kind", tuple(_EMPTY_ROWS))
def test_messages_guardrail_empty_user_row_is_sent_like_an_empty_string(rig: _Rig, kind: str) -> None:
    marker: Final = new_marker()
    ask: Final = final_text(marker)
    body: Final[dict[str, JsonValue]] = {
        "model": _MODEL,
        "max_tokens": 64,
        "messages": [_ask_row(ask)],
        "guardrails": [f"append-{kind}"],
    }
    _send(rig, "/v1/messages", body)
    assert _provider_turns(rig, marker) == (("user", (ask,)), ("user", (_PLACEHOLDER,)))


def test_a_long_run_of_empty_user_messages_is_sent_as_one_text_turn(rig: _Rig) -> None:
    marker: Final = new_marker()
    ask: Final = final_text(marker)
    _send(
        rig, "/v1/chat/completions", _chat([*({"role": "user", "content": []} for _ in range(10_000)), _ask_row(ask)])
    )
    assert _provider_turns(rig, marker) == (("user", (ask,)),)
