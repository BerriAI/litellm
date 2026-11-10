"""claude-fable-5-2 on Vertex AI: the row's capability flags shape the rawPredict body the proxy sends.

Fable rejects sampling params and forced tool choice and always thinks, so the proxy must strip
or refuse those before the request reaches Vertex, the same way it does on the Anthropic API.
"""

import json
import uuid
from collections.abc import Callable
from typing import Final
from urllib.parse import parse_qs

from integration._support.client import Gateway, Scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "claude-fable-5-2"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "global"
_TARGET: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/{_MODEL}:rawPredict"
_TOKEN: Final = "scripted-vertex-access-token"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_FORCED_TOOL_CHOICES: Final[tuple[JsonValue, ...]] = ("required", {"type": "function", "function": {"name": "lookup"}})
_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a synthetic record.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
    }
]


def _reply(identity: str, usage: dict[str, JsonValue] | None = None) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "type": "message",
            "role": "assistant",
            "model": _MODEL,
            "content": [
                {"type": "thinking", "thinking": "scripted thought", "signature": "scripted-signature"},
                {"type": "text", "text": "done"},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 12, "output_tokens": 3},
        }
    ).encode()


def _deployment(scenario: Scenario, api_base: str, *, drop_params: bool = False) -> str:
    credentials: Final = service_account_json(_PROJECT, api_base)
    if drop_params:
        return scenario.model(
            model=f"vertex_ai/{_MODEL}",
            api_base=api_base,
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=_LOCATION,
            vertex_credentials=credentials,
            num_retries=0,
            drop_params=True,
        )
    return scenario.model(
        model=f"vertex_ai/{_MODEL}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=credentials,
        num_retries=0,
    )


def _peer(on_body: Callable[[dict[str, JsonValue]], bytes]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/_oauth/token":
            form: Final = parse_qs(request.body.decode())
            assert form["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"], form
            return Reply(body=json.dumps({"access_token": _TOKEN, "token_type": "Bearer", "expires_in": 3600}).encode())
        assert request.method == "POST" and request.target == _TARGET, (request.method, request.target)
        assert request.headers["authorization"] == f"Bearer {_TOKEN}", request.headers
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["anthropic_version"] == "vertex-2023-10-16", body
        return Reply(body=on_body(body))

    return respond


def _chat_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "synthetic fable request"}],
        "max_tokens": 64,
        **extra,
    }


def _model_calls(requests: tuple[Request, ...]) -> int:
    return sum(request.target == _TARGET for request in requests)


def test_drop_params_removes_temperature_and_top_p_before_vertex(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def on_body(body: dict[str, JsonValue]) -> bytes:
        assert "temperature" not in body and "top_p" not in body, body
        assert body["max_tokens"] == 64, body
        return _reply(identity)

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url, drop_params=True)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, temperature=0.2, top_p=0.9))
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "done", response.text
        assert _model_calls(wire.drain()) == 1


def test_sampling_params_are_refused_before_reaching_vertex_without_drop_params(gateway: Gateway) -> None:
    def on_body(body: dict[str, JsonValue]) -> bytes:
        raise AssertionError(f"temperature must not reach Vertex: {body}")

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, temperature=0.2))
        assert response.status_code == 400, response.text
        assert "temperature" in response.text and "drop_params" in response.text, response.text
        assert _model_calls(wire.drain()) == 0


def test_drop_params_downgrades_forced_and_named_tool_choice_to_auto(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def on_body(body: dict[str, JsonValue]) -> bytes:
        assert body["tool_choice"] == {"type": "auto"}, body
        tools: Final = body["tools"]
        assert isinstance(tools, list) and [tool["name"] for tool in tools if isinstance(tool, dict)] == ["lookup"], (
            body
        )
        return _reply(identity)

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url, drop_params=True)
        responses: Final = tuple(
            gateway.request("POST", "/v1/chat/completions", _chat_body(model, tools=_TOOLS, tool_choice=tool_choice))
            for tool_choice in _FORCED_TOOL_CHOICES
        )
        assert [response.status_code for response in responses] == [200, 200], [response.text for response in responses]
        assert _model_calls(wire.drain()) == 2


def test_forced_tool_choice_is_refused_before_reaching_vertex_without_drop_params(gateway: Gateway) -> None:
    def on_body(body: dict[str, JsonValue]) -> bytes:
        raise AssertionError(f"forced tool_choice must not reach Vertex: {body}")

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, tools=_TOOLS, tool_choice="required")
        )
        assert response.status_code == 400, response.text
        assert "tool_choice" in response.text, response.text
        assert _model_calls(wire.drain()) == 0


def test_thinking_disabled_is_dropped_for_an_always_on_model(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def on_body(body: dict[str, JsonValue]) -> bytes:
        assert "thinking" not in body, body
        return _reply(identity)

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, thinking={"type": "disabled"})
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["reasoning_content"] == "scripted thought", response.text
        assert _model_calls(wire.drain()) == 1


def test_reasoning_effort_becomes_adaptive_thinking_with_output_config_effort(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def on_body(body: dict[str, JsonValue]) -> bytes:
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}, body
        assert body["output_config"] == {"effort": "max"}, body
        assert "budget_tokens" not in json.dumps(body), body
        return _reply(identity)

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, reasoning_effort="max"))
        assert response.status_code == 200, response.text
        assert _model_calls(wire.drain()) == 1


def test_messages_one_hour_cache_control_reaches_vertex_and_its_usage_comes_back(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    system: Final[list[JsonValue]] = [
        {"type": "text", "text": "synthetic cached system prompt", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 9,
        "output_tokens": 3,
        "cache_creation_input_tokens": 7225,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 7225},
    }

    def on_body(body: dict[str, JsonValue]) -> bytes:
        assert body["system"] == system, body
        return _reply(identity, usage)

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "system": system,
                "messages": [{"role": "user", "content": "synthetic fable request"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["cache_creation"] == {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 7225,
        }, response.text
        assert response.json()["usage"]["cache_creation_input_tokens"] == 7225, response.text
        assert _model_calls(wire.drain()) == 1
