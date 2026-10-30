from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final
from uuid import uuid4

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.wire import Reply, Request, wire_server

_UPSTREAM_REPLIES: Final[Mapping[str, Mapping[str, JsonValue]]] = {
    "/v1/chat/completions": {
        "id": "chatcmpl_team_update",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "/v1/responses": {
        "id": "resp_team_update",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.6",
        "output": [
            {
                "type": "message",
                "id": "msg_team_update",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
}


def _respond(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target in _UPSTREAM_REPLIES, request.target
    return Reply(body=json.dumps(_UPSTREAM_REPLIES[request.target]).encode())


def _request(proxy: Gateway, route: str, model: str, key: str) -> httpx.Response:
    marker: Final = f"team update cache {uuid4().hex}"
    if route == "/v1/messages":
        return proxy.request(
            "POST",
            route,
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
            key=key,
            headers={"anthropic-version": "2023-06-01"},
        )
    if route == "/v1/responses":
        return proxy.request(
            "POST",
            route,
            {"model": model, "input": marker, "max_output_tokens": 16, "store": False},
            key=key,
        )
    return proxy.request(
        "POST", route, {"model": model, "messages": [{"role": "user", "content": marker}]}, key=key
    )


def _chat(proxy: Gateway, model: str, key: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"team limit {uuid4().hex}"}]},
        key=key,
    )


def _team_limit_error(response: httpx.Response) -> None:
    assert response.status_code == 429, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    error: Final = object_value(body["error"])
    message: Final = string_value(error["message"])
    assert "Current limit: 1" in message, response.text


def _scenario_model(scenario: Scenario) -> str:
    return scenario.model()


def _wire_model(scenario: Scenario, upstream_url: str) -> str:
    return scenario.model(
        model="openai/gpt-5.6",
        api_base=upstream_url + "/v1",
        api_key="synthetic-openai-key",
    )


@pytest.mark.parametrize(
    "route",
    (
        pytest.param("/v1/chat/completions", id="chat-completions"),
        pytest.param("/v1/messages", id="messages"),
        pytest.param("/v1/responses", id="responses"),
    ),
)
def test_team_update_adds_model_on_peer_for_inference_routes(gateway: Gateway, peer: Gateway, route: str) -> None:
    with wire_server(_respond) as upstream, gateway.scenario() as scenario:
        model_a: Final = _wire_model(scenario, upstream.url)
        model_b: Final = _wire_model(scenario, upstream.url)
        team: Final = scenario.team(models=[model_a])
        key: Final = scenario.key(team_id=team)

        peer_a: Final = _request(peer, route, model_a, key)
        assert peer_a.status_code == 200, peer_a.text
        peer_b_before: Final = _request(peer, route, model_b, key)
        assert peer_b_before.status_code == 403, peer_b_before.text

        update: Final = gateway.request("POST", "/team/update", {"team_id": team, "models": [model_a, model_b]})
        assert update.status_code == 200, update.text

        peer_b_after: Final = eventually(
            lambda: _request(peer, route, model_b, key),
            lambda response: response.status_code == 200,
            seconds=10,
            return_last_on_timeout=True,
        )
        requests: Final = upstream.drain()
        assert peer_b_after.status_code == 200, peer_b_after.text
        expected_target: Final = "/v1/chat/completions" if route == "/v1/chat/completions" else "/v1/responses"
        assert tuple(request.target for request in requests) == (expected_target, expected_target), peer_b_after.text


def test_team_update_raising_rpm_limit_is_enforced_by_peer(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _scenario_model(scenario)
        team: Final = scenario.team(models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team)

        peer_first: Final = _chat(peer, model, key)
        assert peer_first.status_code == 200, peer_first.text
        peer_limited: Final = _chat(peer, model, key)
        _team_limit_error(peer_limited)

        update: Final = gateway.request("POST", "/team/update", {"team_id": team, "rpm_limit": 1000})
        assert update.status_code == 200, update.text

        peer_after: Final = eventually(
            lambda: _chat(peer, model, key),
            lambda response: response.status_code == 200,
            seconds=10,
            return_last_on_timeout=True,
        )
        assert peer_after.status_code == 200, peer_after.text


def test_team_update_lowering_rpm_limit_is_enforced_by_peer(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _scenario_model(scenario)
        team: Final = scenario.team(models=[model], rpm_limit=1000)
        key: Final = scenario.key(team_id=team)

        peer_warm: Final = _chat(peer, model, key)
        assert peer_warm.status_code == 200, peer_warm.text

        update: Final = gateway.request("POST", "/team/update", {"team_id": team, "rpm_limit": 1})
        assert update.status_code == 200, update.text

        peer_limited: Final = eventually(
            lambda: _chat(peer, model, key),
            lambda response: response.status_code == 429,
            seconds=10,
            return_last_on_timeout=True,
        )
        _team_limit_error(peer_limited)


@pytest.mark.parametrize(
    ("route", "initial_models", "expected_before", "expected_after"),
    (
        pytest.param("/team/model/add", "one", 403, 200, id="add"),
        pytest.param("/team/model/delete", "both", 200, 403, id="delete"),
    ),
)
def test_team_model_add_delete_is_enforced_by_peer(
    gateway: Gateway,
    peer: Gateway,
    route: str,
    initial_models: str,
    expected_before: int,
    expected_after: int,
) -> None:
    with gateway.scenario() as scenario:
        model_a: Final = _scenario_model(scenario)
        model_b: Final = _scenario_model(scenario)
        team: Final = scenario.team(models=[model_a] if initial_models == "one" else [model_a, model_b])
        key: Final = scenario.key(team_id=team)

        peer_before: Final = _chat(peer, model_b, key)
        assert peer_before.status_code == expected_before, peer_before.text

        update: Final = gateway.request("POST", route, {"team_id": team, "models": [model_b]})
        assert update.status_code == 200, update.text

        peer_after: Final = eventually(
            lambda: _chat(peer, model_b, key),
            lambda response: response.status_code == expected_after,
            seconds=10,
            return_last_on_timeout=True,
        )
        assert peer_after.status_code == expected_after, peer_after.text
