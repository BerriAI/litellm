import json
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server


def _write_access_config(directory: Path, *, default_deny: bool, model_name: str, api_base: str) -> Path:
    source: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **source,
        "model_list": [
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": api_base,
                },
            }
        ],
        "general_settings": {
            **source["general_settings"],
            "agent_access_default_deny": default_deny,
        },
    }
    path: Final = directory / f"agent-access-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _peer_response(request: Request, wire: Wire, agent_name: str) -> Reply:
    if request.method == "GET":
        assert request.target in ("/.well-known/agent-card.json", "/.well-known/agent.json")
        return Reply(
            body=json.dumps(
                {
                    "protocolVersion": "0.3",
                    "name": agent_name,
                    "description": "Synthetic permission peer",
                    "version": "1.0.0",
                    "url": wire.url + "/",
                    "capabilities": {"streaming": False},
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "skills": [],
                }
            ).encode()
        )
    assert request.method == "POST" and request.target == "/"
    body: Final = json.loads(request.body)
    assert body["jsonrpc"] == "2.0" and body["method"] == "message/send"
    message: Final = body["params"]["message"]
    assert message["role"] == "user"
    assert message["kind"] == "message"
    assert message["parts"] in (
        [{"kind": "text", "text": "synthetic ping"}],
        [{"kind": "text", "text": "user: synthetic ping"}],
    )
    return Reply(
        body=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "kind": "message",
                    "role": "agent",
                    "messageId": message["messageId"] + "-out",
                    "parts": [{"kind": "text", "text": "synthetic pong"}],
                },
            }
        ).encode()
    )


def _create_agent(candidate: Gateway, wire: Wire, agent_name: str, cleanups: ExitStack) -> str:
    created: Final = candidate.post(
        "/v1/agents",
        {
            "agent_name": agent_name,
            "agent_card_params": {
                "protocolVersion": "0.3",
                "name": agent_name,
                "description": "Synthetic permission peer",
                "version": "1.0.0",
                "url": wire.url + "/",
                "capabilities": {"streaming": False},
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "skills": [],
            },
            "litellm_params": {"make_public": False},
        },
    )
    agent_id: Final = string_value(created["agent_id"])
    cleanups.callback(_delete_agent, candidate, agent_id)
    return agent_id


def _delete_agent(candidate: Gateway, agent_id: str) -> None:
    deleted: Final = candidate.request("DELETE", f"/v1/agents/{agent_id}")
    assert deleted.status_code == 200, deleted.text
    assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (agent_id,)) == []


def _card(candidate: Gateway, agent_id: str, key: str) -> httpx.Response:
    return candidate.request(
        "GET",
        f"/a2a/{agent_id}/.well-known/agent-card.json",
        key=key,
        headers={"Connection": "close"},
    )


def _send(candidate: Gateway, agent_id: str, key: str, marker: str) -> httpx.Response:
    return candidate.request(
        "POST",
        f"/a2a/{agent_id}",
        {
            "jsonrpc": "2.0",
            "id": marker,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": marker + "-in",
                    "parts": [{"kind": "text", "text": "synthetic ping"}],
                }
            },
        },
        key=key,
        headers={"Connection": "close"},
    )


def _chat_completion(candidate: Gateway, agent_name: str, key: str) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": f"a2a/{agent_name}",
            "messages": [{"role": "user", "content": "synthetic ping"}],
        },
        key=key,
        headers={"Connection": "close"},
    )


def _model_names(candidate: Gateway, path: str, key: str) -> frozenset[str]:
    response: Final = candidate.request("GET", path, key=key, headers={"Connection": "close"})
    assert response.status_code == 200, response.text
    payload: Final = response.json()
    raw_entries: Final = payload if isinstance(payload, list) else payload["data"]
    entries: Final = tuple(object_value(entry) for entry in raw_entries)
    if path == "/model_group/info":
        return frozenset(string_value(entry["model_group"]) for entry in entries)
    return frozenset(string_value(entry["model_name"]) for entry in entries)


def _listed_agent_ids(candidate: Gateway, key: str) -> frozenset[str]:
    response: Final = candidate.request("GET", "/v1/agents", key=key, headers={"Connection": "close"})
    assert response.status_code == 200, response.text
    return frozenset(string_value(object_value(agent)["agent_id"]) for agent in response.json())


def _assert_successful_send(response: httpx.Response, marker: str) -> None:
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["jsonrpc"] == "2.0" and body["id"] == marker and "error" not in body, response.text
    assert body["result"]["messageId"] == marker + "-in-out", response.text
    assert body["result"]["parts"] == [{"kind": "text", "text": "synthetic pong"}], response.text


def _assert_denial(response: httpx.Response, agent_id: str) -> None:
    assert response.status_code == 403, response.text
    assert response.json() == {
        "detail": f"Agent '{agent_id}' is not allowed for your key/team. Contact proxy admin for access."
    }, response.text


@pytest.mark.timeout(180)
def test_agent_access_default_deny_denies_ungranted_key_but_keeps_explicit_grants(
    gateway: Gateway, tmp_path: Path
) -> None:
    agent_name: Final = "least-privilege-" + uuid.uuid4().hex
    v2_model: Final = "listing-" + agent_name
    config: Final = _write_access_config(
        tmp_path, default_deny=True, model_name=v2_model, api_base=gateway.upstream_url + "/v1"
    )

    def upstream(request: Request) -> Reply:
        return _peer_response(request, wire, agent_name)

    with (
        wire_server(upstream) as wire,
        owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate,
        candidate.scenario() as scenario,
    ):
        agent_id: Final = _create_agent(candidate, wire, agent_name, scenario.cleanups)
        ungranted_team: Final = scenario.team()
        key_a: Final = scenario.key(team_id=ungranted_team)
        key_b: Final = scenario.key(object_permission={"agents": [agent_id]})
        granted_team: Final = scenario.team(object_permission={"agents": [agent_id]})
        key_c: Final = scenario.key(team_id=granted_team)
        wire.drain()

        denied_cards: Final = tuple(_card(candidate, agent_id, key_a) for _ in range(8))
        _assert_denial(denied_cards[0], agent_id)
        assert all(response.json() == denied_cards[0].json() for response in denied_cards), [
            response.text for response in denied_cards
        ]

        denied_sends: Final = tuple(
            _send(candidate, agent_id, key_a, f"denied-{index}-{uuid.uuid4().hex}") for index in range(8)
        )
        _assert_denial(denied_sends[0], agent_id)
        assert all(response.json() == denied_sends[0].json() for response in denied_sends), [
            response.text for response in denied_sends
        ]
        denied_completion: Final = _chat_completion(candidate, agent_name, key_a)
        assert denied_completion.status_code == 403, denied_completion.text
        assert denied_completion.json() == {
            "error": {
                "message": f"Agent '{agent_name}' is not allowed for your key/team. Contact proxy admin for access.",
                "type": "permission_error",
                "param": None,
                "code": "403",
            }
        }, denied_completion.text
        denied_agents: Final = tuple(
            candidate.request("GET", f"/v1/agents/{agent_id}", key=key_a, headers={"Connection": "close"})
            for _ in range(8)
        )
        _assert_denial(denied_agents[0], agent_id)
        assert all(response.json() == denied_agents[0].json() for response in denied_agents), [
            response.text for response in denied_agents
        ]
        assert v2_model in _model_names(candidate, "/v2/model/info", candidate.key)
        assert _listed_agent_ids(candidate, key_a) == frozenset()
        assert f"a2a/{agent_name}" not in _model_names(candidate, "/v2/model/info", key_a)
        assert f"a2a/{agent_name}" not in _model_names(candidate, "/model_group/info", key_a)
        assert tuple(item for item in wire.drain() if item.method == "POST") == ()

        for key in (key_b, key_c, candidate.key):
            cards: Final = tuple(_card(candidate, agent_id, key) for _ in range(8))
            assert all(response.status_code == 200 for response in cards), [response.text for response in cards]
            completion: Final = _chat_completion(candidate, agent_name, key)
            assert completion.status_code == 200, completion.text
            assert completion.json()["choices"][0]["message"]["content"] == "synthetic pong", completion.text
            wire.drain()
            for index in range(8):
                marker: Final = f"granted-{index}-{uuid.uuid4().hex}"
                _assert_successful_send(_send(candidate, agent_id, key, marker), marker)
                posts: Final = tuple(item for item in wire.drain() if item.method == "POST")
                assert len(posts) == 1, posts
            if key != candidate.key:
                assert agent_id in _listed_agent_ids(candidate, key)
                assert f"a2a/{agent_name}" in _model_names(candidate, "/v2/model/info", key)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("workers", [2, 4])
def test_agent_access_default_deny_toggles_at_runtime_through_config_api(
    gateway: Gateway, tmp_path: Path, workers: int
) -> None:
    agent_name: Final = "runtime-agent-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        return _peer_response(request, wire, agent_name)

    with (
        wire_server(upstream) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {},
            config=Path("tests/integration/proxy_config.yaml"),
            workers=workers,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        agent_id: Final = _create_agent(candidate, wire, agent_name, scenario.cleanups)
        ungranted_key: Final = scenario.key()
        wire.drain()
        assert _card(candidate, agent_id, ungranted_key).status_code == 200

        enabled: Final = candidate.request(
            "POST",
            "/config/field/update",
            {
                "config_type": "general_settings",
                "field_name": "agent_access_default_deny",
                "field_value": "true",
            },
        )
        assert enabled.status_code == 200, enabled.text
        config_fields: Final = candidate.request(
            "GET",
            "/config/list",
            params={"config_type": "general_settings"},
        )
        assert config_fields.status_code == 200, config_fields.text
        setting: Final = next(
            object_value(field)
            for field in config_fields.json()
            if object_value(field)["field_name"] == "agent_access_default_deny"
        )
        assert setting["field_type"] == "Boolean" and setting["field_value"] == "true", setting
        eventually(
            lambda: tuple(_card(candidate, agent_id, ungranted_key) for _ in range(8)),
            lambda responses: all(response.status_code == 403 for response in responses),
            seconds=40,
        )

        disabled: Final = candidate.request(
            "POST",
            "/config/field/update",
            {
                "config_type": "general_settings",
                "field_name": "agent_access_default_deny",
                "field_value": False,
            },
        )
        assert disabled.status_code == 200, disabled.text
        eventually(
            lambda: tuple(_card(candidate, agent_id, ungranted_key) for _ in range(8)),
            lambda responses: all(response.status_code == 200 for response in responses),
            seconds=40,
        )
