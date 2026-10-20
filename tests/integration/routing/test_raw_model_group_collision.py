import json
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, Scenario, gateway_from_environment, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(240)

SHARED_MODEL: Final = "openai/integration-shared-raw-model"
SOLO_MODEL: Final = "openai/integration-solo-raw-model"
DEPLOYMENTS: Final = (
    ("collision-chain-a", "collision-a-1", SHARED_MODEL),
    ("collision-chain-a", "collision-a-2", SHARED_MODEL),
    ("collision-chain-b", "collision-b-1", SHARED_MODEL),
    ("collision-solo", "collision-solo-1", SOLO_MODEL),
)
CHAIN_A: Final = frozenset({"collision-a-1", "collision-a-2"})
TEAM_MODEL: Final = "openai/integration-team-raw-model"
TEAM_DEPLOYMENTS: Final = frozenset({"collision-team-1", "collision-team-2"})
ALIAS: Final = "collision-alias"
POOLED_CALLS: Final = 48
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

Build = Callable[[str, str], tuple[str, dict[str, JsonValue]]]


def _json_reply(body: dict[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _reply(request: Request) -> Reply:
    if request.target.endswith("/chat/completions"):
        return _json_reply(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "integration-shared-raw-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        )
    if request.target.endswith("/responses"):
        return _json_reply(
            {
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "integration-shared-raw-model",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            }
        )
    if request.target.endswith("/embeddings"):
        return _json_reply(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "integration-shared-raw-model",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            }
        )
    return Reply(status=404)


def _served_by(requests: tuple[Request, ...]) -> tuple[str, ...]:
    return tuple(request.target.split("/")[1] for request in requests)


def _deployment(wire: Wire, group: str, identity: str, model: str) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {"model": model, "api_base": f"{wire.url}/{identity}/v1", "api_key": "synthetic-key"},
        "model_info": {"id": identity},
    }


def _config(
    wire: Wire,
    directory: Path,
    router_settings: dict[str, JsonValue],
    extra_deployments: tuple[tuple[str, str, str], ...] = (),
) -> Path:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config["model_list"] = [_deployment(wire, *deployment) for deployment in (*DEPLOYMENTS, *extra_deployments)]
    config["router_settings"] = {**JSON_OBJECT.validate_python(config["router_settings"]), **router_settings}
    path: Final = directory / "raw-model-group-collision.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def peer() -> Iterator[Wire]:
    with wire_server(_reply) as wire:
        yield wire


@pytest.fixture(scope="module")
def guarded(peer: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("raw-model-collision-guarded")
    path: Final = _config(peer, directory, {"enable_cross_model_group_collision_check": True})
    with gateway_from_environment() as base, owned_proxy(base, directory, {}, config=path) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def unguarded(peer: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("raw-model-collision-default")
    path: Final = _config(peer, directory, {})
    with gateway_from_environment() as base, owned_proxy(base, directory, {}, config=path) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def guarded_without_specific_deployment(peer: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("raw-model-collision-wildcard")
    path: Final = _config(
        peer,
        directory,
        {"enable_cross_model_group_collision_check": True, "model_group_alias": {ALIAS: SHARED_MODEL}},
        (("collision-wild/*", "collision-wild-1", "openai/*"),),
    )
    with gateway_from_environment() as base, owned_proxy(base, directory, {}, config=path) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def toggled(peer: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("raw-model-collision-toggled")
    path: Final = _config(peer, directory, {})
    with gateway_from_environment() as base, owned_proxy(base, directory, {}, config=path) as proxy:
        yield proxy


def _chat(model: str, marker: str) -> tuple[str, dict[str, JsonValue]]:
    return "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}


def _messages(model: str, marker: str) -> tuple[str, dict[str, JsonValue]]:
    return "/v1/messages", {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]}


def _responses(model: str, marker: str) -> tuple[str, dict[str, JsonValue]]:
    return "/v1/responses", {"model": model, "input": marker}


def _embeddings(model: str, marker: str) -> tuple[str, dict[str, JsonValue]]:
    return "/v1/embeddings", {"model": model, "input": [marker]}


@pytest.mark.parametrize(
    "build",
    (
        pytest.param(_chat, id="chat-completions"),
        pytest.param(_messages, id="anthropic-messages"),
        pytest.param(_responses, id="responses"),
        pytest.param(_embeddings, id="embeddings"),
    ),
)
def test_a_raw_model_string_shared_by_two_groups_is_refused_before_any_upstream_call(
    guarded: Gateway, peer: Wire, build: Build
) -> None:
    with guarded.scenario() as scenario:
        key: Final = scenario.key()
        refused_marker: Final = f"collision-refused-{uuid.uuid4().hex}"
        control_marker: Final = f"collision-control-{uuid.uuid4().hex}"
        peer.drain()
        refused: Final = guarded.request("POST", *build(SHARED_MODEL, refused_marker), key=key)
        control: Final = guarded.request("POST", *build("collision-chain-b", control_marker), key=key)
        received: Final = peer.drain()

        assert refused.status_code == 400, refused.text
        assert SHARED_MODEL in refused.text
        assert control.status_code == 200, control.text
        assert _served_by(received) == ("collision-b-1",)
        assert control_marker in received[0].body.decode()
        assert refused_marker not in received[0].body.decode()


@pytest.mark.parametrize("model", (pytest.param(SHARED_MODEL, id="raw-model"), pytest.param(ALIAS, id="alias")))
def test_the_check_also_refuses_when_a_wildcard_route_sends_the_request_down_the_generic_lookup(
    guarded_without_specific_deployment: Gateway, peer: Wire, model: str
) -> None:
    gateway: Final = guarded_without_specific_deployment
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        peer.drain()
        refused: Final = gateway.request("POST", *_chat(model, uuid.uuid4().hex), key=key)
        control: Final = gateway.request("POST", *_chat("collision-chain-b", uuid.uuid4().hex), key=key)
        received: Final = peer.drain()

        assert refused.status_code == 400, refused.text
        assert SHARED_MODEL in refused.text
        assert control.status_code == 200, control.text
        assert _served_by(received) == ("collision-b-1",)


def test_a_model_group_name_still_routes_inside_its_own_group_when_the_check_is_on(
    guarded: Gateway, peer: Wire
) -> None:
    with guarded.scenario() as scenario:
        key: Final = scenario.key()
        peer.drain()
        responses: Final = tuple(
            guarded.request("POST", *_chat("collision-chain-a", uuid.uuid4().hex), key=key) for _ in range(8)
        )
        served: Final = _served_by(peer.drain())

        assert [response.status_code for response in responses] == [200] * 8, [r.text for r in responses]
        assert len(served) == 8
        assert set(served) <= CHAIN_A


def test_a_raw_model_string_used_by_one_group_is_still_served_when_the_check_is_on(
    guarded: Gateway, peer: Wire
) -> None:
    with guarded.scenario() as scenario:
        peer.drain()
        response: Final = guarded.request("POST", *_chat(SOLO_MODEL, uuid.uuid4().hex), key=scenario.key())

        assert response.status_code == 200, response.text
        assert _served_by(peer.drain()) == ("collision-solo-1",)


def test_without_the_setting_a_shared_raw_model_string_keeps_pooling_every_group(
    unguarded: Gateway, peer: Wire
) -> None:
    with unguarded.scenario() as scenario:
        key: Final = scenario.key()
        peer.drain()
        responses: Final = tuple(
            unguarded.request("POST", *_chat(SHARED_MODEL, uuid.uuid4().hex), key=key) for _ in range(POOLED_CALLS)
        )
        served: Final = set(_served_by(peer.drain()))

        assert [response.status_code for response in responses] == [200] * POOLED_CALLS, [r.text for r in responses]
        assert "collision-b-1" in served
        assert served & CHAIN_A
        assert served <= CHAIN_A | {"collision-b-1"}


def _team_deployment(
    scenario: Scenario, wire: Wire, team_id: str, public_name: str, identity: str, model: str = TEAM_MODEL
) -> None:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": public_name,
            "litellm_params": {
                "model": model,
                "api_base": f"{wire.url}/{identity}/v1",
                "api_key": "synthetic-key",
            },
            "model_info": {"team_id": team_id},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))


def test_the_deployments_of_one_team_model_count_as_one_model_group(guarded: Gateway, peer: Wire) -> None:
    with guarded.scenario() as scenario:
        team: Final = scenario.team()
        public_name: Final = f"collision-team-model-{uuid.uuid4().hex}"
        for identity in sorted(TEAM_DEPLOYMENTS):
            _team_deployment(scenario, peer, team, public_name, identity)
        key: Final = scenario.key(team_id=team)
        peer.drain()
        responses: Final = tuple(
            guarded.request("POST", *_chat(TEAM_MODEL, uuid.uuid4().hex), key=key) for _ in range(8)
        )
        served: Final = _served_by(peer.drain())

        assert [response.status_code for response in responses] == [200] * 8, [r.text for r in responses]
        assert len(served) == 8
        assert set(served) <= TEAM_DEPLOYMENTS


def test_another_teams_model_on_the_same_raw_model_string_does_not_refuse_this_team(
    guarded: Gateway, peer: Wire
) -> None:
    with guarded.scenario() as scenario:
        team: Final = scenario.team()
        other_team: Final = scenario.team()
        public_name: Final = f"collision-team-model-{uuid.uuid4().hex}"
        for identity in sorted(TEAM_DEPLOYMENTS):
            _team_deployment(scenario, peer, team, public_name, identity)
        _team_deployment(scenario, peer, other_team, public_name, "collision-other-team-1")
        key: Final = scenario.key(team_id=team)
        peer.drain()
        responses: Final = tuple(
            guarded.request("POST", *_chat(TEAM_MODEL, uuid.uuid4().hex), key=key) for _ in range(8)
        )
        served: Final = _served_by(peer.drain())

        assert [response.status_code for response in responses] == [200] * 8, [r.text for r in responses]
        assert len(served) == 8
        assert set(served) <= TEAM_DEPLOYMENTS


def test_a_team_model_on_a_raw_model_string_a_proxy_group_also_uses_is_refused_for_that_team(
    guarded: Gateway, peer: Wire
) -> None:
    with guarded.scenario() as scenario:
        team: Final = scenario.team()
        _team_deployment(
            scenario, peer, team, f"collision-team-model-{uuid.uuid4().hex}", "collision-team-solo-1", SOLO_MODEL
        )
        team_key: Final = scenario.key(team_id=team)
        other_key: Final = scenario.key()
        peer.drain()
        refused: Final = guarded.request("POST", *_chat(SOLO_MODEL, uuid.uuid4().hex), key=team_key)
        served: Final = guarded.request("POST", *_chat(SOLO_MODEL, uuid.uuid4().hex), key=other_key)
        received: Final = _served_by(peer.drain())

        assert refused.status_code == 400, refused.text
        assert SOLO_MODEL in refused.text
        assert served.status_code == 200, served.text
        assert received == ("collision-solo-1",)


def _router_setting(gateway: Gateway, name: str) -> JsonValue:
    fields: Final = gateway.get("/router/settings")["fields"]
    assert isinstance(fields, list)
    return next(field["field_value"] for field in fields if isinstance(field, dict) and field["field_name"] == name)


def test_the_admin_ui_config_update_turns_the_check_on_and_off_without_a_restart(toggled: Gateway, peer: Wire) -> None:
    with toggled.scenario() as scenario:
        key: Final = scenario.key()
        peer.drain()
        before: Final = toggled.request("POST", *_chat(SHARED_MODEL, uuid.uuid4().hex), key=key)
        toggled.post("/config/update", {"router_settings": {"enable_cross_model_group_collision_check": True}})
        enabled_value: Final = _router_setting(toggled, "enable_cross_model_group_collision_check")
        refused: Final = toggled.request("POST", *_chat(SHARED_MODEL, uuid.uuid4().hex), key=key)
        toggled.post("/config/update", {"router_settings": {"enable_cross_model_group_collision_check": False}})
        after: Final = toggled.request("POST", *_chat(SHARED_MODEL, uuid.uuid4().hex), key=key)
        served: Final = _served_by(peer.drain())

        assert before.status_code == 200, before.text
        assert enabled_value is True
        assert refused.status_code == 400, refused.text
        assert after.status_code == 200, after.text
        assert len(served) == 2
