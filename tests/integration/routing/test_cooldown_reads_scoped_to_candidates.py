from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.redis_process import OwnedRedis, owned_redis, redis_commands
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = [pytest.mark.timeout(300)]

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MODEL_INFO_ENTRIES: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
OPENAI_MODEL: Final = "gpt-4o-mini"
MASTER_KEY: Final = "sk-integration-cooldown-read-scope"
API_KEY: Final = "synthetic-cooldown-read-scope-key"
COOLDOWN_KEY: Final = re.compile(r"^deployment:(.+):cooldown$")
REDIS_READS: Final = frozenset({"GET", "MGET"})
ENDPOINTS: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
OTHER_GROUP_SIZE: Final = 4
CHAT_RESPONSE: Final = json.dumps(
    {
        "id": "chatcmpl_cooldown_read_scope",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "scoped"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
    }
).encode()
RESPONSES_RESPONSE: Final = json.dumps(
    {
        "id": "resp_cooldown_read_scope",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": OPENAI_MODEL,
        "output": [
            {
                "id": "msg_cooldown_read_scope",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "scoped", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
    }
).encode()


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    redis: OwnedRedis
    wire: Wire
    groups: Mapping[str, tuple[str, frozenset[str]]]
    wildcard_prefix: str
    wildcard_ids: frozenset[str]
    other_ids: frozenset[str]

    @property
    def configured_ids(self) -> frozenset[str]:
        return frozenset().union(*(ids for _, ids in self.groups.values()), self.wildcard_ids, self.other_ids)


def _reply(request: Request) -> Reply:
    if request.target == "/v1/models":
        return Reply(body=json.dumps({"object": "list", "data": [{"id": OPENAI_MODEL, "object": "model"}]}).encode())
    return Reply(body=RESPONSES_RESPONSE if request.target == "/v1/responses" else CHAT_RESPONSE)


def _deployments(model_name: str, model: str, api_base: str, ids: frozenset[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "model_name": model_name,
            "litellm_params": {"model": model, "api_base": api_base, "api_key": API_KEY},
            "model_info": {"id": deployment_id},
        }
        for deployment_id in sorted(ids)
    )


def _ids(prefix: str, count: int) -> frozenset[str]:
    return frozenset(f"{prefix}-{uuid.uuid4().hex[:8]}" for _ in range(count))


def _registered_ids(gateway: Gateway) -> frozenset[str]:
    entries: Final = MODEL_INFO_ENTRIES.validate_python(gateway.get("/model/info")["data"])
    return frozenset(string_value(object_value(entry["model_info"])["id"]) for entry in entries)


def _payload(endpoint: str, model: str) -> dict[str, JsonValue]:
    marker: Final = f"cooldown-scope-{uuid.uuid4().hex}"
    if endpoint == "/v1/responses":
        return {"model": model, "input": marker, "max_output_tokens": 16, "store": False}
    return {"model": model, "messages": [{"role": "user", "content": marker}], "max_tokens": 16}


def _headers(endpoint: str) -> Mapping[str, str]:
    return {"anthropic-version": "2023-06-01"} if endpoint == "/v1/messages" else {}


def _cooldown_reads(commands: tuple[tuple[str, ...], ...]) -> tuple[frozenset[str], ...]:
    """The deployment ids whose cooldown key each Redis read carried, one entry per read that carried any"""
    reads: Final = tuple(command for command in commands if command and command[0].upper() in REDIS_READS)
    matched: Final = tuple(
        frozenset(match.group(1) for argument in command[1:] if (match := COOLDOWN_KEY.match(argument)))
        for command in reads
    )
    return tuple(ids for ids in matched if ids)


@pytest.fixture(scope="module", params=("simple-shuffle", "usage-based-routing-v2"))
def rig(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    strategy: Final = str(request.param)
    tmp_path: Final = tmp_path_factory.mktemp(f"cooldown-scope-{strategy}")
    with owned_redis(tmp_path) as cache, wire_server(_reply) as wire:
        run_id: Final = uuid.uuid4().hex[:8]
        api_base: Final = f"{wire.url}/v1"
        groups: Final = MappingProxyType(
            {
                endpoint: (f"scoped-{index}-{run_id}", _ids(f"scoped-{index}", 2))
                for index, endpoint in enumerate(ENDPOINTS)
            }
        )
        wildcard_prefix: Final = f"wild-{run_id}"
        wildcard_ids: Final = _ids("wildcard", 1)
        other_ids: Final = _ids("other", OTHER_GROUP_SIZE)
        model_list: Final = (
            *(
                deployment
                for name, ids in groups.values()
                for deployment in _deployments(name, f"openai/{OPENAI_MODEL}", api_base, ids)
            ),
            *_deployments(f"{wildcard_prefix}/*", "openai/*", api_base, wildcard_ids),
            *_deployments(f"other-{run_id}", f"openai/{OPENAI_MODEL}", api_base, other_ids),
        )
        configuration: Final = JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        config_path: Final = tmp_path / "cooldown-scope.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    **configuration,
                    "general_settings": {
                        **JSON_OBJECT.validate_python(configuration["general_settings"]),
                        "store_model_in_db": True,
                    },
                    "model_list": list(model_list),
                    "router_settings": {
                        "routing_strategy": strategy,
                        "redis_host": cache.host,
                        "redis_port": cache.port,
                    },
                }
            )
        )
        with httpx.Client(base_url=wire.url, timeout=15, trust_env=False) as bootstrap_client:
            bootstrap: Final = Gateway(bootstrap_client, MASTER_KEY, wire.url)
            with owned_proxy(
                bootstrap,
                tmp_path,
                {
                    "REDIS_HOST": cache.host,
                    "REDIS_PORT": str(cache.port),
                    "STORE_MODEL_IN_DB": "True",
                    "DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS": "0.001",
                },
                config=config_path,
            ) as gateway:
                built: Final = Rig(gateway, cache, wire, groups, wildcard_prefix, wildcard_ids, other_ids)
                eventually(
                    lambda: _registered_ids(gateway),
                    lambda registered: built.configured_ids <= registered,
                    seconds=30,
                )
                yield built


def _send(rig: Rig, endpoint: str, model: str, key: str | None = None) -> tuple[httpx.Response, frozenset[str]]:
    """Sends one request and returns it with the ids of every cooldown key the proxy read from Redis for it"""
    rig.wire.drain()
    with redis_commands(rig.redis.host, rig.redis.port) as commands:
        response: Final = rig.gateway.request(
            "POST", endpoint, _payload(endpoint, model), key=key, headers=_headers(endpoint)
        )
    captured: Final = commands.drain()
    reads: Final = _cooldown_reads(captured)
    assert len(reads) == 1, f"cooldown keys should arrive in one Redis read, saw {reads} in {captured}"
    assert len(rig.wire.drain()) == 1
    return response, reads[0]


@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=("chat-completions", "messages", "responses"))
def test_request_reads_cooldown_keys_of_its_model_group_only(rig: Rig, endpoint: str) -> None:
    model_name, group_ids = rig.groups[endpoint]

    response, read_ids = _send(rig, endpoint, model_name)

    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] in group_ids
    assert read_ids == group_ids, f"read {sorted(read_ids - group_ids)} outside group {sorted(group_ids)}"


def test_wildcard_route_reads_cooldown_key_of_matching_deployment_only(rig: Rig) -> None:
    response, read_ids = _send(rig, "/v1/chat/completions", f"{rig.wildcard_prefix}/{OPENAI_MODEL}")

    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] in rig.wildcard_ids
    assert read_ids == rig.wildcard_ids, f"read {sorted(read_ids - rig.wildcard_ids)} outside the wildcard route"


def test_team_route_reads_cooldown_key_of_team_deployment_only(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        public_name: Final = f"team-scoped-{uuid.uuid4().hex[:8]}"
        team_model_id: Final = _team_model(scenario, public_name, team_id, rig.wire.url)
        team_key: Final = scenario.key(team_id=team_id)
        eventually(
            lambda: _registered_ids(rig.gateway),
            lambda registered: team_model_id in registered,
            seconds=30,
        )

        response, read_ids = _send(rig, "/v1/chat/completions", public_name, key=team_key)

        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-model-id"] == team_model_id
        assert read_ids == {team_model_id}, f"read {sorted(read_ids - {team_model_id})} outside the team route"


def _team_model(scenario: Scenario, public_name: str, team_id: str, upstream_url: str) -> str:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": public_name,
            "litellm_params": {"model": f"openai/{OPENAI_MODEL}", "api_key": API_KEY, "api_base": f"{upstream_url}/v1"},
            "model_info": {"team_id": team_id},
        },
    )
    model_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return model_id
