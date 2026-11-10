import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlsplit

import pytest
from integration._support.client import (
    DEPLOYMENT_IDS,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import owned_config
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_AZURE_API_VERSION: Final = "2024-10-21"
_BACKEND: Final = "gpt-5.4-mini"
_BATCH_ENDPOINT: Final = "/v1/batch"
_CHAT_ENDPOINT: Final = "/v1/chat/completions"
_RESPONSES_ENDPOINT: Final = "/v1/responses"
_TEAM_GROUP: Final = "batch-team-mix"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])
_ENDPOINTS: Final = TypeAdapter(list[str] | None)
_OWNED_PROXY_TIMEOUT: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

Provider = Literal["azure", "openai"]
Mode = Literal["chat", "batch"]


@dataclass(frozen=True, slots=True)
class _Sku:
    provider: Provider
    mode: Mode
    key: str
    weight: int | None
    endpoints: JsonValue
    team_id: str | None


@dataclass(frozen=True, slots=True)
class _Create:
    status: int
    body: str


@dataclass(frozen=True, slots=True)
class _TeamRig:
    owned: OwnedProxy
    wire: Wire
    team_key: str
    other_team_key: str
    no_team_key: str
    chat: _Sku
    batch: _Sku


def _sku(
    provider: Provider,
    mode: Mode,
    *,
    weight: int | None = None,
    endpoints: JsonValue = None,
    team_id: str | None = None,
) -> _Sku:
    return _Sku(provider, mode, f"{mode}-sku-{uuid.uuid4().hex}", weight, endpoints, team_id)


def _serves_batches(key: str) -> bool:
    return key.startswith("batch-sku-")


def _deployment(name: str, wire: Wire, sku: _Sku) -> dict[str, JsonValue]:
    endpoint: Final[dict[str, JsonValue]] = (
        {"model": f"azure/{_BACKEND}", "api_base": wire.url, "api_version": _AZURE_API_VERSION}
        if sku.provider == "azure"
        else {"model": f"openai/{_BACKEND}", "api_base": wire.url + "/v1"}
    )
    weight: Final[dict[str, JsonValue]] = {} if sku.weight is None else {"weight": sku.weight}
    endpoints: Final[dict[str, JsonValue]] = {} if sku.endpoints is None else {"supported_endpoints": sku.endpoints}
    team: Final[dict[str, JsonValue]] = {} if sku.team_id is None else {"team_id": sku.team_id}
    return {
        "model_name": name,
        "litellm_params": {**endpoint, "api_key": sku.key, **weight},
        "model_info": {"mode": sku.mode, **endpoints, **team},
    }


def _json_reply(body: Mapping[str, JsonValue], status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode())


def _provider_key(request: Request) -> str:
    return request.headers.get("api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")


def _is_azure(request: Request) -> bool:
    return urlsplit(request.target).path.startswith("/openai/")


def _posted_to(request: Request, suffix: str) -> bool:
    return request.method == "POST" and urlsplit(request.target).path.endswith(suffix)


def _vendor(request: Request) -> Reply:
    key: Final = _provider_key(request)
    if _posted_to(request, "/files"):
        return _json_reply(
            {
                "id": f"file-{key}",
                "object": "file",
                "bytes": len(request.body),
                "created_at": 1700000000,
                "filename": "in.jsonl",
                "purpose": "batch",
                "status": "processed",
            }
        )
    if _posted_to(request, "/batches") and _is_azure(request) and not _serves_batches(key):
        return _json_reply(
            {"error": {"code": "invalid_deployment_type", "message": "The deployment does not support batch jobs."}},
            400,
        )
    if _posted_to(request, "/batches"):
        return _json_reply(
            {
                "id": f"batch-{uuid.uuid4().hex}",
                "object": "batch",
                "endpoint": _CHAT_ENDPOINT,
                "input_file_id": f"file-{key}",
                "status": "validating",
                "created_at": 1700000000,
                "completion_window": "24h",
            }
        )
    if _posted_to(request, "/chat/completions") and _is_azure(request) and _serves_batches(key):
        return _json_reply(
            {"error": {"code": "OperationNotSupported", "message": "Batch deployments do not serve chat."}}, 400
        )
    if _posted_to(request, "/chat/completions"):
        return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", _BACKEND, "served", stream=False)
    return _json_reply({"data": [], "object": "list"})


def _deployment_ids_on_a_worker(gateway: Gateway, name: str) -> tuple[str, frozenset[str]]:
    response: Final = gateway.request(
        "GET",
        "/health/integration-deployment-ids",
        params={"model_name": name},
        headers={"connection": "close"},
    )
    assert response.status_code == 200, response.text
    return response.headers["x-integration-worker-pid"], frozenset(
        DEPLOYMENT_IDS.validate_json(response.content)["model_ids"]
    )


def _wait_until_every_worker_serves(gateway: Gateway, name: str, identities: frozenset[str]) -> None:
    from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

    def answers() -> tuple[tuple[str, frozenset[str]], ...]:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return tuple(pool.map(lambda _: _deployment_ids_on_a_worker(gateway, name), range(16)))

    eventually(
        answers,
        lambda seen: (
            len({pid for pid, _ in seen}) >= gateway.workers and all(identities <= served for _, served in seen)
        ),
        seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
    )


def _add_deployment(scenario: Scenario, deployment: Mapping[str, JsonValue]) -> str:
    created: Final = scenario.gateway.post("/model/new", deployment)
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def _group(scenario: Scenario, wire: Wire, skus: Sequence[_Sku]) -> str:
    name: Final = f"integration-batch-{uuid.uuid4().hex}"
    identities: Final = frozenset(_add_deployment(scenario, _deployment(name, wire, sku)) for sku in skus)
    _wait_until_every_worker_serves(scenario.gateway, name, identities)
    return name


def _managed_upload(gateway: Gateway, group: str, key: str) -> str:
    line: Final = {"custom_id": "r1", "method": "POST", "url": _CHAT_ENDPOINT, "body": {"model": group}}
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": group},
        {"file": ("in.jsonl", json.dumps(line).encode() + b"\n", "application/jsonl")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    return string_value(_JSON_OBJECT.validate_json(uploaded.content)["id"])


def _create(gateway: Gateway, managed_file: str, key: str) -> _Create:
    response: Final = gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": managed_file, "endpoint": _CHAT_ENDPOINT, "completion_window": "24h"},
        key=key,
    )
    return _Create(response.status_code, response.text)


def _chat(gateway: Gateway, group: str) -> int:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": group, "messages": [{"role": "user", "content": "hello"}], "cache": {"no-cache": True}},
    ).status_code


def _keys_posted_to(requests: Sequence[Request], suffix: str) -> tuple[str, ...]:
    return tuple(_provider_key(request) for request in requests if _posted_to(request, suffix))


def _assert_creates_reach_only(wire: Wire, creates: Sequence[_Create], sku: _Sku) -> None:
    assert [create.status for create in creates] == [200] * len(creates), creates
    assert _keys_posted_to(wire.drain(), "/batches") == (sku.key,) * len(creates)


@pytest.mark.parametrize(
    ("provider", "creates_reach"),
    [("azure", "batch"), ("openai", "chat")],
    ids=["azure-skips-chat", "openai-keeps-chat"],
)
def test_a_batch_create_on_a_mixed_group_reaches_the_deployment_its_provider_allows(
    gateway: Gateway, provider: Provider, creates_reach: Mode
) -> None:
    chat: Final = _sku(provider, "chat", weight=1)
    batch: Final = _sku(provider, "batch", weight=0)
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        group: Final = _group(scenario, wire, (chat, batch))
        managed_file: Final = _managed_upload(gateway, group, gateway.key)
        assert set(_keys_posted_to(wire.drain(), "/files")) == {chat.key, batch.key}

        creates: Final = tuple(_create(gateway, managed_file, gateway.key) for _ in range(3))

        _assert_creates_reach_only(wire, creates, batch if creates_reach == "batch" else chat)


def test_concurrent_batch_creates_and_chats_each_reach_their_own_deployment(gateway: Gateway) -> None:
    chat: Final = _sku("azure", "chat", weight=1)
    batch: Final = _sku("azure", "batch", weight=0)
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        group: Final = _group(scenario, wire, (chat, batch))
        managed_file: Final = _managed_upload(gateway, group, gateway.key)
        wire.drain()

        with ThreadPoolExecutor(max_workers=16) as pool:
            burst: Final = tuple(
                pool.map(
                    lambda turn: _create(gateway, managed_file, gateway.key) if turn % 2 else _chat(gateway, group),
                    range(24),
                )
            )
        after: Final = tuple(_chat(gateway, group) for _ in range(6))
        received: Final = wire.drain()

        chat_statuses: Final = tuple(result for result in burst if isinstance(result, int))
        creates: Final = tuple(result for result in burst if isinstance(result, _Create))
        assert chat_statuses + after == (200,) * 18, (burst, after)
        assert [create.status for create in creates] == [200] * 12, creates
        assert _keys_posted_to(received, "/chat/completions") == (chat.key,) * 18
        assert _keys_posted_to(received, "/batches") == (batch.key,) * 12


def _group_endpoints(gateway: Gateway, group: str, key: str | None = None) -> list[str] | None:
    response: Final = gateway.request("GET", "/model_group/info", params={"model_group": group}, key=key)
    assert response.status_code == 200, response.text
    (entry,) = list_value(_JSON_OBJECT.validate_json(response.content)["data"])
    return _ENDPOINTS.validate_python(object_value(entry).get("supported_endpoints"))


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param(
            (
                ("azure", "chat", [_CHAT_ENDPOINT, _RESPONSES_ENDPOINT, _BATCH_ENDPOINT]),
                ("azure", "batch", [_CHAT_ENDPOINT]),
            ),
            [_BATCH_ENDPOINT, _CHAT_ENDPOINT, _RESPONSES_ENDPOINT],
            id="azure-chat-and-batch",
        ),
        pytest.param(
            (("azure", "chat", [_CHAT_ENDPOINT, _BATCH_ENDPOINT]),),
            [_CHAT_ENDPOINT],
            id="azure-chat-only",
        ),
        pytest.param(
            (("azure", "batch", [_CHAT_ENDPOINT, _RESPONSES_ENDPOINT]),),
            [_BATCH_ENDPOINT],
            id="azure-batch-only",
        ),
        pytest.param(
            (
                ("openai", "chat", [_CHAT_ENDPOINT, _BATCH_ENDPOINT]),
                ("openai", "batch", [_RESPONSES_ENDPOINT]),
            ),
            [_BATCH_ENDPOINT, _CHAT_ENDPOINT, _RESPONSES_ENDPOINT],
            id="openai-chat-and-batch",
        ),
    ],
)
def test_model_group_info_lists_the_endpoints_its_deployments_serve(
    gateway: Gateway, shape: tuple[tuple[Provider, Mode, list[str]], ...], expected: list[str]
) -> None:
    skus: Final = tuple(_sku(provider, mode, endpoints=endpoints) for provider, mode, endpoints in shape)
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        group: Final = _group(scenario, wire, skus)

        listed: Final = _group_endpoints(gateway, group)

        assert listed is not None and sorted(listed) == expected, listed


@pytest.mark.parametrize("malformed", ["/v1/batch", [7], {"batch": True}], ids=["string", "number-list", "object"])
def test_a_malformed_endpoint_list_keeps_group_info_and_chat_answering(gateway: Gateway, malformed: JsonValue) -> None:
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        group: Final = _group(scenario, wire, (_sku("openai", "chat", endpoints=malformed),))

        _group_endpoints(gateway, group)

        assert _chat(gateway, group) == 200


@pytest.fixture(scope="module")
def team_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_TeamRig]:
    directory: Final = tmp_path_factory.mktemp("batch-team")
    team_id: Final = f"integration-batch-team-{uuid.uuid4().hex}"
    chat: Final = _sku("azure", "chat", endpoints=[_CHAT_ENDPOINT])
    batch: Final = _sku("azure", "batch", team_id=team_id)
    with gateway_from_environment() as gateway, wire_server(_vendor) as wire:
        config: Final = owned_config(
            directory,
            [_deployment(_TEAM_GROUP, wire, chat), _deployment(_TEAM_GROUP, wire, batch)],
            litellm_settings={"public_model_groups": [_TEAM_GROUP]},
            router_settings={"routing_strategy": "usage-based-routing"},
        )
        with owned_proxy_process(gateway, directory, {}, config=config) as owned, owned.gateway.scenario() as scenario:
            scenario.team(team_id=team_id)
            yield _TeamRig(
                owned=owned,
                wire=wire,
                team_key=scenario.key(team_id=team_id),
                other_team_key=scenario.key(team_id=scenario.team()),
                no_team_key=scenario.key(),
                chat=chat,
                batch=batch,
            )


def _caller_key(rig: _TeamRig, caller: str) -> str:
    return {"team": rig.team_key, "other-team": rig.other_team_key, "no-team": rig.no_team_key}[caller]


@_OWNED_PROXY_TIMEOUT
@pytest.mark.parametrize(
    ("caller", "lists_batch"),
    [("team", True), ("other-team", False), ("no-team", False)],
    ids=["team", "other-team", "no-team"],
)
def test_model_group_info_lists_a_team_owned_batch_endpoint_only_to_that_team(
    team_rig: _TeamRig, caller: str, lists_batch: bool
) -> None:
    listed: Final = _group_endpoints(team_rig.owned.gateway, _TEAM_GROUP, _caller_key(team_rig, caller))

    assert listed is not None and _CHAT_ENDPOINT in listed, listed
    assert (_BATCH_ENDPOINT in listed) is lists_batch, listed


@_OWNED_PROXY_TIMEOUT
def test_public_model_hub_never_lists_a_team_owned_batch_endpoint(team_rig: _TeamRig) -> None:
    response: Final = team_rig.owned.gateway.client.get("/public/model_hub")
    assert response.status_code == 200, response.text
    (entry,) = [
        object_value(row)
        for row in _JSON_LIST.validate_json(response.content)
        if object_value(row)["model_group"] == _TEAM_GROUP
    ]

    assert _BATCH_ENDPOINT not in (_ENDPOINTS.validate_python(entry.get("supported_endpoints")) or []), entry


@_OWNED_PROXY_TIMEOUT
def test_a_legacy_strategy_sends_a_team_batch_create_to_its_batch_deployment(team_rig: _TeamRig) -> None:
    gateway: Final = team_rig.owned.gateway
    team_rig.wire.drain()
    managed_file: Final = _managed_upload(gateway, _TEAM_GROUP, team_rig.team_key)
    assert team_rig.batch.key in _keys_posted_to(team_rig.wire.drain(), "/files")

    creates: Final = tuple(_create(gateway, managed_file, team_rig.team_key) for _ in range(3))

    _assert_creates_reach_only(team_rig.wire, creates, team_rig.batch)


@_OWNED_PROXY_TIMEOUT
@pytest.mark.parametrize("caller", ["other-team", "no-team"])
def test_a_legacy_strategy_keeps_other_callers_off_a_team_batch_deployment(team_rig: _TeamRig, caller: str) -> None:
    gateway: Final = team_rig.owned.gateway
    key: Final = _caller_key(team_rig, caller)
    team_rig.wire.drain()
    managed_file: Final = _managed_upload(gateway, _TEAM_GROUP, key)
    team_rig.wire.drain()

    tuple(_create(gateway, managed_file, key) for _ in range(3))

    assert _keys_posted_to(team_rig.wire.drain(), "/batches") == (team_rig.chat.key,) * 3
