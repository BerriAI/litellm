import json
import uuid
from collections.abc import Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(4 * graceful_stop_seconds() + 240)

_RELOAD_SECONDS: Final = 5
_PROPAGATION_SECONDS: Final = 6 * _RELOAD_SECONDS
_ATTRIBUTION_ON: Final = MappingProxyType(
    {
        "LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True",
        "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": str(_RELOAD_SECONDS),
    }
)
_SCIM_HEADERS: Final = MappingProxyType({"Content-Type": "application/scim+json"})
_SHARES_REQUIRED: Final = "ptu_shares must map at least one team_id to a positive whole number of PTUs"
_DEPLOYMENT_IDS: Final = TypeAdapter(dict[str, list[str]])
_SAMPLES: Final = 16


def _reply(_request: Request) -> Reply:
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", "gpt-4.1", "served", stream=False)


@dataclass(frozen=True, slots=True)
class _Guarded:
    owned: OwnedProxy
    wire_url: str

    @property
    def proxy(self) -> Gateway:
        return self.owned.gateway


@pytest.fixture(scope="module")
def guarded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Guarded]:
    directory: Final = tmp_path_factory.mktemp("ptu-share-guards")
    with (
        gateway_from_environment() as gateway,
        wire_server(_reply) as wire,
        owned_proxy_process(gateway, directory, _ATTRIBUTION_ON, workers=2) as owned,
    ):
        yield _Guarded(owned, wire.url)


@pytest.fixture
def cleanups() -> Iterator[ExitStack]:
    with ExitStack() as stack:
        yield stack


def _ptu_terms(shares: Mapping[str, JsonValue] | None, count: int) -> dict[str, JsonValue]:
    return {
        "base_model": "azure/gpt-4.1",
        "ptu_count": count,
        "cost_per_ptu_per_hour": 2.0,
        "ptu_effective_from": "2026-01-01T00:00:00Z",
        "ptu_shares": dict(shares) if shares is not None else None,
    }


def _deployment(wire_url: str, name: str, model_info: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": "azure/ptu-guard",
            "api_base": wire_url,
            "api_key": "synthetic-azure-key",
            "api_version": "2025-03-01-preview",
        },
        "model_info": dict(model_info),
    }


def _model_name() -> str:
    return f"ptu-guard-{uuid.uuid4().hex}"


def _stored_models(name: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT model_id, model_info FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s', (name,))


def _team_rows(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))


def _organization_rows(organization_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT organization_id FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (organization_id,)
    )


def _delete_team_if_present(proxy: Gateway, team_id: str) -> None:
    if _team_rows(team_id):
        proxy.post("/team/delete", {"team_ids": [team_id]})
    assert _team_rows(team_id) == []


def _team(proxy: Gateway, cleanups: ExitStack, **fields: JsonValue) -> str:
    team_id: Final = f"ptu-guard-team-{uuid.uuid4().hex}"
    proxy.post("/team/new", {"team_id": team_id, "team_alias": team_id, **fields})
    cleanups.callback(_delete_team_if_present, proxy, team_id)
    return team_id


def _team_key(proxy: Gateway, team_id: str) -> str:
    return string_value(proxy.post("/key/generate", {"team_id": team_id})["key"])


def _delete_organization_if_present(proxy: Gateway, organization_id: str, budget_id: str) -> None:
    if _organization_rows(organization_id):
        deleted: Final = proxy.request("DELETE", "/organization/delete", {"organization_ids": [organization_id]})
        assert deleted.status_code == 200, deleted.text
    assert _organization_rows(organization_id) == []
    proxy.post("/budget/delete", {"id": budget_id})


def _organization(proxy: Gateway, cleanups: ExitStack) -> str:
    created: Final = proxy.post("/organization/new", {"organization_alias": f"ptu-guard-org-{uuid.uuid4().hex}"})
    organization_id: Final = string_value(created["organization_id"])
    cleanups.callback(_delete_organization_if_present, proxy, organization_id, string_value(created["budget_id"]))
    return organization_id


def _worker_deployment_ids(owned: OwnedProxy, model_name: str) -> tuple[int, frozenset[str]]:
    with httpx.Client(
        base_url=str(owned.gateway.client.base_url),
        timeout=10,
        trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0),
    ) as client:
        response: Final = client.get("/health/integration-deployment-ids", params={"model_name": model_name})
    assert response.status_code == 200, response.text
    return (
        int(response.headers["x-integration-worker-pid"]),
        frozenset(_DEPLOYMENT_IDS.validate_python(response.json())["model_ids"]),
    )


def _served_on_no_worker(owned: OwnedProxy, model_name: str) -> bool:
    with ThreadPoolExecutor(max_workers=8) as pool:
        samples: Final = tuple(pool.map(partial(_worker_deployment_ids, owned), (model_name,) * _SAMPLES))
    workers: Final = owned.worker_pids()
    return (
        bool(workers)
        and workers <= frozenset(pid for pid, _ in samples)
        and all(not model_ids for _, model_ids in samples)
    )


def _delete_model(guarded: _Guarded, name: str, identity: str) -> None:
    guarded.proxy.post("/model/delete", {"id": identity})
    eventually(lambda: _served_on_no_worker(guarded.owned, name), bool, seconds=_PROPAGATION_SECONDS)


@contextmanager
def _shared_model(guarded: _Guarded, model_info: Mapping[str, JsonValue]) -> Generator[tuple[str, str]]:
    name: Final = _model_name()
    created: Final = guarded.proxy.post("/model/new", _deployment(guarded.wire_url, name, model_info))
    identity: Final = string_value(object_value(created["model_info"])["id"])
    try:
        eventually(
            lambda: guarded.owned.serves_only_on_every_worker(name, identity), bool, seconds=_PROPAGATION_SECONDS
        )
        yield name, identity
    finally:
        _delete_model(guarded, name, identity)


def _create(guarded: _Guarded, name: str, model_info: Mapping[str, JsonValue]) -> httpx.Response:
    return guarded.proxy.request("POST", "/model/new", _deployment(guarded.wire_url, name, model_info))


def _create_raw(guarded: _Guarded, name: str, raw_shares: str, count: int) -> httpx.Response:
    terms: Final = {key: value for key, value in _ptu_terms(None, count).items() if key != "ptu_shares"}
    body: Final = json.dumps(_deployment(guarded.wire_url, name, terms))
    spliced: Final = f'{body[:-2]}, "ptu_shares": {raw_shares}}}}}'
    return guarded.proxy.client.post(
        "/model/new",
        content=spliced,
        headers={"Authorization": f"Bearer {guarded.proxy.key}", "Content-Type": "application/json"},
    )


def _holds(team_id: str, identity: str) -> str:
    return f"Team {team_id} holds a PTU share on deployment {identity}"


def _chat_statuses(proxy: Gateway, key: str, model: str) -> tuple[int, ...]:
    def status(_index: int) -> int:
        with httpx.Client(
            base_url=str(proxy.client.base_url),
            timeout=30,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client:
            return client.post(
                "/v1/chat/completions",
                json={"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": uuid.uuid4().hex}]},
                headers={"Authorization": f"Bearer {key}"},
            ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        return tuple(pool.map(status, range(_SAMPLES)))


def test_a_share_for_a_team_that_does_not_exist_is_refused(guarded: _Guarded, cleanups: ExitStack) -> None:
    holder: Final = _team(guarded.proxy, cleanups)
    missing: Final = f"ptu-guard-missing-{uuid.uuid4().hex}"
    name: Final = _model_name()

    response: Final = _create(guarded, name, _ptu_terms({holder: 1, missing: 1}, 2))

    assert response.status_code == 400, response.text
    assert f"Team id={missing} does not exist in db" in response.text
    assert _stored_models(name) == []


def test_shares_that_do_not_add_up_to_the_count_are_refused(guarded: _Guarded, cleanups: ExitStack) -> None:
    holder: Final = _team(guarded.proxy, cleanups)
    name: Final = _model_name()

    response: Final = _create(guarded, name, _ptu_terms({holder: 1}, 2))

    assert response.status_code == 400, response.text
    assert "ptu_shares must add up to ptu_count (1 of 2 allocated)" in response.text
    assert _stored_models(name) == []


def test_team_id_and_shares_together_are_refused(guarded: _Guarded, cleanups: ExitStack) -> None:
    holder: Final = _team(guarded.proxy, cleanups)
    name: Final = _model_name()

    response: Final = _create(guarded, name, {**_ptu_terms({holder: 1}, 1), "team_id": holder})

    assert response.status_code == 400, response.text
    assert "team_id and ptu_shares cannot both be set" in response.text
    assert _stored_models(name) == []


@pytest.mark.parametrize(
    ("raw_shares", "count", "status", "reason"),
    [
        pytest.param('["{holder}"]', 1, 422, "ptu_shares", id="list"),
        pytest.param('"{holder}"', 1, 422, "ptu_shares", id="string"),
        pytest.param("{{}}", 1, 400, _SHARES_REQUIRED, id="empty"),
        pytest.param('{{"{holder}": -1}}', 1, 400, _SHARES_REQUIRED, id="negative"),
        pytest.param('{{"{holder}": 0}}', 1, 400, _SHARES_REQUIRED, id="zero"),
        pytest.param('{{"{holder}": 1.5}}', 2, 422, "ptu_shares", id="fraction"),
        pytest.param('{{"{holder}": true}}', 1, 422, "ptu_shares", id="boolean"),
        pytest.param('{{"{holder}": "1"}}', 1, 422, "ptu_shares", id="numeric-string"),
        pytest.param("null", 1, 400, "team_id or ptu_shares is required when PTU fields are set", id="null"),
        pytest.param(
            '{{"{holder}": 1, "{holder}": 1}}',
            2,
            400,
            "ptu_shares must add up to ptu_count (1 of 2 allocated)",
            id="dup",
        ),
        pytest.param('{{"{long_team}": 1}}', 1, 400, "does not exist in db", id="5kb-team-id"),
    ],
)
def test_malformed_shares_are_refused_with_the_reason(
    guarded: _Guarded, cleanups: ExitStack, raw_shares: str, count: int, status: int, reason: str
) -> None:
    holder: Final = _team(guarded.proxy, cleanups)
    name: Final = _model_name()

    response: Final = _create_raw(guarded, name, raw_shares.format(holder=holder, long_team="t" * 5000), count)

    assert response.status_code == status, response.text
    assert reason in response.text
    assert _stored_models(name) == []
    assert guarded.proxy.get("/model/info")["data"]


def test_with_attribution_off_shares_cannot_be_stored() -> None:
    name: Final = _model_name()
    with gateway_from_environment() as gateway:
        try:
            response: Final = gateway.request(
                "POST",
                "/model/new",
                _deployment(gateway.upstream_url, name, {"ptu_shares": {f"ptu-guard-team-{uuid.uuid4().hex}": 1}}),
            )

            assert response.status_code == 400, response.text
            assert "PTU cost attribution is disabled, so ptu_shares cannot be set" in response.text
            assert _stored_models(name) == []
        finally:
            for row in _stored_models(name):
                gateway.post("/model/delete", {"id": string_value(row["model_id"])})


def test_moving_a_share_to_a_team_that_does_not_exist_keeps_the_stored_shares(
    guarded: _Guarded, cleanups: ExitStack
) -> None:
    holder: Final = _team(guarded.proxy, cleanups)
    missing: Final = f"ptu-guard-missing-{uuid.uuid4().hex}"
    name, identity = cleanups.enter_context(_shared_model(guarded, _ptu_terms({holder: 1}, 1)))

    response: Final = guarded.proxy.request(
        "PATCH", f"/model/{identity}/update", {"model_info": {"ptu_shares": {missing: 1}}}
    )

    assert response.status_code == 400, response.text
    assert f"Team id={missing} does not exist in db" in response.text
    assert [object_value(row["model_info"])["ptu_shares"] for row in _stored_models(name)] == [{holder: 1}]


def test_deleting_a_team_that_holds_a_share_is_refused(guarded: _Guarded, cleanups: ExitStack) -> None:
    holder: Final = _team(guarded.proxy, cleanups)
    _name, identity = cleanups.enter_context(_shared_model(guarded, _ptu_terms({holder: 1}, 1)))

    response: Final = guarded.proxy.request("POST", "/team/delete", {"team_ids": [holder]})

    assert response.status_code == 400, response.text
    assert _holds(holder, identity) in response.text
    assert _team_rows(holder) == [{"team_id": holder}]


def test_deleting_an_organization_whose_team_holds_a_share_is_refused(guarded: _Guarded, cleanups: ExitStack) -> None:
    organization_id: Final = _organization(guarded.proxy, cleanups)
    holder: Final = _team(guarded.proxy, cleanups, organization_id=organization_id)
    _name, identity = cleanups.enter_context(_shared_model(guarded, _ptu_terms({holder: 1}, 1)))

    response: Final = guarded.proxy.request("DELETE", "/organization/delete", {"organization_ids": [organization_id]})

    assert response.status_code == 400, response.text
    assert _holds(holder, identity) in response.text
    assert _organization_rows(organization_id) == [{"organization_id": organization_id}]
    assert _team_rows(holder) == [{"team_id": holder}]


def test_deleting_a_scim_group_that_holds_a_share_is_refused(guarded: _Guarded, cleanups: ExitStack) -> None:
    holder: Final = _team(guarded.proxy, cleanups)
    _name, identity = cleanups.enter_context(_shared_model(guarded, _ptu_terms({holder: 1}, 1)))

    response: Final = guarded.proxy.request("DELETE", f"/scim/v2/Groups/{holder}", headers=_SCIM_HEADERS)

    assert response.status_code == 400, response.text
    assert _holds(holder, identity) in response.text
    assert _team_rows(holder) == [{"team_id": holder}]


def test_a_team_without_a_share_deletes_while_another_team_holds_one(guarded: _Guarded, cleanups: ExitStack) -> None:
    holder: Final = _team(guarded.proxy, cleanups)
    bystander: Final = _team(guarded.proxy, cleanups)
    cleanups.enter_context(_shared_model(guarded, _ptu_terms({holder: 1}, 1)))

    response: Final = guarded.proxy.request("POST", "/team/delete", {"team_ids": [bystander]})

    assert response.status_code == 200, response.text
    assert _team_rows(bystander) == []


def test_after_a_share_moves_the_former_holder_deletes_and_the_new_holder_is_served(
    guarded: _Guarded, cleanups: ExitStack
) -> None:
    former: Final = _team(guarded.proxy, cleanups)
    successor: Final = _team(guarded.proxy, cleanups)
    successor_key: Final = _team_key(guarded.proxy, successor)
    name, identity = cleanups.enter_context(_shared_model(guarded, _ptu_terms({former: 1}, 1)))

    moved: Final = guarded.proxy.request(
        "PATCH", f"/model/{identity}/update", {"model_info": {"ptu_shares": {successor: 1}}}
    )
    assert moved.status_code == 200, moved.text

    deleted: Final = eventually(
        lambda: guarded.proxy.request("POST", "/team/delete", {"team_ids": [former]}),
        lambda response: response.status_code != 400,
        seconds=_PROPAGATION_SECONDS,
    )
    assert deleted.status_code == 200, deleted.text
    assert _team_rows(former) == []
    assert [object_value(row["model_info"])["ptu_shares"] for row in _stored_models(name)] == [{successor: 1}]
    served: Final = eventually(
        lambda: _chat_statuses(guarded.proxy, successor_key, name),
        lambda statuses: set(statuses) == {200},
        seconds=_PROPAGATION_SECONDS,
        return_last_on_timeout=True,
    )
    assert set(served) == {200}, served


def test_shares_are_stored_as_whole_ptus_and_served_on_every_worker(guarded: _Guarded, cleanups: ExitStack) -> None:
    first: Final = _team(guarded.proxy, cleanups)
    second: Final = _team(guarded.proxy, cleanups)
    first_key: Final = _team_key(guarded.proxy, first)
    name, identity = cleanups.enter_context(_shared_model(guarded, _ptu_terms({first: 2.0, second: 1}, 3)))

    info: Final = guarded.proxy.get("/model/info", {"litellm_model_id": identity})["data"]
    assert isinstance(info, list) and len(info) == 1, info
    assert object_value(object_value(info[0])["model_info"])["ptu_shares"] == {first: 2, second: 1}
    assert [object_value(row["model_info"])["ptu_shares"] for row in _stored_models(name)] == [{first: 2, second: 1}]
    assert set(_chat_statuses(guarded.proxy, first_key, name)) == {200}
