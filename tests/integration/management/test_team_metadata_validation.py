from collections.abc import Iterator, Mapping
from pathlib import Path
import subprocess
import sys
from typing import Final
from uuid import uuid4

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from tests.integration._support.process import owned_proxy

_SERVICE_URL: Final = "http://127.0.0.1:9414"
_CONFIG: Final = Path(__file__).with_name("team_metadata_validation_proxy_config.yaml")
_IMPLS: Final = ("allowlist", "http", "immutable")
_REQUIRED_MESSAGES: Final = {
    "allowlist": "cost_center is required in team metadata",
    "http": "cost_center missing per cost center service",
    "immutable": "cost_center is required in team metadata",
}
_UNKNOWN_MESSAGES: Final = {
    "allowlist": "is not recognized",
    "http": "rejected by cost center service",
}
_UNAVAILABLE_MESSAGE: Final = (
    "Cost center validation is unavailable right now; the team was not saved. Contact FinOps."
)


def _meta(impl: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"_e2e_validator_impl": impl, **fields}


def _create_team(
    gateway: Gateway,
    metadata: Mapping[str, JsonValue] | None,
    team_id: str | None = None,
) -> httpx.Response:
    body: Final = {
        "team_alias": f"meta-validate-{uuid4().hex[:8]}",
        **({"team_id": team_id} if team_id is not None else {}),
        **({"metadata": dict(metadata)} if metadata is not None else {}),
    }
    return gateway.request("POST", "/team/new", body)


def _patch_team(gateway: Gateway, team_id: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("PATCH", f"/team/{team_id}", body)


def _post_update(gateway: Gateway, team_id: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/team/update", {"team_id": team_id, **body})


def _team_info(gateway: Gateway, team_id: str) -> httpx.Response:
    return gateway.request("GET", "/team/info", params={"team_id": team_id})


def _delete_team(gateway: Gateway, team_id: str) -> None:
    response: Final = gateway.request("POST", "/team/delete", {"team_ids": [team_id]})
    assert response.status_code == 200, response.text


def _service_health() -> httpx.Response | None:
    try:
        return httpx.get(f"{_SERVICE_URL}/health", timeout=2, trust_env=False)
    except httpx.ConnectError:
        return None


@pytest.fixture(scope="module", autouse=True)
def cost_center_service() -> Iterator[None]:
    service: Final = Path(__file__).with_name("cost_center_service.py")
    process: Final = subprocess.Popen(
        [sys.executable, str(service), "--host", "127.0.0.1", "--port", "9414"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health: Final = eventually(
            _service_health,
            lambda response: response is not None and response.status_code == 200,
            seconds=20,
        )
        assert health is not None and health.status_code == 200
        yield
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


@pytest.fixture(scope="module")
def gateway(cost_center_service: None, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with (
        gateway_from_environment() as rig,
        owned_proxy(
            rig,
            tmp_path_factory.mktemp("team-metadata-validation"),
            {"TEAM_METADATA_VALIDATION_SERVICE_URL": f"{_SERVICE_URL}/validate"},
            config=_CONFIG,
        ) as validated,
    ):
        yield validated


@pytest.fixture
def team_with_cost_center(
    request: pytest.FixtureRequest, gateway: Gateway
) -> Iterator[tuple[str, str]]:
    impl: Final = request.param
    team_id: Final = f"meta-validate-{impl}-{uuid4().hex[:8]}"
    response: Final = _create_team(gateway, _meta(impl, cost_center="CC-1001"), team_id)
    assert response.status_code == 200, response.text
    yield impl, team_id
    _delete_team(gateway, team_id)


@pytest.mark.parametrize("impl", _IMPLS)
def test_create_with_valid_cost_center_succeeds(gateway: Gateway, impl: str) -> None:
    response: Final = _create_team(gateway, _meta(impl, cost_center="CC-1001"))
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    team_id: Final = string_value(body["team_id"])
    try:
        assert object_value(body["metadata"])["cost_center"] == "CC-1001"
    finally:
        _delete_team(gateway, team_id)


@pytest.mark.parametrize("impl", _IMPLS)
def test_create_without_cost_center_is_rejected(gateway: Gateway, impl: str) -> None:
    team_id: Final = f"meta-validate-reject-{impl}-{uuid4().hex[:8]}"
    response: Final = _create_team(gateway, _meta(impl), team_id)
    assert response.status_code == 400, response.text
    assert _REQUIRED_MESSAGES[impl] in response.text
    assert _team_info(gateway, team_id).status_code == 404


@pytest.mark.parametrize("impl", _IMPLS)
def test_create_with_unknown_cost_center(gateway: Gateway, impl: str) -> None:
    response: Final = _create_team(gateway, _meta(impl, cost_center="CC-9999"))
    if impl == "immutable":
        assert response.status_code == 200, response.text
        _delete_team(gateway, string_value(object_value(response.json())["team_id"]))
        return
    assert response.status_code == 400, response.text
    assert _UNKNOWN_MESSAGES[impl] in response.text


@pytest.mark.parametrize("team_with_cost_center", _IMPLS, indirect=True)
def test_patch_changing_cost_center(
    gateway: Gateway, team_with_cost_center: tuple[str, str]
) -> None:
    impl: Final
    team_id: Final
    impl, team_id = team_with_cost_center
    response: Final = _patch_team(gateway, team_id, {"metadata": {"cost_center": "CC-1002"}})
    if impl == "immutable":
        assert response.status_code == 400, response.text
        assert "immutable once set" in response.text
        metadata: Final = object_value(object_value(_team_info(gateway, team_id).json())["team_info"])[
            "metadata"
        ]
        assert object_value(metadata)["cost_center"] == "CC-1001"
        return
    assert response.status_code == 200, response.text
    assert object_value(object_value(response.json())["metadata"])["cost_center"] == "CC-1002"


@pytest.mark.parametrize("team_with_cost_center", _IMPLS, indirect=True)
def test_patch_unrelated_key_validates_merged_result(
    gateway: Gateway, team_with_cost_center: tuple[str, str]
) -> None:
    _, team_id = team_with_cost_center
    response: Final = _patch_team(gateway, team_id, {"metadata": {"team_notes": "hello"}})
    assert response.status_code == 200, response.text
    metadata: Final = object_value(object_value(response.json())["metadata"])
    assert metadata == {"cost_center": "CC-1001", "team_notes": "hello", "_e2e_validator_impl": team_with_cost_center[0]}


@pytest.mark.parametrize("team_with_cost_center", _IMPLS, indirect=True)
def test_patch_null_deleting_cost_center_is_rejected(
    gateway: Gateway, team_with_cost_center: tuple[str, str]
) -> None:
    impl: Final
    team_id: Final
    impl, team_id = team_with_cost_center
    response: Final = _patch_team(gateway, team_id, {"metadata": {"cost_center": None}})
    assert response.status_code == 400, response.text
    assert _REQUIRED_MESSAGES[impl] in response.text
    metadata: Final = object_value(
        object_value(_team_info(gateway, team_id).json())["team_info"]
    )["metadata"]
    assert object_value(metadata)["cost_center"] == "CC-1001"


@pytest.mark.parametrize("team_with_cost_center", _IMPLS, indirect=True)
def test_post_update_dropping_cost_center_is_rejected(
    gateway: Gateway, team_with_cost_center: tuple[str, str]
) -> None:
    impl: Final
    team_id: Final
    impl, team_id = team_with_cost_center
    response: Final = _post_update(gateway, team_id, {"metadata": _meta(impl, team_notes="only-notes")})
    assert response.status_code == 400, response.text
    assert _REQUIRED_MESSAGES[impl] in response.text


@pytest.mark.parametrize("team_with_cost_center", _IMPLS, indirect=True)
def test_update_without_metadata_skips_validation(
    gateway: Gateway, team_with_cost_center: tuple[str, str]
) -> None:
    _, team_id = team_with_cost_center
    response: Final = _post_update(gateway, team_id, {"tpm_limit": 55})
    assert response.status_code == 200, response.text
    team: Final = object_value(object_value(_team_info(gateway, team_id).json())["team_info"])
    assert team["tpm_limit"] == 55


def test_http_service_outage_fails_closed_with_configured_message(gateway: Gateway) -> None:
    response: Final = _create_team(gateway, _meta("http_down", cost_center="CC-1001"))
    assert response.status_code == 503, response.text
    assert _UNAVAILABLE_MESSAGE in response.text


def test_metadata_without_dispatch_key_is_untouched(gateway: Gateway) -> None:
    response: Final = _create_team(gateway, {"any_key": "any_value"})
    assert response.status_code == 200, response.text
    team_id: Final = string_value(object_value(response.json())["team_id"])
    try:
        update: Final = _post_update(gateway, team_id, {"metadata": {"any_key": "changed"}})
        assert update.status_code == 200, update.text
    finally:
        _delete_team(gateway, team_id)
