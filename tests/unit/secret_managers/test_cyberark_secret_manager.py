import json
from pathlib import Path
from typing import Final, TypedDict, cast

import httpx
import pytest
import respx

import litellm
import litellm.proxy.proxy_server
from litellm.secret_managers.cyberark_secret_manager import CyberArkSecretManager

FIXTURE_PATH: Final = Path(__file__).resolve().parents[3] / "litellm-rust/crates/secrets-cyberark/tests/fixtures/parity.json"


class ParitySecret(TypedDict):
    name: str
    path: str
    policy_body: str


class ParityFixture(TypedDict):
    endpoint: str
    account: str
    username: str
    api_key: str
    authenticate_path: str
    token_json: str
    authorization_header: str
    policy_path: str
    secrets: list[ParitySecret]


def _fixture() -> ParityFixture:
    return cast(ParityFixture, json.loads(FIXTURE_PATH.read_text()))


def _configure_manager(monkeypatch: pytest.MonkeyPatch, fixture: ParityFixture) -> CyberArkSecretManager:
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setenv("CYBERARK_API_BASE", fixture["endpoint"])
    monkeypatch.setenv("CYBERARK_ACCOUNT", fixture["account"])
    monkeypatch.setenv("CYBERARK_USERNAME", fixture["username"])
    monkeypatch.setenv("CYBERARK_API_KEY", fixture["api_key"])
    monkeypatch.setenv("CYBERARK_REFRESH_INTERVAL", "300")
    monkeypatch.delenv("CYBERARK_CLIENT_CERT", raising=False)
    monkeypatch.delenv("CYBERARK_CLIENT_KEY", raising=False)
    return CyberArkSecretManager()


def _respond(
    route: respx.Route,
    *,
    status_code: int = 200,
    content: str | bytes | None = None,
    text: str | None = None,
) -> respx.Route:
    return route.respond(  # pyright: ignore[reportUnknownMemberType]  # respx route stubs leave response builder partially unknown
        status_code=status_code,
        content=content,
        text=text,
    )


@respx.mock
def test_sync_read_matches_parity_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture: Final = _fixture()
    manager: Final = _configure_manager(monkeypatch, fixture)
    endpoint: Final = fixture["endpoint"]
    token_json: Final = fixture["token_json"]
    auth_route: Final = _respond(
        respx.post(endpoint + fixture["authenticate_path"]),
        content=token_json.encode(),
    )
    routes: Final = [
        _respond(respx.get(endpoint + secret["path"]), text="value")
        for secret in fixture["secrets"]
    ]

    for secret in fixture["secrets"]:
        assert manager.sync_read_secret(secret["name"]) == "value"  # pyright: ignore[reportUnknownMemberType]  # legacy secret manager API is untyped

    expected_authorization: Final = fixture["authorization_header"]
    assert auth_route.calls.last.request.content == fixture["api_key"].encode()
    assert all(route.calls.last.request.headers["Authorization"] == expected_authorization for route in routes)
    assert all(
        route.calls.last.request.url.raw_path.decode() == secret["path"]
        for route, secret in zip(routes, fixture["secrets"], strict=True)
    )


@pytest.mark.asyncio
@respx.mock
async def test_async_write_matches_parity_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    fixture: Final = _fixture()
    manager: Final = _configure_manager(monkeypatch, fixture)
    secret: Final = fixture["secrets"][0]
    endpoint: Final = fixture["endpoint"]
    token_json: Final = fixture["token_json"]
    _respond(respx.post(endpoint + fixture["authenticate_path"]), content=token_json.encode())
    policy_route: Final = _respond(respx.post(endpoint + fixture["policy_path"]), status_code=201)
    value_route: Final = _respond(respx.post(endpoint + secret["path"]), status_code=201)

    await manager.async_write_secret(secret["name"], "v")  # pyright: ignore[reportUnknownMemberType]  # legacy secret manager API is untyped

    assert policy_route.calls.last.request.content.decode() == secret["policy_body"]
    assert policy_route.calls.last.request.headers["Content-Type"] == "application/x-yaml"
    assert value_route.calls.last.request.content == b"v"


@pytest.mark.asyncio
@respx.mock
async def test_async_write_retries_policy_load_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    fixture: Final = _fixture()
    manager: Final = _configure_manager(monkeypatch, fixture)
    secret: Final = fixture["secrets"][0]
    endpoint: Final = fixture["endpoint"]
    _respond(respx.post(endpoint + fixture["authenticate_path"]), content=fixture["token_json"].encode())
    policy_route: Final = respx.post(endpoint + fixture["policy_path"]).mock(
        side_effect=[httpx.Response(409), httpx.Response(409), httpx.Response(201)]
    )
    value_route: Final = respx.post(endpoint + secret["path"]).mock(
        side_effect=lambda _: httpx.Response(201 if policy_route.call_count == 3 else 404)
    )

    result: Final = await manager.async_write_secret(secret["name"], "v")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy secret manager API is untyped

    assert policy_route.call_count == 3
    assert value_route.call_count == 1
    assert result["status"] == "success"


def test_missing_credentials_raise_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    for name in (
        "CYBERARK_API_KEY",
        "CYBERARK_CLIENT_CERT",
        "CYBERARK_CLIENT_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Missing CyberArk credentials"):
        CyberArkSecretManager()
