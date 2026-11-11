import asyncio
import json
from pathlib import Path
from typing import Final, TypedDict, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
import yaml

import litellm
import litellm.proxy.proxy_server
from litellm._uuid import uuid
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy_outcome",
    [422, 500, httpx.ConnectError("conjur unreachable")],
    ids=["unprocessable", "server_error", "unreachable"],
)
@respx.mock
async def test_async_write_does_not_retry_non_conflict_policy_failures(
    monkeypatch: pytest.MonkeyPatch, policy_outcome: int | httpx.ConnectError
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    fixture: Final = _fixture()
    manager: Final = _configure_manager(monkeypatch, fixture)
    secret: Final = fixture["secrets"][0]
    endpoint: Final = fixture["endpoint"]
    _respond(respx.post(endpoint + fixture["authenticate_path"]), content=fixture["token_json"].encode())
    policy_route: Final = respx.post(endpoint + fixture["policy_path"])
    if isinstance(policy_outcome, int):
        _respond(policy_route, status_code=policy_outcome)
    else:
        policy_route.mock(side_effect=policy_outcome)
    value_route: Final = _respond(respx.post(endpoint + secret["path"]), status_code=201)

    await manager.async_write_secret(secret["name"], "v")  # pyright: ignore[reportUnknownMemberType]  # legacy secret manager API is untyped

    assert policy_route.call_count == 1
    assert value_route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_concurrent_async_writes_load_policy_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    fixture: Final = _fixture()
    manager: Final = _configure_manager(monkeypatch, fixture)
    endpoint: Final = fixture["endpoint"]
    _respond(respx.post(endpoint + fixture["authenticate_path"]), content=fixture["token_json"].encode())
    in_flight: Final = asyncio.Semaphore(1)

    async def load_policy(_: httpx.Request) -> httpx.Response:
        if in_flight.locked():
            return httpx.Response(409)
        async with in_flight:
            await asyncio.sleep(0.05)
        return httpx.Response(201)

    policy_route: Final = respx.post(endpoint + fixture["policy_path"]).mock(side_effect=load_policy)
    respx.post(url__startswith=endpoint + "/secrets/").respond(status_code=201)  # pyright: ignore[reportUnknownMemberType]  # respx route stubs leave response builder partially unknown

    results: Final = await asyncio.gather(
        *(manager.async_write_secret(f"concurrent-{index}", "v") for index in range(4))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy secret manager API is untyped
    )

    assert policy_route.call_count == 4
    assert [result["status"] for result in results] == ["success"] * 4


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


@pytest.fixture
def cyberark_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CYBERARK_API_KEY", "test-cyberark-api-key-909")
    monkeypatch.setenv("CYBERARK_API_BASE", "http://0.0.0.0:8080")
    monkeypatch.setenv("CYBERARK_ACCOUNT", "default")
    monkeypatch.setenv("CYBERARK_USERNAME", "admin")


def create_mock_response(status_code: int, text: str = ""):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = text
    mock_response.raise_for_status = MagicMock()

    if status_code >= 400:
        error = httpx.HTTPStatusError(message=f"HTTP {status_code}", request=MagicMock(), response=mock_response)
        mock_response.raise_for_status.side_effect = error

    return mock_response


@pytest.mark.asyncio
async def test_cyberark_write_secret_rejects_yaml_injection(cyberark_env):
    with patch("litellm.proxy.proxy_server.premium_user", True):
        malicious_secret_name = "foo\n- !grant\n  role: !!admin\n  member: attacker"

        mock_sync_client = MagicMock()
        mock_async_client = AsyncMock()

        with (
            patch(
                "litellm.secret_managers.cyberark_secret_manager.get_httpx_client",
                return_value=mock_sync_client,
            ),
            patch(
                "litellm.secret_managers.cyberark_secret_manager.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            cyberark_manager = CyberArkSecretManager()

            response = await cyberark_manager.async_write_secret(
                secret_name=malicious_secret_name,
                secret_value="sk-9876",
            )

            assert response["status"] == "error"
            assert "Invalid secret_name" in response["message"]
            mock_sync_client.client.post.assert_not_called()
            mock_async_client.post.assert_not_called()


@pytest.mark.parametrize(
    "secret_name",
    [
        "foo: bar",
        "foo # bar",
        "plain-alias",
        "team/user@example.com",
    ],
)
@pytest.mark.asyncio
async def test_cyberark_ensure_variable_exists_escapes_yaml_metacharacters(cyberark_env, secret_name):
    with patch("litellm.proxy.proxy_server.premium_user", True):
        captured = {}

        async def _capture_post(url, headers=None, content=None):
            captured["content"] = content
            return create_mock_response(status_code=201, text="")

        mock_sync_client = MagicMock()
        mock_sync_client.client.post.return_value = create_mock_response(status_code=200, text="mock-token")
        mock_async_client = MagicMock()
        mock_async_client.client.post.side_effect = _capture_post

        with patch(
            "litellm.secret_managers.cyberark_secret_manager.get_httpx_client",
            return_value=mock_sync_client,
        ):
            cyberark_manager = CyberArkSecretManager()
            await cyberark_manager._ensure_variable_exists(secret_name, mock_async_client)

        policy_yaml = captured["content"]
        parsed = yaml.compose(policy_yaml)
        assert len(parsed.value) == 1
        node = parsed.value[0]
        assert node.tag == "!variable"
        assert node.value == secret_name


@pytest.mark.asyncio
async def test_cyberark_write_and_read_secret(cyberark_env):
    with patch("litellm.proxy.proxy_server.premium_user", True):
        secret_name = f"test-secret-{uuid.uuid4()}"
        secret_value = f"test-value-{uuid.uuid4()}"

        mock_sync_client = MagicMock()
        mock_sync_client.client.post.return_value = create_mock_response(status_code=200, text="mock-token")
        mock_sync_client.client.get.return_value = create_mock_response(status_code=200, text=secret_value)

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = create_mock_response(status_code=201, text="")

        with (
            patch(
                "litellm.secret_managers.cyberark_secret_manager.get_httpx_client",
                return_value=mock_sync_client,
            ),
            patch(
                "litellm.secret_managers.cyberark_secret_manager.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            cyberark_manager = CyberArkSecretManager()

            write_response = await cyberark_manager.async_write_secret(
                secret_name=secret_name,
                secret_value=secret_value,
            )

            assert write_response["status"] == "success"

            read_value = cyberark_manager.sync_read_secret(secret_name=secret_name)

            assert read_value is not None
            assert read_value == secret_value


@pytest.mark.asyncio
async def test_cyberark_rotate_secret(cyberark_env):
    with patch("litellm.proxy.proxy_server.premium_user", True):
        secret_alias = f"test-rotation-key-{uuid.uuid4()}"
        initial_key_value = f"sk-initial-{uuid.uuid4()}"
        rotated_key_value = f"sk-rotated-{uuid.uuid4()}"

        current_value = {"value": initial_key_value}

        mock_sync_client = MagicMock()
        mock_sync_client.client.post.return_value = create_mock_response(status_code=200, text="mock-token")

        def get_mock_sync_read_response(*args, **kwargs):
            return create_mock_response(status_code=200, text=current_value["value"])

        mock_sync_client.client.get.side_effect = get_mock_sync_read_response

        mock_async_client = AsyncMock()

        async def mock_async_post(*args, **kwargs):
            content = kwargs.get("content", "")
            if content:
                current_value["value"] = content
            return create_mock_response(status_code=201, text="")

        mock_async_client.post.side_effect = mock_async_post

        async def get_mock_async_read_response(*args, **kwargs):
            return create_mock_response(status_code=200, text=current_value["value"])

        mock_async_client.get.side_effect = get_mock_async_read_response

        with (
            patch(
                "litellm.secret_managers.cyberark_secret_manager.get_httpx_client",
                return_value=mock_sync_client,
            ),
            patch(
                "litellm.secret_managers.cyberark_secret_manager.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            cyberark_manager = CyberArkSecretManager()

            write_response = await cyberark_manager.async_write_secret(
                secret_name=secret_alias,
                secret_value=initial_key_value,
            )
            assert write_response["status"] == "success"

            initial_read = cyberark_manager.sync_read_secret(secret_name=secret_alias)
            assert initial_read == initial_key_value

            rotation_response = await cyberark_manager.async_rotate_secret(
                current_secret_name=secret_alias,
                new_secret_name=secret_alias,
                new_secret_value=rotated_key_value,
            )
            assert rotation_response["status"] == "success"

            cyberark_manager.cache.flush_cache()

            rotated_read = cyberark_manager.sync_read_secret(secret_name=secret_alias)

            assert rotated_read is not None
            assert rotated_read == rotated_key_value
            assert rotated_read != initial_key_value


@pytest.mark.asyncio
async def test_cyberark_rotate_secret_with_new_alias(cyberark_env):
    with patch("litellm.proxy.proxy_server.premium_user", True):
        base_alias = f"test-alias-change-{uuid.uuid4()}"
        old_alias = f"{base_alias}-v1"
        new_alias = f"{base_alias}-v2"
        old_value = f"sk-old-{uuid.uuid4()}"
        new_value = f"sk-new-{uuid.uuid4()}"

        secrets_store = {}

        mock_sync_client = MagicMock()
        mock_sync_client.client.post.return_value = create_mock_response(status_code=200, text="mock-token")

        def get_mock_sync_read(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            for secret_name, secret_val in secrets_store.items():
                if secret_name in url:
                    return create_mock_response(status_code=200, text=secret_val)
            return create_mock_response(status_code=404, text="Not found")

        mock_sync_client.client.get.side_effect = get_mock_sync_read

        mock_async_client = AsyncMock()

        async def mock_async_post(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            content = kwargs.get("content", "")

            if old_alias in url:
                secrets_store[old_alias] = content
            elif new_alias in url:
                secrets_store[new_alias] = content

            return create_mock_response(status_code=201, text="")

        mock_async_client.post.side_effect = mock_async_post

        async def get_mock_async_read(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            for secret_name, secret_val in secrets_store.items():
                if secret_name in url:
                    return create_mock_response(status_code=200, text=secret_val)
            return create_mock_response(status_code=404, text="Not found")

        mock_async_client.get.side_effect = get_mock_async_read

        with (
            patch(
                "litellm.secret_managers.cyberark_secret_manager.get_httpx_client",
                return_value=mock_sync_client,
            ),
            patch(
                "litellm.secret_managers.cyberark_secret_manager.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            cyberark_manager = CyberArkSecretManager()

            write_response = await cyberark_manager.async_write_secret(
                secret_name=old_alias,
                secret_value=old_value,
            )
            assert write_response["status"] == "success"

            rotation_response = await cyberark_manager.async_rotate_secret(
                current_secret_name=old_alias,
                new_secret_name=new_alias,
                new_secret_value=new_value,
            )
            assert rotation_response["status"] == "success"

            cyberark_manager.cache.flush_cache()

            new_read = cyberark_manager.sync_read_secret(secret_name=new_alias)
            assert new_read == new_value

            old_read = cyberark_manager.sync_read_secret(secret_name=old_alias)
            assert old_read == old_value
