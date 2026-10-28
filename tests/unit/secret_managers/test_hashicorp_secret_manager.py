import datetime
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from pydantic import JsonValue, TypeAdapter

import litellm.proxy.proxy_server
from litellm.secret_managers.hashicorp_secret_manager import HashicorpSecretManager

VAULT_ADDR: Final = "http://vault.test:8200"
LOGIN_RESPONSE: Final = {"auth": {"client_token": "hvs.login-token", "lease_duration": 3600}}
SECRET_RESPONSE: Final = {"data": {"data": {"key": "sk-from-vault", "password": "pw-from-vault"}}}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

NAMESPACE_ENV_VARS: Final = ("HCP_VAULT_NAMESPACE", "HCP_VAULT_LOGIN_NAMESPACE", "HCP_VAULT_SECRET_NAMESPACE")
PARITY_ENV_VARS: Final = (
    "HCP_VAULT_ADDR",
    "HCP_VAULT_TOKEN",
    "HCP_VAULT_NAMESPACE",
    "HCP_VAULT_LOGIN_NAMESPACE",
    "HCP_VAULT_SECRET_NAMESPACE",
    "HCP_VAULT_MOUNT_NAME",
    "HCP_VAULT_PATH_PREFIX",
    "HCP_VAULT_APPROLE_ROLE_ID",
    "HCP_VAULT_APPROLE_SECRET_ID",
    "HCP_VAULT_APPROLE_MOUNT_PATH",
    "HCP_VAULT_CLIENT_CERT",
    "HCP_VAULT_CLIENT_KEY",
    "HCP_VAULT_CERT_ROLE",
    "HCP_VAULT_REFRESH_INTERVAL",
    "SECRET_MANAGER_REFRESH_INTERVAL",
)


def _build_manager(monkeypatch: pytest.MonkeyPatch, env: Mapping[str, str]) -> HashicorpSecretManager:
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    for name in NAMESPACE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in ("HCP_VAULT_CLIENT_CERT", "HCP_VAULT_CLIENT_KEY", "HCP_VAULT_CERT_ROLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HCP_VAULT_ADDR", VAULT_ADDR)
    monkeypatch.delenv("HCP_VAULT_TOKEN", raising=False)
    monkeypatch.setenv("HCP_VAULT_APPROLE_ROLE_ID", "role-id")
    monkeypatch.setenv("HCP_VAULT_APPROLE_SECRET_ID", "secret-id")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return HashicorpSecretManager()


@pytest.mark.parametrize(
    ("env", "expected_login_namespace", "expected_secret_namespace"),
    [
        ({"HCP_VAULT_LOGIN_NAMESPACE": "root", "HCP_VAULT_SECRET_NAMESPACE": "teams/team-a"}, "root", "teams/team-a"),
        ({"HCP_VAULT_NAMESPACE": "admin"}, "admin", "admin"),
        ({"HCP_VAULT_NAMESPACE": "admin", "HCP_VAULT_LOGIN_NAMESPACE": "root"}, "root", "admin"),
        ({"HCP_VAULT_NAMESPACE": "admin", "HCP_VAULT_SECRET_NAMESPACE": "teams/team-a"}, "admin", "teams/team-a"),
    ],
)
@respx.mock
def test_sync_read_uses_login_namespace_for_approle_and_secret_namespace_for_url(
    monkeypatch: pytest.MonkeyPatch,
    env: Mapping[str, str],
    expected_login_namespace: str,
    expected_secret_namespace: str,
) -> None:
    manager: Final = _build_manager(monkeypatch, env)
    login_route: Final = respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    read_route: Final = respx.get(f"{VAULT_ADDR}/v1/{expected_secret_namespace}/secret/data/OPENAI_API_KEY").respond(
        json=SECRET_RESPONSE
    )

    assert manager.sync_read_secret("OPENAI_API_KEY") == "sk-from-vault"

    assert login_route.call_count == 1
    assert login_route.calls.last.request.headers["X-Vault-Namespace"] == expected_login_namespace
    assert read_route.call_count == 1
    read_request: Final = read_route.calls.last.request
    assert read_request.headers["X-Vault-Token"] == "hvs.login-token"
    assert "X-Vault-Namespace" not in read_request.headers


@respx.mock
def test_login_header_is_omitted_when_no_namespace_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    manager: Final = _build_manager(monkeypatch, {})
    login_route: Final = respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    read_route: Final = respx.get(f"{VAULT_ADDR}/v1/secret/data/OPENAI_API_KEY").respond(json=SECRET_RESPONSE)

    assert manager.sync_read_secret("OPENAI_API_KEY") == "sk-from-vault"

    assert "X-Vault-Namespace" not in login_route.calls.last.request.headers
    assert read_route.call_count == 1


@respx.mock
def test_sync_read_per_secret_namespace_overrides_secret_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    manager: Final = _build_manager(
        monkeypatch, {"HCP_VAULT_LOGIN_NAMESPACE": "root", "HCP_VAULT_SECRET_NAMESPACE": "teams/team-a"}
    )
    login_route: Final = respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    read_route: Final = respx.get(f"{VAULT_ADDR}/v1/teams/team-b/kv-prod/data/virtual-keys/DB_PASSWORD").respond(
        json=SECRET_RESPONSE
    )
    optional_params: Final = {
        "secret_manager_settings": {
            "namespace": "teams/team-b",
            "mount": "kv-prod",
            "path_prefix": "virtual-keys",
            "data": "password",
        }
    }

    assert manager.sync_read_secret("DB_PASSWORD", optional_params=optional_params) == "pw-from-vault"

    assert login_route.calls.last.request.headers["X-Vault-Namespace"] == "root"
    assert read_route.call_count == 1


@respx.mock
def test_sync_read_caches_per_resolved_target(monkeypatch: pytest.MonkeyPatch) -> None:
    manager: Final = _build_manager(monkeypatch, {"HCP_VAULT_SECRET_NAMESPACE": "teams/team-a"})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    team_a_route: Final = respx.get(f"{VAULT_ADDR}/v1/teams/team-a/secret/data/SHARED").respond(
        json={"data": {"data": {"key": "team-a-value"}}}
    )
    team_b_route: Final = respx.get(f"{VAULT_ADDR}/v1/teams/team-b/secret/data/SHARED").respond(
        json={"data": {"data": {"key": "team-b-value"}}}
    )
    team_b_params: Final = {"secret_manager_settings": {"namespace": "teams/team-b"}}

    assert manager.sync_read_secret("SHARED") == "team-a-value"
    assert manager.sync_read_secret("SHARED", optional_params=team_b_params) == "team-b-value"
    assert manager.sync_read_secret("SHARED") == "team-a-value"

    assert team_a_route.call_count == 1
    assert team_b_route.call_count == 1


@respx.mock
def test_sync_read_caches_per_data_key_for_the_same_secret_path(monkeypatch: pytest.MonkeyPatch) -> None:
    manager: Final = _build_manager(monkeypatch, {"HCP_VAULT_SECRET_NAMESPACE": "teams/team-a"})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    respx.get(f"{VAULT_ADDR}/v1/teams/team-a/secret/data/DB_CREDS").respond(json=SECRET_RESPONSE)
    password_params: Final = {"secret_manager_settings": {"data": "password"}}

    assert manager.sync_read_secret("DB_CREDS") == "sk-from-vault"
    assert manager.sync_read_secret("DB_CREDS", optional_params=password_params) == "pw-from-vault"
    assert manager.sync_read_secret("DB_CREDS") == "sk-from-vault"


@pytest.mark.asyncio
@respx.mock
async def test_async_delete_evicts_every_cached_field_of_the_secret_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {"HCP_VAULT_SECRET_NAMESPACE": "teams/team-a"})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    secret_url: Final = f"{VAULT_ADDR}/v1/teams/team-a/secret/data/DB_CREDS"
    read_route: Final = respx.get(secret_url).respond(json=SECRET_RESPONSE)
    respx.delete(secret_url).respond(status_code=204)
    password_params: Final = {"secret_manager_settings": {"data": "password"}}

    assert await manager.async_read_secret("DB_CREDS", optional_params=password_params) == "pw-from-vault"
    assert await manager.async_delete_secret("DB_CREDS")
    assert await manager.async_read_secret("DB_CREDS", optional_params=password_params) == "pw-from-vault"

    assert read_route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_async_read_uses_secret_namespace_and_login_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(
        monkeypatch, {"HCP_VAULT_LOGIN_NAMESPACE": "root", "HCP_VAULT_SECRET_NAMESPACE": "teams/team-a"}
    )
    login_route: Final = respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    read_route: Final = respx.get(f"{VAULT_ADDR}/v1/teams/team-a/secret/data/OPENAI_API_KEY").respond(
        json=SECRET_RESPONSE
    )

    assert await manager.async_read_secret("OPENAI_API_KEY") == "sk-from-vault"

    assert login_route.calls.last.request.headers["X-Vault-Namespace"] == "root"
    assert read_route.call_count == 1
    assert "X-Vault-Namespace" not in read_route.calls.last.request.headers


@pytest.mark.asyncio
@respx.mock
async def test_async_write_and_read_share_the_secret_namespace_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(
        monkeypatch, {"HCP_VAULT_LOGIN_NAMESPACE": "root", "HCP_VAULT_SECRET_NAMESPACE": "teams/team-a"}
    )
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    write_route: Final = respx.post(f"{VAULT_ADDR}/v1/teams/team-a/secret/data/VIRTUAL_KEY").respond(
        json={"data": {"version": 1}}
    )
    read_route: Final = respx.get(f"{VAULT_ADDR}/v1/teams/team-a/secret/data/VIRTUAL_KEY").respond(
        json={"data": {"data": {"key": "sk-virtual"}}}
    )

    await manager.async_write_secret("VIRTUAL_KEY", "sk-virtual")
    assert await manager.async_read_secret("VIRTUAL_KEY") == "sk-virtual"

    assert write_route.call_count == 1
    assert read_route.call_count == 1


def _write_self_signed_cert(directory: Path) -> tuple[Path, Path]:
    private_key: Final = ec.derive_private_key(1, ec.SECP256R1())
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "litellm-test")])
    valid_from: Final = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    valid_until: Final = datetime.datetime(2049, 12, 31, tzinfo=datetime.timezone.utc)
    certificate: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(1)
        .not_valid_before(valid_from)
        .not_valid_after(valid_until)
        .sign(private_key, hashes.SHA256())
    )
    cert_path: Final = directory / "client.crt"
    key_path: Final = directory / "client.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@respx.mock
def test_tls_login_uses_login_namespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cert, key = _write_self_signed_cert(tmp_path)
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    for name in NAMESPACE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HCP_VAULT_TOKEN", raising=False)
    monkeypatch.delenv("HCP_VAULT_APPROLE_ROLE_ID", raising=False)
    monkeypatch.delenv("HCP_VAULT_APPROLE_SECRET_ID", raising=False)
    monkeypatch.setenv("HCP_VAULT_ADDR", VAULT_ADDR)
    monkeypatch.setenv("HCP_VAULT_CLIENT_CERT", str(cert))
    monkeypatch.setenv("HCP_VAULT_CLIENT_KEY", str(key))
    monkeypatch.setenv("HCP_VAULT_CERT_ROLE", "test-role")
    monkeypatch.setenv("HCP_VAULT_NAMESPACE", "admin")
    monkeypatch.setenv("HCP_VAULT_LOGIN_NAMESPACE", "root")
    manager: Final = HashicorpSecretManager()
    login_route: Final = respx.post(f"{VAULT_ADDR}/v1/auth/cert/login").respond(json=LOGIN_RESPONSE)

    assert manager._auth_via_tls_cert() == "hvs.login-token"
    assert login_route.calls.last.request.headers["X-Vault-Namespace"] == "root"
    assert _JSON_OBJECT.validate_json(login_route.calls.last.request.content) == {"name": "test-role"}
    assert manager.cache.get_cache("hcp_vault_token") == "hvs.login-token"


@respx.mock
def test_hashicorp_secret_manager_get_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    manager: Final = _build_manager(monkeypatch, {"HCP_VAULT_NAMESPACE": "admin"})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    read_route: Final = respx.get(f"{VAULT_ADDR}/v1/admin/secret/data/sample-secret-mock").respond(
        json=SECRET_RESPONSE
    )

    assert manager.sync_read_secret("sample-secret-mock") == "sk-from-vault"

    assert read_route.call_count == 1
    assert read_route.calls.last.request.headers["X-Vault-Token"] == "hvs.login-token"


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_write_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {"HCP_VAULT_NAMESPACE": "admin"})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    write_route: Final = respx.post(f"{VAULT_ADDR}/v1/admin/secret/data/sample-secret").respond(
        json={"data": {"version": 1}}
    )

    response: Final = await manager.async_write_secret("sample-secret", "value-mock")

    assert response == {"data": {"version": 1}}
    assert write_route.call_count == 1
    assert _JSON_OBJECT.validate_json(write_route.calls.last.request.content) == {"data": {"key": "value-mock"}}


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_write_secret_with_team_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    write_route: Final = respx.post(
        f"{VAULT_ADDR}/v1/team-namespace/kv-team/data/teams/custom/team-secret"
    ).respond(json={"data": {"version": 1}})
    settings: Final = {
        "secret_manager_settings": {
            "namespace": "team-namespace",
            "mount": "kv-team",
            "path_prefix": "teams/custom",
            "data": "password",
        }
    }

    response: Final = await manager.async_write_secret(
        "team-secret", "value-mock", optional_params=settings
    )

    assert response == {"data": {"version": 1}}
    assert _JSON_OBJECT.validate_json(write_route.calls.last.request.content) == {"data": {"password": "value-mock"}}


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_delete_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {"HCP_VAULT_NAMESPACE": "admin"})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    delete_route: Final = respx.delete(f"{VAULT_ADDR}/v1/admin/secret/data/sample-secret").respond(
        status_code=204
    )

    response: Final = await manager.async_delete_secret("sample-secret")

    assert response == {
        "status": "success",
        "message": "Secret sample-secret deleted successfully",
    }
    assert delete_route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_delete_secret_with_team_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    delete_route: Final = respx.delete(
        f"{VAULT_ADDR}/v1/team-namespace/kv-team/data/teams/custom/team-secret"
    ).respond(status_code=204)
    settings: Final = {
        "secret_manager_settings": {
            "namespace": "team-namespace",
            "mount": "kv-team",
            "path_prefix": "teams/custom",
        }
    }

    response: Final = await manager.async_delete_secret("team-secret", optional_params=settings)

    assert response == {
        "status": "success",
        "message": "Secret team-secret deleted successfully",
    }
    assert delete_route.call_count == 1


@respx.mock
def test_hashicorp_secret_manager_approle_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    manager: Final = _build_manager(
        monkeypatch,
        {
            "HCP_VAULT_APPROLE_ROLE_ID": "test-role-id-123",
            "HCP_VAULT_APPROLE_SECRET_ID": "test-secret-id-456",
            "HCP_VAULT_APPROLE_MOUNT_PATH": "approle",
        },
    )
    login_route: Final = respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)

    assert manager._auth_via_approle() == "hvs.login-token"

    assert _JSON_OBJECT.validate_json(login_route.calls.last.request.content) == {
        "role_id": "test-role-id-123",
        "secret_id": "test-secret-id-456",
    }
    assert manager.cache.get_cache("hcp_vault_approle_token") == "hvs.login-token"


def test_hashicorp_custom_mount_and_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    manager: Final = _build_manager(monkeypatch, {"HCP_VAULT_NAMESPACE": "admin"})

    assert manager.get_url("my-secret") == f"{VAULT_ADDR}/v1/admin/secret/data/my-secret"
    assert manager.get_url("my-secret", mount_name="kv") == f"{VAULT_ADDR}/v1/admin/kv/data/my-secret"
    assert manager.get_url("my-secret", path_prefix="myapp") == (
        f"{VAULT_ADDR}/v1/admin/secret/data/myapp/my-secret"
    )
    assert manager.get_url("my-secret", mount_name="kv", path_prefix="production") == (
        f"{VAULT_ADDR}/v1/admin/kv/data/production/my-secret"
    )


@pytest.mark.parametrize(
    "malicious_secret_name",
    [
        "../../../other-app/creds",
        "litellm/../../secret",
        "foo\nbar",
        "foo\u2028bar",
        "foo\u2029bar",
        "foo\x85bar",
    ],
)
def test_hashicorp_get_url_rejects_path_traversal(
    monkeypatch: pytest.MonkeyPatch, malicious_secret_name: str
) -> None:
    manager: Final = _build_manager(monkeypatch, {})

    with pytest.raises(ValueError, match="Invalid secret_name"):
        manager.get_url(malicious_secret_name)


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_rotate_secret_different_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    current_route: Final = respx.get(f"{VAULT_ADDR}/v1/secret/data/old-secret").respond(
        json={"data": {"data": {"key": "old-secret-value"}}}
    )
    write_route: Final = respx.post(f"{VAULT_ADDR}/v1/secret/data/new-secret").respond(
        json={"data": {"version": 1}}
    )
    new_route: Final = respx.get(f"{VAULT_ADDR}/v1/secret/data/new-secret").respond(
        json={"data": {"data": {"key": "new-secret-value"}}}
    )
    delete_route: Final = respx.delete(f"{VAULT_ADDR}/v1/secret/data/old-secret").respond(
        status_code=204
    )

    response: Final = await manager.async_rotate_secret(
        "old-secret", "new-secret", "new-secret-value"
    )

    assert response == {"data": {"version": 1}}
    assert current_route.call_count == 1
    assert _JSON_OBJECT.validate_json(write_route.calls.last.request.content) == {
        "data": {"key": "new-secret-value", "description": "Rotated from old-secret"}
    }
    assert new_route.call_count == 1
    assert delete_route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_rotate_secret_same_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    secret_route: Final = respx.get(f"{VAULT_ADDR}/v1/secret/data/same-secret").mock(
        side_effect=[
            httpx.Response(200, json={"data": {"data": {"key": "old-secret-value"}}}),
            httpx.Response(200, json={"data": {"data": {"key": "updated-secret-value"}}}),
        ]
    )
    write_route: Final = respx.post(f"{VAULT_ADDR}/v1/secret/data/same-secret").respond(
        json={"data": {"version": 1}}
    )
    delete_route: Final = respx.delete(f"{VAULT_ADDR}/v1/secret/data/same-secret").respond(
        status_code=204
    )

    response: Final = await manager.async_rotate_secret(
        "same-secret", "same-secret", "updated-secret-value"
    )

    assert response == {"data": {"version": 1}}
    assert secret_route.call_count == 2
    assert write_route.call_count == 1
    assert delete_route.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_rotate_secret_current_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    respx.get(f"{VAULT_ADDR}/v1/secret/data/non-existent").respond(
        status_code=404, text="Not Found"
    )

    response: Final = await manager.async_rotate_secret(
        "non-existent", "new-secret", "new-value"
    )

    assert response["status"] == "error"
    assert "non-existent" in response["message"]
    assert "not found" in response["message"].lower()


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_rotate_secret_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    respx.get(f"{VAULT_ADDR}/v1/secret/data/old-secret").respond(
        json={"data": {"data": {"key": "old-secret-value"}}}
    )
    write_route: Final = respx.post(f"{VAULT_ADDR}/v1/secret/data/new-secret").respond(
        json={"status": "error", "message": "Write failed"}
    )

    response: Final = await manager.async_rotate_secret(
        "old-secret", "new-secret", "new-value"
    )

    assert response == {"status": "error", "message": "Write failed"}
    assert write_route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_rotate_secret_with_team_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    current_route: Final = respx.get(
        f"{VAULT_ADDR}/v1/team-namespace/kv-team/data/teams/custom/team-old-secret"
    ).respond(json={"data": {"data": {"password": "old-secret-value"}}})
    write_route: Final = respx.post(
        f"{VAULT_ADDR}/v1/team-namespace/kv-team/data/teams/custom/team-new-secret"
    ).respond(json={"data": {"version": 1}})
    new_route: Final = respx.get(
        f"{VAULT_ADDR}/v1/team-namespace/kv-team/data/teams/custom/team-new-secret"
    ).respond(json={"data": {"data": {"password": "new-team-secret-value"}}})
    delete_route: Final = respx.delete(
        f"{VAULT_ADDR}/v1/team-namespace/kv-team/data/teams/custom/team-old-secret"
    ).respond(status_code=204)
    settings: Final = {
        "secret_manager_settings": {
            "namespace": "team-namespace",
            "mount": "kv-team",
            "path_prefix": "teams/custom",
            "data": "password",
        }
    }

    response: Final = await manager.async_rotate_secret(
        "team-old-secret", "team-new-secret", "new-team-secret-value", optional_params=settings
    )

    assert response == {"data": {"version": 1}}
    assert current_route.call_count == 1
    assert _JSON_OBJECT.validate_json(write_route.calls.last.request.content) == {
        "data": {
            "password": "new-team-secret-value",
            "description": "Rotated from team-old-secret",
        }
    }
    assert new_route.call_count == 1
    assert delete_route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_hashicorp_secret_manager_rotate_secret_value_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    manager: Final = _build_manager(monkeypatch, {})
    respx.post(f"{VAULT_ADDR}/v1/auth/approle/login").respond(json=LOGIN_RESPONSE)
    respx.get(f"{VAULT_ADDR}/v1/secret/data/old-secret").respond(
        json={"data": {"data": {"key": "old-secret-value"}}}
    )
    respx.post(f"{VAULT_ADDR}/v1/secret/data/new-secret").respond(
        json={"data": {"version": 1}}
    )
    respx.get(f"{VAULT_ADDR}/v1/secret/data/new-secret").respond(
        json={"data": {"data": {"key": "different-value"}}}
    )

    response: Final = await manager.async_rotate_secret(
        "old-secret", "new-secret", "expected-value"
    )

    assert response["status"] == "error"
    assert "mismatch" in response["message"].lower()
    assert "expected-value" in response["message"]


with Path(__file__).with_name("hashicorp_vault_parity.json").open() as parity_file:
    PARITY_CASES: Final = json.load(parity_file)


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case["name"])
def test_configuration_matches_native_parity_fixture(
    monkeypatch: pytest.MonkeyPatch, case: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    for name in PARITY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in case["env"].items():
        monkeypatch.setenv(name, value)

    manager: Final = HashicorpSecretManager()
    env: Final = case["env"]
    expected_login_url: Final = case["expected_login_url"]
    if env.get("HCP_VAULT_APPROLE_ROLE_ID") and env.get("HCP_VAULT_APPROLE_SECRET_ID"):
        login_url: str | None = (
            f"{manager.vault_addr}/v1/auth/{manager.approle_mount_path}/login"
        )
    elif env.get("HCP_VAULT_CLIENT_CERT") and env.get("HCP_VAULT_CLIENT_KEY"):
        login_url = f"{manager.vault_addr}/v1/auth/cert/login"
    else:
        login_url = None

    assert manager.get_url(case["secret_name"]) == case["expected_secret_url"]
    assert manager.vault_login_namespace == case["expected_login_namespace"]
    assert manager.vault_secret_namespace == case["expected_secret_namespace"]
    assert login_url == expected_login_url
