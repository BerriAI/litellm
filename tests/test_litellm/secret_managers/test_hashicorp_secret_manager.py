import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
import respx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import litellm.proxy.proxy_server
from litellm.secret_managers.hashicorp_secret_manager import HashicorpSecretManager

VAULT_ADDR: Final = "http://vault.test:8200"
LOGIN_RESPONSE: Final = {"auth": {"client_token": "hvs.login-token", "lease_duration": 3600}}
SECRET_RESPONSE: Final = {"data": {"data": {"key": "sk-from-vault", "password": "pw-from-vault"}}}

NAMESPACE_ENV_VARS: Final = ("HCP_VAULT_NAMESPACE", "HCP_VAULT_LOGIN_NAMESPACE", "HCP_VAULT_SECRET_NAMESPACE")


def _build_manager(monkeypatch: pytest.MonkeyPatch, env: Mapping[str, str]) -> HashicorpSecretManager:
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    for name in NAMESPACE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HCP_VAULT_ADDR", VAULT_ADDR)
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
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "litellm-test")])
    now: Final = datetime.datetime.now(datetime.timezone.utc)
    certificate: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
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
    monkeypatch.delenv("HCP_VAULT_APPROLE_ROLE_ID", raising=False)
    monkeypatch.delenv("HCP_VAULT_APPROLE_SECRET_ID", raising=False)
    monkeypatch.setenv("HCP_VAULT_ADDR", VAULT_ADDR)
    monkeypatch.setenv("HCP_VAULT_CLIENT_CERT", str(cert))
    monkeypatch.setenv("HCP_VAULT_CLIENT_KEY", str(key))
    monkeypatch.setenv("HCP_VAULT_NAMESPACE", "admin")
    monkeypatch.setenv("HCP_VAULT_LOGIN_NAMESPACE", "root")
    manager: Final = HashicorpSecretManager()
    login_route: Final = respx.post(f"{VAULT_ADDR}/v1/auth/cert/login").respond(json=LOGIN_RESPONSE)

    assert manager._auth_via_tls_cert() == "hvs.login-token"
    assert login_route.calls.last.request.headers["X-Vault-Namespace"] == "root"
