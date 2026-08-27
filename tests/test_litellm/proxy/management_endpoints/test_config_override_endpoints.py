import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prisma.errors import RecordNotFoundError

import litellm
import litellm.proxy.proxy_server as ps
from litellm.proxy._types import KeyManagementSystem, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.config_override_endpoints import (
    CYBERARK_ENV_VAR_MAPPING,
    HASHICORP_ENV_VAR_MAPPING,
    _build_field_schema,
    _set_env_vars,
)
from litellm.proxy.proxy_server import app
from litellm.types.proxy.management_endpoints.config_overrides import (
    CyberArkConfig,
    HashicorpVaultConfig,
)

VAULT_URL = "/config_overrides/hashicorp_vault"
CYBERARK_URL = "/config_overrides/cyberark"


@pytest.fixture
def client():
    return TestClient(app)


def _make_mock_db():
    mock = MagicMock()
    mock.find_unique = AsyncMock(return_value=None)
    mock.upsert = AsyncMock(return_value=None)
    mock.delete = AsyncMock(return_value=None)
    prisma = MagicMock()
    prisma.db.litellm_configoverrides = mock
    return prisma, mock


def _make_mock_proxy_config():
    cfg = MagicMock()
    cfg.initialize_secret_manager = MagicMock()
    cfg._last_hashicorp_vault_config = None
    cfg._encrypt_env_variables = MagicMock(
        side_effect=lambda d: {k: f"enc_{v}" for k, v in d.items()}
    )
    cfg._decrypt_db_variables = MagicMock(
        side_effect=lambda d: {
            k: v.replace("enc_", "") if isinstance(v, str) else v for k, v in d.items()
        }
    )
    return cfg


def _upserted_data(mock_db):
    return json.loads(mock_db.upsert.call_args.kwargs["data"]["create"]["config_value"])


def _db_record(data):
    rec = MagicMock()
    rec.config_value = json.dumps(data)
    return rec


def _cleanup():
    app.dependency_overrides.pop(ps.user_api_key_auth, None)
    for env_var in HASHICORP_ENV_VAR_MAPPING.values():
        os.environ.pop(env_var, None)
    for env_var in CYBERARK_ENV_VAR_MAPPING.values():
        os.environ.pop(env_var, None)


def _set_admin():
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )


@pytest.mark.asyncio
async def test_hashicorp_vault_crud_lifecycle(client, monkeypatch):
    """Create → read (masked) → partial update (merge from DB) → clear field →
    only-provided fields → delete → idempotent delete → env fallback →
    merge from env → helpers → encrypt/decrypt roundtrip."""
    mock_prisma, mock_db = _make_mock_db()
    mock_cfg = _make_mock_proxy_config()
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_cfg)
    old_client, old_kms = litellm.secret_manager_client, litellm._key_management_system
    _set_admin()

    try:
        # 1. POST: create
        r = client.post(
            VAULT_URL,
            json={
                "vault_addr": "https://vault.example.com",
                "vault_token": "my-secret-vault-token",
                "vault_namespace": "admin",
                "vault_mount_name": "secret",
            },
        )
        assert r.status_code == 200
        assert os.environ["HCP_VAULT_ADDR"] == "https://vault.example.com"
        data = _upserted_data(mock_db)
        assert data["vault_token"] == "enc_my-secret-vault-token"
        mock_cfg.initialize_secret_manager.assert_called_with(
            key_management_system="hashicorp_vault"
        )
        assert mock_cfg._last_hashicorp_vault_config is not None

        # 2. GET: sensitive fields masked
        mock_db.find_unique = AsyncMock(return_value=_db_record(data))
        r = client.get(VAULT_URL)
        assert r.status_code == 200
        vals = r.json()["values"]
        assert vals["vault_addr"] == "https://vault.example.com"
        assert "*" in vals["vault_token"]
        assert "properties" in r.json()["field_schema"]

        # 3. POST partial: omitted fields merge from DB
        r = client.post(VAULT_URL, json={"vault_addr": "https://vault.new.com"})
        assert r.status_code == 200
        data = _upserted_data(mock_db)
        assert data["vault_addr"] == "enc_https://vault.new.com"
        assert data["vault_token"] == "enc_my-secret-vault-token"
        assert data["vault_namespace"] == "enc_admin"

        # 4. POST empty string: clears field, preserves others
        step3 = {
            **data,
            "approle_role_id": "enc_role",
            "approle_secret_id": "enc_secret",
        }
        mock_db.find_unique = AsyncMock(return_value=_db_record(step3))
        mock_db.upsert = AsyncMock(return_value=None)
        r = client.post(VAULT_URL, json={"vault_token": ""})
        assert r.status_code == 200
        data = _upserted_data(mock_db)
        assert "vault_token" not in data
        assert data["approle_role_id"] == "enc_role"

        # 5. POST only provided fields (clean slate)
        for v in HASHICORP_ENV_VAR_MAPPING.values():
            os.environ.pop(v, None)
        mock_db.find_unique = AsyncMock(return_value=None)
        mock_db.upsert = AsyncMock(return_value=None)
        r = client.post(
            VAULT_URL, json={"vault_addr": "https://v.com", "vault_token": "tok"}
        )
        assert r.status_code == 200
        assert _upserted_data(mock_db) == {
            "vault_addr": "enc_https://v.com",
            "vault_token": "enc_tok",
        }

        # 6. DELETE: clears everything
        litellm.secret_manager_client = MagicMock()
        litellm._key_management_system = KeyManagementSystem.HASHICORP_VAULT
        r = client.delete(VAULT_URL)
        assert r.status_code == 200
        assert os.environ.get("HCP_VAULT_ADDR") is None
        assert litellm.secret_manager_client is None

        # 7. DELETE idempotent
        mock_db.delete = AsyncMock(
            side_effect=RecordNotFoundError(
                data={"clientVersion": "0.0.0"}, message="Not found"
            )
        )
        assert client.delete(VAULT_URL).status_code == 200

        # 8. GET: env var fallback
        mock_db.find_unique = AsyncMock(return_value=None)
        monkeypatch.setenv("HCP_VAULT_ADDR", "https://vault.env.com")
        monkeypatch.setenv("HCP_VAULT_NAMESPACE", "env-ns")
        r = client.get(VAULT_URL)
        assert r.json()["values"]["vault_addr"] == "https://vault.env.com"

        # 9. POST: merge from env vars
        monkeypatch.setenv("HCP_VAULT_TOKEN", "env-token")
        monkeypatch.setenv("HCP_VAULT_MOUNT_NAME", "env-mount")
        mock_cfg.initialize_secret_manager = MagicMock()
        mock_db.upsert = AsyncMock(return_value=None)
        r = client.post(VAULT_URL, json={"vault_addr": "https://vault.merged.com"})
        assert r.status_code == 200
        data = _upserted_data(mock_db)
        assert data["vault_token"] == "enc_env-token"
        assert data["vault_mount_name"] == "enc_env-mount"

        # 10. _set_env_vars: empty string unsets
        monkeypatch.setenv("HCP_VAULT_TOKEN", "existing")
        _set_env_vars({"vault_token": "", "vault_addr": "https://v.com"})
        assert os.environ.get("HCP_VAULT_TOKEN") is None
        assert os.environ["HCP_VAULT_ADDR"] == "https://v.com"

        # 11. _build_field_schema
        schema = _build_field_schema(HashicorpVaultConfig)
        assert "vault_addr" in schema["properties"]
        assert len(schema["properties"]["vault_addr"]["description"]) > 0

        # 12. encrypt/decrypt roundtrip
        from litellm.proxy.proxy_server import ProxyConfig

        monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key")
        pc = ProxyConfig()
        orig = {"vault_addr": "https://v.com", "vault_token": "secret"}
        encrypted = pc._encrypt_env_variables(orig)
        assert all(encrypted[k] != orig[k] for k in orig)
        decrypted = pc._decrypt_db_variables(encrypted)
        assert all(decrypted[k] == orig[k] for k in orig)

    finally:
        litellm.secret_manager_client = old_client
        litellm._key_management_system = old_kms
        _cleanup()


@pytest.mark.asyncio
async def test_hashicorp_vault_validation_errors_and_access_control(
    client, monkeypatch
):
    """Validation (missing fields, init failure rollback), DELETE preserves
    non-Vault secret managers, non-admin 403 on all endpoints."""
    mock_prisma, mock_db = _make_mock_db()
    mock_cfg = MagicMock()
    mock_cfg._last_hashicorp_vault_config = {"vault_addr": "old"}
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_cfg)
    old_client, old_kms = litellm.secret_manager_client, litellm._key_management_system
    _set_admin()

    try:
        # 1. Missing vault_addr → 400
        r = client.post(VAULT_URL, json={"vault_token": "tok"})
        assert r.status_code == 400
        assert "Vault Address" in r.json()["detail"]

        # 2. Missing auth → 400
        r = client.post(VAULT_URL, json={"vault_addr": "https://v.com"})
        assert r.status_code == 400
        assert "authentication" in r.json()["detail"].lower()

        # 3. Init failure → 500, env vars restored
        mock_cfg.initialize_secret_manager = MagicMock(side_effect=Exception("fail"))
        monkeypatch.setenv("HCP_VAULT_ADDR", "https://vault.old.com")
        monkeypatch.setenv("HCP_VAULT_TOKEN", "old-token")
        r = client.post(
            VAULT_URL, json={"vault_addr": "https://bad.com", "vault_token": "bad"}
        )
        assert r.status_code == 500
        assert os.environ["HCP_VAULT_ADDR"] == "https://vault.old.com"
        mock_db.upsert.assert_not_awaited()

        # 4. DELETE preserves non-Vault secret manager
        aws = MagicMock()
        litellm.secret_manager_client = aws
        litellm._key_management_system = KeyManagementSystem.AWS_SECRET_MANAGER
        assert client.delete(VAULT_URL).status_code == 200
        assert litellm.secret_manager_client is aws
        assert litellm._key_management_system == KeyManagementSystem.AWS_SECRET_MANAGER

        # 5. Non-admin → 403
        app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER, user_id="user"
        )
        assert client.get(VAULT_URL).status_code == 403
        assert (
            client.post(VAULT_URL, json={"vault_addr": "https://v.com"}).status_code
            == 403
        )
        assert client.delete(VAULT_URL).status_code == 403

    finally:
        litellm.secret_manager_client = old_client
        litellm._key_management_system = old_kms
        _cleanup()


@pytest.mark.asyncio
async def test_cyberark_crud_lifecycle(client, monkeypatch):
    """Create → read (masked) → partial update (merge from DB) → clear field →
    delete → idempotent delete → env fallback → merge from env → schema."""
    mock_prisma, mock_db = _make_mock_db()
    mock_cfg = _make_mock_proxy_config()
    mock_cfg._last_cyberark_config = None
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_cfg)
    old_client, old_kms = litellm.secret_manager_client, litellm._key_management_system
    _set_admin()

    try:
        # 1. POST: create with API-key auth
        r = client.post(
            CYBERARK_URL,
            json={
                "cyberark_api_base": "https://conjur.example.com",
                "cyberark_account": "myorg",
                "cyberark_username": "litellm-user",
                "cyberark_api_key": "my-secret-api-key",
            },
        )
        assert r.status_code == 200
        assert os.environ["CYBERARK_API_BASE"] == "https://conjur.example.com"
        assert os.environ["CYBERARK_API_KEY"] == "my-secret-api-key"
        data = _upserted_data(mock_db)
        assert data["cyberark_api_key"] == "enc_my-secret-api-key"
        mock_cfg.initialize_secret_manager.assert_called_with(
            key_management_system="cyberark"
        )
        assert mock_cfg._last_cyberark_config is not None

        # 2. GET: sensitive fields masked
        mock_db.find_unique = AsyncMock(return_value=_db_record(data))
        r = client.get(CYBERARK_URL)
        assert r.status_code == 200
        vals = r.json()["values"]
        assert vals["cyberark_api_base"] == "https://conjur.example.com"
        assert "*" in vals["cyberark_api_key"]
        assert "properties" in r.json()["field_schema"]

        # 3. POST partial: omitted fields merge from DB
        r = client.post(CYBERARK_URL, json={"cyberark_api_base": "https://conjur.new.com"})
        assert r.status_code == 200
        data = _upserted_data(mock_db)
        assert data["cyberark_api_base"] == "enc_https://conjur.new.com"
        assert data["cyberark_api_key"] == "enc_my-secret-api-key"
        assert data["cyberark_account"] == "enc_myorg"

        # 4. POST empty string: clears field, switches to cert auth
        step3 = {
            **data,
            "client_cert": "enc_/certs/client.pem",
            "client_key": "enc_/certs/client.key",
        }
        mock_db.find_unique = AsyncMock(return_value=_db_record(step3))
        mock_db.upsert = AsyncMock(return_value=None)
        r = client.post(CYBERARK_URL, json={"cyberark_api_key": ""})
        assert r.status_code == 200
        data = _upserted_data(mock_db)
        assert "cyberark_api_key" not in data
        assert data["client_cert"] == "enc_/certs/client.pem"

        # 5. DELETE: clears everything
        litellm.secret_manager_client = MagicMock()  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        litellm._key_management_system = KeyManagementSystem.CYBERARK  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        r = client.delete(CYBERARK_URL)
        assert r.status_code == 200
        assert os.environ.get("CYBERARK_API_BASE") is None
        assert litellm.secret_manager_client is None
        assert mock_cfg._last_cyberark_config is None

        # 6. DELETE idempotent
        mock_db.delete = AsyncMock(
            side_effect=RecordNotFoundError(
                data={"clientVersion": "0.0.0"}, message="Not found"
            )
        )
        assert client.delete(CYBERARK_URL).status_code == 200

        # 7. GET: env var fallback with masking
        mock_db.find_unique = AsyncMock(return_value=None)
        monkeypatch.setenv("CYBERARK_API_BASE", "https://conjur.env.com")
        monkeypatch.setenv("CYBERARK_API_KEY", "env-api-key")
        r = client.get(CYBERARK_URL)
        vals = r.json()["values"]
        assert vals["cyberark_api_base"] == "https://conjur.env.com"
        assert "*" in vals["cyberark_api_key"]

        # 8. POST: merge from env vars
        mock_cfg.initialize_secret_manager = MagicMock()
        mock_db.upsert = AsyncMock(return_value=None)
        r = client.post(CYBERARK_URL, json={"cyberark_api_base": "https://conjur.merged.com"})
        assert r.status_code == 200
        data = _upserted_data(mock_db)
        assert data["cyberark_api_key"] == "enc_env-api-key"

        # 9. _build_field_schema
        schema = _build_field_schema(CyberArkConfig)
        assert "cyberark_api_base" in schema["properties"]
        assert len(schema["properties"]["cyberark_api_base"]["description"]) > 0

    finally:
        litellm.secret_manager_client = old_client  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        litellm._key_management_system = old_kms  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        _cleanup()


@pytest.mark.asyncio
async def test_cyberark_validation_errors_and_access_control(client, monkeypatch):
    """Validation (missing api base, missing auth, init failure rollback),
    DELETE preserves non-CyberArk secret managers, non-admin 403."""
    mock_prisma, mock_db = _make_mock_db()
    mock_cfg = MagicMock()
    mock_cfg._last_cyberark_config = {"cyberark_api_base": "old"}
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_cfg)
    old_client, old_kms = litellm.secret_manager_client, litellm._key_management_system
    _set_admin()

    try:
        # 1. Missing cyberark_api_base → 400
        r = client.post(CYBERARK_URL, json={"cyberark_api_key": "key"})
        assert r.status_code == 400
        assert "API Base" in r.json()["detail"]

        # 2. Missing auth → 400 (cert without key is not valid auth)
        r = client.post(
            CYBERARK_URL,
            json={"cyberark_api_base": "https://c.com", "client_cert": "/c.pem"},
        )
        assert r.status_code == 400
        assert "authentication" in r.json()["detail"].lower()

        # 3. Init failure → 500, env vars restored, nothing persisted
        mock_cfg.initialize_secret_manager = MagicMock(side_effect=Exception("fail"))
        monkeypatch.setenv("CYBERARK_API_BASE", "https://conjur.old.com")
        monkeypatch.setenv("CYBERARK_API_KEY", "old-key")
        r = client.post(
            CYBERARK_URL,
            json={"cyberark_api_base": "https://bad.com", "cyberark_api_key": "bad"},
        )
        assert r.status_code == 500
        assert os.environ["CYBERARK_API_BASE"] == "https://conjur.old.com"
        mock_db.upsert.assert_not_awaited()

        # 4. DELETE preserves non-CyberArk secret manager
        aws = MagicMock()
        litellm.secret_manager_client = aws  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        litellm._key_management_system = KeyManagementSystem.AWS_SECRET_MANAGER  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        assert client.delete(CYBERARK_URL).status_code == 200
        assert litellm.secret_manager_client is aws

        # 5. Non-admin → 403
        app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER, user_id="user"
        )
        assert client.get(CYBERARK_URL).status_code == 403
        assert (
            client.post(
                CYBERARK_URL, json={"cyberark_api_base": "https://c.com"}
            ).status_code
            == 403
        )
        assert client.delete(CYBERARK_URL).status_code == 403
        assert client.post(CYBERARK_URL + "/test_connection").status_code == 403

    finally:
        litellm.secret_manager_client = old_client  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        litellm._key_management_system = old_kms  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        _cleanup()


@pytest.mark.asyncio
async def test_cyberark_audit_log_redacts_values(client, monkeypatch):
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    mock_prisma, mock_db = _make_mock_db()
    mock_cfg = _make_mock_proxy_config()
    mock_cfg._last_cyberark_config = None
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_cfg)
    _set_admin()

    audit_calls = []

    async def capture(request_data):
        audit_calls.append(request_data)

    try:
        with patch(  # test-quality-ok: patching proxy-internal collaborator to isolate the endpoint
            "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
            new=capture,
        ):
            r = client.post(
                CYBERARK_URL,
                json={
                    "cyberark_api_base": "https://conjur.example.com",
                    "cyberark_api_key": "my-very-secret-key",
                },
            )
            assert r.status_code == 200
            for _ in range(3):
                await asyncio.sleep(0)

        assert len(audit_calls) == 1
        log = audit_calls[0]
        assert log.action == "created"
        assert log.object_id == "cyberark"
        assert "my-very-secret-key" not in log.updated_values
        assert "conjur.example.com" not in log.updated_values
        after = json.loads(log.updated_values)
        assert "cyberark_api_key" in after["config"]
        assert "cyberark_api_base" in after["config"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_cyberark_test_connection(client, monkeypatch):
    """400 when not configured; success path authenticates and hits /whoami."""
    from litellm.secret_managers.cyberark_secret_manager import CyberArkSecretManager

    old_client, old_kms = litellm.secret_manager_client, litellm._key_management_system
    _set_admin()

    try:
        # Not configured → 400
        litellm.secret_manager_client = None  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        r = client.post(CYBERARK_URL + "/test_connection")
        assert r.status_code == 400
        assert "not configured" in r.json()["detail"].lower()

        # Configured → authenticates and calls /whoami
        mock_manager = MagicMock(spec=CyberArkSecretManager)
        mock_manager.conjur_addr = "https://conjur.example.com"
        mock_manager.ssl_verify = True
        mock_manager._get_request_headers = MagicMock(
            return_value={"Authorization": "Token abc"}
        )
        litellm.secret_manager_client = mock_manager  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        with patch(  # test-quality-ok: patching proxy-internal collaborator to isolate the endpoint
            "litellm.proxy.management_endpoints.config_override_endpoints.get_async_httpx_client",
            return_value=mock_http,
        ):
            r = client.post(CYBERARK_URL + "/test_connection")
        assert r.status_code == 200
        assert "conjur.example.com" in r.json()["message"]
        called_url = mock_http.get.call_args.args[0]
        assert called_url == "https://conjur.example.com/whoami"

        # Auth failure → 502
        mock_manager._get_request_headers = MagicMock(
            side_effect=Exception("bad credentials")
        )
        r = client.post(CYBERARK_URL + "/test_connection")
        assert r.status_code == 502
        assert "authentication failed" in r.json()["detail"].lower()
    finally:
        litellm.secret_manager_client = old_client  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        litellm._key_management_system = old_kms  # test-quality-ok: endpoint hot-reloads litellm globals; test must set and restore them
        _cleanup()


# ── Audit-log emission for /config_overrides/hashicorp_vault ─────────────────


class TestHashicorpVaultAuditLog:
    """The KMS config endpoint controls every secret retrieval on the proxy.
    A mutation (create/update/delete) must emit an audit-log row when
    ``litellm.store_audit_logs`` is True, with credential values redacted
    so the audit table can't itself be a credential-harvest sink."""

    @pytest.mark.asyncio
    async def test_post_emits_audit_log_with_redacted_values(self, client, monkeypatch):
        monkeypatch.setattr(litellm, "store_audit_logs", True)
        mock_prisma, mock_db = _make_mock_db()
        mock_cfg = _make_mock_proxy_config()
        monkeypatch.setattr(ps, "prisma_client", mock_prisma)
        monkeypatch.setattr(ps, "proxy_config", mock_cfg)
        _set_admin()

        audit_calls = []

        async def capture(request_data):
            audit_calls.append(request_data)

        try:
            with patch(
                "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
                new=capture,
            ):
                r = client.post(
                    VAULT_URL,
                    json={
                        "vault_addr": "https://vault.example.com",
                        "vault_token": "my-very-secret-token",
                    },
                )
                assert r.status_code == 200
                for _ in range(3):
                    await asyncio.sleep(0)

            assert len(audit_calls) == 1
            log = audit_calls[0]
            assert log.action == "created"
            assert log.object_id == "hashicorp_vault"
            # Plaintext credentials must NOT appear anywhere in the row.
            assert "my-very-secret-token" not in log.updated_values
            assert "vault.example.com" not in log.updated_values
            # Field names are kept so the auditor can see what changed.
            after = json.loads(log.updated_values)
            assert "vault_token" in after["config"]
            assert "vault_addr" in after["config"]
        finally:
            _cleanup()

    @pytest.mark.asyncio
    async def test_post_action_is_updated_when_row_exists_with_null_config_value(
        self, client, monkeypatch
    ):
        """A row can exist in ``litellm_configoverrides`` with a NULL
        ``config_value`` (e.g. an earlier failed write left a stub).
        Re-POSTing must label the audit row as ``updated`` — the row
        already exists — not ``created``."""
        monkeypatch.setattr(litellm, "store_audit_logs", True)
        mock_prisma, mock_db = _make_mock_db()
        mock_cfg = _make_mock_proxy_config()
        monkeypatch.setattr(ps, "prisma_client", mock_prisma)
        monkeypatch.setattr(ps, "proxy_config", mock_cfg)
        _set_admin()

        # Row exists but ``config_value`` is NULL.
        null_record = MagicMock()
        null_record.config_value = None
        mock_db.find_unique = AsyncMock(return_value=null_record)

        audit_calls = []

        async def capture(request_data):
            audit_calls.append(request_data)

        try:
            with patch(
                "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
                new=capture,
            ):
                r = client.post(
                    VAULT_URL,
                    json={
                        "vault_addr": "https://vault.example.com",
                        "vault_token": "tok",
                    },
                )
                assert r.status_code == 200
                for _ in range(3):
                    await asyncio.sleep(0)

            assert len(audit_calls) == 1
            assert audit_calls[0].action == "updated"
        finally:
            _cleanup()

    @pytest.mark.asyncio
    async def test_delete_emits_audit_log_only_when_row_existed(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(litellm, "store_audit_logs", True)
        mock_prisma, mock_db = _make_mock_db()
        mock_cfg = _make_mock_proxy_config()
        monkeypatch.setattr(ps, "prisma_client", mock_prisma)
        monkeypatch.setattr(ps, "proxy_config", mock_cfg)
        _set_admin()

        audit_calls = []

        async def capture(request_data):
            audit_calls.append(request_data)

        try:
            with patch(
                "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
                new=capture,
            ):
                # Idempotent delete on an empty table → no row, no audit log.
                mock_db.find_unique = AsyncMock(return_value=None)
                mock_db.delete = AsyncMock(side_effect=RecordNotFoundError(MagicMock()))
                r = client.delete(VAULT_URL)
                assert r.status_code == 200
                for _ in range(3):
                    await asyncio.sleep(0)
                assert audit_calls == []

                # Delete a real row → audit log fires with action="deleted".
                mock_db.find_unique = AsyncMock(
                    return_value=_db_record(
                        {
                            "vault_addr": "enc_https://v.example.com",
                            "vault_token": "enc_t",
                        }
                    )
                )
                mock_db.delete = AsyncMock(return_value=None)
                r = client.delete(VAULT_URL)
                assert r.status_code == 200
                for _ in range(3):
                    await asyncio.sleep(0)

            assert len(audit_calls) == 1
            log = audit_calls[0]
            assert log.action == "deleted"
            # The before-snapshot must redact the token before logging.
            assert "enc_t" not in log.before_value
            assert "v.example.com" not in log.before_value
        finally:
            _cleanup()

    @pytest.mark.asyncio
    async def test_no_audit_when_store_audit_logs_is_off(self, client, monkeypatch):
        monkeypatch.setattr(litellm, "store_audit_logs", False)
        mock_prisma, mock_db = _make_mock_db()
        mock_cfg = _make_mock_proxy_config()
        monkeypatch.setattr(ps, "prisma_client", mock_prisma)
        monkeypatch.setattr(ps, "proxy_config", mock_cfg)
        _set_admin()

        audit_calls = []

        async def capture(request_data):
            audit_calls.append(request_data)

        try:
            with patch(
                "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
                new=capture,
            ):
                r = client.post(
                    VAULT_URL,
                    json={
                        "vault_addr": "https://vault.example.com",
                        "vault_token": "tok",
                    },
                )
                assert r.status_code == 200

            assert audit_calls == []
        finally:
            _cleanup()
