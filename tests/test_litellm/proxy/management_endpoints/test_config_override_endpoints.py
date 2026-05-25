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
    HASHICORP_ENV_VAR_MAPPING,
    _build_field_schema,
    _set_env_vars,
)
from litellm.proxy.proxy_server import app
from litellm.types.proxy.management_endpoints.config_overrides import (
    HashicorpVaultConfig,
)

VAULT_URL = "/config_overrides/hashicorp_vault"


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
