import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.config_override_endpoints import (
    HASHICORP_SENSITIVE_FIELDS,
    _build_field_schema,
    _decrypt_sensitive_fields,
    _encrypt_sensitive_fields,
)
from litellm.proxy.proxy_server import app
from litellm.types.proxy.management_endpoints.config_overrides import (
    HashicorpVaultConfig,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_encrypt_decrypt_sensitive_fields_roundtrip():
    """Sensitive fields should be encrypted, non-sensitive fields left as-is."""
    data = {
        "vault_addr": "https://vault.example.com:8200",
        "vault_token": "my-secret-token",
        "approle_role_id": "role-123",
        "approle_secret_id": "secret-456",
        "vault_namespace": "admin",
        "vault_mount_name": "secret",
        "client_key": None,
    }

    with patch(
        "litellm.proxy.management_endpoints.config_override_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.side_effect = lambda v: f"enc_{v}"

        encrypted = _encrypt_sensitive_fields(data, HASHICORP_SENSITIVE_FIELDS)

        # Non-sensitive fields unchanged
        assert encrypted["vault_addr"] == "https://vault.example.com:8200"
        assert encrypted["vault_namespace"] == "admin"
        assert encrypted["vault_mount_name"] == "secret"

        # Sensitive fields encrypted
        assert encrypted["vault_token"] == "enc_my-secret-token"
        assert encrypted["approle_role_id"] == "enc_role-123"
        assert encrypted["approle_secret_id"] == "enc_secret-456"

        # None values should not be encrypted
        assert encrypted["client_key"] is None

    with patch(
        "litellm.proxy.management_endpoints.config_override_endpoints.decrypt_value_helper"
    ) as mock_decrypt:
        mock_decrypt.side_effect = lambda v, **kwargs: v.replace("enc_", "")

        decrypted = _decrypt_sensitive_fields(encrypted, HASHICORP_SENSITIVE_FIELDS)

        # Round-trip: values should match original
        assert decrypted["vault_addr"] == "https://vault.example.com:8200"
        assert decrypted["vault_token"] == "my-secret-token"
        assert decrypted["approle_role_id"] == "role-123"
        assert decrypted["approle_secret_id"] == "secret-456"
        assert decrypted["vault_namespace"] == "admin"
        assert decrypted["client_key"] is None


def test_build_field_schema():
    """Field schema should include description and type for all HashicorpVaultConfig fields."""
    schema = _build_field_schema(HashicorpVaultConfig)

    assert "properties" in schema
    assert "vault_addr" in schema["properties"]
    assert "vault_token" in schema["properties"]
    assert "approle_role_id" in schema["properties"]

    # Check that descriptions are populated
    assert len(schema["properties"]["vault_addr"]["description"]) > 0
    assert len(schema["properties"]["vault_token"]["description"]) > 0


@pytest.mark.asyncio
async def test_get_hashicorp_config_fallback_to_env(client, monkeypatch):
    """When no DB record exists, GET should return env var values."""
    mock_configoverrides = MagicMock()
    mock_configoverrides.find_unique = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setenv("HCP_VAULT_ADDR", "https://vault.env.com")
    monkeypatch.setenv("HCP_VAULT_NAMESPACE", "env-ns")

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.get("/config_overrides/hashicorp_vault")
        assert response.status_code == 200
        data = response.json()
        assert data["config_type"] == "hashicorp_vault"
        assert data["values"]["vault_addr"] == "https://vault.env.com"
        assert data["values"]["vault_namespace"] == "env-ns"
        assert "field_schema" in data
        assert "properties" in data["field_schema"]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_get_hashicorp_config_from_db(client, monkeypatch):
    """When a DB record exists, GET should return masked sensitive values."""
    mock_record = MagicMock()
    mock_record.config_value = {
        "vault_addr": "https://vault.db.com",
        "vault_token": "encrypted_token",
        "vault_namespace": "db-ns",
    }

    mock_configoverrides = MagicMock()
    mock_configoverrides.find_unique = AsyncMock(return_value=mock_record)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    with patch(
        "litellm.proxy.management_endpoints.config_override_endpoints.decrypt_value_helper"
    ) as mock_decrypt:
        mock_decrypt.return_value = "decrypted_token_value"

        app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
        )

        try:
            response = client.get("/config_overrides/hashicorp_vault")
            assert response.status_code == 200
            data = response.json()
            assert data["config_type"] == "hashicorp_vault"
            # Non-sensitive fields returned as-is
            assert data["values"]["vault_addr"] == "https://vault.db.com"
            assert data["values"]["vault_namespace"] == "db-ns"
            # Sensitive fields should be masked, not plaintext
            vault_token_value = data["values"]["vault_token"]
            assert "*" in vault_token_value
            assert vault_token_value != "decrypted_token_value"
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_update_hashicorp_config_success(client, monkeypatch):
    """POST should set env vars, encrypt sensitive fields, upsert DB, and reinit secret manager."""
    mock_configoverrides = MagicMock()
    mock_configoverrides.find_unique = AsyncMock(return_value=None)
    mock_configoverrides.upsert = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides

    mock_proxy_config = MagicMock()
    mock_proxy_config.initialize_secret_manager = MagicMock()
    mock_proxy_config._last_hashicorp_vault_config = None

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_proxy_config)

    with patch(
        "litellm.proxy.management_endpoints.config_override_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.side_effect = lambda v: f"enc_{v}"

        app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
        )

        try:
            response = client.post(
                "/config_overrides/hashicorp_vault",
                json={
                    "vault_addr": "https://vault.new.com",
                    "vault_token": "new-token",
                    "vault_namespace": "new-ns",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

            # Verify env vars were set
            assert os.environ.get("HCP_VAULT_ADDR") == "https://vault.new.com"
            assert os.environ.get("HCP_VAULT_TOKEN") == "new-token"
            assert os.environ.get("HCP_VAULT_NAMESPACE") == "new-ns"

            # Verify DB upsert was called
            mock_configoverrides.upsert.assert_awaited_once()
            upsert_call = mock_configoverrides.upsert.call_args
            create_data = json.loads(
                upsert_call.kwargs["data"]["create"]["config_value"]
            )
            assert create_data["vault_token"] == "enc_new-token"
            assert (
                create_data["vault_addr"] == "https://vault.new.com"
            )  # not encrypted

            # Verify secret manager was reinitialized
            mock_proxy_config.initialize_secret_manager.assert_called_once_with(
                key_management_system="hashicorp_vault"
            )

            # Verify change-detection cache was updated
            assert mock_proxy_config._last_hashicorp_vault_config is not None
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)
            # Clean up env vars
            os.environ.pop("HCP_VAULT_ADDR", None)
            os.environ.pop("HCP_VAULT_TOKEN", None)
            os.environ.pop("HCP_VAULT_NAMESPACE", None)


@pytest.mark.asyncio
async def test_update_hashicorp_config_excludes_none_fields(client, monkeypatch):
    """POST with partial fields should only store provided fields (None fields excluded)."""
    mock_configoverrides = MagicMock()
    mock_configoverrides.find_unique = AsyncMock(return_value=None)
    mock_configoverrides.upsert = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides

    mock_proxy_config = MagicMock()
    mock_proxy_config.initialize_secret_manager = MagicMock()
    mock_proxy_config._last_hashicorp_vault_config = None

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_proxy_config)

    with patch(
        "litellm.proxy.management_endpoints.config_override_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.side_effect = lambda v: f"enc_{v}"

        app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
        )

        try:
            response = client.post(
                "/config_overrides/hashicorp_vault",
                json={
                    "vault_addr": "https://vault.partial.com",
                    "vault_token": "tok",
                },
            )
            assert response.status_code == 200

            # Only vault_addr and vault_token should be in the upserted data
            upsert_call = mock_configoverrides.upsert.call_args
            create_data = json.loads(
                upsert_call.kwargs["data"]["create"]["config_value"]
            )
            assert create_data == {
                "vault_addr": "https://vault.partial.com",
                "vault_token": "enc_tok",
            }
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)
            os.environ.pop("HCP_VAULT_ADDR", None)
            os.environ.pop("HCP_VAULT_TOKEN", None)


@pytest.mark.asyncio
async def test_update_hashicorp_config_preserves_existing_sensitive_fields(
    client, monkeypatch
):
    """POST without sensitive fields should merge them from the existing DB record."""
    existing_record = MagicMock()
    existing_record.config_value = json.dumps(
        {
            "vault_addr": "https://vault.old.com",
            "vault_token": "enc_old-token",
            "approle_role_id": "enc_old-role",
        }
    )

    mock_configoverrides = MagicMock()
    mock_configoverrides.find_unique = AsyncMock(return_value=existing_record)
    mock_configoverrides.upsert = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides

    mock_proxy_config = MagicMock()
    mock_proxy_config.initialize_secret_manager = MagicMock()
    mock_proxy_config._last_hashicorp_vault_config = None

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_proxy_config)

    with patch(
        "litellm.proxy.management_endpoints.config_override_endpoints.encrypt_value_helper"
    ) as mock_encrypt, patch(
        "litellm.proxy.management_endpoints.config_override_endpoints.decrypt_value_helper"
    ) as mock_decrypt:
        mock_encrypt.side_effect = lambda v: f"enc_{v}"
        mock_decrypt.side_effect = lambda v, **kwargs: v.replace("enc_", "")

        app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
        )

        try:
            # Only send vault_addr — no vault_token or approle_role_id
            response = client.post(
                "/config_overrides/hashicorp_vault",
                json={"vault_addr": "https://vault.new.com"},
            )
            assert response.status_code == 200

            # Verify the upserted data preserved existing sensitive fields
            upsert_call = mock_configoverrides.upsert.call_args
            create_data = json.loads(
                upsert_call.kwargs["data"]["create"]["config_value"]
            )
            assert create_data["vault_addr"] == "https://vault.new.com"
            assert create_data["vault_token"] == "enc_old-token"
            assert create_data["approle_role_id"] == "enc_old-role"
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)
            os.environ.pop("HCP_VAULT_ADDR", None)
            os.environ.pop("HCP_VAULT_TOKEN", None)
            os.environ.pop("HCP_VAULT_APPROLE_ROLE_ID", None)


@pytest.mark.asyncio
async def test_update_hashicorp_config_init_failure_restores_env_vars(
    client, monkeypatch
):
    """When initialize_secret_manager fails, env vars should be restored to previous values and DB should not be updated."""
    mock_configoverrides = MagicMock()
    mock_configoverrides.find_unique = AsyncMock(return_value=None)
    mock_configoverrides.upsert = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides

    mock_proxy_config = MagicMock()
    mock_proxy_config.initialize_secret_manager = MagicMock(
        side_effect=Exception("Vault connection refused")
    )

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_proxy_config)

    # Set pre-existing env vars that should be restored on failure
    monkeypatch.setenv("HCP_VAULT_ADDR", "https://vault.old.com")
    monkeypatch.setenv("HCP_VAULT_TOKEN", "old-token")

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.post(
            "/config_overrides/hashicorp_vault",
            json={
                "vault_addr": "https://vault.bad.com",
                "vault_token": "bad-token",
            },
        )
        assert response.status_code == 500
        assert "Vault connection refused" in response.json()["detail"]["error"]

        # Env vars should be restored to previous values, not wiped
        assert os.environ.get("HCP_VAULT_ADDR") == "https://vault.old.com"
        assert os.environ.get("HCP_VAULT_TOKEN") == "old-token"

        # DB should NOT have been updated
        mock_configoverrides.upsert.assert_not_awaited()
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_update_hashicorp_config_missing_vault_addr(client, monkeypatch):
    """POST without vault_addr should return 400."""
    mock_configoverrides = MagicMock()
    mock_configoverrides.find_unique = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.post(
            "/config_overrides/hashicorp_vault",
            json={"vault_token": "some-token"},
        )
        assert response.status_code == 400
        assert "Vault Address" in response.json()["detail"]["error"]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_update_hashicorp_config_missing_auth(client, monkeypatch):
    """POST with vault_addr but no auth method should return 400."""
    mock_configoverrides = MagicMock()
    mock_configoverrides.find_unique = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.post(
            "/config_overrides/hashicorp_vault",
            json={"vault_addr": "https://vault.example.com"},
        )
        assert response.status_code == 400
        assert "authentication" in response.json()["detail"]["error"].lower()
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_admin_only_access_get(client, monkeypatch):
    """Non-admin users should get 403 on GET."""
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="normal_user"
    )

    try:
        response = client.get("/config_overrides/hashicorp_vault")
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_admin_only_access_post(client, monkeypatch):
    """Non-admin users should get 403 on POST."""
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="normal_user"
    )

    try:
        response = client.post(
            "/config_overrides/hashicorp_vault",
            json={"vault_addr": "https://vault.example.com"},
        )
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
