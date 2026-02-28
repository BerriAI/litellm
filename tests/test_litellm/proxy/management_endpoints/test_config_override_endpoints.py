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
    _get_current_env_values,
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


def test_encrypt_sensitive_fields_skips_none_values():
    """None values should not be encrypted."""
    data = {
        "vault_addr": "https://vault.example.com",
        "vault_token": None,
        "approle_role_id": None,
    }

    with patch(
        "litellm.proxy.management_endpoints.config_override_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        encrypted = _encrypt_sensitive_fields(data, HASHICORP_SENSITIVE_FIELDS)

        mock_encrypt.assert_not_called()
        assert encrypted["vault_token"] is None
        assert encrypted["approle_role_id"] is None


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


def test_get_current_env_values(monkeypatch):
    """Should return current env var values using the mapping."""
    from litellm.proxy.management_endpoints.config_override_endpoints import (
        HASHICORP_ENV_VAR_MAPPING,
    )

    monkeypatch.setenv("HCP_VAULT_ADDR", "https://vault.test.com")
    monkeypatch.setenv("HCP_VAULT_NAMESPACE", "test-ns")
    # Don't set HCP_VAULT_TOKEN — should be None

    values = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)

    assert values["vault_addr"] == "https://vault.test.com"
    assert values["vault_namespace"] == "test-ns"
    assert values["vault_token"] is None


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
    """When a DB record exists, GET should return decrypted values."""
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
        mock_decrypt.return_value = "decrypted_token"

        app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
        )

        try:
            response = client.get("/config_overrides/hashicorp_vault")
            assert response.status_code == 200
            data = response.json()
            assert data["config_type"] == "hashicorp_vault"
            assert data["values"]["vault_addr"] == "https://vault.db.com"
            assert data["values"]["vault_token"] == "decrypted_token"
            assert data["values"]["vault_namespace"] == "db-ns"
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_update_hashicorp_config_success(client, monkeypatch):
    """POST should set env vars, encrypt sensitive fields, upsert DB, and reinit secret manager."""
    mock_configoverrides = MagicMock()
    mock_configoverrides.upsert = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides

    mock_proxy_config = MagicMock()
    mock_proxy_config.initialize_secret_manager = MagicMock()

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
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)
            # Clean up env vars
            os.environ.pop("HCP_VAULT_ADDR", None)
            os.environ.pop("HCP_VAULT_TOKEN", None)
            os.environ.pop("HCP_VAULT_NAMESPACE", None)


@pytest.mark.asyncio
async def test_update_hashicorp_config_excludes_none_fields(client, monkeypatch):
    """POST with partial fields should only set provided fields."""
    mock_configoverrides = MagicMock()
    mock_configoverrides.upsert = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_configoverrides = mock_configoverrides

    mock_proxy_config = MagicMock()
    mock_proxy_config.initialize_secret_manager = MagicMock()

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    monkeypatch.setattr(ps, "proxy_config", mock_proxy_config)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.post(
            "/config_overrides/hashicorp_vault",
            json={"vault_addr": "https://vault.partial.com"},
        )
        assert response.status_code == 200

        # Only vault_addr should be in the upserted data
        upsert_call = mock_configoverrides.upsert.call_args
        create_data = json.loads(
            upsert_call.kwargs["data"]["create"]["config_value"]
        )
        assert create_data == {"vault_addr": "https://vault.partial.com"}
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        os.environ.pop("HCP_VAULT_ADDR", None)


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
