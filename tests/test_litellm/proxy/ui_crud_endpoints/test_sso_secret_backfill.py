"""Regression tests for #38177: a partial SSO settings update must not overwrite
a stored client secret with the masked placeholder the UI sends back (or lose
it when the field is omitted), while still allowing an intentional clear, a
genuinely new secret, and never echoing the plaintext secret in the response."""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_keys
from litellm.proxy.config_resolvers.sso import SSO_SECRET_FIELDS
from litellm.proxy.proxy_server import app

client = TestClient(app)

REAL_SECRET = "real_generic_secret_ABCD1234"


@pytest.fixture
def mock_auth():
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    async def _override():
        return {"user_id": "test_user"}

    app.dependency_overrides[user_api_key_auth] = _override
    yield
    app.dependency_overrides.pop(user_api_key_auth, None)


def _masked(secret):
    return mask_sensitive_keys({"generic_client_secret": secret}, set(SSO_SECRET_FIELDS))["generic_client_secret"]


def _mock_prisma(monkeypatch, existing_settings):
    """Wire a prisma client whose SSO row returns existing_settings (or None)."""
    record = None
    if existing_settings is not None:
        record = MagicMock()
        record.sso_settings = existing_settings

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=record)
    mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()
    mock_prisma.db.litellm_config = MagicMock()
    mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_config.update = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", AsyncMock(return_value={}))
    monkeypatch.setattr(proxy_config, "_encrypt_env_variables", lambda environment_variables: environment_variables)
    monkeypatch.setattr(proxy_config, "_decrypt_db_variables", lambda stored: stored)
    return mock_prisma


def _stored_secret(mock_prisma):
    return json.loads(mock_prisma.db.litellm_ssoconfig.upsert.call_args.kwargs["data"]["update"]["sso_settings"])


def test_database_stored_secret_is_preserved(mock_auth, monkeypatch):
    """A masked round-trip must keep a database-stored secret."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.delenv("GENERIC_CLIENT_SECRET", raising=False)
    mock_prisma = _mock_prisma(
        monkeypatch,
        {"generic_client_id": "cid", "generic_client_secret": REAL_SECRET, "proxy_base_url": "https://old.example.com"},
    )

    edited = {
        "generic_client_id": "cid",
        "generic_client_secret": _masked(REAL_SECRET),
        "proxy_base_url": "https://new.example.com",
    }
    resp = client.patch("/update/sso_settings", json=edited)
    assert resp.status_code == 200

    stored = _stored_secret(mock_prisma)
    assert stored["generic_client_secret"] == REAL_SECRET
    assert stored["proxy_base_url"] == "https://new.example.com"
    # The restored plaintext must not leak back to the caller.
    assert resp.json()["settings"]["generic_client_secret"] == _masked(REAL_SECRET)


def test_environment_sourced_secret_is_preserved(mock_auth, monkeypatch):
    """A masked round-trip must keep a secret configured via env, even with no DB row."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.setenv("GENERIC_CLIENT_SECRET", REAL_SECRET)
    mock_prisma = _mock_prisma(monkeypatch, None)  # nothing in the database

    edited = {
        "generic_client_id": "cid",
        "generic_client_secret": _masked(REAL_SECRET),
        "proxy_base_url": "https://new.example.com",
    }
    resp = client.patch("/update/sso_settings", json=edited)
    assert resp.status_code == 200

    stored = _stored_secret(mock_prisma)
    assert stored["generic_client_secret"] == REAL_SECRET


def test_empty_secret_clears_intentionally(mock_auth, monkeypatch):
    """An empty incoming value is an intentional clear, not a masked round-trip."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.delenv("GENERIC_CLIENT_SECRET", raising=False)
    mock_prisma = _mock_prisma(
        monkeypatch,
        {"generic_client_id": "cid", "generic_client_secret": REAL_SECRET},
    )

    edited = {"generic_client_id": "cid", "generic_client_secret": "", "proxy_base_url": "https://x.example.com"}
    resp = client.patch("/update/sso_settings", json=edited)
    assert resp.status_code == 200

    stored = _stored_secret(mock_prisma)
    assert stored["generic_client_secret"] == ""


def test_new_secret_is_saved(mock_auth, monkeypatch):
    """A genuinely new secret (no mask character) replaces the stored one."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.delenv("GENERIC_CLIENT_SECRET", raising=False)
    mock_prisma = _mock_prisma(
        monkeypatch,
        {"generic_client_id": "cid", "generic_client_secret": REAL_SECRET},
    )

    edited = {
        "generic_client_id": "cid",
        "generic_client_secret": "brand_new_secret_9999",
        "proxy_base_url": "https://x",
    }
    resp = client.patch("/update/sso_settings", json=edited)
    assert resp.status_code == 200

    stored = _stored_secret(mock_prisma)
    assert stored["generic_client_secret"] == "brand_new_secret_9999"


def test_omitted_secret_is_preserved(mock_auth, monkeypatch):
    """A partial payload that leaves the secret out entirely must keep it."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.delenv("GENERIC_CLIENT_SECRET", raising=False)
    mock_prisma = _mock_prisma(
        monkeypatch,
        {"generic_client_id": "cid", "generic_client_secret": REAL_SECRET},
    )

    resp = client.patch("/update/sso_settings", json={"generic_client_id": "cid", "proxy_base_url": "https://x"})
    assert resp.status_code == 200

    assert _stored_secret(mock_prisma)["generic_client_secret"] == REAL_SECRET


def test_explicit_null_clears_intentionally(mock_auth, monkeypatch):
    """The dashboard's "Clear SSO Settings" sends null; that must clear, not restore."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.delenv("GENERIC_CLIENT_SECRET", raising=False)
    mock_prisma = _mock_prisma(
        monkeypatch,
        {"generic_client_id": "cid", "generic_client_secret": REAL_SECRET},
    )

    resp = client.patch("/update/sso_settings", json={"generic_client_id": "cid", "generic_client_secret": None})
    assert resp.status_code == 200

    assert _stored_secret(mock_prisma)["generic_client_secret"] is None
    assert "GENERIC_CLIENT_SECRET" not in os.environ


def test_new_secret_containing_asterisk_is_saved(mock_auth, monkeypatch):
    """Only the exact mask of the stored secret counts as "unchanged"."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.delenv("GENERIC_CLIENT_SECRET", raising=False)
    mock_prisma = _mock_prisma(
        monkeypatch,
        {"generic_client_id": "cid", "generic_client_secret": REAL_SECRET},
    )

    resp = client.patch(
        "/update/sso_settings", json={"generic_client_id": "cid", "generic_client_secret": "new*secret*with*stars"}
    )
    assert resp.status_code == 200

    assert _stored_secret(mock_prisma)["generic_client_secret"] == "new*secret*with*stars"
    assert resp.json()["settings"]["generic_client_secret"] != "new*secret*with*stars"
