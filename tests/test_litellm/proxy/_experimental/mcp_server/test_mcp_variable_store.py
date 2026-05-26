"""Tests for the per-user MCP variable storage backend (DB vs credential store)."""

import json
from unittest.mock import AsyncMock

import pytest

from litellm.proxy._experimental.mcp_server import db as _db
from litellm.proxy._experimental.mcp_server import mcp_variable_store as store
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.secret_managers.base_secret_manager import BaseSecretManager

_SALT_KEY = "test-salt-key-for-variable-store-tests-1234"


@pytest.fixture
def salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", _SALT_KEY)


@pytest.fixture(autouse=True)
def _clean_store(monkeypatch):
    """Each test starts with a clean cache and no store env configuration."""
    monkeypatch.delenv("LITELLM_MCP_VARIABLE_STORE", raising=False)
    monkeypatch.delenv("LITELLM_MCP_VARIABLE_STORE_PREFIX", raising=False)
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "general_settings", {}, raising=False)
    store.reset_cache()
    yield
    store.reset_cache()


class _FakeSecretManager(BaseSecretManager):
    def __init__(self):
        self.written = {}

    async def async_read_secret(self, secret_name, optional_params=None, timeout=None):
        return self.written.get(secret_name)

    def sync_read_secret(self, secret_name, optional_params=None, timeout=None):
        return self.written.get(secret_name)

    async def async_write_secret(
        self,
        secret_name,
        secret_value,
        description=None,
        optional_params=None,
        timeout=None,
        tags=None,
    ):
        self.written[secret_name] = secret_value
        return {"ok": True}

    async def async_delete_secret(
        self, secret_name, recovery_window_in_days=7, optional_params=None, timeout=None
    ):
        self.written.pop(secret_name, None)
        return {"ok": True}


# ── provider resolution ─────────────────────────────────────────────────────


def test_resolve_provider_none_when_unset():
    assert store._resolve_provider() is None


@pytest.mark.parametrize("value", ["database", "db", "  DB  ", ""])
def test_resolve_provider_database_means_db(monkeypatch, value):
    monkeypatch.setenv("LITELLM_MCP_VARIABLE_STORE", value)
    assert store._resolve_provider() is None


def test_resolve_provider_normalises_case(monkeypatch):
    monkeypatch.setenv("LITELLM_MCP_VARIABLE_STORE", "HashiCorp_Vault")
    assert store._resolve_provider() == "hashicorp_vault"


def test_resolve_provider_prefers_general_settings(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(
        ps, "general_settings", {"mcp_variable_store": "vault"}, raising=False
    )
    assert store._resolve_provider() == "vault"


# ── secret key building ──────────────────────────────────────────────────────


def test_secret_key_default_prefix():
    assert store._secret_key("u1") == "litellm/mcp/user/u1"


def test_secret_key_prefix_override(monkeypatch):
    monkeypatch.setenv("LITELLM_MCP_VARIABLE_STORE_PREFIX", "secret/data/mcp/")
    assert store._secret_key("u1") == "secret/data/mcp/u1"


# ── manager building ─────────────────────────────────────────────────────────


def test_build_manager_unknown_provider_returns_none():
    assert store._build_manager("not-a-real-provider") is None


def test_build_manager_kms_reuses_global_client(monkeypatch):
    import litellm

    fake = _FakeSecretManager()
    monkeypatch.setattr(litellm, "secret_manager_client", fake, raising=False)
    assert store._build_manager("key_management_system") is fake


def test_build_manager_kms_without_client_returns_none(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "secret_manager_client", None, raising=False)
    assert store._build_manager("kms") is None


# ── DB fallback (no store configured) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_falls_back_to_db(monkeypatch):
    monkeypatch.setattr(store, "_get_manager", lambda: None)
    captured = {}

    async def fake_get(prisma, user_id):
        captured["args"] = (prisma, user_id)
        return {"X": "1"}

    monkeypatch.setattr(_db, "get_user_variables", fake_get)
    assert await store.get_user_variables("PRISMA", "alice") == {"X": "1"}
    assert captured["args"] == ("PRISMA", "alice")


@pytest.mark.asyncio
async def test_store_falls_back_to_db(monkeypatch):
    monkeypatch.setattr(store, "_get_manager", lambda: None)
    captured = {}

    async def fake_store(prisma, user_id, values):
        captured["args"] = (prisma, user_id, values)

    monkeypatch.setattr(_db, "store_user_variables", fake_store)
    await store.store_user_variables("PRISMA", "alice", {"A": "1"})
    assert captured["args"] == ("PRISMA", "alice", {"A": "1"})


@pytest.mark.asyncio
async def test_delete_falls_back_to_db(monkeypatch):
    monkeypatch.setattr(store, "_get_manager", lambda: None)
    captured = {}

    async def fake_delete(prisma, user_id):
        captured["args"] = (prisma, user_id)

    monkeypatch.setattr(_db, "delete_user_variables", fake_delete)
    await store.delete_user_variables("PRISMA", "alice")
    assert captured["args"] == ("PRISMA", "alice")


# ── credential store path (store exclusive) ──────────────────────────────────


@pytest.mark.asyncio
async def test_store_writes_encrypted_blob_to_manager(salt_key, monkeypatch):
    mgr = _FakeSecretManager()
    monkeypatch.setattr(store, "_get_manager", lambda: mgr)
    payload = {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}

    await store.store_user_variables(None, "alice", payload)

    key = store._secret_key("alice")
    assert key in mgr.written
    blob = mgr.written[key]
    # The stored blob is encrypted (no plaintext) but round-trips back.
    assert "s3cret" not in blob
    assert _db._decode_user_variables(blob) == payload


@pytest.mark.asyncio
async def test_get_reads_and_decodes_from_manager(salt_key, monkeypatch):
    payload = {"A": "1", "B": "2"}
    mgr = _FakeSecretManager()
    mgr.written[store._secret_key("alice")] = encrypt_value_helper(json.dumps(payload))
    monkeypatch.setattr(store, "_get_manager", lambda: mgr)

    assert await store.get_user_variables(None, "alice") == payload


@pytest.mark.asyncio
async def test_get_returns_empty_when_manager_has_no_secret(monkeypatch):
    mgr = _FakeSecretManager()  # nothing stored
    monkeypatch.setattr(store, "_get_manager", lambda: mgr)
    assert await store.get_user_variables(None, "alice") == {}


@pytest.mark.asyncio
async def test_delete_removes_secret_from_manager(monkeypatch):
    mgr = _FakeSecretManager()
    key = store._secret_key("alice")
    mgr.written[key] = "blob"
    monkeypatch.setattr(store, "_get_manager", lambda: mgr)

    await store.delete_user_variables(None, "alice")
    assert key not in mgr.written


@pytest.mark.asyncio
async def test_store_then_get_round_trip_through_manager(salt_key, monkeypatch):
    mgr = _FakeSecretManager()
    monkeypatch.setattr(store, "_get_manager", lambda: mgr)
    payload = {"TOKEN": "abc123"}

    await store.store_user_variables(None, "bob", payload)
    assert await store.get_user_variables(None, "bob") == payload
