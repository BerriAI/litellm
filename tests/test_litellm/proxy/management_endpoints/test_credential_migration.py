"""
Tests for the at-rest credential re-encryption migration engine.

The pure engine (classify / reencrypt / selective-dict) is tested directly; the
DB walkers are tested against an AsyncMock Prisma client. Live end-to-end
proof-of-fix (real proxy + DB) is performed separately on the repro server.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import proxy_server
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    _V2_GCM_PREFIX,
    encrypt_value_helper,
)
from litellm.proxy.management_endpoints import credential_migration as cm


@pytest.fixture
def salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-migration-salt-1234")
    monkeypatch.setattr(proxy_server, "general_settings", {})
    return "sk-migration-salt-1234"


def _legacy_ct(value: str, monkeypatch) -> str:
    """Produce a legacy (nacl) ciphertext with the AES gate off."""
    monkeypatch.setattr(proxy_server, "general_settings", {})
    return encrypt_value_helper(value)


def _enable_aes(monkeypatch):
    monkeypatch.setattr(
        proxy_server, "general_settings", {"encryption_algorithm": "aes-256-gcm"}
    )


# --------------------------- pure engine ---------------------------


def test_classify_value(salt_key, monkeypatch):
    legacy = _legacy_ct("secret", monkeypatch)
    _enable_aes(monkeypatch)
    migrated = encrypt_value_helper("secret")

    assert cm.classify_value(legacy) == "legacy"
    assert cm.classify_value(migrated) == "migrated"
    assert cm.classify_value("just-plaintext") == "plaintext"
    assert cm.classify_value("") == "plaintext"
    assert cm.classify_value(123) == "not-a-string"
    assert cm.classify_value(None) == "not-a-string"


def test_is_migrated(salt_key, monkeypatch):
    _enable_aes(monkeypatch)
    assert cm.is_migrated(encrypt_value_helper("x")) is True
    assert cm.is_migrated("plaintext") is False
    assert cm.is_migrated(5) is False


def test_reencrypt_value_legacy_to_v2(salt_key, monkeypatch):
    legacy = _legacy_ct("secret", monkeypatch)
    _enable_aes(monkeypatch)

    out = cm.reencrypt_value(legacy)
    assert out != legacy
    assert out.startswith(_V2_GCM_PREFIX)


def test_reencrypt_value_is_idempotent(salt_key, monkeypatch):
    _enable_aes(monkeypatch)
    v2 = encrypt_value_helper("secret")
    # Already v2 -> returned byte-for-byte unchanged (no re-wrap).
    assert cm.reencrypt_value(v2) == v2


def test_reencrypt_value_preserves_non_string_and_empty(salt_key, monkeypatch):
    _enable_aes(monkeypatch)
    assert cm.reencrypt_value(42) == 42
    assert cm.reencrypt_value("") == ""
    assert cm.reencrypt_value(None) is None


def test_reencrypt_value_skips_undecryptable(salt_key, monkeypatch):
    """A value that does not decrypt (legacy plaintext or corrupt) is preserved."""
    _enable_aes(monkeypatch)
    plaintext = "not-actually-encrypted"
    assert cm.reencrypt_value(plaintext) == plaintext


def test_reencrypt_selective_dict(salt_key, monkeypatch):
    legacy_key = _legacy_ct("the-api-key", monkeypatch)
    _enable_aes(monkeypatch)

    data = {"api_key": legacy_key, "base_url": "https://x", "integration_token": None}
    out = cm.reencrypt_selective_dict(data, ["api_key", "integration_token"])

    assert out["api_key"].startswith(_V2_GCM_PREFIX)
    assert out["base_url"] == "https://x"  # untouched non-sensitive
    assert out["integration_token"] is None  # null skipped


# --------------------------- gate enforcement ---------------------------


@pytest.mark.asyncio
async def test_migrate_requires_aes_gate(salt_key, monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {})  # gate off
    with pytest.raises(RuntimeError, match="encryption_algorithm"):
        await cm.migrate_encryption(
            prisma_client=MagicMock(), user_api_key_dict=MagicMock()
        )


# --------------------------- config-row walker ---------------------------


def _config_prisma(record):
    """Build an AsyncMock prisma client whose litellm_config returns `record`."""
    client = MagicMock()
    client.db.litellm_config.find_unique = AsyncMock(return_value=record)
    client.db.litellm_config.update = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_vantage_walker_migrates_legacy_field(salt_key, monkeypatch):
    legacy_api_key = _legacy_ct("vantage-secret", monkeypatch)
    _enable_aes(monkeypatch)
    record = SimpleNamespace(
        param_value={
            "api_key": legacy_api_key,
            "integration_token": None,
            "base_url": "https://api.vantage.sh",
        }
    )
    client = _config_prisma(record)

    report = await cm._migrate_config_settings_row(
        client, "vantage_settings", cm._VANTAGE_SENSITIVE, dry_run=False
    )

    assert report.migrated == 1
    assert report.legacy == 1
    client.db.litellm_config.update.assert_awaited_once()
    written = json.loads(
        client.db.litellm_config.update.call_args.kwargs["data"]["param_value"]
    )
    assert written["api_key"].startswith(_V2_GCM_PREFIX)
    assert written["base_url"] == "https://api.vantage.sh"  # non-sensitive untouched


@pytest.mark.asyncio
async def test_vantage_walker_idempotent_no_write(salt_key, monkeypatch):
    _enable_aes(monkeypatch)
    record = SimpleNamespace(
        param_value={"api_key": encrypt_value_helper("already-v2"), "base_url": "x"}
    )
    client = _config_prisma(record)

    report = await cm._migrate_config_settings_row(
        client, "vantage_settings", cm._VANTAGE_SENSITIVE, dry_run=False
    )

    assert report.already_v2 == 1
    assert report.migrated == 0
    client.db.litellm_config.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_config_walker_dry_run_does_not_write(salt_key, monkeypatch):
    legacy_api_key = _legacy_ct("vantage-secret", monkeypatch)
    _enable_aes(monkeypatch)
    record = SimpleNamespace(param_value={"api_key": legacy_api_key})
    client = _config_prisma(record)

    report = await cm._migrate_config_settings_row(
        client, "vantage_settings", cm._VANTAGE_SENSITIVE, dry_run=True
    )

    assert report.legacy == 1
    assert report.migrated == 1  # counted as would-migrate
    client.db.litellm_config.update.assert_not_awaited()  # but no write in dry-run


@pytest.mark.asyncio
async def test_config_walker_handles_missing_row(salt_key, monkeypatch):
    _enable_aes(monkeypatch)
    client = _config_prisma(None)
    report = await cm._migrate_config_settings_row(
        client, "cloudzero_settings", cm._CLOUDZERO_SENSITIVE, dry_run=False
    )
    assert report.scanned == 0
    client.db.litellm_config.update.assert_not_awaited()


# --------------------------- --check scanner ---------------------------


@pytest.mark.asyncio
async def test_check_reports_residual_legacy(salt_key, monkeypatch):
    legacy_api_key = _legacy_ct("vantage-secret", monkeypatch)
    _enable_aes(monkeypatch)

    client = MagicMock()
    # Net-new walker tables: empty team / token / sso, one legacy vantage field.
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)
    client.db.litellm_config.update = AsyncMock()

    def _find_unique(where):
        if where.get("param_name") == "vantage_settings":
            return SimpleNamespace(param_value={"api_key": legacy_api_key})
        return None

    client.db.litellm_config.find_unique = AsyncMock(side_effect=_find_unique)

    report = await cm.check_encryption(client)

    assert report.residual_legacy == 1
    client.db.litellm_config.update.assert_not_awaited()  # read-only


@pytest.mark.asyncio
async def test_check_reports_zero_after_migration(salt_key, monkeypatch):
    _enable_aes(monkeypatch)
    client = MagicMock()
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)
    client.db.litellm_config.update = AsyncMock()

    def _find_unique(where):
        if where.get("param_name") == "vantage_settings":
            return SimpleNamespace(
                param_value={"api_key": encrypt_value_helper("already-v2")}
            )
        return None

    client.db.litellm_config.find_unique = AsyncMock(side_effect=_find_unique)

    report = await cm.check_encryption(client)
    assert report.residual_legacy == 0


# --------------------------- callback_vars walker ---------------------------


@pytest.mark.asyncio
async def test_callback_vars_walker_migrates_team_metadata(salt_key, monkeypatch):
    """A team row with a legacy-encrypted callback var is rewritten to v2."""
    from litellm.proxy.common_utils.callback_utils import encrypt_callback_vars

    # Legacy-encrypt a callback var via the real callback path (gate off).
    monkeypatch.setattr(proxy_server, "general_settings", {})
    legacy_meta = encrypt_callback_vars(
        {"logging": [{"callback_vars": {"gcs_path_service_account": "sa-secret"}}]}
    )
    _enable_aes(monkeypatch)

    team_row = SimpleNamespace(team_id="team-1", metadata=legacy_meta)
    client = MagicMock()
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_row])
    client.db.litellm_teamtable.update = AsyncMock()

    report = await cm._migrate_callback_vars_table(client, "team", dry_run=False)

    assert report.migrated == 1
    client.db.litellm_teamtable.update.assert_awaited_once()
    written = json.loads(
        client.db.litellm_teamtable.update.call_args.kwargs["data"]["metadata"]
    )
    inner = written["logging"][0]["callback_vars"]["gcs_path_service_account"]
    assert "v2:gcm:" in inner
