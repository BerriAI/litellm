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
    decrypt_value_helper,
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


def _empty_covered_tables(client):
    """Wire every table the scanner walks on `client` to return no rows.

    Lets a `check_encryption` / scanner test isolate the location under test
    without the other tables raising on an unconfigured mock.
    """
    for _, db_attr, _, _ in cm._COVERED_TABLE_SPECS:
        getattr(client.db, db_attr).find_many = AsyncMock(return_value=[])
    for _, db_attr, _, _ in cm._SETTINGS_ROW_SPECS:
        getattr(client.db, db_attr).find_many = AsyncMock(return_value=[])


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


async def _migrate_sso(client, dry_run, policy=cm.ALGORITHM_POLICY):
    """Run the settings-row walker over the SSO config table."""
    return await cm._migrate_settings_rows(
        client, "sso_config", "litellm_ssoconfig", "id", "sso_settings", dry_run, policy=policy
    )


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
    assert report.legacy == 0  # migrated -> no longer residual legacy
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

    # A dry run reports residual legacy only; nothing is migrated (no write), so
    # `migrated` and `residual_legacy` are never contradictory in --check output.
    assert report.legacy == 1
    assert report.migrated == 0
    client.db.litellm_config.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_config_walker_handles_missing_row(salt_key, monkeypatch):
    _enable_aes(monkeypatch)
    client = _config_prisma(None)
    report = await cm._migrate_config_settings_row(
        client, "cloudzero_settings", cm._CLOUDZERO_SENSITIVE, dry_run=False
    )
    assert report.scanned == 0
    client.db.litellm_config.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_sso_walker_real_run_migrates_and_clears_residual(salt_key, monkeypatch):
    """SSO real run: a migrated field is counted as migrated, not residual legacy."""
    legacy = _legacy_ct("client-secret", monkeypatch)
    _enable_aes(monkeypatch)
    record = SimpleNamespace(id="sso_config", sso_settings={"client_secret": legacy, "client_id": "id"})
    client = MagicMock()
    client.db.litellm_ssoconfig.find_many = AsyncMock(return_value=[record])
    client.db.litellm_ssoconfig.update = AsyncMock()

    report = await _migrate_sso(client, dry_run=False)

    assert report.migrated == 1
    assert report.legacy == 0  # migrated -> no longer residual
    client.db.litellm_ssoconfig.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_sso_walker_dry_run_reports_residual_not_migrated(salt_key, monkeypatch):
    """SSO dry run: residual legacy only; migrated stays 0 (never contradictory)."""
    legacy = _legacy_ct("client-secret", monkeypatch)
    _enable_aes(monkeypatch)
    record = SimpleNamespace(id="sso_config", sso_settings={"client_secret": legacy})
    client = MagicMock()
    client.db.litellm_ssoconfig.find_many = AsyncMock(return_value=[record])
    client.db.litellm_ssoconfig.update = AsyncMock()

    report = await _migrate_sso(client, dry_run=True)

    assert report.legacy == 1
    assert report.migrated == 0
    client.db.litellm_ssoconfig.update.assert_not_awaited()


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
    _empty_covered_tables(client)

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
    _empty_covered_tables(client)

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
    assert report.scanned == 1  # one field examined, not "post-v2" count
    client.db.litellm_teamtable.update.assert_awaited_once()
    written = json.loads(
        client.db.litellm_teamtable.update.call_args.kwargs["data"]["metadata"]
    )
    inner = written["logging"][0]["callback_vars"]["gcs_path_service_account"]
    assert "v2:gcm:" in inner


@pytest.mark.asyncio
async def test_callback_vars_walker_dry_run_reports_legacy(salt_key, monkeypatch):
    """In --check (dry-run) mode, a legacy callback var counts as residual legacy."""
    from litellm.proxy.common_utils.callback_utils import encrypt_callback_vars

    monkeypatch.setattr(proxy_server, "general_settings", {})
    legacy_meta = encrypt_callback_vars(
        {"logging": [{"callback_vars": {"gcs_path_service_account": "sa-secret"}}]}
    )
    _enable_aes(monkeypatch)

    team_row = SimpleNamespace(team_id="team-1", metadata=legacy_meta)
    client = MagicMock()
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_row])
    client.db.litellm_teamtable.update = AsyncMock()

    report = await cm._migrate_callback_vars_table(client, "team", dry_run=True)

    assert report.scanned == 1
    assert report.legacy == 1  # would-migrate -> residual legacy in attestation
    assert report.migrated == 0
    client.db.litellm_teamtable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_vars_walker_migrates_callback_settings_shape(
    salt_key, monkeypatch
):
    """Regression: credentials under ``metadata.callback_settings.callback_vars``
    with no top-level ``logging`` key must be migrated, not skipped.

    The walker previously early-continued on ``"logging" not in metadata``, so
    this credential shape (which ``encrypt_callback_vars`` does encrypt) was left
    in legacy format at rest while the migration still reported success.
    """
    from litellm.proxy.common_utils.callback_utils import encrypt_callback_vars

    monkeypatch.setattr(proxy_server, "general_settings", {})
    legacy_meta = encrypt_callback_vars(
        {
            "callback_settings": {
                "callback_vars": {"gcs_path_service_account": "sa-secret"}
            }
        }
    )
    _enable_aes(monkeypatch)
    assert "logging" not in legacy_meta  # the shape that used to be skipped

    team_row = SimpleNamespace(team_id="team-1", metadata=legacy_meta)
    client = MagicMock()
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_row])
    client.db.litellm_teamtable.update = AsyncMock()

    report = await cm._migrate_callback_vars_table(client, "team", dry_run=False)

    assert report.migrated == 1
    assert report.scanned == 1
    client.db.litellm_teamtable.update.assert_awaited_once()
    written = json.loads(
        client.db.litellm_teamtable.update.call_args.kwargs["data"]["metadata"]
    )
    inner = written["callback_settings"]["callback_vars"]["gcs_path_service_account"]
    assert "v2:gcm:" in inner


@pytest.mark.asyncio
async def test_check_reports_callback_var_legacy_with_gate_off(salt_key, monkeypatch):
    """check_encryption must report residual legacy callback vars even when the
    AES gate is OFF.

    Detection is decrypt-based, not a re-encrypt delta, so it does not depend on
    the write gate. A heuristic that re-encrypts and counts new v2 values would
    read zero here (gate off -> no v2 produced) and emit a false-clean
    attestation -- exactly the compliance trap this guards against.
    """
    from litellm.proxy.common_utils.callback_utils import encrypt_callback_vars

    # Legacy-encrypt a callback var, and leave the gate OFF for the check itself.
    monkeypatch.setattr(proxy_server, "general_settings", {})
    legacy_meta = encrypt_callback_vars(
        {"logging": [{"callback_vars": {"gcs_path_service_account": "sa-secret"}}]}
    )
    team_row = SimpleNamespace(team_id="team-1", metadata=legacy_meta)

    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[team_row])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.db.litellm_teamtable.update = AsyncMock()

    report = await cm.check_encryption(client)

    assert report.residual_legacy == 1
    assert report.as_dict()["locations"]["team.callback_vars"]["legacy"] == 1
    client.db.litellm_teamtable.update.assert_not_awaited()  # read-only


# --------------------------- covered-tables scanner ---------------------------


@pytest.mark.asyncio
async def test_scan_covered_tables_classifies_legacy_and_v2(salt_key, monkeypatch):
    """The read-only scanner classifies the model and credentials tables."""
    legacy = _legacy_ct("model-secret", monkeypatch)
    _enable_aes(monkeypatch)
    v2 = encrypt_value_helper("cred-secret")

    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_proxymodeltable.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(litellm_params={"api_key": legacy, "model": "gpt-4"})
        ]
    )
    client.db.litellm_credentialstable.find_many = AsyncMock(
        return_value=[SimpleNamespace(credential_values={"api_key": v2})]
    )
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)

    by_loc = {r.location: r for r in await cm._scan_covered_tables(client)}

    assert by_loc["model_table"].legacy == 1
    assert by_loc["model_table"].plaintext == 1  # "gpt-4" model name, not ciphertext
    assert by_loc["credentials"].already_v2 == 1
    assert by_loc["credentials"].legacy == 0


@pytest.mark.asyncio
async def test_check_counts_covered_table_residual(salt_key, monkeypatch):
    """check_encryption now scans the rotation-covered tables (model table here),
    so a legacy value there counts toward residual_legacy (the P1 attestation gap).
    """
    legacy = _legacy_ct("model-secret", monkeypatch)
    _enable_aes(monkeypatch)

    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.db.litellm_config.update = AsyncMock()
    client.db.litellm_proxymodeltable.find_many = AsyncMock(
        return_value=[SimpleNamespace(litellm_params={"api_key": legacy})]
    )

    report = await cm.check_encryption(client)

    assert report.residual_legacy == 1
    assert report.as_dict()["locations"]["model_table"]["legacy"] == 1
    client.db.litellm_config.update.assert_not_awaited()  # read-only


@pytest.mark.asyncio
async def test_migrate_covered_tables_reports_real_counts(salt_key, monkeypatch):
    """_migrate_covered_tables derives real per-table counts from pre/post scans,
    instead of the always-zero report Greptile flagged (P1).
    """
    legacy = _legacy_ct("model-secret", monkeypatch)
    _enable_aes(monkeypatch)
    v2 = encrypt_value_helper("model-secret")

    row = SimpleNamespace(litellm_params={"api_key": legacy})
    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[row])
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)

    async def fake_rotate(**kwargs):
        # Stand in for _rotate_master_key: re-encrypt the model api_key in place.
        row.litellm_params["api_key"] = v2

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints._rotate_master_key",
        fake_rotate,
    )

    by_loc = {
        r.location: r for r in await cm._migrate_covered_tables(client, MagicMock())
    }

    assert by_loc["model_table"].migrated == 1  # was legacy pre, v2 post
    assert by_loc["model_table"].legacy == 0  # residual zero after rotation
    assert by_loc["model_table"].already_v2 == 1


# --------------------------- salt-key rotation ---------------------------


@pytest.fixture
def rotated_salt_key(monkeypatch):
    """Simulate a completed key swap: new key active, old one retired."""
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-old")
    old_ct = encrypt_value_helper("provider-api-key")
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-new")
    monkeypatch.setenv("LITELLM_SALT_KEY_PREVIOUS", "sk-salt-old")
    return old_ct


def test_salt_key_policy_classifies_retired_ciphertext_as_legacy(rotated_salt_key):
    assert cm.classify_value(rotated_salt_key, policy=cm.SALT_KEY_POLICY) == "legacy"
    # The algorithm pass has nothing to do here: the format is already correct.
    assert cm.classify_value(rotated_salt_key, policy=cm.ALGORITHM_POLICY) == "legacy"


def test_salt_key_policy_reencrypts_under_active_key(rotated_salt_key, monkeypatch):
    out = cm.reencrypt_value(rotated_salt_key, policy=cm.SALT_KEY_POLICY)

    assert out != rotated_salt_key
    # Readable with the retired key removed from the environment: the whole point.
    monkeypatch.delenv("LITELLM_SALT_KEY_PREVIOUS")
    assert decrypt_value_helper(out, key="t") == "provider-api-key"
    assert cm.classify_value(out, policy=cm.SALT_KEY_POLICY) == "migrated"


def test_salt_key_policy_preserves_the_legacy_format(rotated_salt_key, monkeypatch):
    """A salt-only rotation must not silently switch algorithms.

    The configured write algorithm is AES here, so a rewrite that went through
    ``encrypt_value_helper`` would upgrade the format as a side effect of a
    key-only rotation. The two axes are independent, so the format must survive.
    """
    _enable_aes(monkeypatch)

    out = cm.reencrypt_value(rotated_salt_key, policy=cm.SALT_KEY_POLICY)

    assert out != rotated_salt_key
    assert not out.startswith(_V2_GCM_PREFIX)


def test_salt_key_policy_preserves_the_aes_format(monkeypatch):
    """Regression: a key-only rotation must never downgrade AES-GCM ciphertext.

    A value written in AES under the retired key, rotated while the configured
    write algorithm sits at its legacy default, used to come back as nacl: the
    rewrite went through ``encrypt_value_helper``, which reads that setting.
    """
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-old")
    _enable_aes(monkeypatch)
    old_aes = encrypt_value_helper("provider-api-key")
    assert old_aes.startswith(_V2_GCM_PREFIX)

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-new")
    monkeypatch.setenv("LITELLM_SALT_KEY_PREVIOUS", "sk-salt-old")
    monkeypatch.setattr(proxy_server, "general_settings", {})  # legacy write algo

    out = cm.reencrypt_value(old_aes, policy=cm.SALT_KEY_POLICY)

    assert out != old_aes
    assert out.startswith(_V2_GCM_PREFIX)
    monkeypatch.delenv("LITELLM_SALT_KEY_PREVIOUS")
    assert decrypt_value_helper(out, key="t") == "provider-api-key"


@pytest.mark.asyncio
async def test_salt_key_rotation_preserves_each_algorithm_in_a_mixed_row(monkeypatch):
    """Regression: a callback row mixing a retired-key value with an active-key
    AES value must have only the former rewritten.

    The walker used to round-trip the whole row through
    ``encrypt_callback_vars(decrypt_callback_vars(...))``, which pushed every
    value through the configured write algorithm and downgraded the AES one.
    """
    from litellm.proxy.common_utils.callback_utils import (
        _CALLBACK_VAR_ENCRYPTED_PREFIX,
    )

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-old")
    monkeypatch.setattr(proxy_server, "general_settings", {})
    retired = _CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper("sa-secret")

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-new")
    _enable_aes(monkeypatch)
    active_aes = _CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper("lf-secret")

    monkeypatch.setenv("LITELLM_SALT_KEY_PREVIOUS", "sk-salt-old")
    monkeypatch.setattr(proxy_server, "general_settings", {})  # legacy write algo

    row = SimpleNamespace(
        team_id="t1",
        metadata={
            "logging": [
                {
                    "callback_vars": {
                        "gcs_path_service_account": retired,
                        "langfuse_secret_key": active_aes,
                    }
                }
            ]
        },
    )
    client = MagicMock()
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[row])
    client.db.litellm_teamtable.update = AsyncMock()

    report = await cm._migrate_callback_vars_table(
        client, "team", dry_run=False, policy=cm.SALT_KEY_POLICY
    )

    assert report.migrated == 1
    assert report.already_v2 == 1
    written = json.loads(
        client.db.litellm_teamtable.update.call_args.kwargs["data"]["metadata"]
    )
    cvs = written["logging"][0]["callback_vars"]

    # Already under the active key: left byte-for-byte alone, so no downgrade.
    assert cvs["langfuse_secret_key"] == active_aes

    # Under the retired key: moved to the active key, keeping its own format.
    rotated = cvs["gcs_path_service_account"]
    assert rotated != retired
    inner = rotated.removeprefix(_CALLBACK_VAR_ENCRYPTED_PREFIX)
    assert not inner.startswith(_V2_GCM_PREFIX)
    monkeypatch.delenv("LITELLM_SALT_KEY_PREVIOUS")
    assert decrypt_value_helper(inner, key="t") == "sa-secret"


@pytest.mark.asyncio
async def test_check_counts_sso_identity_assertion_residual(rotated_salt_key):
    """Regression: the SSO identity assertion store is part of the attestation.

    ``_rotate_master_key`` rotates it but logs and carries on when that store
    fails, so a scan that skipped the table could report ``residual_legacy == 0``
    while an assertion still depended on the retired key. Dropping
    ``LITELLM_SALT_KEY_PREVIOUS`` then reads that assertion as absent.
    """
    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.db.litellm_ssoidentityassertion.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="u1", assertion_b64=rotated_salt_key)]
    )

    report = await cm.check_encryption(client, policy=cm.SALT_KEY_POLICY)

    assert report.as_dict()["locations"]["sso_identity_assertions"]["legacy"] == 1
    assert report.residual_legacy == 1


@pytest.mark.asyncio
async def test_check_counts_mcp_oauth_client_residual(rotated_salt_key):
    """Regression: the MCP OAuth client table is part of the attestation.

    The shared rotation path rewrites those DCR client credentials, so leaving
    the table out of the scan let the check report ``residual_legacy == 0``
    while a client secret still decrypted only under the retired key.
    """
    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.db.litellm_mcpserveroauthclient.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(
                server_id="srv-1",
                credentials={"client_id": "public-id", "client_secret": rotated_salt_key},
            )
        ]
    )

    report = await cm.check_encryption(client, policy=cm.SALT_KEY_POLICY)

    assert report.as_dict()["locations"]["mcp_oauth_clients"]["legacy"] == 1
    assert report.residual_legacy == 1


@pytest.mark.asyncio
async def test_salt_key_migration_refuses_to_downgrade_oauth_client_credentials(monkeypatch):
    """Regression: the downgrade preflight covers the MCP OAuth client table.

    Its credentials go through the same rotation path, which writes via the
    configured algorithm, so an AES client secret would be silently rewritten in
    the legacy format by a key-only rotation.
    """
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-old")
    _enable_aes(monkeypatch)
    old_aes = encrypt_value_helper("dcr-client-secret")

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-new")
    monkeypatch.setenv("LITELLM_SALT_KEY_PREVIOUS", "sk-salt-old")
    monkeypatch.setattr(proxy_server, "general_settings", {})  # legacy write algo

    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_mcpserveroauthclient.find_many = AsyncMock(
        return_value=[SimpleNamespace(server_id="srv-1", credentials={"client_secret": old_aes})]
    )
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="aes-256-gcm"):
        await cm.migrate_encryption(
            prisma_client=client,
            user_api_key_dict=MagicMock(),
            policy=cm.SALT_KEY_POLICY,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location,db_attr,pk,pk_value,column,field",
    [
        ("cache_config", "litellm_cacheconfig", "id", "cache_config", "cache_settings", "redis_password"),
        (
            "config_overrides",
            "litellm_configoverrides",
            "config_type",
            "hashicorp_vault",
            "config_value",
            "HCP_VAULT_TOKEN",
        ),
    ],
)
async def test_salt_key_pass_rotates_settings_rows(
    rotated_salt_key, monkeypatch, location, db_attr, pk, pk_value, column, field
):
    """Regression: the cache-config and config-override rows are rotated too.

    Both hold live secrets written through the same encryption path and neither
    has a master-key rotation path, so skipping them would strand a Redis
    password or a Vault token under the retired key while the check read clean.
    """
    row = SimpleNamespace(**{pk: pk_value, column: {field: rotated_salt_key, "host": "localhost"}})
    client = MagicMock()
    table = getattr(client.db, db_attr)
    table.find_many = AsyncMock(return_value=[row])
    table.update = AsyncMock()

    report = await cm._migrate_settings_rows(
        client, location, db_attr, pk, column, dry_run=False, policy=cm.SALT_KEY_POLICY
    )

    assert report.migrated == 1
    written = json.loads(table.update.call_args.kwargs["data"][column])
    assert written["host"] == "localhost"  # non-ciphertext left alone
    monkeypatch.delenv("LITELLM_SALT_KEY_PREVIOUS")
    assert decrypt_value_helper(written[field], key="t") == "provider-api-key"


@pytest.mark.asyncio
async def test_check_counts_settings_row_residual(rotated_salt_key):
    """A secret left under the retired key in either settings row blocks the attestation."""
    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.db.litellm_cacheconfig.find_many = AsyncMock(
        return_value=[SimpleNamespace(id="cache_config", cache_settings={"redis_password": rotated_salt_key})]
    )
    client.db.litellm_configoverrides.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(config_type="hashicorp_vault", config_value={"HCP_VAULT_TOKEN": rotated_salt_key})
        ]
    )

    report = await cm.check_encryption(client, policy=cm.SALT_KEY_POLICY)
    locations = report.as_dict()["locations"]

    assert locations["cache_config"]["legacy"] == 1
    assert locations["config_overrides"]["legacy"] == 1
    assert report.residual_legacy == 2
    client.db.litellm_cacheconfig.update.assert_not_called()  # read-only


@pytest.mark.asyncio
@pytest.mark.parametrize("db_attr,location", [("litellm_proxymodeltable", "model_table"), ("litellm_cacheconfig", "cache_config")])
async def test_check_names_a_store_it_could_not_read(rotated_salt_key, db_attr, location):
    """Regression: a store the scan cannot open is unknown, never clean.

    Both scanners swallow a driver failure and return zero of every counter, so
    without this the check answered `residual_legacy: 0` for a table nobody
    read, and the operator dropped the retired key on that word.
    """
    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)
    getattr(client.db, db_attr).find_many = AsyncMock(side_effect=RuntimeError("connection refused"))

    report = await cm.check_encryption(client, policy=cm.SALT_KEY_POLICY)

    assert report.residual_legacy == 0  # nothing was read, so nothing counted
    assert location in report.unreadable_locations
    assert report.as_dict()["unreadable_locations"] == [location]


@pytest.mark.asyncio
async def test_check_reports_no_unreadable_stores_on_a_healthy_scan(rotated_salt_key):
    """The unreadable list stays empty when every store answers, so it can only
    ever mean a real read failure."""
    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)

    report = await cm.check_encryption(client, policy=cm.SALT_KEY_POLICY)

    assert report.unreadable_locations == ()
    assert report.residual_legacy == 0


@pytest.mark.asyncio
async def test_salt_key_migration_refuses_to_downgrade_covered_tables(monkeypatch):
    """The rotation-covered tables are re-encrypted by ``_rotate_master_key``,
    which writes through the configured algorithm and so cannot preserve a
    value's own. Refuse the run rather than downgrade AES ciphertext silently.
    """
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-old")
    _enable_aes(monkeypatch)
    old_aes = encrypt_value_helper("model-secret")

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-new")
    monkeypatch.setenv("LITELLM_SALT_KEY_PREVIOUS", "sk-salt-old")
    monkeypatch.setattr(proxy_server, "general_settings", {})  # legacy write algo

    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_proxymodeltable.find_many = AsyncMock(
        return_value=[SimpleNamespace(litellm_params={"api_key": old_aes})]
    )
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="aes-256-gcm"):
        await cm.migrate_encryption(
            prisma_client=client,
            user_api_key_dict=MagicMock(),
            policy=cm.SALT_KEY_POLICY,
        )


def test_salt_key_policy_is_idempotent(rotated_salt_key):
    once = cm.reencrypt_value(rotated_salt_key, policy=cm.SALT_KEY_POLICY)
    assert cm.reencrypt_value(once, policy=cm.SALT_KEY_POLICY) == once


def test_salt_key_policy_does_not_require_the_aes_gate(rotated_salt_key):
    """The AES gate guards the algorithm pass only, never the salt-key pass."""
    assert cm.reencrypt_value(rotated_salt_key, policy=cm.SALT_KEY_POLICY) != rotated_salt_key


@pytest.mark.asyncio
async def test_salt_key_migration_requires_previous_keys(monkeypatch):
    """Without the retired key, old ciphertext reads as plaintext and is skipped,
    so the pass would report a clean run while leaving values behind.
    """
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-new")
    monkeypatch.delenv("LITELLM_SALT_KEY_PREVIOUS", raising=False)

    with pytest.raises(RuntimeError, match="LITELLM_SALT_KEY_PREVIOUS"):
        await cm.migrate_encryption(
            prisma_client=MagicMock(),
            user_api_key_dict=MagicMock(),
            policy=cm.SALT_KEY_POLICY,
        )


def test_policy_for_mode():
    assert cm.policy_for_mode("algorithm") is cm.ALGORITHM_POLICY
    assert cm.policy_for_mode("salt-key") is cm.SALT_KEY_POLICY


@pytest.mark.asyncio
async def test_salt_key_check_reports_residual(rotated_salt_key):
    client = MagicMock()
    _empty_covered_tables(client)
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    client.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)
    client.db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.db.litellm_config.update = AsyncMock()
    client.db.litellm_proxymodeltable.find_many = AsyncMock(
        return_value=[SimpleNamespace(litellm_params={"api_key": rotated_salt_key})]
    )

    report = await cm.check_encryption(client, policy=cm.SALT_KEY_POLICY)

    assert report.residual_legacy == 1
    assert report.as_dict()["locations"]["model_table"]["legacy"] == 1
    client.db.litellm_config.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_salt_key_pass_walks_callback_vars(monkeypatch):
    """Locations with no master-key rotation path are covered by the salt pass too."""
    from litellm.proxy.common_utils.callback_utils import (
        decrypt_callback_vars,
        encrypt_callback_vars,
    )

    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-old")
    old_meta = encrypt_callback_vars(
        {"logging": [{"callback_vars": {"gcs_path_service_account": "sa-secret"}}]}
    )
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-new")
    monkeypatch.setenv("LITELLM_SALT_KEY_PREVIOUS", "sk-salt-old")

    row = SimpleNamespace(team_id="t1", metadata=old_meta)
    client = MagicMock()
    client.db.litellm_teamtable.find_many = AsyncMock(return_value=[row])
    client.db.litellm_teamtable.update = AsyncMock()

    report = await cm._migrate_callback_vars_table(
        client, "team", dry_run=False, policy=cm.SALT_KEY_POLICY
    )

    assert report.migrated == 1
    written = json.loads(
        client.db.litellm_teamtable.update.call_args.kwargs["data"]["metadata"]
    )
    # Readable once the retired key is gone: the rewrite used the active key.
    monkeypatch.delenv("LITELLM_SALT_KEY_PREVIOUS")
    rotated = decrypt_callback_vars(written)
    assert rotated["logging"][0]["callback_vars"]["gcs_path_service_account"] == "sa-secret"
