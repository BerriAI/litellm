"""
Tests for the at-rest credential encryption chokepoint.

Covers the AES-256-GCM default (``v3:gcm:`` written under HKDF-SHA256), the
``encryption_algorithm`` legacy opt-in and its FIPS refusal, and the dual-read
guarantees that keep ``v2:gcm:`` (raw SHA-256 AES) and unprefixed
XSalsa20-Poly1305 (nacl) ciphertext decrypting after the default flip.
"""

import base64
import hashlib
import logging
import os
import sys

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from litellm.proxy import proxy_server
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    _V2_GCM_PREFIX,
    _V3_GCM_PREFIX,
    LegacyEncryptionUnavailableError,
    _get_encryption_algorithm,
    decrypt_if_encrypted_with,
    decrypt_value_helper,
    encrypt_value,
    encrypt_value_helper,
    is_versioned_gcm,
    legacy_encryption_available,
    require_legacy_reader,
)
from litellm.proxy.common_utils.fips import FipsModeError

_SALT_KEY = "sk-salt-aes-1234"


def _use_legacy(monkeypatch):
    """Opt the write-time algorithm back into XSalsa20-Poly1305 for the duration of a test."""
    monkeypatch.setattr(proxy_server, "general_settings", {"encryption_algorithm": "xsalsa20-poly1305"})


def _sha256_key(signing_key: str = _SALT_KEY) -> bytes:
    return hashlib.sha256(signing_key.encode()).digest()


def _hkdf_key(signing_key: str = _SALT_KEY) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"litellm-at-rest-v3").derive(
        signing_key.encode()
    )


def _v2_ciphertext(plaintext: str, key: bytes) -> str:
    """Build a ``v2:gcm:`` value the way the previous release wrote it: AES-GCM under raw SHA-256."""
    nonce = os.urandom(12)
    return (
        _V2_GCM_PREFIX + base64.urlsafe_b64encode(nonce + AESGCM(key).encrypt(nonce, plaintext.encode(), None)).decode()
    )


def _legacy_nacl_ciphertext(plaintext: str, key: bytes) -> str:
    """Build an unprefixed value the way pre-AES releases wrote it: XSalsa20-Poly1305 under raw SHA-256."""
    import nacl.secret

    return base64.urlsafe_b64encode(bytes(nacl.secret.SecretBox(key).encrypt(plaintext.encode()))).decode()


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", _SALT_KEY)
    monkeypatch.delenv("LITELLM_FIPS_MODE", raising=False)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    yield


def test_default_write_is_v3_gcm_and_round_trips():
    ct = encrypt_value_helper("super-secret")

    assert ct.startswith(_V3_GCM_PREFIX), ct
    assert decrypt_value_helper(ct, key="t") == "super-secret"


def test_explicit_aes_setting_writes_v3_gcm(monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {"encryption_algorithm": "AES-256-GCM"})

    assert encrypt_value_helper("secret").startswith(_V3_GCM_PREFIX)


def test_unknown_algorithm_falls_back_to_the_aes_default(monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {"encryption_algorithm": "rot13"})

    ct = encrypt_value_helper("secret")
    assert ct.startswith(_V3_GCM_PREFIX), ct
    assert decrypt_value_helper(ct, key="t") == "secret"


def test_v3_value_is_aes_gcm_under_hkdf_sha256_not_raw_sha256():
    ct = encrypt_value_helper("hkdf-secret")
    raw = base64.urlsafe_b64decode(ct[len(_V3_GCM_PREFIX) :])
    nonce, blob = raw[:12], raw[12:]

    assert AESGCM(_hkdf_key()).decrypt(nonce, blob, None) == b"hkdf-secret"
    with pytest.raises(InvalidTag):
        AESGCM(_sha256_key()).decrypt(nonce, blob, None)


def test_each_v3_write_uses_a_fresh_nonce():
    first, second = encrypt_value_helper("same-secret"), encrypt_value_helper("same-secret")

    assert first != second
    assert (
        base64.urlsafe_b64decode(first[len(_V3_GCM_PREFIX) :])[:12]
        != base64.urlsafe_b64decode(second[len(_V3_GCM_PREFIX) :])[:12]
    )


def test_v2_value_written_under_raw_sha256_still_decrypts():
    legacy_v2 = _v2_ciphertext("v2-secret", _sha256_key())

    assert decrypt_value_helper(legacy_v2, key="t") == "v2-secret"
    assert decrypt_if_encrypted_with(legacy_v2, _SALT_KEY) == "v2-secret"


def test_v2_value_is_not_read_with_the_v3_derivation():
    """A v2 body relabelled as v3 must fail: the prefix, not the caller, picks the derivation."""
    v2_body = _v2_ciphertext("v2-secret", _sha256_key())[len(_V2_GCM_PREFIX) :]

    assert decrypt_if_encrypted_with(_V3_GCM_PREFIX + v2_body, _SALT_KEY) is None
    assert decrypt_value_helper(_V3_GCM_PREFIX + v2_body, key="t", exception_type="debug") is None


def test_legacy_nacl_value_still_decrypts_under_the_aes_default():
    legacy = _legacy_nacl_ciphertext("legacy-secret", _sha256_key())
    assert not is_versioned_gcm(legacy)

    assert decrypt_value_helper(legacy, key="t") == "legacy-secret"
    assert decrypt_if_encrypted_with(legacy, _SALT_KEY) == "legacy-secret"
    assert encrypt_value_helper("fresh").startswith(_V3_GCM_PREFIX)


def test_legacy_opt_in_writes_unprefixed_nacl_that_still_reads_back(monkeypatch):
    _use_legacy(monkeypatch)

    ct = encrypt_value_helper("legacy-secret")

    assert not is_versioned_gcm(ct), ct
    assert decrypt_value_helper(ct, key="t") == "legacy-secret"
    assert _get_encryption_algorithm() == "xsalsa20-poly1305"


def test_legacy_opt_in_is_refused_under_fips_mode(monkeypatch):
    _use_legacy(monkeypatch)
    monkeypatch.setenv("LITELLM_FIPS_MODE", "true")

    with pytest.raises(FipsModeError, match="xsalsa20-poly1305") as refused:
        encrypt_value_helper("secret")
    assert "aes-256-gcm" in str(refused.value)


def test_fips_mode_with_the_default_setting_writes_v3_gcm(monkeypatch):
    monkeypatch.setenv("LITELLM_FIPS_MODE", "true")

    ct = encrypt_value_helper("fips-secret")
    assert ct.startswith(_V3_GCM_PREFIX), ct
    assert decrypt_value_helper(ct, key="t") == "fips-secret"


def test_versioned_gcm_values_never_import_nacl(monkeypatch):
    legacy_v2 = _v2_ciphertext("v2-secret", _sha256_key())
    monkeypatch.setitem(sys.modules, "nacl", None)
    monkeypatch.setitem(sys.modules, "nacl.secret", None)

    assert decrypt_value_helper(encrypt_value_helper("v3-secret"), key="t") == "v3-secret"
    assert decrypt_value_helper(legacy_v2, key="t") == "v2-secret"


def test_legacy_ciphertext_without_pynacl_logs_the_reencrypt_path_and_never_returns_the_blob(monkeypatch, caplog):
    legacy = _legacy_nacl_ciphertext("legacy-secret", _sha256_key())
    assert legacy_encryption_available()
    monkeypatch.setitem(sys.modules, "nacl", None)
    monkeypatch.setitem(sys.modules, "nacl.secret", None)
    assert not legacy_encryption_available()

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
        assert decrypt_value_helper(legacy, key="t", exception_type="debug", return_original_value=True) is None
        assert decrypt_value_helper(legacy, key="t", exception_type="debug") is None
        assert decrypt_if_encrypted_with(legacy, _SALT_KEY) is None
    assert len(caplog.records) == 3, caplog.text
    for record in caplog.records:
        message = record.getMessage()
        assert "PyNaCl is not installed" in message
        assert "legacy-encryption" in message
        assert "/credentials/migrate-encryption" in message
    assert "legacy-secret" not in caplog.text


def test_require_legacy_reader_refuses_rewrite_passes_without_pynacl(monkeypatch):
    require_legacy_reader("rotate the master key")
    monkeypatch.setitem(sys.modules, "nacl", None)
    monkeypatch.setitem(sys.modules, "nacl.secret", None)

    with pytest.raises(LegacyEncryptionUnavailableError, match=r"rotate the master key.*legacy-encryption"):
        require_legacy_reader("rotate the master key")


def test_legacy_opt_in_without_pynacl_fails_the_write_not_silently(monkeypatch):
    _use_legacy(monkeypatch)
    monkeypatch.setitem(sys.modules, "nacl", None)
    monkeypatch.setitem(sys.modules, "nacl.secret", None)

    with pytest.raises(LegacyEncryptionUnavailableError, match="encrypt"):
        encrypt_value_helper("secret")


def test_v3_prefix_is_the_idempotent_migration_marker():
    ct = encrypt_value_helper("secret")
    assert is_versioned_gcm(ct)

    again = encrypt_value_helper(decrypt_value_helper(ct, key="t"))
    assert again.startswith(_V3_GCM_PREFIX)
    assert decrypt_value_helper(again, key="t") == "secret"


@pytest.mark.parametrize("prefix", [_V3_GCM_PREFIX, _V2_GCM_PREFIX])
def test_aes_decrypt_failure_returns_none_not_raise(prefix: str):
    garbled = prefix + "not-valid-base64-or-ciphertext!!!"
    assert decrypt_value_helper(garbled, key="t", exception_type="debug") is None


def test_aes_decrypt_failure_returns_original_when_requested():
    garbled = _V3_GCM_PREFIX + "###"
    assert decrypt_value_helper(garbled, key="t", exception_type="debug", return_original_value=True) == garbled


def test_empty_string_round_trips_under_aes():
    ct = encrypt_value_helper("")
    assert ct.startswith(_V3_GCM_PREFIX)
    assert decrypt_value_helper(ct, key="t") == ""


def test_callback_prefix_composes_with_v3():
    """litellm_enc:: + v3:gcm:... round-trips through the callback read path."""
    from litellm.proxy.common_utils.callback_utils import (
        _CALLBACK_VAR_ENCRYPTED_PREFIX,
        _decrypt_or_passthrough,
        _encrypt_if_plaintext,
    )

    stored = _encrypt_if_plaintext("gcs_path_service_account", "my-sa-secret")

    assert stored.startswith(_CALLBACK_VAR_ENCRYPTED_PREFIX)
    inner = stored[len(_CALLBACK_VAR_ENCRYPTED_PREFIX) :]
    assert inner.startswith(_V3_GCM_PREFIX)
    assert _decrypt_or_passthrough("gcs_path_service_account", stored) == "my-sa-secret"


def test_decrypt_failure_debug_log_omits_raw_value(monkeypatch):
    """Regression for LIT-4152: the decrypt-failure debug breadcrumb must not
    embed the raw value.

    A DB ``environment_variables`` secret (e.g. a ``DATABASE_URL`` connection
    string) reaches this path when it cannot be decrypted, for example after a
    salt or master key change, and previously printed in cleartext when the
    module regex scrubber was bypassed. The failing key still names the pair so
    the breadcrumb keeps its debugging value. Uses a dedicated handler rather
    than caplog because caplog is unreliable under pytest-xdist.
    """
    import logging

    import litellm._logging as _logging_module
    from litellm._logging import verbose_proxy_logger

    monkeypatch.setattr(_logging_module, "_ENABLE_SECRET_REDACTION", False)

    secret = "postgresql://leak_user:leak_pw_decrypt@leak-host:5432/leak_db"

    class LogRecordHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = LogRecordHandler()
    handler.setLevel(logging.DEBUG)
    original_level = verbose_proxy_logger.level
    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_proxy_logger.addHandler(handler)
    try:
        result = decrypt_value_helper(secret, key="DATABASE_URL", return_original_value=True)
        rendered = " ".join(record.getMessage() for record in handler.records)
    finally:
        verbose_proxy_logger.removeHandler(handler)
        verbose_proxy_logger.setLevel(original_level)

    assert secret not in rendered, f"raw value leaked in decrypt-failure log: {rendered!r}"
    assert "leak_pw_decrypt" not in rendered
    assert any("DATABASE_URL" in record.getMessage() for record in handler.records), (
        "the failing key should still be named in the breadcrumb"
    )
    assert result == secret


@pytest.mark.parametrize("use_legacy", [False, True])
def test_explicit_key_decrypt_reads_only_values_written_under_that_key(monkeypatch, use_legacy: bool):
    if use_legacy:
        _use_legacy(monkeypatch)
    written_with_previous_key = encrypt_value_helper("stored-secret", new_encryption_key="sk-1234")

    assert decrypt_if_encrypted_with(written_with_previous_key, "sk-1234") == "stored-secret"
    assert decrypt_if_encrypted_with(written_with_previous_key, "sk-another-key") is None
    assert decrypt_value_helper(written_with_previous_key, key="t", exception_type="debug") is None


@pytest.mark.parametrize(
    "not_a_ciphertext",
    [
        "",
        "gpt-5.4-mini",
        "https://example.invalid/v1",
        "v2:gcm:",
        "v3:gcm:",
        "aGVsbG8=",
        "*",
        "-",
        "_",
        "...",
        " ",
        "{}",
        "[]",
        "=",
    ],
)
def test_explicit_key_decrypt_rejects_values_that_are_not_ciphertexts(not_a_ciphertext: str):
    assert decrypt_if_encrypted_with(not_a_ciphertext, "sk-1234") is None


@pytest.mark.parametrize("use_legacy", [False, True])
def test_explicit_key_decrypt_tells_an_encrypted_empty_string_from_no_ciphertext(monkeypatch, use_legacy: bool):
    if use_legacy:
        _use_legacy(monkeypatch)

    assert decrypt_if_encrypted_with(encrypt_value_helper("", new_encryption_key="sk-1234"), "sk-1234") == ""


def test_explicit_key_decrypt_supports_the_empty_master_key():
    written_with_empty_key = encrypt_value(value="stored-secret", signing_key="")

    assert decrypt_if_encrypted_with(base64.urlsafe_b64encode(written_with_empty_key).decode(), "") == "stored-secret"
