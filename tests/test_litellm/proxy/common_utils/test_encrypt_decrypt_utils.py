"""
Tests for the at-rest credential encryption chokepoint.

Covers the AES-256-GCM (``v2:gcm:``) path, the ``encryption_algorithm`` config
gate, and the backward-compatibility guarantees that let legacy XSalsa20-Poly1305
(nacl) ciphertext and new AES values coexist and decrypt correctly.
"""

import base64
import json
from types import MappingProxyType

import pytest

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.proxy import proxy_server
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    _V2_GCM_PREFIX,
    decrypt_config_section,
    decrypt_if_encrypted_with,
    decrypt_json_strings,
    decrypt_stored_json_object,
    decrypt_value_helper,
    encrypt_config_section,
    encrypt_json_strings,
    encrypt_stored_json_object,
    encrypt_value,
    encrypt_value_helper,
    json_value,
)

SALT_KEY = "sk-salt-aes-1234"


def _use_aes(monkeypatch):
    """Flip the write-time algorithm to AES-256-GCM for the duration of a test."""
    monkeypatch.setattr(proxy_server, "general_settings", {"encryption_algorithm": "aes-256-gcm"})


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    # Dominant convention in the test_litellm/ tree: set the key via env.
    monkeypatch.setenv("LITELLM_SALT_KEY", SALT_KEY)
    # Ensure the legacy default is in force unless a test opts into AES.
    monkeypatch.setattr(proxy_server, "general_settings", {})
    yield


def test_aes_gcm_round_trip(monkeypatch):
    """A value written under AES-256-GCM is tagged v2:gcm: and decrypts back."""
    _use_aes(monkeypatch)

    ct = encrypt_value_helper("super-secret")

    assert ct.startswith(_V2_GCM_PREFIX)
    assert decrypt_value_helper(ct, key="t") == "super-secret"


def test_default_is_legacy_algorithm(monkeypatch):
    """With no config, writes stay on the legacy algorithm (no v2: marker)."""
    ct = encrypt_value_helper("legacy-secret")

    assert not ct.startswith(_V2_GCM_PREFIX)
    assert decrypt_value_helper(ct, key="t") == "legacy-secret"


def test_legacy_nacl_value_still_decrypts_after_flag_flip(monkeypatch):
    """A value written under the old algorithm decrypts unchanged once AES is on.

    This is the mixed-format readback guarantee: decrypt is format-detecting, so
    flipping the flag forward never strands previously-written data.
    """
    legacy = encrypt_value_helper("legacy-secret")  # default = xsalsa20
    assert not legacy.startswith(_V2_GCM_PREFIX)

    _use_aes(monkeypatch)
    # New writes are now AES, but the old value must still come back.
    assert decrypt_value_helper(legacy, key="t") == "legacy-secret"
    assert encrypt_value_helper("fresh").startswith(_V2_GCM_PREFIX)


def test_v2_prefix_is_idempotent_marker(monkeypatch):
    """The migration's skip-check: an already-v2 value is recognized by its prefix.

    Re-encrypting an AES value yields a fresh (different nonce) AES value, but the
    prefix is what lets a migration skip already-migrated rows without decrypting.
    """
    _use_aes(monkeypatch)

    ct = encrypt_value_helper("secret")
    assert ct.startswith(_V2_GCM_PREFIX)

    # Round-tripping does not change the plaintext, and the marker is stable.
    again = encrypt_value_helper(decrypt_value_helper(ct, key="t"))
    assert again.startswith(_V2_GCM_PREFIX)
    assert decrypt_value_helper(again, key="t") == "secret"


def test_aes_decrypt_failure_returns_none_not_raise(monkeypatch):
    """Decrypt contract preserved: a garbled v2 value returns None, never raises."""
    _use_aes(monkeypatch)

    garbled = _V2_GCM_PREFIX + "not-valid-base64-or-ciphertext!!!"
    # exception_type="debug" exercises the swallow path; must not raise.
    assert decrypt_value_helper(garbled, key="t", exception_type="debug") is None


def test_aes_decrypt_failure_returns_original_when_requested(monkeypatch):
    """With return_original_value=True a bad v2 value comes back as-is, not None."""
    _use_aes(monkeypatch)

    garbled = _V2_GCM_PREFIX + "###"
    assert decrypt_value_helper(garbled, key="t", exception_type="debug", return_original_value=True) == garbled


def test_empty_string_round_trips_under_aes(monkeypatch):
    """Empty string is preserved through the AES path (parity with legacy)."""
    _use_aes(monkeypatch)

    ct = encrypt_value_helper("")
    assert ct.startswith(_V2_GCM_PREFIX)
    assert decrypt_value_helper(ct, key="t") == ""


def test_callback_prefix_composes_with_v2(monkeypatch):
    """litellm_enc:: + v2:gcm:... round-trips through the callback read path.

    Callback vars are stored as ``litellm_enc::<helper output>``; the read path
    strips ``litellm_enc::`` then calls the helper, so the value handed to the
    helper is ``v2:gcm:...``. Ordering must work end to end.
    """
    from litellm.proxy.common_utils.callback_utils import (
        _CALLBACK_VAR_ENCRYPTED_PREFIX,
        _decrypt_or_passthrough,
        _encrypt_if_plaintext,
    )

    _use_aes(monkeypatch)

    # "gcs_path_service_account" is a known-sensitive callback key.
    stored = _encrypt_if_plaintext("gcs_path_service_account", "my-sa-secret")

    assert stored.startswith(_CALLBACK_VAR_ENCRYPTED_PREFIX)
    inner = stored[len(_CALLBACK_VAR_ENCRYPTED_PREFIX) :]
    assert inner.startswith(_V2_GCM_PREFIX)
    assert _decrypt_or_passthrough("gcs_path_service_account", stored) == "my-sa-secret"


def test_unknown_algorithm_falls_back_to_legacy(monkeypatch):
    """An unrecognized encryption_algorithm value does not produce v2 writes."""
    monkeypatch.setattr(proxy_server, "general_settings", {"encryption_algorithm": "rot13"})

    ct = encrypt_value_helper("secret")
    assert not ct.startswith(_V2_GCM_PREFIX)
    assert decrypt_value_helper(ct, key="t") == "secret"


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


@pytest.mark.parametrize("use_aes", [False, True])
def test_explicit_key_decrypt_reads_only_values_written_under_that_key(monkeypatch, use_aes: bool):
    if use_aes:
        _use_aes(monkeypatch)
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


@pytest.mark.parametrize("use_aes", [False, True])
def test_explicit_key_decrypt_tells_an_encrypted_empty_string_from_no_ciphertext(monkeypatch, use_aes: bool):
    if use_aes:
        _use_aes(monkeypatch)

    assert decrypt_if_encrypted_with(encrypt_value_helper("", new_encryption_key="sk-1234"), "sk-1234") == ""


def test_explicit_key_decrypt_supports_the_empty_master_key():
    written_with_empty_key = encrypt_value(value="stored-secret", signing_key="")

    assert decrypt_if_encrypted_with(base64.urlsafe_b64encode(written_with_empty_key).decode(), "") == "stored-secret"


_NESTED_PARAMS = {
    "model": "openai/gpt-5.4-mini",
    "extra_headers": {"Authorization": "Bearer gateway-secret", "X-Trace": "trace-1"},
    "fallbacks": [{"gpt-5.5-mini": ["gpt-5.4-mini"]}],
    "rpm": 10,
    "enabled": True,
    "api_base": None,
}


def _nested_string_leaves(stored: dict) -> list:
    return [
        stored["model"],
        stored["extra_headers"]["Authorization"],
        stored["extra_headers"]["X-Trace"],
        stored["fallbacks"][0]["gpt-5.5-mini"][0],
    ]


@pytest.mark.parametrize("use_aes", [False, True])
def test_encrypt_json_strings_encrypts_every_nested_string_leaf(monkeypatch, use_aes: bool):
    if use_aes:
        _use_aes(monkeypatch)

    stored = encrypt_json_strings(_NESTED_PARAMS)

    assert (stored["rpm"], stored["enabled"], stored["api_base"]) == (10, True, None)
    plaintexts = _nested_string_leaves(_NESTED_PARAMS)
    assert all(leaf != plain for leaf, plain in zip(_nested_string_leaves(stored), plaintexts))
    assert [decrypt_if_encrypted_with(leaf, SALT_KEY) for leaf in _nested_string_leaves(stored)] == plaintexts
    assert decrypt_json_strings(stored) == _NESTED_PARAMS


def test_encrypt_json_strings_reencrypts_ciphertext_under_the_new_key_without_double_wrapping():
    stored = encrypt_json_strings({"api_key": "top-secret", "nested": {"token": encrypt_value_helper("nested-secret")}})

    rotated = encrypt_json_strings(stored, new_encryption_key="sk-rotated")

    assert decrypt_if_encrypted_with(rotated["api_key"], "sk-rotated") == "top-secret"
    assert decrypt_if_encrypted_with(rotated["nested"]["token"], "sk-rotated") == "nested-secret"
    assert decrypt_if_encrypted_with(rotated["api_key"], SALT_KEY) is None


def test_encrypt_json_strings_treats_a_legacy_plaintext_row_as_plaintext():
    legacy_row = {"api_key": "plain-secret", "extra_headers": {"Authorization": "Bearer plain"}}

    assert decrypt_json_strings(legacy_row) == legacy_row
    assert decrypt_json_strings(encrypt_json_strings(legacy_row)) == legacy_row


def test_encrypt_json_strings_without_any_key_stores_the_value_as_sent(monkeypatch):
    monkeypatch.delenv("LITELLM_SALT_KEY")
    monkeypatch.setattr(proxy_server, "master_key", None)

    assert encrypt_json_strings(_NESTED_PARAMS) == _NESTED_PARAMS
    assert decrypt_json_strings(_NESTED_PARAMS) == _NESTED_PARAMS


def test_encrypt_json_strings_refuses_a_value_nested_past_the_recursion_cap():
    value = {"secret": "deep-secret"}
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH + 1):
        value = {"child": value}

    with pytest.raises(ValueError, match=f"nested deeper than {DEFAULT_MAX_RECURSE_DEPTH} levels"):
        encrypt_json_strings(value)
    assert decrypt_json_strings(value) == value


def test_encrypt_json_strings_encrypts_a_value_nested_at_the_recursion_cap():
    value = "deep-secret"
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH):
        value = {"child": value}

    stored = encrypt_json_strings(value)

    deepest = stored
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH):
        deepest = deepest["child"]
    assert deepest != "deep-secret"
    assert decrypt_json_strings(stored) == value


def test_json_value_coerces_non_json_containers():
    assert json_value(MappingProxyType({"headers": ("a", "b")})) == {"headers": ["a", "b"]}


@pytest.mark.parametrize("stored", ["not json", "[1]", ["a"], None, 3])
def test_decrypt_stored_json_object_returns_an_empty_object_for_a_non_object_row(stored: object):
    assert decrypt_stored_json_object(stored) == {}


def test_decrypt_stored_json_object_reads_json_text_and_parsed_rows_alike():
    stored = encrypt_json_strings(_NESTED_PARAMS)

    assert decrypt_stored_json_object(json.dumps(stored)) == _NESTED_PARAMS
    assert decrypt_stored_json_object(stored) == _NESTED_PARAMS


def test_encrypt_config_section_encrypts_router_settings_string_leaves_only():
    router_settings = {
        "redis_password": "redis-pw",
        "num_retries": 2,
        "fallbacks": [{"gpt-5.5-mini": ["gpt-5.4-mini"]}],
    }

    stored = encrypt_config_section("router_settings", router_settings)

    assert stored["num_retries"] == 2
    assert stored["redis_password"] != "redis-pw"
    assert decrypt_if_encrypted_with(stored["redis_password"], SALT_KEY) == "redis-pw"
    assert decrypt_if_encrypted_with(stored["fallbacks"][0]["gpt-5.5-mini"][0], SALT_KEY) == "gpt-5.4-mini"
    assert decrypt_config_section("router_settings", json.dumps(stored)) == router_settings


def test_encrypt_config_section_leaves_other_sections_readable():
    general_settings = {"alerting": ["slack"], "proxy_batch_write_at": 60}

    assert encrypt_config_section("general_settings", general_settings) == general_settings
    assert decrypt_config_section("general_settings", json.dumps(general_settings)) == general_settings


def test_encrypt_config_section_rekeys_router_settings_for_master_key_rotation():
    stored = encrypt_config_section("router_settings", {"redis_password": "redis-pw"})

    rotated = encrypt_config_section("router_settings", stored, new_encryption_key="sk-rotated")

    assert decrypt_if_encrypted_with(rotated["redis_password"], "sk-rotated") == "redis-pw"


def test_decrypt_json_strings_reads_values_encrypted_under_the_given_key():
    stored = encrypt_json_strings(_NESTED_PARAMS, new_encryption_key="sk-other-key")

    assert decrypt_json_strings(stored, signing_key="sk-other-key") == _NESTED_PARAMS
    assert decrypt_json_strings(stored)["extra_headers"]["Authorization"] != "Bearer gateway-secret"


def test_encrypt_stored_json_object_rekeys_a_row_stored_as_a_json_string():
    stored_row = json.dumps(encrypt_json_strings({"api_key": "vendor-secret", "default_on": False}))

    rotated = encrypt_stored_json_object(stored_row, new_encryption_key="sk-rotated")

    assert rotated is not None
    assert decrypt_if_encrypted_with(rotated["api_key"], "sk-rotated") == "vendor-secret"
    assert rotated["default_on"] is False


@pytest.mark.parametrize("stored_row", ("not json", json.dumps(["api_key"]), 7, None))
def test_encrypt_stored_json_object_returns_none_for_a_row_that_is_not_a_json_object(stored_row):
    assert encrypt_stored_json_object(stored_row, new_encryption_key="sk-rotated") is None


def test_decrypt_stored_json_object_reports_a_malformed_row_without_its_contents(caplog, monkeypatch):
    import logging

    from litellm._logging import verbose_proxy_logger

    monkeypatch.setattr(verbose_proxy_logger, "propagate", True)
    caplog.set_level(logging.ERROR, logger=verbose_proxy_logger.name)

    assert decrypt_stored_json_object("leaked-secret-text") == {}
    assert decrypt_stored_json_object(None) == {}

    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert "leaked-secret-text" not in caplog.text
