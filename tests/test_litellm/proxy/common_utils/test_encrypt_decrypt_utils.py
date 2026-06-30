"""
Tests for the at-rest credential encryption chokepoint.

Covers the AES-256-GCM (``v2:gcm:``) path, the ``encryption_algorithm`` config
gate, and the backward-compatibility guarantees that let legacy XSalsa20-Poly1305
(nacl) ciphertext and new AES values coexist and decrypt correctly.
"""

import pytest

from litellm.proxy import proxy_server
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    _V2_GCM_PREFIX,
    decrypt_value_helper,
    encrypt_value_helper,
)


def _use_aes(monkeypatch):
    """Flip the write-time algorithm to AES-256-GCM for the duration of a test."""
    monkeypatch.setattr(
        proxy_server, "general_settings", {"encryption_algorithm": "aes-256-gcm"}
    )


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    # Dominant convention in the test_litellm/ tree: set the key via env.
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-aes-1234")
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
    assert (
        decrypt_value_helper(
            garbled, key="t", exception_type="debug", return_original_value=True
        )
        == garbled
    )


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
    monkeypatch.setattr(
        proxy_server, "general_settings", {"encryption_algorithm": "rot13"}
    )

    ct = encrypt_value_helper("secret")
    assert not ct.startswith(_V2_GCM_PREFIX)
    assert decrypt_value_helper(ct, key="t") == "secret"
