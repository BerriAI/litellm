"""Tests for password hashing and verification utilities."""

import base64
import hashlib
import logging
import os

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import hash_password, needs_password_rehash, verify_password


def _scrypt_row(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return "scrypt:" + base64.b64encode(salt + derived).decode()


def _pbkdf2_row(password: str, iterations: int) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2:sha256:{iterations}:{base64.b64encode(salt).decode()}:{base64.b64encode(derived).decode()}"


class TestHashPassword:
    def test_produces_pbkdf2_prefix_at_owasp_floor(self):
        h = hash_password("test")
        assert h.startswith("pbkdf2:sha256:600000:")
        iterations = int(h.split(":")[2])
        assert iterations >= 600_000

    def test_unique_salt_per_call(self):
        assert hash_password("same") != hash_password("same")

    def test_round_trip_and_wrong_password(self):
        h = hash_password("correct")
        assert verify_password("correct", h) is True
        assert verify_password("wrong", h) is False

    def test_empty_password(self):
        h = hash_password("")
        assert verify_password("", h) is True
        assert verify_password("notempty", h) is False

    def test_unicode_password(self):
        h = hash_password("pässwörd")
        assert verify_password("pässwörd", h) is True
        assert verify_password("password", h) is False

    def test_long_password(self):
        pw = "a" * 1000
        h = hash_password(pw)
        assert verify_password(pw, h) is True


class TestVerifyPasswordFormats:
    def test_row_with_higher_iteration_count_verifies(self):
        stored = _pbkdf2_row("iterated", 700_000)
        assert verify_password("iterated", stored) is True
        assert verify_password("other", stored) is False

    def test_scrypt_row_verifies_when_fips_off(self, monkeypatch):
        monkeypatch.delenv("LITELLM_FIPS_MODE", raising=False)
        stored = _scrypt_row("legacy-scrypt-pass")
        assert verify_password("legacy-scrypt-pass", stored) is True
        assert verify_password("wrong", stored) is False

    def test_scrypt_row_rejected_and_logged_when_fips_on(self, monkeypatch, caplog):
        monkeypatch.setenv("LITELLM_FIPS_MODE", "true")
        stored = _scrypt_row("legacy-scrypt-pass")
        with caplog.at_level(logging.ERROR, logger=verbose_proxy_logger.name):
            assert verify_password("legacy-scrypt-pass", stored) is False
        assert "scrypt" in caplog.text
        assert "/user/update" in caplog.text

    def test_sha256_fallback_verifies_in_both_modes(self, monkeypatch):
        stored = hashlib.sha256(b"oldpass").hexdigest()
        monkeypatch.delenv("LITELLM_FIPS_MODE", raising=False)
        assert verify_password("oldpass", stored) is True
        assert verify_password("wrong", stored) is False
        monkeypatch.setenv("LITELLM_FIPS_MODE", "true")
        assert verify_password("oldpass", stored) is True

    @pytest.mark.parametrize(
        "stored",
        (
            "pbkdf2:sha256:600000",
            "pbkdf2:sha256:600000:c2FsdA==",
            "pbkdf2:sha256:600000:c2FsdA==:a2V5:extra",
            "pbkdf2:sha256:not-an-int:c2FsdA==:a2V5",
            "pbkdf2:sha256:600000:not-base64-!!:a2V5",
            "pbkdf2:sha256:600000:c2FsdA==:not-base64-!!",
            "pbkdf2:md5:600000:c2FsdA==:a2V5",
        ),
        ids=(
            "missing_fields",
            "four_fields",
            "six_fields",
            "non_int_iterations",
            "bad_salt_base64",
            "bad_key_base64",
            "wrong_digest",
        ),
    )
    def test_malformed_pbkdf2_rows_return_false(self, stored):
        assert verify_password("test", stored) is False

    @pytest.mark.parametrize(
        "iterations",
        (10_000_001, 10**30, 0),
        ids=("above_max", "overflows_c_long", "zero"),
    )
    def test_out_of_range_iteration_counts_return_false(self, iterations):
        stored = _pbkdf2_row("test", 600_000).replace(":600000:", f":{iterations}:")
        assert verify_password("test", stored) is False

    def test_scrypt_invalid_base64_rejected(self):
        assert verify_password("test", "scrypt:not-valid-base64!!!") is False


class TestVerifyPasswordFallbacks:
    def test_sha256_fallback(self):
        stored = hashlib.sha256(b"oldpass").hexdigest()
        assert verify_password("oldpass", stored) is True
        assert verify_password("wrong", stored) is False

    def test_no_plaintext_fallback(self):
        # Plaintext fallback removed to prevent pass-the-hash attacks
        assert verify_password("plaintext", "plaintext") is False

    def test_sha256_not_confused_with_plaintext(self):
        # A 64-char hex string that isn't a valid SHA256 of the password
        fake_hex = "a" * 64
        assert verify_password("test", fake_hex) is False


class TestNeedsPasswordRehash:
    def test_legacy_rows_need_rehash(self):
        assert needs_password_rehash(_scrypt_row("x")) is True
        assert needs_password_rehash(hashlib.sha256(b"x").hexdigest()) is True
        assert needs_password_rehash("plaintext") is True

    def test_pbkdf2_rows_do_not_need_rehash(self):
        assert needs_password_rehash(hash_password("x")) is False
        assert needs_password_rehash(_pbkdf2_row("x", 700_000)) is False
