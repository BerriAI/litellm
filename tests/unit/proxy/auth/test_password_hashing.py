"""Tests for password hashing and verification utilities."""

import hashlib

import pytest

from litellm.proxy.utils import hash_password, verify_password


class TestHashPassword:
    def test_produces_scrypt_prefix(self):
        assert hash_password("test").startswith("scrypt:")

    def test_unique_salt_per_call(self):
        assert hash_password("same") != hash_password("same")

    def test_output_length(self):
        # "scrypt:" (7) + base64(48 bytes) (64) = 71
        assert len(hash_password("test")) == 71


class TestVerifyPassword:
    def test_correct_password(self):
        h = hash_password("correct")
        assert verify_password("correct", h) is True

    def test_wrong_password(self):
        h = hash_password("correct")
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


class TestVerifyPasswordFallbacks:
    def test_sha256_fallback(self):
        stored = hashlib.sha256("oldpass".encode()).hexdigest()
        assert verify_password("oldpass", stored) is True
        assert verify_password("wrong", stored) is False

    def test_no_plaintext_fallback(self):
        # Plaintext fallback removed to prevent pass-the-hash attacks
        assert verify_password("plaintext", "plaintext") is False

    def test_scrypt_preferred_over_fallbacks(self):
        h = hash_password("test")
        # Scrypt hash should not accidentally match as plaintext or SHA256
        assert verify_password("test", h) is True
        assert h.startswith("scrypt:")

    def test_sha256_not_confused_with_plaintext(self):
        # A 64-char hex string that isn't a valid SHA256 of the password
        fake_hex = "a" * 64
        assert verify_password("test", fake_hex) is False

    def test_scrypt_invalid_base64_rejected(self):
        assert verify_password("test", "scrypt:not-valid-base64!!!") is False
