"""Tests for FIPS-compliant encryption and password hashing.

Covers:
  - AES-256-GCM encryption/decryption (encrypt_decrypt_utils)
  - PBKDF2-HMAC-SHA256 password hashing (proxy/utils)
  - Backward-compatibility: decrypt legacy nacl values in non-FIPS mode
  - Cross-format password verification (pbkdf2, scrypt, sha256)
  - _rehash_password_if_needed triggers correctly in both modes
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_fips(monkeypatch, enabled: bool):
    monkeypatch.setenv("LITELLM_FIPS_MODE", "true" if enabled else "false")


# ---------------------------------------------------------------------------
# AES-256-GCM low-level helpers
# ---------------------------------------------------------------------------


class TestAesGcmHelpers:
    def test_roundtrip(self):
        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            _decrypt_aes_gcm,
            _encrypt_aes_gcm,
        )

        plaintext = "hello FIPS world"
        signing_key = "my-secret-key"
        ciphertext = _encrypt_aes_gcm(plaintext, signing_key)
        assert isinstance(ciphertext, bytes)
        assert _decrypt_aes_gcm(ciphertext, signing_key) == plaintext

    def test_wrong_key_raises(self):
        from cryptography.exceptions import InvalidTag

        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            _decrypt_aes_gcm,
            _encrypt_aes_gcm,
        )

        ciphertext = _encrypt_aes_gcm("secret", "key-A")
        with pytest.raises(InvalidTag):
            _decrypt_aes_gcm(ciphertext, "key-B")

    def test_nonce_is_random(self):
        from litellm.proxy.common_utils.encrypt_decrypt_utils import _encrypt_aes_gcm

        c1 = _encrypt_aes_gcm("same", "key")
        c2 = _encrypt_aes_gcm("same", "key")
        # Different nonces → different ciphertexts
        assert c1 != c2

    def test_short_ciphertext_raises(self):
        from litellm.proxy.common_utils.encrypt_decrypt_utils import _decrypt_aes_gcm

        with pytest.raises(ValueError, match="too short"):
            _decrypt_aes_gcm(b"\x00" * 5, "key")

    def test_unicode_roundtrip(self):
        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            _decrypt_aes_gcm,
            _encrypt_aes_gcm,
        )

        plaintext = "pässwörd 🔐"
        ct = _encrypt_aes_gcm(plaintext, "key")
        assert _decrypt_aes_gcm(ct, "key") == plaintext


# ---------------------------------------------------------------------------
# encrypt_value_helper / decrypt_value_helper — FIPS mode
# ---------------------------------------------------------------------------


class TestEncryptDecryptHelperFipsMode:
    def test_fips_mode_uses_aes_gcm_prefix(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.common_utils import encrypt_decrypt_utils

        monkeypatch.setattr(encrypt_decrypt_utils, "_get_salt_key", lambda: "test-key")
        result = encrypt_decrypt_utils.encrypt_value_helper("my-secret")
        assert result.startswith("aes256gcm:")

    def test_fips_mode_roundtrip(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.common_utils import encrypt_decrypt_utils

        monkeypatch.setattr(encrypt_decrypt_utils, "_get_salt_key", lambda: "test-key")
        encrypted = encrypt_decrypt_utils.encrypt_value_helper("my-api-key")
        decrypted = encrypt_decrypt_utils.decrypt_value_helper(
            value=encrypted, key="any"
        )
        assert decrypted == "my-api-key"

    def test_non_fips_uses_nacl_format(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.common_utils import encrypt_decrypt_utils

        monkeypatch.setattr(encrypt_decrypt_utils, "_get_salt_key", lambda: "test-key")
        result = encrypt_decrypt_utils.encrypt_value_helper("my-secret")
        assert not result.startswith("aes256gcm:")

    def test_non_fips_roundtrip(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.common_utils import encrypt_decrypt_utils

        monkeypatch.setattr(encrypt_decrypt_utils, "_get_salt_key", lambda: "test-key")
        encrypted = encrypt_decrypt_utils.encrypt_value_helper("hello")
        decrypted = encrypt_decrypt_utils.decrypt_value_helper(
            value=encrypted, key="any"
        )
        assert decrypted == "hello"

    def test_decrypt_aes_gcm_values_in_non_fips_mode(self, monkeypatch):
        """Values encrypted in FIPS mode can be decrypted in non-FIPS mode."""
        from litellm.proxy.common_utils import encrypt_decrypt_utils

        monkeypatch.setattr(
            encrypt_decrypt_utils, "_get_salt_key", lambda: "shared-key"
        )

        # Encrypt with FIPS mode ON
        _set_fips(monkeypatch, True)
        encrypted = encrypt_decrypt_utils.encrypt_value_helper("cross-mode-value")

        # Decrypt with FIPS mode OFF
        _set_fips(monkeypatch, False)
        decrypted = encrypt_decrypt_utils.decrypt_value_helper(
            value=encrypted, key="any"
        )
        assert decrypted == "cross-mode-value"

    def test_decrypt_nacl_values_in_fips_mode(self, monkeypatch):
        """Legacy nacl-encrypted values can still be decrypted when FIPS mode is on."""
        from litellm.proxy.common_utils import encrypt_decrypt_utils

        monkeypatch.setattr(
            encrypt_decrypt_utils, "_get_salt_key", lambda: "shared-key"
        )

        # Encrypt with FIPS mode OFF (nacl)
        _set_fips(monkeypatch, False)
        encrypted = encrypt_decrypt_utils.encrypt_value_helper("legacy-value")

        # Decrypt with FIPS mode ON
        _set_fips(monkeypatch, True)
        decrypted = encrypt_decrypt_utils.decrypt_value_helper(
            value=encrypted, key="any"
        )
        assert decrypted == "legacy-value"

    def test_non_string_value_returned_unchanged(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.common_utils import encrypt_decrypt_utils

        monkeypatch.setattr(encrypt_decrypt_utils, "_get_salt_key", lambda: "test-key")
        result = encrypt_decrypt_utils.encrypt_value_helper(123)  # type: ignore
        assert result == 123

    def test_decrypt_error_returns_none_by_default(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.common_utils import encrypt_decrypt_utils

        monkeypatch.setattr(encrypt_decrypt_utils, "_get_salt_key", lambda: "test-key")
        result = encrypt_decrypt_utils.decrypt_value_helper(
            value="notvalidbase64!!!", key="k"
        )
        assert result is None


# ---------------------------------------------------------------------------
# is_fips_mode detection
# ---------------------------------------------------------------------------


class TestIsFipsMode:
    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("LITELLM_FIPS_MODE", val)
        from litellm.proxy.common_utils.encrypt_decrypt_utils import _is_fips_mode

        assert _is_fips_mode() is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "", "off"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("LITELLM_FIPS_MODE", val)
        from litellm.proxy.common_utils.encrypt_decrypt_utils import _is_fips_mode

        assert _is_fips_mode() is False

    def test_unset_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv("LITELLM_FIPS_MODE", raising=False)
        from litellm.proxy.common_utils.encrypt_decrypt_utils import _is_fips_mode

        assert _is_fips_mode() is False


# ---------------------------------------------------------------------------
# Password hashing — FIPS mode (PBKDF2)
# ---------------------------------------------------------------------------


class TestHashPasswordFipsMode:
    def test_fips_mode_produces_pbkdf2_prefix(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password

        result = hash_password("secret")
        assert result.startswith("pbkdf2:")

    def test_non_fips_mode_produces_scrypt_prefix(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.utils import hash_password

        result = hash_password("secret")
        assert result.startswith("scrypt:")

    def test_pbkdf2_unique_salt_per_call(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password

        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_pbkdf2_output_length(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password

        h = hash_password("test")
        # "pbkdf2:" (7) + base64(16+32=48 bytes) = 7 + 64 = 71
        assert len(h) == 71

    def test_pbkdf2_correct_password_verifies(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password, verify_password

        h = hash_password("correct")
        assert verify_password("correct", h) is True

    def test_pbkdf2_wrong_password_fails(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password, verify_password

        h = hash_password("correct")
        assert verify_password("wrong", h) is False

    def test_pbkdf2_empty_password(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password, verify_password

        h = hash_password("")
        assert verify_password("", h) is True
        assert verify_password("notempty", h) is False

    def test_pbkdf2_unicode_password(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password, verify_password

        h = hash_password("pässwörd")
        assert verify_password("pässwörd", h) is True
        assert verify_password("password", h) is False


# ---------------------------------------------------------------------------
# verify_password — cross-format compatibility
# ---------------------------------------------------------------------------


class TestVerifyPasswordCrossFormat:
    """Ensure verify_password handles all stored hash formats correctly."""

    def test_verify_pbkdf2_format(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password, verify_password

        h = hash_password("pw")
        assert h.startswith("pbkdf2:")
        assert verify_password("pw", h) is True

    def test_verify_scrypt_format(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.utils import hash_password, verify_password

        h = hash_password("pw")
        assert h.startswith("scrypt:")
        assert verify_password("pw", h) is True

    def test_verify_sha256_fallback(self):
        from litellm.proxy.utils import verify_password

        stored = hashlib.sha256("oldpass".encode()).hexdigest()
        assert verify_password("oldpass", stored) is True
        assert verify_password("wrong", stored) is False

    def test_verify_scrypt_hash_in_fips_mode(self, monkeypatch):
        """Legacy scrypt hashes can still be verified even when FIPS mode is on."""
        _set_fips(monkeypatch, False)
        from litellm.proxy.utils import hash_password

        scrypt_hash = hash_password("mypassword")

        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import verify_password

        assert verify_password("mypassword", scrypt_hash) is True

    def test_verify_pbkdf2_hash_in_non_fips_mode(self, monkeypatch):
        """PBKDF2 hashes can be verified even when FIPS mode is off."""
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password

        pbkdf2_hash = hash_password("mypassword")

        _set_fips(monkeypatch, False)
        from litellm.proxy.utils import verify_password

        assert verify_password("mypassword", pbkdf2_hash) is True

    def test_invalid_pbkdf2_base64_rejected(self):
        from litellm.proxy.utils import verify_password

        assert verify_password("test", "pbkdf2:not-valid-base64!!!") is False

    def test_no_plaintext_fallback(self):
        from litellm.proxy.utils import verify_password

        assert verify_password("plaintext", "plaintext") is False

    def test_unknown_format_rejected(self):
        from litellm.proxy.utils import verify_password

        assert verify_password("test", "argon2:somehash") is False


# ---------------------------------------------------------------------------
# migrate_passwords_to_scrypt_async — includes pbkdf2 skip
# ---------------------------------------------------------------------------


class TestMigratePasswordsAsync:
    @pytest.mark.asyncio
    async def test_skips_pbkdf2_passwords(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.utils import hash_password, migrate_passwords_to_scrypt_async

        pbkdf2_hash = hash_password("existing")
        user = MagicMock()
        user.password = pbkdf2_hash
        user.user_id = "u1"

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[user])

        result = await migrate_passwords_to_scrypt_async(prisma_client)
        assert result == "No plaintext passwords found"
        prisma_client.db.litellm_usertable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_scrypt_passwords(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.utils import hash_password, migrate_passwords_to_scrypt_async

        scrypt_hash = hash_password("existing")
        user = MagicMock()
        user.password = scrypt_hash
        user.user_id = "u1"

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[user])

        result = await migrate_passwords_to_scrypt_async(prisma_client)
        assert result == "No plaintext passwords found"

    @pytest.mark.asyncio
    async def test_migrates_plaintext_passwords(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.utils import migrate_passwords_to_scrypt_async

        user = MagicMock()
        user.password = "plaintext_password"
        user.user_id = "u2"

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[user])
        prisma_client.db.litellm_usertable.update = AsyncMock()

        result = await migrate_passwords_to_scrypt_async(prisma_client)
        assert "Migrated 1" in result
        prisma_client.db.litellm_usertable.update.assert_called_once()


# ---------------------------------------------------------------------------
# _rehash_password_if_needed
# ---------------------------------------------------------------------------


class TestRehashPasswordIfNeeded:
    @pytest.mark.asyncio
    async def test_no_rehash_when_pbkdf2_in_fips_mode(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.auth.login_utils import _rehash_password_if_needed

        with patch("litellm.proxy.auth.login_utils.hash_password") as mock_hash:
            await _rehash_password_if_needed("u1", "pw", "pbkdf2:somehash")
            mock_hash.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_rehash_when_scrypt_in_non_fips_mode(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.auth.login_utils import _rehash_password_if_needed

        with patch("litellm.proxy.auth.login_utils.hash_password") as mock_hash:
            await _rehash_password_if_needed("u1", "pw", "scrypt:somehash")
            mock_hash.assert_not_called()

    @pytest.mark.asyncio
    async def test_rehashes_sha256_to_scrypt_in_non_fips_mode(self, monkeypatch):
        _set_fips(monkeypatch, False)
        from litellm.proxy.auth.login_utils import _rehash_password_if_needed

        sha256_stored = hashlib.sha256("pw".encode()).hexdigest()
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.update = AsyncMock()

        # prisma_client is imported lazily inside the function from proxy_server
        with (
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma, create=True),
            patch(
                "litellm.proxy.auth.login_utils.hash_password",
                return_value="scrypt:newhash",
            ) as mock_hash,
        ):
            await _rehash_password_if_needed("u1", "pw", sha256_stored)
            mock_hash.assert_called_once_with("pw")
            mock_prisma.db.litellm_usertable.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_rehashes_scrypt_to_pbkdf2_in_fips_mode(self, monkeypatch):
        _set_fips(monkeypatch, True)
        from litellm.proxy.auth.login_utils import _rehash_password_if_needed

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.update = AsyncMock()

        # prisma_client is imported lazily inside the function from proxy_server
        with (
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma, create=True),
            patch(
                "litellm.proxy.auth.login_utils.hash_password",
                return_value="pbkdf2:newhash",
            ) as mock_hash,
        ):
            await _rehash_password_if_needed("u1", "pw", "scrypt:oldhash")
            mock_hash.assert_called_once_with("pw")
            mock_prisma.db.litellm_usertable.update.assert_called_once()
