"""Tests for password hashing and verification utilities."""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.utils import (
    _is_already_hashed,
    hash_password,
    verify_password,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_return(value):
    """Return an async mock that resolves to `value`."""
    mock = AsyncMock(return_value=value)
    return mock


def _make_scrypt_hash(password: str) -> str:
    import base64
    import os

    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return "scrypt:" + base64.b64encode(salt + dk).decode()


# ---------------------------------------------------------------------------
# hash_password
# ---------------------------------------------------------------------------


class TestHashPassword:
    def test_produces_pbkdf2_prefix(self):
        assert hash_password("test").startswith("pbkdf2:")

    def test_unique_salt_per_call(self):
        assert hash_password("same") != hash_password("same")

    def test_output_length(self):
        # "pbkdf2:" (7) + base64(48 bytes) (64) = 71
        assert len(hash_password("test")) == 71

    def test_no_longer_produces_scrypt_prefix(self):
        assert not hash_password("test").startswith("scrypt:")

    def test_empty_password(self):
        h = hash_password("")
        assert h.startswith("pbkdf2:")

    def test_unicode_password(self):
        h = hash_password("pässwörd")
        assert h.startswith("pbkdf2:")
        assert verify_password("pässwörd", h) is True

    def test_long_password(self):
        pw = "a" * 1000
        h = hash_password(pw)
        assert h.startswith("pbkdf2:")
        assert verify_password(pw, h) is True


# ---------------------------------------------------------------------------
# verify_password — pbkdf2 hashes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# verify_password — legacy fallbacks
# ---------------------------------------------------------------------------


class TestVerifyPasswordFallbacks:
    def test_sha256_fallback(self):
        stored = hashlib.sha256("oldpass".encode()).hexdigest()
        assert verify_password("oldpass", stored) is True
        assert verify_password("wrong", stored) is False

    def test_no_plaintext_fallback(self):
        assert verify_password("plaintext", "plaintext") is False

    def test_pbkdf2_preferred_over_fallbacks(self):
        h = hash_password("test")
        assert verify_password("test", h) is True
        assert h.startswith("pbkdf2:")

    def test_sha256_not_confused_with_plaintext(self):
        fake_hex = "a" * 64
        assert verify_password("test", fake_hex) is False

    def test_pbkdf2_invalid_base64_rejected(self):
        assert verify_password("test", "pbkdf2:not-valid-base64!!!") is False

    def test_scrypt_invalid_base64_rejected(self):
        assert verify_password("test", "scrypt:not-valid-base64!!!") is False

    def test_unknown_prefix_rejected(self):
        assert verify_password("test", "argon2:somehash") is False


# ---------------------------------------------------------------------------
# verify_password — scrypt backward compatibility
# ---------------------------------------------------------------------------


class TestScryptBackwardCompatibility:
    """Ensure existing scrypt hashes can still be verified (backward compat)."""

    def test_scrypt_hash_verifies_correctly(self):
        h = _make_scrypt_hash("mypassword")
        assert verify_password("mypassword", h) is True

    def test_scrypt_hash_rejects_wrong_password(self):
        h = _make_scrypt_hash("mypassword")
        assert verify_password("wrong", h) is False

    def test_scrypt_unavailable_in_fips_mode_returns_false(self):
        """When scrypt raises ValueError (FIPS mode), verify_password returns False
        instead of propagating the exception."""
        import base64

        salt = b"\x00" * 16
        dummy_dk = b"\x00" * 32
        stored = "scrypt:" + base64.b64encode(salt + dummy_dk).decode()

        with patch("hashlib.scrypt", side_effect=ValueError("unsupported")):
            result = verify_password("somepassword", stored)

        assert result is False

    def test_scrypt_unavailable_logs_warning(self, caplog):
        import base64
        import logging

        salt = b"\x00" * 16
        dummy_dk = b"\x00" * 32
        stored = "scrypt:" + base64.b64encode(salt + dummy_dk).decode()

        with patch("hashlib.scrypt", side_effect=ValueError("unsupported")):
            with caplog.at_level(logging.WARNING):
                verify_password("somepassword", stored)

        assert any("FIPS" in r.message or "scrypt" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _is_already_hashed
# ---------------------------------------------------------------------------


class TestIsAlreadyHashed:
    @pytest.mark.parametrize(
        "stored,expected",
        [
            ("pbkdf2:abc", True),
            ("scrypt:abc", True),
            ("a" * 64, True),
            ("plaintext", False),
            ("short", False),
            ("a" * 63, False),
            ("g" * 64, False),  # 64 chars but not valid hex
        ],
    )
    def test_detection(self, stored, expected):
        assert _is_already_hashed(stored) is expected


# ---------------------------------------------------------------------------
# migrate_passwords_to_pbkdf2_async
# ---------------------------------------------------------------------------


class TestMigratePasswordsToPbkdf2:
    @pytest.mark.asyncio
    async def test_migrates_plaintext_passwords(self):
        from litellm.proxy.utils import migrate_passwords_to_pbkdf2_async

        user = MagicMock()
        user.user_id = "u1"
        user.password = "plaintext"

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_many = _async_return([user])
        prisma_client.db.litellm_usertable.update = _async_return(None)

        result = await migrate_passwords_to_pbkdf2_async(prisma_client)

        assert "1" in result
        call_args = prisma_client.db.litellm_usertable.update.call_args
        new_pw = call_args.kwargs["data"]["password"]
        assert new_pw.startswith("pbkdf2:")

    @pytest.mark.asyncio
    async def test_skips_pbkdf2_hashed_passwords(self):
        from litellm.proxy.utils import migrate_passwords_to_pbkdf2_async

        user = MagicMock()
        user.user_id = "u1"
        user.password = hash_password("already-hashed")

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_many = _async_return([user])
        prisma_client.db.litellm_usertable.update = _async_return(None)

        result = await migrate_passwords_to_pbkdf2_async(prisma_client)

        assert "No plaintext" in result
        prisma_client.db.litellm_usertable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_scrypt_hashed_passwords(self):
        """Scrypt hashes are NOT touched at startup; they migrate on next login."""
        from litellm.proxy.utils import migrate_passwords_to_pbkdf2_async

        scrypt_hash = _make_scrypt_hash("pw")
        user = MagicMock()
        user.user_id = "u1"
        user.password = scrypt_hash

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_many = _async_return([user])
        prisma_client.db.litellm_usertable.update = _async_return(None)

        result = await migrate_passwords_to_pbkdf2_async(prisma_client)

        assert "No plaintext" in result
        prisma_client.db.litellm_usertable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_sha256_hashed_passwords(self):
        """SHA256 hashes are NOT touched at startup; they migrate on next login."""
        from litellm.proxy.utils import migrate_passwords_to_pbkdf2_async

        sha256_hash = hashlib.sha256(b"pw").hexdigest()
        user = MagicMock()
        user.user_id = "u1"
        user.password = sha256_hash

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_many = _async_return([user])
        prisma_client.db.litellm_usertable.update = _async_return(None)

        result = await migrate_passwords_to_pbkdf2_async(prisma_client)

        assert "No plaintext" in result
        prisma_client.db.litellm_usertable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_users_with_passwords(self):
        from litellm.proxy.utils import migrate_passwords_to_pbkdf2_async

        prisma_client = MagicMock()
        prisma_client.db.litellm_usertable.find_many = _async_return([])
        prisma_client.db.litellm_usertable.update = _async_return(None)

        result = await migrate_passwords_to_pbkdf2_async(prisma_client)

        assert "No plaintext" in result

    @pytest.mark.asyncio
    async def test_backward_compat_alias_still_works(self):
        """migrate_passwords_to_scrypt_async is kept as an alias."""
        from litellm.proxy.utils import (
            migrate_passwords_to_pbkdf2_async,
            migrate_passwords_to_scrypt_async,
        )

        assert migrate_passwords_to_scrypt_async is migrate_passwords_to_pbkdf2_async


# ---------------------------------------------------------------------------
# _rehash_password_if_needed (login-time rehash helper)
# ---------------------------------------------------------------------------


def _make_mock_proxy_server(prisma_client_value):
    """Build a minimal sys.modules stub for litellm.proxy.proxy_server."""
    import types

    mod = types.ModuleType("litellm.proxy.proxy_server")
    mod.prisma_client = prisma_client_value
    return mod


class TestRehashPasswordIfNeeded:
    """_rehash_password_if_needed lazy-imports prisma_client from proxy_server,
    so we inject a mock module into sys.modules to avoid the heavy import chain."""

    @pytest.mark.asyncio
    async def test_does_not_rehash_pbkdf2_password(self, monkeypatch):
        import sys

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.update = _async_return(None)

        monkeypatch.setitem(
            sys.modules,
            "litellm.proxy.proxy_server",
            _make_mock_proxy_server(mock_prisma),
        )

        from litellm.proxy.auth.login_utils import _rehash_password_if_needed

        stored = hash_password("pw")
        assert stored.startswith("pbkdf2:")

        await _rehash_password_if_needed("uid", "pw", stored)

        mock_prisma.db.litellm_usertable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_rehashes_scrypt_to_pbkdf2_on_login(self, monkeypatch):
        import sys

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.update = _async_return(None)

        monkeypatch.setitem(
            sys.modules,
            "litellm.proxy.proxy_server",
            _make_mock_proxy_server(mock_prisma),
        )

        from litellm.proxy.auth.login_utils import _rehash_password_if_needed

        scrypt_stored = _make_scrypt_hash("pw")
        await _rehash_password_if_needed("uid", "pw", scrypt_stored)

        mock_prisma.db.litellm_usertable.update.assert_called_once()
        new_pw = mock_prisma.db.litellm_usertable.update.call_args.kwargs["data"][
            "password"
        ]
        assert new_pw.startswith("pbkdf2:")

    @pytest.mark.asyncio
    async def test_rehashes_sha256_to_pbkdf2_on_login(self, monkeypatch):
        import sys

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.update = _async_return(None)

        monkeypatch.setitem(
            sys.modules,
            "litellm.proxy.proxy_server",
            _make_mock_proxy_server(mock_prisma),
        )

        from litellm.proxy.auth.login_utils import _rehash_password_if_needed

        sha256_stored = hashlib.sha256(b"pw").hexdigest()
        await _rehash_password_if_needed("uid", "pw", sha256_stored)

        mock_prisma.db.litellm_usertable.update.assert_called_once()
        new_pw = mock_prisma.db.litellm_usertable.update.call_args.kwargs["data"][
            "password"
        ]
        assert new_pw.startswith("pbkdf2:")
