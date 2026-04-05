"""
Tests for _strip_credential_value and whitespace handling in credential storage.
"""

import base64
import json
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._experimental.mcp_server.db import (
    _strip_credential_value,
    encrypt_credentials,
    store_user_credential,
    store_user_oauth_credential,
)


class TestStripCredentialValue:
    def test_returns_none_for_none(self):
        assert _strip_credential_value(None) is None

    def test_no_change_for_clean_value(self):
        token = "ya29.a0ATkoCc6VikZRWgeS5J9dd_gxP9wCmAARKc7m1lAjbF94"
        assert _strip_credential_value(token) == token

    def test_strips_leading_trailing_spaces(self):
        assert _strip_credential_value("  my-token  ") == "my-token"

    def test_strips_leading_trailing_newlines(self):
        assert _strip_credential_value("\nmy-token\n") == "my-token"

    def test_strips_embedded_newline_from_line_wrap(self):
        """Simulates terminal line-wrapping that inserts newline in the middle."""
        token_with_wrap = "ya29.a0ATkoCc6Vik\nZRWgeS5J9dd_gxP9w"
        expected = "ya29.a0ATkoCc6VikZRWgeS5J9dd_gxP9w"
        assert _strip_credential_value(token_with_wrap) == expected

    def test_strips_crlf_line_endings(self):
        """Handles Windows-style CRLF line endings from copy-paste."""
        assert _strip_credential_value("token\r\nvalue") == "tokenvalue"
        assert _strip_credential_value("abc\r\n  def") == "abc  def"

    def test_strips_tabs(self):
        assert _strip_credential_value("my\ttoken") == "mytoken"

    def test_preserves_internal_spaces(self):
        """Internal spaces are preserved for passphrase-style credentials."""
        assert _strip_credential_value("my secret passphrase") == "my secret passphrase"

    def test_empty_string(self):
        assert _strip_credential_value("") is None

    def test_whitespace_only_returns_none(self):
        """All-whitespace input should be treated as absent."""
        assert _strip_credential_value("   ") is None
        assert _strip_credential_value("\n\t ") is None


class TestEncryptCredentialsStripsWhitespace:
    """Verify that encrypt_credentials strips whitespace before encrypting."""

    @patch(
        "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
        side_effect=lambda value, new_encryption_key: value,
    )
    def test_auth_value_newlines_stripped(self, mock_encrypt):
        """Ensure newlines in auth_value are removed before encryption."""
        creds = {"auth_value": "ya29.abc123\ndef456ghi789"}
        result = encrypt_credentials(creds, encryption_key=None)
        assert result["auth_value"] == "ya29.abc123def456ghi789"

    @patch(
        "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
        side_effect=lambda value, new_encryption_key: value,
    )
    def test_client_id_and_secret_stripped(self, mock_encrypt):
        creds = {"client_id": " my-client-id \n", "client_secret": "\n secret \t"}
        result = encrypt_credentials(creds, encryption_key=None)
        assert result["client_id"] == "my-client-id"
        assert result["client_secret"] == "secret"

    @patch(
        "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
        side_effect=lambda value, new_encryption_key: value,
    )
    def test_aws_fields_stripped(self, mock_encrypt):
        creds = {
            "aws_access_key_id": " AKIA123 ",
            "aws_secret_access_key": "secret\nkey",
            "aws_session_token": " token\n\tvalue ",
        }
        result = encrypt_credentials(creds, encryption_key=None)
        assert result["aws_access_key_id"] == "AKIA123"
        assert result["aws_secret_access_key"] == "secretkey"
        assert result["aws_session_token"] == "tokenvalue"

    @patch(
        "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
        side_effect=lambda value, new_encryption_key: value,
    )
    def test_whitespace_only_field_removed_from_dict(self, mock_encrypt):
        """Whitespace-only values must be removed, not left unencrypted."""
        creds = {"auth_value": "   \n  ", "client_id": "valid-id"}
        result = encrypt_credentials(creds, encryption_key=None)
        assert "auth_value" not in result
        assert result["client_id"] == "valid-id"


class TestStoreUserCredentialStripsWhitespace:
    """Verify that store_user_credential strips whitespace."""

    @pytest.mark.asyncio
    async def test_strips_whitespace_before_storing(self):
        mock_prisma = AsyncMock()
        await store_user_credential(mock_prisma, "user1", "server1", " my-token\n ")
        call_args = mock_prisma.db.litellm_mcpusercredentials.upsert.call_args
        create_data = call_args.kwargs["data"]["create"]
        stored = base64.urlsafe_b64decode(create_data["credential_b64"]).decode()
        assert stored == "my-token"

    @pytest.mark.asyncio
    async def test_rejects_whitespace_only_credential(self):
        mock_prisma = AsyncMock()
        with pytest.raises(ValueError, match="must not be empty"):
            await store_user_credential(mock_prisma, "user1", "server1", "   \n  ")


class TestStoreUserOAuthCredentialStripsWhitespace:
    """Verify that store_user_oauth_credential strips whitespace."""

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_tokens(self):
        mock_prisma = AsyncMock()
        mock_prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(
            return_value=None
        )
        await store_user_oauth_credential(
            mock_prisma,
            "user1",
            "server1",
            access_token=" token\n123 ",
            refresh_token=" refresh\n456 ",
        )
        call_args = mock_prisma.db.litellm_mcpusercredentials.upsert.call_args
        create_data = call_args.kwargs["data"]["create"]
        payload = json.loads(
            base64.urlsafe_b64decode(create_data["credential_b64"]).decode()
        )
        assert payload["access_token"] == "token123"
        assert payload["refresh_token"] == "refresh456"

    @pytest.mark.asyncio
    async def test_rejects_whitespace_only_access_token(self):
        mock_prisma = AsyncMock()
        with pytest.raises(ValueError, match="must not be empty"):
            await store_user_oauth_credential(
                mock_prisma, "user1", "server1", access_token="   "
            )

    @pytest.mark.asyncio
    async def test_whitespace_only_refresh_token_discarded(self):
        """Whitespace-only refresh_token should be silently discarded."""
        mock_prisma = AsyncMock()
        mock_prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(
            return_value=None
        )
        await store_user_oauth_credential(
            mock_prisma,
            "user1",
            "server1",
            access_token="valid-token",
            refresh_token="   \n  ",
        )
        call_args = mock_prisma.db.litellm_mcpusercredentials.upsert.call_args
        create_data = call_args.kwargs["data"]["create"]
        payload = json.loads(
            base64.urlsafe_b64decode(create_data["credential_b64"]).decode()
        )
        assert "refresh_token" not in payload
