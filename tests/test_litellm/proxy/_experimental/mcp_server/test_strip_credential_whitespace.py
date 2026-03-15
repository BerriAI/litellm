"""
Tests for _strip_credential_value and whitespace handling in encrypt_credentials.
"""

from unittest.mock import patch

from litellm.proxy._experimental.mcp_server.db import (
    _strip_credential_value,
    encrypt_credentials,
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
        token_with_wrap = "ya29.a0ATkoCc6Vik\n  ZRWgeS5J9dd_gxP9w"
        expected = "ya29.a0ATkoCc6VikZRWgeS5J9dd_gxP9w"
        assert _strip_credential_value(token_with_wrap) == expected

    def test_strips_tabs(self):
        assert _strip_credential_value("my\ttoken") == "mytoken"

    def test_empty_string(self):
        assert _strip_credential_value("") == ""


class TestEncryptCredentialsStripsWhitespace:
    """Verify that encrypt_credentials strips whitespace before encrypting."""

    @patch(
        "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
        side_effect=lambda value, new_encryption_key: value,
    )
    def test_auth_value_whitespace_stripped(self, mock_encrypt):
        """Ensure whitespace in auth_value is removed before encryption."""
        creds = {"auth_value": "ya29.abc123\n  def456ghi789"}
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
            "aws_session_token": " token\n  value ",
        }
        result = encrypt_credentials(creds, encryption_key=None)
        assert result["aws_access_key_id"] == "AKIA123"
        assert result["aws_secret_access_key"] == "secretkey"
        assert result["aws_session_token"] == "tokenvalue"
