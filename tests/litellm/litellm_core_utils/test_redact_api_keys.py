"""
Tests for the API key redaction utility.

Ensures that various API key formats are properly masked in error messages
to prevent credential leakage to end users.
"""

import pytest

from litellm.litellm_core_utils.redact_api_keys import redact_api_keys


class TestRedactApiKeys:
    """Tests for the redact_api_keys function."""

    def test_returns_none_for_none_input(self):
        assert redact_api_keys(None) is None

    def test_returns_non_string_as_is(self):
        assert redact_api_keys(123) == 123

    def test_no_keys_returns_unchanged(self):
        msg = "Connection timeout after 30 seconds"
        assert redact_api_keys(msg) == msg

    def test_redacts_openai_sk_key(self):
        msg = "Error: Invalid API key: sk-1234567890abcdefghijklmnopqrstuv"
        result = redact_api_keys(msg)
        assert "sk-1234567890abcdefghijklmnopqrstuv" not in result
        assert "sk-****" in result

    def test_redacts_anthropic_sk_ant_key(self):
        msg = "AuthenticationError: Invalid API key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_api_keys(msg)
        assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890" not in result
        assert "sk-ant-****" in result

    def test_redacts_bearer_token(self):
        msg = "Authorization failed with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        result = redact_api_keys(msg)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "Bearer [REDACTED]" in result

    def test_redacts_key_in_url_param(self):
        msg = "Request to https://api.example.com/v1/chat?key=abcdef1234567890ghij failed"
        result = redact_api_keys(msg)
        assert "abcdef1234567890ghij" not in result
        assert "key=[REDACTED]" in result

    def test_redacts_api_key_url_param(self):
        msg = "Error calling https://api.example.com?api_key=sk_test_abc123def456ghi789 - 401"
        result = redact_api_keys(msg)
        assert "sk_test_abc123def456ghi789" not in result
        assert "api_key=[REDACTED]" in result

    def test_redacts_azure_api_key_header(self):
        msg = "api-key: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4 was rejected"
        result = redact_api_keys(msg)
        assert "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" not in result
        assert "api-key: [REDACTED]" in result

    def test_redacts_authorization_header_value(self):
        msg = "Authorization: sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        result = redact_api_keys(msg)
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz123456" not in result

    def test_preserves_error_context(self):
        msg = "OpenAIException - Error code: 401 - Invalid API key provided: sk-abcdefghijklmnopqrstuvwxyz12345678. You can find your API key at https://platform.openai.com/account/api-keys."
        result = redact_api_keys(msg)
        assert "sk-abcdefghijklmnopqrstuvwxyz12345678" not in result
        assert "Error code: 401" in result
        assert "Invalid API key provided" in result

    def test_redacts_multiple_keys_in_same_message(self):
        msg = "Tried key sk-aaaabbbbccccddddeeeeffffgggg1234 then sk-1111222233334444555566667777abcd"
        result = redact_api_keys(msg)
        assert "sk-aaaabbbbccccddddeeeeffffgggg1234" not in result
        assert "sk-1111222233334444555566667777abcd" not in result

    def test_real_world_openai_error_with_key(self):
        """Simulate a real OpenAI error that includes the API key in the message."""
        msg = (
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ123456. "
            "You can find your API key at https://platform.openai.com/account/api-keys.', "
            "'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"
        )
        result = redact_api_keys(msg)
        assert "sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ123456" not in result
        assert "sk-" in result  # prefix should still be visible
        assert "invalid_api_key" in result  # error context preserved

    def test_real_world_azure_error_with_key(self):
        """Simulate an Azure error that includes the API key."""
        msg = "Access denied with api-key=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6. Check your credentials."
        result = redact_api_keys(msg)
        assert "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6" not in result

    def test_empty_string(self):
        assert redact_api_keys("") == ""

    def test_short_values_not_redacted(self):
        """Short values that don't look like keys should not be redacted."""
        msg = "key=abc is too short"
        assert redact_api_keys(msg) == msg
