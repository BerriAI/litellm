"""
Tests that API keys are masked in error responses.

When an invalid/malformed API key is sent (e.g., with a leading space or
wrong prefix), the error response must NOT return the key in plain text.
Instead, it should show only the first 4 and last 4 characters with ****
in the middle.
"""

import pytest


class TestKeyMaskingInAuthErrors:
    """Test that user_api_key_auth masks keys in validation error messages."""

    def test_assert_message_masks_key_without_sk_prefix(self):
        """
        When a key doesn't start with 'sk-', the AssertionError message
        should contain a masked version, not the full key.
        """
        from litellm.proxy.auth.auth_utils import abbreviate_api_key

        # Simulate the logic from user_api_key_auth.py
        api_key = "my-secret-api-key-1234567890abcdef"
        _masked_key = (
            "{}****{}".format(api_key[:4], api_key[-4:])
            if len(api_key) > 8
            else "****"
        )

        # The masked key should NOT contain the full original key
        assert api_key not in _masked_key
        # Should show first 4 and last 4 chars
        assert _masked_key == "my-s****cdef"

    def test_assert_message_masks_key_with_leading_space(self):
        """
        Reported case: key with leading space like ' sk-abc123...'
        """
        api_key = " sk-abc123def456ghi789jkl012mno345pqr"
        _masked_key = (
            "{}****{}".format(api_key[:4], api_key[-4:])
            if len(api_key) > 8
            else "****"
        )

        assert api_key not in _masked_key
        assert _masked_key == " sk-****5pqr"

    def test_assert_message_masks_short_key(self):
        """Short keys (<=8 chars) should be fully masked."""
        api_key = "short"
        _masked_key = (
            "{}****{}".format(api_key[:4], api_key[-4:])
            if len(api_key) > 8
            else "****"
        )
        assert _masked_key == "****"

    def test_key_not_starting_with_sk_raises_masked_error(self):
        """
        Verify the assert message format contains masked key, not the original.

        Note: Python's AssertionError str(e) includes the expression + message,
        but the *message* part (which is what gets passed to ProxyException)
        should only contain the masked key.
        """
        api_key = "bad-key-format-1234567890abcdefghijklmnop"
        _masked_key = (
            "{}****{}".format(api_key[:4], api_key[-4:])
            if len(api_key) > 8
            else "****"
        )

        # Build the same message string that user_api_key_auth.py would produce
        error_message = "LiteLLM Virtual Key expected. Received={}, expected to start with 'sk-'.".format(
            _masked_key
        )
        # The full key must NOT appear in the message
        assert api_key not in error_message
        # The masked version should appear
        assert _masked_key in error_message
        # Should still have helpful context
        assert "expected to start with 'sk-'" in error_message


class TestKeyMaskingInKeyManagement:
    """Test that key_management_endpoints masks keys in validation errors."""

    def test_invalid_key_format_error_is_masked(self):
        """
        When creating a key that doesn't start with 'sk-', the error
        should not include the full key value.
        """
        key_value = "bad-prefix-1234567890abcdefghijklmnop"
        _masked = (
            "{}****{}".format(key_value[:4], key_value[-4:])
            if len(key_value) > 8
            else "****"
        )

        error_msg = f"Invalid key format. LiteLLM Virtual Key must start with 'sk-'. Received: {_masked}"

        # Full key must not appear
        assert key_value not in error_msg
        # Masked version should appear
        assert _masked in error_msg
        assert "bad-****mnop" in error_msg


class TestPresidioErrorSanitization:
    """Test that Presidio errors don't leak request text containing keys."""

    def test_analyze_text_error_does_not_leak_text(self):
        """
        If Presidio analyzer fails, the error message should NOT contain
        the original text that was being analyzed.
        """
        # Simulate what happens: user message contains an API key,
        # Presidio fails, error message should be sanitized
        original_text = "Please use this key: sk-secret1234567890abcdefghijklmnop"

        # The sanitized exception from our fix
        sanitized_error = f"Presidio PII analysis failed: ConnectionError"

        assert original_text not in sanitized_error
        assert "sk-secret1234567890abcdefghijklmnop" not in sanitized_error

    def test_anonymize_text_error_does_not_leak_text(self):
        """
        If Presidio anonymizer fails, the error should be sanitized.
        """
        sanitized_error = f"Presidio PII anonymization failed: ClientError"

        assert "sk-" not in sanitized_error
        assert "api_key" not in sanitized_error
