"""
Tests for _mask_secret_fields_for_logging to ensure JWT tokens and other
sensitive auth headers are masked in debug logs.
"""

import pytest

from litellm.utils import _mask_secret_fields_for_logging


SAMPLE_JWT = (
    "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImxpdGVsbG0tbW9jay1qd3Qta2V5LTEiLCJ0eXAiOiJKV1QifQ"
    ".eyJzdWIiOiJkZWZhdWx0X3VzZXJfaWQiLCJyb2xlcyI6WyJBRE1JTiJdLCJpYXQiOjE3NzA5NDM0NDR9"
    ".Yqr0dbk3XbD7nGEz5KzMyHjIfPH_18DNPjXe1tpKw9HMEvKGgIZSNP8MAsDwPnUvjeTH26kIp6xmy"
)

SAMPLE_SECRET_FIELDS = {
    "raw_headers": {
        "host": "localhost:4000",
        "user-agent": "curl/8.7.1",
        "accept": "*/*",
        "authorization": SAMPLE_JWT,
        "content-type": "application/json",
        "content-length": "96",
    }
}


class TestMaskSecretFieldsForLogging:
    """Tests for the proxy version of _mask_secret_fields_for_logging."""

    def test_should_mask_authorization_header(self):
        result = _mask_secret_fields_for_logging(SAMPLE_SECRET_FIELDS)
        auth_value = result["raw_headers"]["authorization"]
        # Should not contain the full JWT
        assert auth_value != SAMPLE_JWT
        # Should contain masking characters
        assert "****" in auth_value
        # Should preserve first 10 chars
        assert auth_value.startswith(SAMPLE_JWT[:10])
        # Should preserve last 4 chars
        assert auth_value.endswith(SAMPLE_JWT[-4:])

    def test_should_not_mask_non_sensitive_headers(self):
        result = _mask_secret_fields_for_logging(SAMPLE_SECRET_FIELDS)
        assert result["raw_headers"]["host"] == "localhost:4000"
        assert result["raw_headers"]["user-agent"] == "curl/8.7.1"
        assert result["raw_headers"]["accept"] == "*/*"
        assert result["raw_headers"]["content-type"] == "application/json"
        assert result["raw_headers"]["content-length"] == "96"

    def test_should_mask_x_api_key_header(self):
        secret_fields = {
            "raw_headers": {
                "x-api-key": "sk-very-secret-api-key-value-1234567890",
                "host": "localhost:4000",
            }
        }
        result = _mask_secret_fields_for_logging(secret_fields)
        assert result["raw_headers"]["x-api-key"] != "sk-very-secret-api-key-value-1234567890"
        assert "****" in result["raw_headers"]["x-api-key"]
        assert result["raw_headers"]["host"] == "localhost:4000"

    def test_should_mask_api_key_header(self):
        secret_fields = {
            "raw_headers": {
                "API-Key": "some-long-azure-api-key-value-here",
                "host": "localhost:4000",
            }
        }
        result = _mask_secret_fields_for_logging(secret_fields)
        assert "****" in result["raw_headers"]["API-Key"]

    def test_should_mask_google_ai_studio_key_header(self):
        secret_fields = {
            "raw_headers": {
                "x-goog-api-key": "AIzaSyB-some-long-google-api-key-1234567890",
                "host": "localhost:4000",
            }
        }
        result = _mask_secret_fields_for_logging(secret_fields)
        assert "****" in result["raw_headers"]["x-goog-api-key"]
        assert result["raw_headers"]["host"] == "localhost:4000"

    def test_should_mask_azure_apim_subscription_key_header(self):
        secret_fields = {
            "raw_headers": {
                "Ocp-Apim-Subscription-Key": "abcdef1234567890abcdef1234567890",
                "host": "localhost:4000",
            }
        }
        result = _mask_secret_fields_for_logging(secret_fields)
        assert "****" in result["raw_headers"]["Ocp-Apim-Subscription-Key"]
        assert result["raw_headers"]["host"] == "localhost:4000"

    def test_should_mask_short_auth_values(self):
        secret_fields = {
            "raw_headers": {
                "authorization": "short-token",
            }
        }
        result = _mask_secret_fields_for_logging(secret_fields)
        assert result["raw_headers"]["authorization"] == "****"

    def test_should_handle_non_dict_input(self):
        assert _mask_secret_fields_for_logging("not a dict") == "not a dict"
        assert _mask_secret_fields_for_logging(123) == 123
        assert _mask_secret_fields_for_logging(None) is None

    def test_should_handle_empty_dict(self):
        assert _mask_secret_fields_for_logging({}) == {}

    def test_should_handle_non_dict_field_values(self):
        secret_fields = {"some_key": "some_string_value"}
        result = _mask_secret_fields_for_logging(secret_fields)
        assert result["some_key"] == "some_string_value"

    def test_should_not_mutate_original(self):
        original = {
            "raw_headers": {
                "authorization": SAMPLE_JWT,
                "host": "localhost:4000",
            }
        }
        _mask_secret_fields_for_logging(original)
        # Original should be unchanged
        assert original["raw_headers"]["authorization"] == SAMPLE_JWT


class TestMaskSecretFieldsReexport:
    """Verify the re-export from litellm_pre_call_utils points to the same function."""

    def test_should_be_same_function_as_proxy_reexport(self):
        from litellm.proxy.litellm_pre_call_utils import (
            _mask_secret_fields_for_logging as proxy_version,
        )

        assert proxy_version is _mask_secret_fields_for_logging
