"""
Unit tests for CLI token utilities
"""

import json
from unittest.mock import mock_open, patch


from litellm.litellm_core_utils.cli_token_utils import (
    get_litellm_gateway_api_key,
    get_stored_base_url,
)


class TestCLITokenUtils:
    """Test CLI token utility functions"""

    def test_get_litellm_gateway_api_key_success(self):
        """Test getting CLI API key when token file exists and is valid"""
        token_data = {
            "key": "sk-test-cli-key-123",
            "user_id": "test-user",
            "user_email": "test@example.com",
            "timestamp": 1234567890,
        }

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(token_data))),
            patch(
                "litellm.litellm_core_utils.cli_token_utils.get_cli_token_file_path",
                return_value="/test/.litellm/token.json",
            ),
        ):

            result = get_litellm_gateway_api_key()

            assert result == "sk-test-cli-key-123"

    def test_get_litellm_gateway_api_key_no_file(self):
        """Test getting CLI API key when token file doesn't exist"""
        with (
            patch("os.path.exists", return_value=False),
            patch(
                "litellm.litellm_core_utils.cli_token_utils.get_cli_token_file_path",
                return_value="/test/.litellm/token.json",
            ),
        ):

            result = get_litellm_gateway_api_key()

            assert result is None

    def test_get_litellm_gateway_api_key_invalid_json(self):
        """Test getting CLI API key when token file has invalid JSON"""
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="invalid json")),
            patch(
                "litellm.litellm_core_utils.cli_token_utils.get_cli_token_file_path",
                return_value="/test/.litellm/token.json",
            ),
        ):

            result = get_litellm_gateway_api_key()

            assert result is None

    def test_get_litellm_gateway_api_key_no_key_field(self):
        """Test getting CLI API key when token file exists but has no key field"""
        token_data = {
            "user_id": "test-user",
            "user_email": "test@example.com",
            # Missing 'key' field
        }

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(token_data))),
            patch(
                "litellm.litellm_core_utils.cli_token_utils.get_cli_token_file_path",
                return_value="/test/.litellm/token.json",
            ),
        ):

            result = get_litellm_gateway_api_key()

            assert result is None


class TestGetStoredBaseUrl:
    """Test reading the proxy base URL stored by `lite login`"""

    def test_returns_stored_base_url(self):
        with patch(
            "litellm.litellm_core_utils.cli_token_utils.load_cli_token",
            return_value={"key": "sk-test", "base_url": "https://llm.acme.com"},
        ):
            assert get_stored_base_url() == "https://llm.acme.com"

    def test_returns_none_when_no_token_file(self):
        with patch(
            "litellm.litellm_core_utils.cli_token_utils.load_cli_token",
            return_value=None,
        ):
            assert get_stored_base_url() is None

    def test_returns_none_when_base_url_missing(self):
        with patch(
            "litellm.litellm_core_utils.cli_token_utils.load_cli_token",
            return_value={"key": "sk-old-token"},
        ):
            assert get_stored_base_url() is None

    def test_returns_none_when_base_url_empty(self):
        with patch(
            "litellm.litellm_core_utils.cli_token_utils.load_cli_token",
            return_value={"key": "sk-test", "base_url": ""},
        ):
            assert get_stored_base_url() is None
