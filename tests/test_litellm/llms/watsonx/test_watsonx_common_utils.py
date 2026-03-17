import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.llms.watsonx.common_utils import generate_iam_token


class TestGenerateIAMToken:
    """Tests for the generate_iam_token function, specifically testing API key fallback logic."""

    @patch("litellm.llms.watsonx.common_utils.iam_token_cache")
    @patch("litellm.llms.watsonx.common_utils.litellm.module_level_client")
    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_generate_iam_token_with_watsonx_zenapikey(
        self, mock_get_secret_str, mock_client, mock_cache
    ):
        """Test that WATSONX_ZENAPIKEY is used when it's the only key available."""
        # Setup mocks
        mock_cache.get_cache.return_value = None  # Cache miss
        mock_get_secret_str.side_effect = lambda key: (
            "zen-api-key-12345" if key == "WATSONX_ZENAPIKEY" else None
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "test-token-12345",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        # Call function without api_key parameter
        result = generate_iam_token()

        # Verify get_secret_str was called with correct keys in order
        # Note: get_watsonx_iam_url() also calls get_secret_str("WATSONX_IAM_URL")
        calls = [
            call[0][0]
            for call in mock_get_secret_str.call_args_list
            if call[0][0] != "WATSONX_IAM_URL"
        ]
        assert "WX_API_KEY" in calls
        assert "WATSONX_API_KEY" in calls
        assert "WATSONX_APIKEY" in calls
        assert "WATSONX_ZENAPIKEY" in calls

        # Verify the token was generated using WATSONX_ZENAPIKEY
        assert result == "test-token-12345"
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["data"]["apikey"] == "zen-api-key-12345"

    @patch("litellm.llms.watsonx.common_utils.iam_token_cache")
    @patch("litellm.llms.watsonx.common_utils.litellm.module_level_client")
    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_generate_iam_token_api_key_priority_order(
        self, mock_get_secret_str, mock_client, mock_cache
    ):
        """Test that API keys are checked in the correct priority order."""
        # Setup mocks
        mock_cache.get_cache.return_value = None  # Cache miss

        # Test priority: WX_API_KEY > WATSONX_API_KEY > WATSONX_APIKEY > WATSONX_ZENAPIKEY
        test_cases = [
            # (env_keys_set, expected_key_used, expected_calls)
            (
                {"WX_API_KEY": "wx-key"},
                "wx-key",
                ["WX_API_KEY"],  # Should stop after first call
            ),
            (
                {"WATSONX_API_KEY": "watsonx-api-key"},
                "watsonx-api-key",
                ["WX_API_KEY", "WATSONX_API_KEY"],  # Should check WX_API_KEY first, then WATSONX_API_KEY
            ),
            (
                {"WATSONX_APIKEY": "watsonx-apikey"},
                "watsonx-apikey",
                ["WX_API_KEY", "WATSONX_API_KEY", "WATSONX_APIKEY"],
            ),
            (
                {"WATSONX_ZENAPIKEY": "watsonx-zenapikey"},
                "watsonx-zenapikey",
                ["WX_API_KEY", "WATSONX_API_KEY", "WATSONX_APIKEY", "WATSONX_ZENAPIKEY"],
            ),
            # Test that higher priority keys take precedence
            (
                {
                    "WX_API_KEY": "wx-key",
                    "WATSONX_ZENAPIKEY": "zen-key",
                },
                "wx-key",
                ["WX_API_KEY"],  # Should stop after first call
            ),
            (
                {
                    "WATSONX_API_KEY": "watsonx-api-key",
                    "WATSONX_ZENAPIKEY": "zen-key",
                },
                "watsonx-api-key",
                ["WX_API_KEY", "WATSONX_API_KEY"],  # Should stop after WATSONX_API_KEY
            ),
            (
                {
                    "WATSONX_APIKEY": "watsonx-apikey",
                    "WATSONX_ZENAPIKEY": "zen-key",
                },
                "watsonx-apikey",
                ["WX_API_KEY", "WATSONX_API_KEY", "WATSONX_APIKEY"],
            ),
        ]

        for env_keys, expected_key, expected_calls in test_cases:
            mock_get_secret_str.reset_mock()
            mock_client.reset_mock()
            mock_cache.reset_mock()

            # Configure mock to return values based on env_keys
            def get_secret_side_effect(key):
                return env_keys.get(key)

            mock_get_secret_str.side_effect = get_secret_side_effect

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "access_token": "test-token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            # Call function
            result = generate_iam_token()

            # Verify the correct key was used
            call_kwargs = mock_client.post.call_args
            assert (
                call_kwargs.kwargs["data"]["apikey"] == expected_key
            ), f"Expected {expected_key} but got {call_kwargs.kwargs['data']['apikey']} for env_keys: {env_keys}"

            # Verify get_secret_str was called with expected keys (checking short-circuit behavior)
            # Note: get_watsonx_iam_url() also calls get_secret_str("WATSONX_IAM_URL"), so we filter that out
            actual_calls = [
                call[0][0]
                for call in mock_get_secret_str.call_args_list
                if call[0][0] != "WATSONX_IAM_URL"
            ]
            assert (
                actual_calls == expected_calls
            ), f"Expected calls {expected_calls} but got {actual_calls} for env_keys: {env_keys}"

    @patch("litellm.llms.watsonx.common_utils.iam_token_cache")
    @patch("litellm.llms.watsonx.common_utils.litellm.module_level_client")
    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_generate_iam_token_with_direct_api_key(
        self, mock_get_secret_str, mock_client, mock_cache
    ):
        """Test that when api_key is passed directly, it's used instead of environment variables."""
        # Setup mocks
        mock_cache.get_cache.return_value = None  # Cache miss
        mock_get_secret_str.return_value = "env-key-should-not-be-used"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "test-token-12345",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        # Call function with direct api_key
        direct_key = "direct-api-key-12345"
        result = generate_iam_token(api_key=direct_key)

        # Verify get_secret_str was NOT called for API keys (since api_key was provided)
        # Note: get_watsonx_iam_url() calls get_secret_str("WATSONX_IAM_URL"), which is expected
        api_key_calls = [
            call[0][0]
            for call in mock_get_secret_str.call_args_list
            if call[0][0] not in ["WATSONX_IAM_URL"]
        ]
        assert (
            len(api_key_calls) == 0
        ), f"Expected no API key calls but got {api_key_calls}"

        # Verify the direct key was used
        assert result == "test-token-12345"
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["data"]["apikey"] == direct_key

    @patch("litellm.llms.watsonx.common_utils.iam_token_cache")
    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_generate_iam_token_no_api_key_raises_error(
        self, mock_get_secret_str, mock_cache
    ):
        """Test that ValueError is raised when no API key is available."""
        # Setup mocks
        mock_cache.get_cache.return_value = None  # Cache miss
        mock_get_secret_str.return_value = None  # No keys available

        # Call function without api_key and expect ValueError
        with pytest.raises(ValueError, match="API key is required"):
            generate_iam_token()

        # Verify get_secret_str was called for all possible API keys
        # Note: get_watsonx_iam_url() also calls get_secret_str("WATSONX_IAM_URL")
        calls = [
            call[0][0]
            for call in mock_get_secret_str.call_args_list
            if call[0][0] != "WATSONX_IAM_URL"
        ]
        assert "WX_API_KEY" in calls
        assert "WATSONX_API_KEY" in calls
        assert "WATSONX_APIKEY" in calls
        assert "WATSONX_ZENAPIKEY" in calls

    @patch("litellm.llms.watsonx.common_utils.iam_token_cache")
    @patch("litellm.llms.watsonx.common_utils.litellm.module_level_client")
    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_generate_iam_token_uses_cache(
        self, mock_get_secret_str, mock_client, mock_cache
    ):
        """Test that cached token is returned when available."""
        # Setup mocks
        cached_token = "cached-token-12345"
        mock_cache.get_cache.return_value = cached_token

        # Call function
        result = generate_iam_token()

        # Verify cached token was returned
        assert result == cached_token

        # Verify get_secret_str and client.post were NOT called (cache hit)
        mock_get_secret_str.assert_not_called()
        mock_client.post.assert_not_called()
