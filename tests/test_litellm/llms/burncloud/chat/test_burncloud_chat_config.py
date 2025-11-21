"""Unit tests for BurnCloud chat transformation."""

import unittest
from typing import Dict, List, Optional
from unittest.mock import patch, Mock

from litellm.llms.burncloud.chat.transformation import BurnCloudChatConfig
from litellm.types.llms.openai import AllMessageValues


class TestBurnCloudChatConfig(unittest.TestCase):
    """Test suite for BurnCloudChatConfig class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = BurnCloudChatConfig()
        self.test_model = "gpt-3.5-turbo"
        self.test_api_key = "test-api-key"
        self.test_api_base = "https://api.burncloud.ai"
        self.test_messages: List[AllMessageValues] = [
            {"role": "user", "content": "Hello"}
        ]

    def test_get_complete_url_with_valid_base(self) -> None:
        """Test get_complete_url with valid API base."""
        result = self.config.get_complete_url(
            api_base=self.test_api_base,
            api_key=self.test_api_key,
            model=self.test_model,
            optional_params={},
            litellm_params={}
        )

        expected = f"{self.test_api_base}/v1/chat/completions"
        self.assertEqual(result, expected)

    def test_get_complete_url_with_v1_base(self) -> None:
        """Test get_complete_url when API base ends with /v1."""
        api_base_with_v1 = f"{self.test_api_base}/v1"

        result = self.config.get_complete_url(
            api_base=api_base_with_v1,
            api_key=self.test_api_key,
            model=self.test_model,
            optional_params={},
            litellm_params={}
        )

        expected = f"{api_base_with_v1}/chat/completions"
        self.assertEqual(result, expected)

    def test_get_complete_url_without_api_base(self) -> None:
        """Test get_complete_url when API base is None."""
        with patch("litellm.llms.burncloud.chat.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_base

            result = self.config.get_complete_url(
                api_base=None,
                api_key=self.test_api_key,
                model=self.test_model,
                optional_params={},
                litellm_params={}
            )

            expected = f"{self.test_api_base}/v1/chat/completions"
            self.assertEqual(result, expected)

    def test_validate_environment_with_api_key(self) -> None:
        """Test environment validation with API key provided."""
        headers: Dict = {}
        optional_params: Dict = {}
        litellm_params: Dict = {}

        with patch("litellm.llms.burncloud.chat.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_key

            result = self.config.validate_environment(
                headers=headers,
                model=self.test_model,
                messages=self.test_messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=None,
                api_base=None
            )

            expected = {
                "Authorization": f"Bearer {self.test_api_key}",
                "Content-Type": "application/json",
            }
            self.assertEqual(result, expected)

    def test_validate_environment_with_existing_auth_header(self) -> None:
        """Test environment validation with existing Authorization header."""
        custom_auth = "Custom token"
        headers = {"Authorization": custom_auth}
        optional_params: Dict = {}
        litellm_params: Dict = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.test_model,
            messages=self.test_messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=self.test_api_key,
            api_base=None
        )

        expected = {
            "Authorization": custom_auth,
            "Content-Type": "application/json",
        }
        self.assertEqual(result, expected)

    def test_validate_environment_with_additional_headers(self) -> None:
        """Test environment validation with additional headers."""
        headers = {"X-Custom-Header": "custom-value"}
        optional_params: Dict = {}
        litellm_params: Dict = {}

        with patch("litellm.llms.burncloud.chat.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_key

            result = self.config.validate_environment(
                headers=headers,
                model=self.test_model,
                messages=self.test_messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=None,
                api_base=None
            )

            self.assertIn("Authorization", result)
            self.assertIn("Content-Type", result)
            self.assertIn("X-Custom-Header", result)
            self.assertEqual(result["X-Custom-Header"], "custom-value")
