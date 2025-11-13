import unittest
from typing import List
from unittest.mock import patch

from litellm.llms.burncloud.chat.transformation import BurnCloudChatConfig
from litellm.types.llms.openai import AllMessageValues


class TestBurnCloudChatConfig(unittest.TestCase):
    """
    Unit tests for BurnCloud chat transformation module
    """

    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        self.burncloud_config = BurnCloudChatConfig()
        self.test_messages: List[AllMessageValues] = [
            {
                "role": "user",
                "content": "Hello, how are you?"
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you for asking!"
            }
        ]
        self.test_model = "burncloud-test-model"

    def tearDown(self):
        """
        Clean up after each test method
        """
        pass

    @patch('litellm.llms.burncloud.chat.transformation.handle_messages_with_content_list_to_str_conversion')
    @patch('litellm.llms.burncloud.chat.transformation.OpenAIGPTConfig._transform_messages')
    def test_transform_messages_sync(self, mock_super_transform, mock_handle_messages):
        """
        Test _transform_messages method in synchronous mode
        """
        # Arrange
        mock_handle_messages.return_value = self.test_messages
        mock_super_transform.return_value = self.test_messages

        # Act
        result = self.burncloud_config._transform_messages(
            messages=self.test_messages,
            model=self.test_model,
            is_async=False
        )

        # Assert
        mock_handle_messages.assert_called_once_with(self.test_messages)
        mock_super_transform.assert_called_once_with(
            messages=self.test_messages,
            model=self.test_model,
            is_async=False
        )
        self.assertEqual(result, self.test_messages)

    @patch('litellm.llms.burncloud.chat.transformation.handle_messages_with_content_list_to_str_conversion')
    @patch('litellm.llms.burncloud.chat.transformation.OpenAIGPTConfig._transform_messages')
    def test_transform_messages_async(self, mock_super_transform, mock_handle_messages):
        """
        Test _transform_messages method in asynchronous mode
        """
        # Arrange
        mock_handle_messages.return_value = self.test_messages
        mock_super_transform.return_value = self.test_messages

        # Act
        result = self.burncloud_config._transform_messages(
            messages=self.test_messages,
            model=self.test_model,
            is_async=True
        )

        # Assert
        mock_handle_messages.assert_called_once_with(self.test_messages)
        mock_super_transform.assert_called_once_with(
            messages=self.test_messages,
            model=self.test_model,
            is_async=True
        )
        self.assertEqual(result, self.test_messages)

    @patch('litellm.llms.burncloud.chat.transformation.get_secret_str')
    def test_get_openai_compatible_provider_info_with_params(self, mock_get_secret):
        """
        Test _get_openai_compatible_provider_info with provided api_base and api_key
        """
        # Arrange
        test_api_base = "https://api.burncloud.example.com"
        test_api_key = "test-api-key"
        mock_get_secret.return_value = None

        # Act
        result_api_base, result_api_key = self.burncloud_config._get_openai_compatible_provider_info(
            api_base=test_api_base,
            api_key=test_api_key
        )

        # Assert
        self.assertEqual(result_api_base, test_api_base)
        self.assertEqual(result_api_key, test_api_key)
        mock_get_secret.assert_not_called()

    @patch('litellm.llms.burncloud.chat.transformation.get_secret_str')
    def test_get_openai_compatible_provider_info_with_secrets(self, mock_get_secret):
        """
        Test _get_openai_compatible_provider_info using secret manager
        """
        # Arrange
        mock_get_secret.side_effect = lambda key: {
            "BURNCLOUD_API_BASE": "https://api.burncloud.example.com",
            "BURNCLOUD_API_KEY": "secret-api-key"
        }.get(key, None)

        # Act
        result_api_base, result_api_key = self.burncloud_config._get_openai_compatible_provider_info(
            api_base=None,
            api_key=None
        )

        # Assert
        self.assertEqual(result_api_base, "https://api.burncloud.example.com")
        self.assertEqual(result_api_key, "secret-api-key")
        mock_get_secret.assert_any_call("BURNCLOUD_API_BASE")
        mock_get_secret.assert_any_call("BURNCLOUD_API_KEY")
