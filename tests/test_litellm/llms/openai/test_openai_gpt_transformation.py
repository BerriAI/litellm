"""
Tests for OpenAI GPT Transformation, specifically focusing on the get_models method
and the wildcard model expansion fix.
"""
import os
import sys
import pytest
from unittest.mock import Mock, patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class TestOpenAIGPTConfig:
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OpenAIGPTConfig()

        # Mock response data matching OpenAI's /v1/models format
        self.mock_models_response = {
            "data": [
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-4-turbo", "object": "model"},
            ]
        }

    @patch('litellm.module_level_client.get')
    def test_get_models_default_api_base(self, mock_get):
        """Test get_models with default OpenAI API base"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models with default parameters
        result = self.config.get_models(api_key="test-key")

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://api.openai.com/v1/models",
            headers={"Authorization": "Bearer test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_custom_api_base_with_port(self, mock_get):
        """Test get_models with custom API base that includes port"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models with custom API base including port
        result = self.config.get_models(
            api_key="test-key",
            api_base="https://custom.openai.com:8080"
        )

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://custom.openai.com:8080/v1/models",
            headers={"Authorization": "Bearer test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_custom_api_base_with_path(self, mock_get):
        """Test get_models with custom API base that includes a path (main fix)"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models with custom API base including path
        result = self.config.get_models(
            api_key="test-key",
            api_base="https://custom.domain.com/api/openai"
        )

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://custom.domain.com/api/openai/v1/models",
            headers={"Authorization": "Bearer test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_api_base_ending_with_v1(self, mock_get):
        """Test get_models when API base already ends with /v1"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models with API base ending in /v1
        result = self.config.get_models(
            api_key="test-key",
            api_base="https://custom.domain.com/api/v1"
        )

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://custom.domain.com/api/v1/models",
            headers={"Authorization": "Bearer test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_api_base_ending_with_v1_slash(self, mock_get):
        """Test get_models when API base ends with /v1/"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models with API base ending in /v1/
        result = self.config.get_models(
            api_key="test-key",
            api_base="https://custom.domain.com/api/v1/"
        )

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://custom.domain.com/api/v1/models",
            headers={"Authorization": "Bearer test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_complex_api_base_with_subdomain_and_path(self, mock_get):
        """Test get_models with complex API base (subdomain + path)"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models with complex API base
        result = self.config.get_models(
            api_key="test-key",
            api_base="https://api.subdomain.example.com/org/project/openai"
        )

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://api.subdomain.example.com/org/project/openai/v1/models",
            headers={"Authorization": "Bearer test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_with_trailing_slash_in_path(self, mock_get):
        """Test get_models strips trailing slashes from paths correctly"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models with trailing slash in path
        result = self.config.get_models(
            api_key="test-key",
            api_base="https://custom.domain.com/api/openai/"
        )

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://custom.domain.com/api/openai/v1/models",
            headers={"Authorization": "Bearer test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_root_path_only(self, mock_get):
        """Test get_models when path is just root '/'"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models with just root path
        result = self.config.get_models(
            api_key="test-key",
            api_base="https://custom.domain.com/"
        )

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://custom.domain.com/v1/models",
            headers={"Authorization": "Bearer test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_error_response(self, mock_get):
        """Test get_models handles error responses correctly"""
        # Setup mock error response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        # Call get_models and expect exception
        with pytest.raises(Exception, match="Failed to get models: Unauthorized"):
            self.config.get_models(api_key="invalid-key")

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env-test-key'})
    @patch('litellm.module_level_client.get')
    def test_get_models_uses_env_api_key(self, mock_get):
        """Test get_models uses environment variable for API key when not provided"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_models_response
        mock_get.return_value = mock_response

        # Call get_models without explicit API key
        result = self.config.get_models()

        # Assertions
        assert result == ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        mock_get.assert_called_once_with(
            url="https://api.openai.com/v1/models",
            headers={"Authorization": "Bearer env-test-key"}
        )

    @patch('litellm.module_level_client.get')
    def test_get_models_empty_response(self, mock_get):
        """Test get_models handles empty models list"""
        # Setup mock response with empty data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        # Call get_models
        result = self.config.get_models(api_key="test-key")

        # Assertions
        assert result == []
        mock_get.assert_called_once()