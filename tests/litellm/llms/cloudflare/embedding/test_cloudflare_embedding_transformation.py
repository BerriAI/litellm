"""
Unit tests for Cloudflare Workers AI embedding transformation.

Tests the CloudflareEmbeddingConfig including request/response transformation,
URL construction, and environment validation.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.cloudflare.embedding.transformation import CloudflareEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


class TestCloudflareEmbeddingConfig:
    """Test CloudflareEmbeddingConfig parameter handling and transformation."""

    def setup_method(self):
        self.config = CloudflareEmbeddingConfig()
        self.model = "@cf/baai/bge-large-en-v1.5"

    def test_get_supported_openai_params(self):
        """Test that supported params returns empty list."""
        params = self.config.get_supported_openai_params(self.model)
        assert params == []

    def test_map_openai_params(self):
        """Test that map_openai_params returns optional_params unchanged."""
        optional_params = {"some_param": "value"}
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )
        assert result == optional_params

    def test_transform_embedding_request_string(self):
        """Test transform request with string input."""
        result = self.config.transform_embedding_request(
            model=self.model,
            input="hello world",
            optional_params={},
            headers={},
        )
        assert result["text"] == "hello world"

    def test_transform_embedding_request_list(self):
        """Test transform request with list input."""
        input_list = ["hello", "world"]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_list,
            optional_params={},
            headers={},
        )
        assert result["text"] == input_list

    def test_transform_embedding_response(self):
        """Test transform response parses Cloudflare embedding format."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "result": {
                "shape": [2, 1024],
                "data": [[0.1] * 1024, [0.2] * 1024],
            },
            "success": True,
        }
        model_response = EmbeddingResponse()

        result = self.config.transform_embedding_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
        )

        assert len(result.data) == 2
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[1]["index"] == 1
        assert len(result.data[0]["embedding"]) == 1024
        assert result.usage.prompt_tokens == 2
        assert result.usage.total_tokens == 2

    def test_transform_embedding_response_single(self):
        """Test transform response with single embedding."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "result": {
                "shape": [1, 768],
                "data": [[0.5] * 768],
            },
            "success": True,
        }
        model_response = EmbeddingResponse()

        result = self.config.transform_embedding_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
        )

        assert len(result.data) == 1
        assert result.data[0]["embedding"] == [0.5] * 768

    def test_validate_environment(self):
        """Test that validate_environment sets correct headers."""
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key-123",
        )
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["content-type"] == "application/json"

    def test_validate_environment_missing_key(self):
        """Test that validate_environment raises error when key is missing."""
        with pytest.raises(ValueError, match="Missing Cloudflare API Key"):
            self.config.validate_environment(
                headers={},
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    @patch("litellm.llms.cloudflare.embedding.transformation.get_secret_str")
    def test_get_complete_url(self, mock_get_secret):
        """Test URL construction with default api_base."""
        mock_get_secret.return_value = "test-account-id"
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert (
            url
            == f"https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/{self.model}"
        )

    def test_get_complete_url_with_api_base(self):
        """Test URL construction with explicit api_base."""
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/ai/run/",
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert url == f"https://custom.api.com/ai/run/{self.model}"
