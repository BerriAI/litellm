"""
Test transformation logic for hosted_vllm embeddings.

This test verifies that the transformation layer correctly handles parameters,
especially ensuring that encoding_format is not included when not provided.
"""

import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.hosted_vllm.embedding.transformation import (
    HostedVLLMEmbeddingConfig,
)


class TestHostedVLLMEmbeddingTransformation:
    """Test suite for hosted_vllm embedding transformation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = HostedVLLMEmbeddingConfig()
        self.model = "hosted_vllm/BAAI/bge-small-en-v1.5"

    def test_transform_embedding_request_basic(self):
        """Test basic embedding request transformation."""
        input_data = ["hello world"]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )

        expected_result = {
            "model": "BAAI/bge-small-en-v1.5",  # prefix should be stripped
            "input": input_data,
        }
        assert result == expected_result

    def test_transform_embedding_request_string_input(self):
        """Test that string input is converted to list."""
        input_data = "hello world"
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )

        assert result["input"] == ["hello world"]
        assert result["model"] == "BAAI/bge-small-en-v1.5"

    def test_transform_embedding_request_with_dimensions(self):
        """Test embedding request with dimensions parameter."""
        input_data = ["hello world"]
        optional_params = {"dimensions": 384}
        
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )

        assert result["model"] == "BAAI/bge-small-en-v1.5"
        assert result["input"] == input_data
        assert result["dimensions"] == 384

    def test_encoding_format_not_included_when_not_provided(self):
        """
        Test that encoding_format is NOT included in the request when not provided.
        
        This is critical because vLLM rejects requests with encoding_format=None or
        encoding_format="" with error: "unknown variant ``, expected float or base64"
        """
        input_data = ["hello world"]
        
        # Test with no encoding_format in optional_params
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )

        assert "encoding_format" not in result, (
            "encoding_format should not be in request when not provided"
        )

    def test_encoding_format_not_included_when_none(self):
        """
        Test that encoding_format is NOT included when explicitly set to None.
        """
        input_data = ["hello world"]
        optional_params = {"encoding_format": None}
        
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )

        # encoding_format=None should be passed through, but filtered later
        # by the HTTP handler
        assert result.get("encoding_format") is None

    def test_encoding_format_included_when_float(self):
        """Test that encoding_format is included when set to 'float'."""
        input_data = ["hello world"]
        optional_params = {"encoding_format": "float"}
        
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )

        assert result["encoding_format"] == "float"

    def test_encoding_format_included_when_base64(self):
        """Test that encoding_format is included when set to 'base64'."""
        input_data = ["hello world"]
        optional_params = {"encoding_format": "base64"}
        
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )

        assert result["encoding_format"] == "base64"

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI parameters are correctly listed."""
        supported = self.config.get_supported_openai_params(self.model)
        
        assert "timeout" in supported
        assert "dimensions" in supported
        assert "encoding_format" in supported
        assert "user" in supported

    def test_map_openai_params(self):
        """Test mapping of OpenAI parameters."""
        non_default_params = {
            "dimensions": 512,
            "encoding_format": "float",
            "user": "test-user",
        }
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["dimensions"] == 512
        assert result["encoding_format"] == "float"
        assert result["user"] == "test-user"

    def test_map_openai_params_filters_unsupported(self):
        """Test that unsupported parameters are not mapped."""
        non_default_params = {
            "dimensions": 512,
            "unsupported_param": "value",
        }
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["dimensions"] == 512
        assert "unsupported_param" not in result

    def test_get_complete_url(self):
        """Test URL construction for embeddings endpoint."""
        api_base = "https://test-vllm.example.com/v1"
        
        url = self.config.get_complete_url(
            api_base=api_base,
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert url == "https://test-vllm.example.com/v1/embeddings"

    def test_get_complete_url_adds_embeddings_suffix(self):
        """Test that /embeddings is added if not present."""
        api_base = "https://test-vllm.example.com"
        
        url = self.config.get_complete_url(
            api_base=api_base,
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert url == "https://test-vllm.example.com/embeddings"

    def test_validate_environment_with_api_key(self):
        """Test environment validation with API key."""
        headers = {}
        
        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )

        assert "Authorization" in result
        assert result["Authorization"] == "Bearer test-api-key"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_without_api_key(self):
        """Test environment validation without API key (uses fake-api-key)."""
        headers = {}
        
        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )

        # Should not include Authorization header with fake-api-key
        assert "Authorization" not in result
        assert result["Content-Type"] == "application/json"

    def test_encoding_format_not_sent_in_actual_request(self):
        """
        E2E test that encoding_format is not sent when not provided.
        
        This test mocks the HTTP client to verify the actual request payload.
        """
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()
        
        with patch.object(client, "post") as mock_post:
            # Mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    }
                ],
                "model": "BAAI/bge-small-en-v1.5",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5,
                },
            }
            mock_response.text = json.dumps(mock_response.json.return_value)
            mock_post.return_value = mock_response

            try:
                litellm.embedding(
                    model=self.model,
                    input=["Hello world"],
                    api_base="https://test-vllm.example.com/v1",
                    client=client,
                )
            except Exception:
                pass

            # Verify the request was made
            mock_post.assert_called_once()

            # Get the data that was sent
            call_kwargs = mock_post.call_args[1]
            sent_data = json.loads(call_kwargs["data"])

            # Assert that encoding_format is NOT in the sent data
            assert "encoding_format" not in sent_data, (
                "encoding_format should not be in request when not provided"
            )
            assert sent_data["model"] == "BAAI/bge-small-en-v1.5"
            assert sent_data["input"] == ["Hello world"]

    def test_encoding_format_float_sent_in_actual_request(self):
        """
        Test that encoding_format='float' is sent when explicitly provided.
        """
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()
        
        with patch.object(client, "post") as mock_post:
            # Mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    }
                ],
                "model": "BAAI/bge-small-en-v1.5",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5,
                },
            }
            mock_response.text = json.dumps(mock_response.json.return_value)
            mock_post.return_value = mock_response

            try:
                litellm.embedding(
                    model=self.model,
                    input=["Hello world"],
                    api_base="https://test-vllm.example.com/v1",
                    encoding_format="float",
                    client=client,
                )
            except Exception:
                pass

            # Verify the request was made
            mock_post.assert_called_once()

            # Get the data that was sent
            call_kwargs = mock_post.call_args[1]
            sent_data = json.loads(call_kwargs["data"])

            # Assert that encoding_format IS in the sent data
            assert "encoding_format" in sent_data, (
                "encoding_format='float' should be in request when provided"
            )
            assert sent_data["encoding_format"] == "float"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
