"""Tests for Isaacus embedding transformation."""

import json
from unittest.mock import MagicMock, Mock

import httpx
import pytest

from litellm.llms.isaacus.embedding.transformation import (
    IsaacusEmbeddingConfig,
    IsaacusError,
)
from litellm.types.utils import EmbeddingResponse


class TestIsaacusEmbeddingConfig:
    """Test IsaacusEmbeddingConfig transformation methods."""

    def test_get_complete_url_default(self):
        """Test getting complete URL with default base."""
        config = IsaacusEmbeddingConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="kanon-2-embedder",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.isaacus.com/v1/embeddings"

    def test_get_complete_url_custom_base(self):
        """Test getting complete URL with custom base."""
        config = IsaacusEmbeddingConfig()
        url = config.get_complete_url(
            api_base="https://custom.api.com/v1",
            api_key=None,
            model="kanon-2-embedder",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/embeddings"

    def test_get_complete_url_custom_base_with_endpoint(self):
        """Test getting complete URL when custom base already has endpoint."""
        config = IsaacusEmbeddingConfig()
        url = config.get_complete_url(
            api_base="https://custom.api.com/v1/embeddings",
            api_key=None,
            model="kanon-2-embedder",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/embeddings"

    def test_get_supported_openai_params(self):
        """Test supported OpenAI parameters."""
        config = IsaacusEmbeddingConfig()
        params = config.get_supported_openai_params("kanon-2-embedder")
        assert params == ["dimensions"]

    def test_map_openai_params(self):
        """Test mapping OpenAI parameters to Isaacus parameters."""
        config = IsaacusEmbeddingConfig()
        optional_params = {}
        result = config.map_openai_params(
            non_default_params={"dimensions": 512},
            optional_params=optional_params,
            model="kanon-2-embedder",
            drop_params=False,
        )
        assert result["dimensions"] == 512

    def test_validate_environment(self):
        """Test environment validation and header setup."""
        config = IsaacusEmbeddingConfig()
        headers = config.validate_environment(
            headers={},
            model="kanon-2-embedder",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_transform_embedding_request_single_string(self):
        """Test transforming single string input."""
        config = IsaacusEmbeddingConfig()
        request = config.transform_embedding_request(
            model="kanon-2-embedder",
            input="This is a test",
            optional_params={},
            headers={},
        )
        assert request["model"] == "kanon-2-embedder"
        assert request["texts"] == ["This is a test"]

    def test_transform_embedding_request_list_of_strings(self):
        """Test transforming list of strings input."""
        config = IsaacusEmbeddingConfig()
        request = config.transform_embedding_request(
            model="kanon-2-embedder",
            input=["First text", "Second text"],
            optional_params={},
            headers={},
        )
        assert request["model"] == "kanon-2-embedder"
        assert request["texts"] == ["First text", "Second text"]

    def test_transform_embedding_request_with_optional_params(self):
        """Test transforming request with optional parameters."""
        config = IsaacusEmbeddingConfig()
        request = config.transform_embedding_request(
            model="kanon-2-embedder",
            input="Test text",
            optional_params={
                "task": "retrieval/query",
                "dimensions": 1024,
                "overflow_strategy": "drop_end",
            },
            headers={},
        )
        assert request["model"] == "kanon-2-embedder"
        assert request["texts"] == ["Test text"]
        assert request["task"] == "retrieval/query"
        assert request["dimensions"] == 1024
        assert request["overflow_strategy"] == "drop_end"

    def test_transform_embedding_request_token_array_error(self):
        """Test that token arrays raise an error."""
        config = IsaacusEmbeddingConfig()
        with pytest.raises(ValueError, match="does not support token array"):
            config.transform_embedding_request(
                model="kanon-2-embedder",
                input=[[1, 2, 3]],
                optional_params={},
                headers={},
            )

    def test_transform_embedding_response(self):
        """Test transforming Isaacus response to OpenAI format."""
        config = IsaacusEmbeddingConfig()

        # Create mock response with actual Isaacus format
        # Isaacus returns embeddings as objects with "embedding" and "index" fields
        mock_response_data = {
            "embeddings": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "usage": {"input_tokens": 10},
        }

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data

        model_response = EmbeddingResponse()
        logging_obj = MagicMock()

        result = config.transform_embedding_response(
            model="kanon-2-embedder",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=logging_obj,
        )

        assert result.model == "kanon-2-embedder"
        assert result.object == "list"
        assert len(result.data) == 2
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[1]["index"] == 1
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 10

    def test_transform_embedding_response_invalid_json(self):
        """Test handling of invalid JSON response."""
        config = IsaacusEmbeddingConfig()

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        mock_response.text = "Invalid JSON"
        mock_response.status_code = 500

        model_response = EmbeddingResponse()
        logging_obj = MagicMock()

        with pytest.raises(IsaacusError) as exc_info:
            config.transform_embedding_response(
                model="kanon-2-embedder",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=logging_obj,
            )

        assert exc_info.value.status_code == 500
        assert "Invalid JSON" in exc_info.value.message

    def test_get_error_class(self):
        """Test getting custom error class."""
        config = IsaacusEmbeddingConfig()
        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={},
        )
        assert isinstance(error, IsaacusError)
        assert error.status_code == 400
        assert error.message == "Test error"


class TestIsaacusError:
    """Test IsaacusError exception class."""

    def test_isaacus_error_creation(self):
        """Test creating IsaacusError."""
        error = IsaacusError(
            status_code=401,
            message="Unauthorized",
        )
        assert error.status_code == 401
        assert error.message == "Unauthorized"
        assert error.request.method == "POST"
        # Note: The actual URL may be set by the base class
        assert error.request is not None
