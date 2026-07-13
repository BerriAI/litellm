"""
Unit tests for Perplexity embedding transformation logic.
"""

import base64
import json
import struct
from unittest.mock import MagicMock

import httpx

from litellm.llms.perplexity.embedding.transformation import (
    PerplexityEmbeddingConfig,
    PerplexityEmbeddingError,
)
from litellm.types.utils import EmbeddingResponse


class TestPerplexityEmbeddingConfig:
    def setup_method(self):
        self.config = PerplexityEmbeddingConfig()
        self.model = "pplx-embed-v1-0.6b"
        self.logging_obj = MagicMock()

    def test_get_complete_url_default(self):
        """Test default URL construction."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.perplexity.ai/v1/embeddings"

    def test_get_complete_url_custom_base(self):
        """Test URL construction with custom api_base."""
        url = self.config.get_complete_url(
            api_base="https://custom.api.com",
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/embeddings"

    def test_get_complete_url_already_has_embeddings(self):
        """Test URL construction when api_base already ends with /embeddings."""
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/v1/embeddings",
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/embeddings"

    def test_get_supported_openai_params(self):
        """Test that supported params are correctly listed."""
        supported = self.config.get_supported_openai_params(self.model)
        assert "dimensions" in supported
        assert "encoding_format" in supported

    def test_map_openai_params_dimensions(self):
        """Test that dimensions parameter is correctly mapped."""
        result = self.config.map_openai_params(
            non_default_params={"dimensions": 512},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["dimensions"] == 512

    def test_map_openai_params_encoding_format(self):
        """Test that encoding_format parameter is correctly mapped."""
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": "base64_int8"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["encoding_format"] == "base64_int8"

    def test_map_openai_params_unsupported_dropped(self):
        """Test that unsupported parameters are not passed through."""
        result = self.config.map_openai_params(
            non_default_params={"dimensions": 256, "user": "test-user"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["dimensions"] == 256
        assert "user" not in result

    def test_validate_environment_with_api_key(self):
        """Test environment validation with explicit API key."""
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="pplx-test-key",
        )
        assert headers["Authorization"] == "Bearer pplx-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_transform_embedding_request_string_input(self):
        """Test request transformation with string input."""
        result = self.config.transform_embedding_request(
            model=self.model,
            input="Hello world",
            optional_params={},
            headers={},
        )
        assert result["model"] == self.model
        assert result["input"] == "Hello world"

    def test_transform_embedding_request_list_input(self):
        """Test request transformation with list input."""
        input_data = ["Hello world", "Testing embeddings"]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        assert result["model"] == self.model
        assert result["input"] == input_data

    def test_transform_embedding_request_with_params(self):
        """Test request transformation with optional params."""
        result = self.config.transform_embedding_request(
            model=self.model,
            input=["Test"],
            optional_params={"dimensions": 256},
            headers={},
        )
        assert result["model"] == self.model
        assert result["input"] == ["Test"]
        assert result["dimensions"] == 256

    def test_transform_embedding_response_float_passthrough(self):
        """Test response transformation when embeddings are already float arrays."""
        mock_response_data = {
            "object": "list",
            "model": "pplx-embed-v1-0.6b",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5,
            },
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.status_code = 200

        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
        )

        assert result.model == "pplx-embed-v1-0.6b"
        assert result.object == "list"
        assert len(result.data) == 1
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.usage.prompt_tokens == 5
        assert result.usage.total_tokens == 5

    def test_transform_embedding_response_base64_int8(self):
        """Test decoding base64_int8 embeddings to float arrays (Perplexity default)."""
        int8_values = [127, -128, 0, 64, -64]
        b64_encoded = base64.b64encode(
            struct.pack(f"{len(int8_values)}b", *int8_values)
        ).decode()

        mock_response_data = {
            "object": "list",
            "model": "pplx-embed-v1-0.6b",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": b64_encoded,
                }
            ],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.status_code = 200

        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
        )

        embedding = result.data[0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) == 5
        assert all(isinstance(v, float) for v in embedding)
        assert abs(embedding[0] - 1.0) < 0.01
        assert abs(embedding[1] - (-128.0 / 127.0)) < 0.01
        assert embedding[2] == 0.0

    def test_decode_base64_embedding_static(self):
        """Test the static decode helper directly."""
        int8_values = [10, -10, 50, -50]
        b64_str = base64.b64encode(struct.pack("4b", *int8_values)).decode()
        result = PerplexityEmbeddingConfig._decode_base64_embedding(b64_str)
        assert len(result) == 4
        assert abs(result[0] - 10.0 / 127.0) < 1e-6
        assert abs(result[1] - (-10.0 / 127.0)) < 1e-6

    def test_decode_base64_embedding_list_passthrough(self):
        """Test that float lists pass through unchanged."""
        floats = [0.5, -0.3, 0.8]
        result = PerplexityEmbeddingConfig._decode_base64_embedding(floats)
        assert result == floats

    def test_transform_embedding_response_error(self):
        """Test that malformed response raises PerplexityEmbeddingError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_response.text = "Server error"
        mock_response.status_code = 500

        model_response = EmbeddingResponse()
        try:
            self.config.transform_embedding_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
            )
            assert False, "Should have raised PerplexityEmbeddingError"
        except PerplexityEmbeddingError as e:
            assert e.status_code == 500
            assert "Server error" in e.message

    def test_get_error_class(self):
        """Test that get_error_class returns the correct error type."""
        error = self.config.get_error_class(
            error_message="Not found",
            status_code=404,
            headers={},
        )
        assert isinstance(error, PerplexityEmbeddingError)
        assert error.status_code == 404
        assert error.message == "Not found"

    def test_transform_embedding_request_4b_model(self):
        """Test request transformation with the 4b model."""
        model = "pplx-embed-v1-4b"
        result = self.config.transform_embedding_request(
            model=model,
            input=["Test text"],
            optional_params={"dimensions": 2560},
            headers={},
        )
        assert result["model"] == model
        assert result["dimensions"] == 2560


class TestPerplexityEmbeddingProviderConfig:
    """Test that Perplexity is correctly registered in ProviderConfigManager."""

    def test_provider_config_returns_perplexity_embedding(self):
        import litellm
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_embedding_config(
            model="pplx-embed-v1-0.6b",
            provider=litellm.LlmProviders.PERPLEXITY,
        )
        assert config is not None
        assert isinstance(config, PerplexityEmbeddingConfig)

    def test_provider_config_returns_perplexity_embedding_4b(self):
        import litellm
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_embedding_config(
            model="pplx-embed-v1-4b",
            provider=litellm.LlmProviders.PERPLEXITY,
        )
        assert config is not None
        assert isinstance(config, PerplexityEmbeddingConfig)


class TestPerplexityEmbeddingModelInfo:
    """Test that Perplexity embedding models are in model_prices_and_context_window."""

    def test_model_info_available(self):
        import litellm

        info = litellm.get_model_info("perplexity/pplx-embed-v1-0.6b")
        assert info is not None
        assert info["mode"] == "embedding"
        assert info["max_input_tokens"] == 32768
        assert info["output_vector_size"] == 1024

    def test_model_info_4b_available(self):
        import litellm

        info = litellm.get_model_info("perplexity/pplx-embed-v1-4b")
        assert info is not None
        assert info["mode"] == "embedding"
        assert info["max_input_tokens"] == 32768
        assert info["output_vector_size"] == 2560
