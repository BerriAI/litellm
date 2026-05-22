"""
Test cases for SageMaker Cohere embedding model integration.

AWS Marketplace Cohere containers expose the native Cohere embed API, which
expects {"texts": [...], "input_type": "..."} rather than the HuggingFace
{"inputs": [...]} format that the default SageMaker config sends.
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.sagemaker.embedding.transformation import (
    SagemakerCohereEmbeddingConfig,
    SagemakerEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse, Usage


class TestSagemakerCohereEmbeddingFactory:
    """Test the factory pattern correctly routes Cohere models."""

    def test_get_model_config_cohere_model(self):
        """Cohere models should return SagemakerCohereEmbeddingConfig."""
        config = SagemakerEmbeddingConfig.get_model_config(
            "cohere.embed-multilingual-v3"
        )
        assert isinstance(config, SagemakerCohereEmbeddingConfig)

    def test_get_model_config_cohere_english(self):
        """cohere-embed-english-v3 should also use Cohere config."""
        config = SagemakerEmbeddingConfig.get_model_config("cohere-embed-english-v3")
        assert isinstance(config, SagemakerCohereEmbeddingConfig)

    def test_get_model_config_cohere_case_insensitive(self):
        """Model detection should be case-insensitive."""
        config = SagemakerEmbeddingConfig.get_model_config("COHERE-EMBED-V3")
        assert isinstance(config, SagemakerCohereEmbeddingConfig)

    def test_get_model_config_hf_model_unaffected(self):
        """Non-Cohere, non-Voyage models still return base SagemakerEmbeddingConfig."""
        config = SagemakerEmbeddingConfig.get_model_config(
            "sentence-transformers-model"
        )
        assert isinstance(config, SagemakerEmbeddingConfig)
        assert not isinstance(config, SagemakerCohereEmbeddingConfig)


class TestSagemakerCohereEmbeddingRequest:
    """Test request transformation for Cohere SageMaker embeddings."""

    def setup_method(self):
        self.config = SagemakerCohereEmbeddingConfig()

    def test_transform_request_uses_texts_key(self):
        """Request body must use 'texts', not 'inputs'."""
        result = self.config.transform_embedding_request(
            model="cohere.embed-multilingual-v3",
            input=["hello", "world"],
            optional_params={},
            headers={},
        )
        assert "texts" in result, "Cohere format requires 'texts' key"
        assert "inputs" not in result, "HF 'inputs' key must not be present"
        assert result["texts"] == ["hello", "world"]

    def test_transform_request_has_input_type(self):
        """Request must include 'input_type' (required by Cohere API)."""
        result = self.config.transform_embedding_request(
            model="cohere.embed-multilingual-v3",
            input=["hello"],
            optional_params={},
            headers={},
        )
        assert "input_type" in result
        assert result["input_type"] == litellm.COHERE_DEFAULT_EMBEDDING_INPUT_TYPE

    def test_transform_request_default_input_type_is_search_document(self):
        """Default input_type should be 'search_document'."""
        result = self.config.transform_embedding_request(
            model="cohere.embed-multilingual-v3",
            input=["hello"],
            optional_params={},
            headers={},
        )
        assert result["input_type"] == "search_document"

    def test_transform_request_optional_params_merged(self):
        """Optional params (e.g. custom input_type) should be merged into request."""
        result = self.config.transform_embedding_request(
            model="cohere.embed-multilingual-v3",
            input=["query text"],
            optional_params={"input_type": "search_query"},
            headers={},
        )
        assert result["input_type"] == "search_query"
        assert result["texts"] == ["query text"]

    def test_transform_request_single_string_input(self):
        """Single string input should be wrapped in a list."""
        result = self.config.transform_embedding_request(
            model="cohere.embed-multilingual-v3",
            input="hello",  # type: ignore[arg-type]
            optional_params={},
            headers={},
        )
        assert result["texts"] == ["hello"]


class TestSagemakerCohereEmbeddingResponse:
    """Test response transformation for Cohere SageMaker embeddings."""

    def setup_method(self):
        self.config = SagemakerCohereEmbeddingConfig()

    def _make_response(self, body: dict) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    def test_transform_response_basic(self):
        """Standard Cohere response should be parsed correctly."""
        cohere_response = {
            "id": "abc123",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "texts": ["hello", "world"],
            "meta": {"api_version": {"version": "1"}},
            "response_type": "embeddings_floats",
        }

        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="cohere.embed-multilingual-v3",
            raw_response=self._make_response(cohere_response),
            model_response=model_response,
            logging_obj=None,
            request_data={"texts": ["hello", "world"], "input_type": "search_document"},
        )

        assert result.object == "list"
        assert result.model == "cohere.embed-multilingual-v3"
        assert len(result.data) == 2
        assert result.data[0] == {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3],
        }
        assert result.data[1] == {
            "object": "embedding",
            "index": 1,
            "embedding": [0.4, 0.5, 0.6],
        }

    def test_transform_response_usage(self):
        """Usage should be calculated from the input texts."""
        cohere_response = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "texts": ["hello world"],
        }

        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="cohere.embed-multilingual-v3",
            raw_response=self._make_response(cohere_response),
            model_response=model_response,
            logging_obj=None,
            request_data={"texts": ["hello world"], "input_type": "search_document"},
        )

        assert isinstance(result.usage, Usage)
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == result.usage.prompt_tokens

    def test_transform_response_missing_embeddings_raises(self):
        """Response without 'embeddings' key should raise SagemakerError."""
        from litellm.llms.sagemaker.common_utils import SagemakerError

        bad_response = {"outputs": [[0.1, 0.2]]}  # wrong key

        model_response = EmbeddingResponse()
        with pytest.raises(SagemakerError, match="embeddings"):
            self.config.transform_embedding_response(
                model="cohere.embed-multilingual-v3",
                raw_response=self._make_response(bad_response),
                model_response=model_response,
                logging_obj=None,
                request_data={"texts": ["hello"], "input_type": "search_document"},
            )

    def test_transform_response_invalid_json_raises(self):
        """Invalid JSON response should raise SagemakerError."""
        from litellm.llms.sagemaker.common_utils import SagemakerError

        bad_response = httpx.Response(
            status_code=200,
            content=b"not valid json",
            headers={"content-type": "application/json"},
        )

        model_response = EmbeddingResponse()
        with pytest.raises(SagemakerError, match="Failed to parse response"):
            self.config.transform_embedding_response(
                model="cohere.embed-multilingual-v3",
                raw_response=bad_response,
                model_response=model_response,
                logging_obj=None,
                request_data={"texts": ["hello"], "input_type": "search_document"},
            )


class TestSagemakerCohereOpenAIParams:
    """Test OpenAI parameter mapping for Cohere SageMaker embeddings."""

    def setup_method(self):
        self.config = SagemakerCohereEmbeddingConfig()

    def test_supported_params(self):
        params = self.config.get_supported_openai_params("cohere.embed-multilingual-v3")
        assert "encoding_format" in params
        assert "dimensions" in params

    def test_map_encoding_format(self):
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": "float"},
            optional_params={},
            model="cohere.embed-multilingual-v3",
            drop_params=False,
        )
        assert result["embedding_types"] == ["float"]

    def test_map_dimensions(self):
        result = self.config.map_openai_params(
            non_default_params={"dimensions": 512},
            optional_params={},
            model="cohere.embed-multilingual-v3",
            drop_params=False,
        )
        assert result["output_dimension"] == 512

    def test_map_encoding_format_already_list(self):
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": ["float", "int8"]},
            optional_params={},
            model="cohere.embed-multilingual-v3",
            drop_params=False,
        )
        assert result["embedding_types"] == ["float", "int8"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
