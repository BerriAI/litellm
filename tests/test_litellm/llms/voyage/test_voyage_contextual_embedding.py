import json
from unittest.mock import MagicMock

import pytest


class TestVoyageContextualEmbeddings:
    def test_contextual_model_detection(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        assert VoyageContextualEmbeddingConfig.is_contextualized_embeddings("voyage-context-3")
        assert VoyageContextualEmbeddingConfig.is_contextualized_embeddings("voyage-context-4")
        assert not VoyageContextualEmbeddingConfig.is_contextualized_embeddings("voyage-3-lite")

    def test_url_generation(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        assert (
            config.get_complete_url(None, None, "voyage-context-4", {}, {})
            == "https://api.voyageai.com/v1/contextualizedembeddings"
        )
        assert (
            config.get_complete_url("https://custom.api.com", None, "voyage-context-4", {}, {})
            == "https://custom.api.com/contextualizedembeddings"
        )
        assert (
            config.get_complete_url(
                "https://custom.api.com/contextualizedembeddings",
                None,
                "voyage-context-4",
                {},
                {},
            )
            == "https://custom.api.com/contextualizedembeddings"
        )

    def test_get_supported_openai_params(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        assert config.get_supported_openai_params("voyage-context-4") == [
            "encoding_format",
            "dimensions",
        ]

    def test_map_openai_params(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        result = config.map_openai_params(
            {"encoding_format": "float", "dimensions": 512}, {}, "voyage-context-4", False
        )
        assert result["encoding_format"] == "float"
        assert result["output_dimension"] == 512

    def test_validate_environment_with_api_key(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        headers = config.validate_environment(
            {}, "voyage-context-4", [], {}, {}, api_key="test-key"
        )
        assert headers == {"Authorization": "Bearer test-key"}

    def test_validate_environment_secret_fallback(self, monkeypatch):
        import litellm.llms.voyage.embedding.transformation_contextual as module
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        def fake_get_secret(name):
            return "secret-key" if name == "VOYAGE_API_KEY" else None

        monkeypatch.setattr(module, "get_secret_str", fake_get_secret)
        config = VoyageContextualEmbeddingConfig()
        headers = config.validate_environment(
            {}, "voyage-context-4", [], {}, {}, api_key=None
        )
        assert headers == {"Authorization": "Bearer secret-key"}

    def test_nested_list_passthrough(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        nested = [["Hello", "world"], ["Test"]]
        transformed = config.transform_embedding_request(
            "voyage-context-4", nested, {}, {}
        )
        assert transformed["inputs"] == nested
        assert transformed["model"] == "voyage-context-4"
        assert "enable_auto_chunking" not in transformed

    def test_flat_list_str_auto_chunked(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        transformed = config.transform_embedding_request(
            "voyage-context-4", ["Hello", "world"], {}, {}
        )
        assert transformed["inputs"] == ["Hello", "world"]
        assert transformed["enable_auto_chunking"] is True
        assert transformed["chunk_size"] == 32000
        assert transformed["input_type"] == "document"

    def test_flat_list_str_query_no_auto_chunk(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        transformed = config.transform_embedding_request(
            "voyage-context-4", ["Hello", "world"], {"input_type": "query"}, {}
        )
        assert transformed["inputs"] == ["Hello", "world"]
        assert transformed["input_type"] == "query"
        assert "enable_auto_chunking" not in transformed

    def test_flat_list_str_document_preserves_input_type(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        transformed = config.transform_embedding_request(
            "voyage-context-4", ["Hello"], {"input_type": "document"}, {}
        )
        assert transformed["input_type"] == "document"
        assert transformed["enable_auto_chunking"] is True

    def test_single_string_auto_chunked(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        transformed = config.transform_embedding_request(
            "voyage-context-4", "Hello", {}, {}
        )
        assert transformed["inputs"] == ["Hello"]
        assert transformed["enable_auto_chunking"] is True
        assert transformed["input_type"] == "document"

    def test_single_string_query_no_auto_chunk(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()
        transformed = config.transform_embedding_request(
            "voyage-context-4", "Hello", {"input_type": "query"}, {}
        )
        assert transformed["inputs"] == ["Hello"]
        assert transformed["input_type"] == "query"
        assert "enable_auto_chunking" not in transformed

    def test_response_transformation(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )
        from litellm.types.utils import EmbeddingResponse

        config = VoyageContextualEmbeddingConfig()
        response_payload = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "voyage-context-4",
            "usage": {"total_tokens": 24},
        }
        raw_response = MagicMock()
        raw_response.json.return_value = response_payload
        raw_response.status_code = 200
        raw_response.text = json.dumps(response_payload)

        model_response = EmbeddingResponse()
        transformed = config.transform_embedding_response(
            "voyage-context-4", raw_response, model_response, MagicMock()
        )
        assert transformed.model == "voyage-context-4"
        assert transformed.object == "list"
        assert transformed.data == response_payload["data"]
        assert transformed.usage.prompt_tokens == 24
        assert transformed.usage.total_tokens == 24

    def test_error_response_and_error_class(self):
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
            VoyageError,
        )
        from litellm.types.utils import EmbeddingResponse

        config = VoyageContextualEmbeddingConfig()
        raw_response = MagicMock()
        raw_response.json.side_effect = ValueError("not json")
        raw_response.status_code = 400
        raw_response.text = "bad request"

        with pytest.raises(VoyageError) as exc_info:
            config.transform_embedding_response(
                "voyage-context-4", raw_response, EmbeddingResponse(), MagicMock()
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.message == "bad request"

        error = config.get_error_class("rate limited", 429, {"x-test": "1"})
        assert isinstance(error, VoyageError)
        assert error.status_code == 429
        assert error.message == "rate limited"
