"""
Test VoyageAI multimodal embeddings transformation and request handling.

Tests the VoyageMultimodalEmbeddingConfig class which handles:
- voyage-multimodal-3 and voyage-multimodal-3.5 models
- Text, image_url, image_base64, video_url, video_base64 content types
- Routing to /v1/multimodalembeddings endpoint
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../.."))

import httpx
import pytest

import litellm
from litellm.llms.voyage.embedding.transformation_multimodal import (
    VoyageMultimodalEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse


class TestVoyageMultimodalEmbeddingConfig:
    """Test VoyageMultimodalEmbeddingConfig transformation class."""

    def test_is_multimodal_embedding_detection(self):
        """Test that multimodal models are correctly detected."""
        config = VoyageMultimodalEmbeddingConfig()

        # Should detect multimodal models
        assert config.is_multimodal_embedding("voyage-multimodal-3") is True
        assert config.is_multimodal_embedding("voyage-multimodal-3.5") is True
        assert config.is_multimodal_embedding("voyage/voyage-multimodal-3") is True
        assert config.is_multimodal_embedding("voyage/voyage-multimodal-3.5") is True

        # Should not detect non-multimodal models
        assert config.is_multimodal_embedding("voyage-3.5") is False
        assert config.is_multimodal_embedding("voyage-context-3") is False
        assert config.is_multimodal_embedding("voyage-code-3") is False

    def test_get_complete_url_default(self):
        """Test that default URL points to multimodalembeddings endpoint."""
        config = VoyageMultimodalEmbeddingConfig()

        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="voyage-multimodal-3.5",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://api.voyageai.com/v1/multimodalembeddings"

    def test_get_complete_url_with_custom_base(self):
        """Test URL generation with custom api_base."""
        config = VoyageMultimodalEmbeddingConfig()

        # Without trailing endpoint
        url = config.get_complete_url(
            api_base="https://custom.api.com/v1",
            api_key="test-key",
            model="voyage-multimodal-3.5",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/multimodalembeddings"

        # With trailing endpoint already
        url = config.get_complete_url(
            api_base="https://custom.api.com/v1/multimodalembeddings",
            api_key="test-key",
            model="voyage-multimodal-3.5",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/multimodalembeddings"

    def test_transform_request_text_only(self):
        """Test request transformation with text-only input."""
        config = VoyageMultimodalEmbeddingConfig()

        result = config.transform_embedding_request(
            model="voyage-multimodal-3.5",
            input=["Hello world"],
            optional_params={},
            headers={},
        )

        assert result["model"] == "voyage-multimodal-3.5"
        assert result["inputs"] == [{"content": [{"type": "text", "text": "Hello world"}]}]

    def test_transform_request_explicit_content_format(self):
        """Test request transformation with explicit content format."""
        config = VoyageMultimodalEmbeddingConfig()

        result = config.transform_embedding_request(
            model="voyage-multimodal-3.5",
            input=[
                {
                    "content": [
                        {"type": "text", "text": "A beach scene"},
                        {"type": "image_url", "image_url": "https://example.com/beach.jpg"},
                    ]
                }
            ],
            optional_params={},
            headers={},
        )

        assert result["model"] == "voyage-multimodal-3.5"
        assert len(result["inputs"]) == 1
        assert result["inputs"][0]["content"][0] == {"type": "text", "text": "A beach scene"}
        assert result["inputs"][0]["content"][1] == {
            "type": "image_url",
            "image_url": "https://example.com/beach.jpg",
        }

    def test_transform_request_with_video(self):
        """Test request transformation with video content (3.5 only)."""
        config = VoyageMultimodalEmbeddingConfig()

        result = config.transform_embedding_request(
            model="voyage-multimodal-3.5",
            input=[
                {
                    "content": [
                        {"type": "text", "text": "Demo video"},
                        {"type": "video_url", "video_url": "https://example.com/demo.mp4"},
                    ]
                }
            ],
            optional_params={},
            headers={},
        )

        assert result["inputs"][0]["content"][1] == {
            "type": "video_url",
            "video_url": "https://example.com/demo.mp4",
        }

    def test_transform_request_with_base64_content(self):
        """Test request transformation with base64-encoded content."""
        config = VoyageMultimodalEmbeddingConfig()

        result = config.transform_embedding_request(
            model="voyage-multimodal-3.5",
            input=[
                {
                    "content": [
                        {"type": "image_base64", "image_base64": "iVBORw0KGgoAAAANS..."},
                        {"type": "video_base64", "video_base64": "AAAAIGZ0eXBpc29..."},
                    ]
                }
            ],
            optional_params={},
            headers={},
        )

        assert result["inputs"][0]["content"][0]["type"] == "image_base64"
        assert result["inputs"][0]["content"][1]["type"] == "video_base64"

    def test_transform_request_with_optional_params(self):
        """Test that optional params are included in request."""
        config = VoyageMultimodalEmbeddingConfig()

        result = config.transform_embedding_request(
            model="voyage-multimodal-3.5",
            input=["Test text"],
            optional_params={"output_dimension": 512, "input_type": "document"},
            headers={},
        )

        assert result["output_dimension"] == 512
        assert result["input_type"] == "document"

    def test_map_openai_params_dimensions(self):
        """Test that OpenAI dimensions param maps to output_dimension."""
        config = VoyageMultimodalEmbeddingConfig()

        result = config.map_openai_params(
            non_default_params={"dimensions": 1024},
            optional_params={},
            model="voyage-multimodal-3.5",
            drop_params=False,
        )

        assert result["output_dimension"] == 1024

    def test_transform_response(self):
        """Test response transformation."""
        config = VoyageMultimodalEmbeddingConfig()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "model": "voyage-multimodal-3.5",
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"total_tokens": 10, "text_tokens": 5},
        }

        model_response = EmbeddingResponse()
        logging_obj = MagicMock()

        result = config.transform_embedding_response(
            model="voyage-multimodal-3.5",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=logging_obj,
        )

        assert result.model == "voyage-multimodal-3.5"
        assert result.data == [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
        assert result.usage.prompt_tokens == 5
        assert result.usage.total_tokens == 10

    def test_get_supported_openai_params(self):
        """Test that supported params are returned."""
        config = VoyageMultimodalEmbeddingConfig()

        params = config.get_supported_openai_params(model="voyage-multimodal-3.5")

        assert "encoding_format" in params
        assert "dimensions" in params
        assert "input_type" in params
        assert "truncation" in params


class TestVoyageMultimodalProviderRouting:
    """Test that multimodal models are routed correctly."""

    def test_multimodal_detected_in_get_llm_provider(self):
        """Test that multimodal models are detected as voyage provider."""
        model, provider, api_base, api_key = litellm.get_llm_provider(
            model="voyage/voyage-multimodal-3.5"
        )
        assert provider == "voyage"
        assert model == "voyage-multimodal-3.5"

    def test_multimodal_config_used_for_supported_params(self):
        """Test that multimodal config is used for get_supported_openai_params."""
        params = litellm.get_supported_openai_params(
            model="voyage-multimodal-3.5", custom_llm_provider="voyage"
        )

        assert params is not None
        assert "dimensions" in params
        assert "input_type" in params
