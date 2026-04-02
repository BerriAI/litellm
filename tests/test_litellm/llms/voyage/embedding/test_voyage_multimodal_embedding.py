"""
Test cases for Voyage multimodal embedding configuration.

Tests the VoyageMultimodalEmbeddingConfig class including model detection,
URL generation, parameter mapping, input transformation, and response handling.
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.voyage.embedding.transformation_multimodal import (
    VoyageMultimodalEmbeddingConfig,
    _convert_content_block,
)
from litellm.types.utils import EmbeddingResponse, Usage


class TestIsMultimodalEmbeddings:
    def test_multimodal_model_detected(self):
        assert (
            VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings(
                "voyage-multimodal-3"
            )
            is True
        )

    def test_multimodal_3_5_detected(self):
        assert (
            VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings(
                "voyage-multimodal-3.5"
            )
            is True
        )

    def test_case_insensitive(self):
        assert (
            VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings(
                "Voyage-Multimodal-3"
            )
            is True
        )

    def test_standard_model_not_detected(self):
        assert (
            VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings("voyage-3")
            is False
        )

    def test_contextual_model_not_detected(self):
        assert (
            VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings("voyage-context-3")
            is False
        )


class TestGetCompleteUrl:
    def setup_method(self):
        self.config = VoyageMultimodalEmbeddingConfig()

    def test_default_url(self):
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="voyage-multimodal-3",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.voyageai.com/v1/multimodalembeddings"

    def test_custom_api_base(self):
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/v1",
            api_key=None,
            model="voyage-multimodal-3",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/multimodalembeddings"

    def test_custom_api_base_already_has_path(self):
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/v1/multimodalembeddings",
            api_key=None,
            model="voyage-multimodal-3",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/multimodalembeddings"


class TestGetSupportedOpenaiParams:
    def test_supported_params(self):
        config = VoyageMultimodalEmbeddingConfig()
        params = config.get_supported_openai_params("voyage-multimodal-3")
        assert "encoding_format" in params
        assert "dimensions" in params
        assert "input_type" in params


class TestMapOpenaiParams:
    def setup_method(self):
        self.config = VoyageMultimodalEmbeddingConfig()

    def test_encoding_format_mapped_to_output_encoding(self):
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": "base64"},
            optional_params={},
            model="voyage-multimodal-3",
            drop_params=False,
        )
        assert result == {"output_encoding": "base64"}

    def test_dimensions_mapped_to_output_dimension(self):
        result = self.config.map_openai_params(
            non_default_params={"dimensions": 1024},
            optional_params={},
            model="voyage-multimodal-3",
            drop_params=False,
        )
        assert result == {"output_dimension": 1024}

    def test_input_type_passed_through(self):
        result = self.config.map_openai_params(
            non_default_params={"input_type": "query"},
            optional_params={},
            model="voyage-multimodal-3",
            drop_params=False,
        )
        assert result == {"input_type": "query"}

    def test_all_params_combined(self):
        result = self.config.map_openai_params(
            non_default_params={
                "encoding_format": "base64",
                "dimensions": 512,
                "input_type": "document",
            },
            optional_params={},
            model="voyage-multimodal-3",
            drop_params=False,
        )
        assert result == {
            "output_encoding": "base64",
            "output_dimension": 512,
            "input_type": "document",
        }


class TestConvertContentBlock:
    def test_text_block(self):
        result = _convert_content_block({"type": "text", "text": "hello"})
        assert result == {"type": "text", "text": "hello"}

    def test_image_url_block_string(self):
        result = _convert_content_block(
            {"type": "image_url", "image_url": "https://example.com/img.jpg"}
        )
        assert result == {
            "type": "image_url",
            "image_url": "https://example.com/img.jpg",
        }

    def test_image_url_block_openai_format(self):
        result = _convert_content_block(
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
        )
        assert result == {
            "type": "image_url",
            "image_url": "https://example.com/img.jpg",
        }

    def test_image_url_block_base64_data_uri(self):
        data_uri = "data:image/jpeg;base64,/9j/4AAQ..."
        result = _convert_content_block({"type": "image_url", "image_url": data_uri})
        assert result == {"type": "image_base64", "image_base64": data_uri}

    def test_image_url_block_base64_data_uri_openai_format(self):
        data_uri = "data:image/jpeg;base64,/9j/4AAQ..."
        result = _convert_content_block(
            {"type": "image_url", "image_url": {"url": data_uri}}
        )
        assert result == {"type": "image_base64", "image_base64": data_uri}

    def test_image_base64_block(self):
        result = _convert_content_block(
            {"type": "image_base64", "image_base64": "data:image/png;base64,abc"}
        )
        assert result == {
            "type": "image_base64",
            "image_base64": "data:image/png;base64,abc",
        }

    def test_video_url_block(self):
        result = _convert_content_block(
            {"type": "video_url", "video_url": "https://example.com/video.mp4"}
        )
        assert result == {
            "type": "video_url",
            "video_url": "https://example.com/video.mp4",
        }

    def test_unknown_type_passed_through(self):
        block = {"type": "unknown", "data": "something"}
        result = _convert_content_block(block)
        assert result == block


class TestTransformEmbeddingRequest:
    def setup_method(self):
        self.config = VoyageMultimodalEmbeddingConfig()

    def test_single_text_string(self):
        result = self.config.transform_embedding_request(
            model="voyage-multimodal-3",
            input="Hello world",
            optional_params={},
            headers={},
        )
        assert result == {
            "inputs": [{"content": [{"type": "text", "text": "Hello world"}]}],
            "model": "voyage-multimodal-3",
        }

    def test_text_list(self):
        result = self.config.transform_embedding_request(
            model="voyage-multimodal-3",
            input=["Hello", "World"],
            optional_params={},
            headers={},
        )
        assert result == {
            "inputs": [
                {"content": [{"type": "text", "text": "Hello"}]},
                {"content": [{"type": "text", "text": "World"}]},
            ],
            "model": "voyage-multimodal-3",
        }

    def test_multimodal_input_with_content_blocks(self):
        input_data = [
            {
                "content": [
                    {"type": "text", "text": "A photo of a cat"},
                    {"type": "image_url", "image_url": "https://example.com/cat.jpg"},
                ]
            }
        ]
        result = self.config.transform_embedding_request(
            model="voyage-multimodal-3",
            input=input_data,
            optional_params={},
            headers={},
        )
        assert result == {
            "inputs": [
                {
                    "content": [
                        {"type": "text", "text": "A photo of a cat"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/cat.jpg",
                        },
                    ]
                }
            ],
            "model": "voyage-multimodal-3",
        }

    def test_openai_style_image_url_converted(self):
        input_data = [
            {
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ]
            }
        ]
        result = self.config.transform_embedding_request(
            model="voyage-multimodal-3",
            input=input_data,
            optional_params={},
            headers={},
        )
        assert result["inputs"][0]["content"][1] == {
            "type": "image_url",
            "image_url": "https://example.com/img.png",
        }

    def test_optional_params_included(self):
        result = self.config.transform_embedding_request(
            model="voyage-multimodal-3",
            input="test",
            optional_params={"input_type": "query", "output_dimension": 512},
            headers={},
        )
        assert result["input_type"] == "query"
        assert result["output_dimension"] == 512


class TestTransformEmbeddingResponse:
    def setup_method(self):
        self.config = VoyageMultimodalEmbeddingConfig()

    def test_standard_response(self):
        voyage_response = {
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "object": "list",
            "model": "voyage-multimodal-3",
            "usage": {
                "text_tokens": 5,
                "image_pixels": 2000000,
                "total_tokens": 1005,
            },
        }
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(voyage_response).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="voyage-multimodal-3",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={"inputs": []},
        )

        assert result.object == "list"
        assert result.model == "voyage-multimodal-3"
        assert len(result.data) == 2
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert isinstance(result.usage, Usage)
        assert result.usage.total_tokens == 1005

    def test_error_response(self):
        mock_response = httpx.Response(
            status_code=400,
            content=b"not json",
            headers={"content-type": "text/plain"},
        )
        model_response = EmbeddingResponse()
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingError,
        )

        with pytest.raises(VoyageMultimodalEmbeddingError):
            self.config.transform_embedding_response(
                model="voyage-multimodal-3",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=None,
                request_data={},
            )


class TestProviderConfigRouting:
    def test_multimodal_model_returns_multimodal_config(self):
        from litellm.utils import ProviderConfigManager
        import litellm

        config = ProviderConfigManager.get_provider_embedding_config(
            model="voyage-multimodal-3",
            provider=litellm.LlmProviders.VOYAGE,
        )
        assert isinstance(config, VoyageMultimodalEmbeddingConfig)

    def test_multimodal_3_5_returns_multimodal_config(self):
        from litellm.utils import ProviderConfigManager
        import litellm

        config = ProviderConfigManager.get_provider_embedding_config(
            model="voyage-multimodal-3.5",
            provider=litellm.LlmProviders.VOYAGE,
        )
        assert isinstance(config, VoyageMultimodalEmbeddingConfig)

    def test_standard_model_returns_standard_config(self):
        from litellm.utils import ProviderConfigManager
        from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig
        import litellm

        config = ProviderConfigManager.get_provider_embedding_config(
            model="voyage-3",
            provider=litellm.LlmProviders.VOYAGE,
        )
        assert isinstance(config, VoyageEmbeddingConfig)

    def test_contextual_model_returns_contextual_config(self):
        from litellm.utils import ProviderConfigManager
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )
        import litellm

        config = ProviderConfigManager.get_provider_embedding_config(
            model="voyage-context-3",
            provider=litellm.LlmProviders.VOYAGE,
        )
        assert isinstance(config, VoyageContextualEmbeddingConfig)


if __name__ == "__main__":
    pytest.main([__file__])
