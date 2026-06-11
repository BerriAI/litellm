import json
from unittest.mock import MagicMock

import pytest


class TestVoyageMultimodalEmbeddings:
    def test_multimodal_model_detection(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        assert VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings(
            "voyage-multimodal-3.5"
        )
        assert VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings(
            "voyage-multimodal-3"
        )
        assert not VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings("voyage-4")

    def test_multimodal_embedding_url_generation(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        config = VoyageMultimodalEmbeddingConfig()
        assert (
            config.get_complete_url(None, None, "voyage-multimodal-3.5", {}, {})
            == "https://api.voyageai.com/v1/multimodalembeddings"
        )
        assert (
            config.get_complete_url(
                "https://custom.api.com", None, "voyage-multimodal-3.5", {}, {}
            )
            == "https://custom.api.com/multimodalembeddings"
        )
        assert (
            config.get_complete_url(
                "https://custom.api.com/multimodalembeddings",
                None,
                "voyage-multimodal-3.5",
                {},
                {},
            )
            == "https://custom.api.com/multimodalembeddings"
        )

    def test_multimodal_embedding_request_transformation(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        config = VoyageMultimodalEmbeddingConfig()
        data_uri = "data:image/png;base64,AAAA"
        request = config.transform_embedding_request(
            "voyage-multimodal-3.5",
            [
                {
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ]
                }
            ],
            {"input_type": "document", "output_dimension": 512},
            {},
        )

        assert request["model"] == "voyage-multimodal-3.5"
        assert "inputs" in request
        assert "input" not in request
        assert request["input_type"] == "document"
        assert request["output_dimension"] == 512
        assert request["inputs"][0]["content"][1] == {
            "type": "image_base64",
            "image_base64": "AAAA",
        }

    def test_multimodal_embedding_string_input_transformation(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        config = VoyageMultimodalEmbeddingConfig()
        request = config.transform_embedding_request(
            "voyage-multimodal-3.5", "hello", {}, {}
        )
        assert request["inputs"] == [
            {"content": [{"type": "text", "text": "hello"}]}
        ]

    def test_multimodal_embedding_response_transformation(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )
        from litellm.types.utils import EmbeddingResponse

        config = VoyageMultimodalEmbeddingConfig()
        response_payload = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
            ],
            "model": "voyage-multimodal-3.5",
            "usage": {
                "text_tokens": 2,
                "image_pixels": 0,
                "video_pixels": 0,
                "total_tokens": 2,
            },
        }
        raw_response = MagicMock()
        raw_response.json.return_value = response_payload
        raw_response.status_code = 200
        raw_response.text = json.dumps(response_payload)

        model_response = EmbeddingResponse()
        transformed = config.transform_embedding_response(
            "voyage-multimodal-3.5", raw_response, model_response, MagicMock()
        )

        assert transformed.model == "voyage-multimodal-3.5"
        assert transformed.object == "list"
        assert transformed.data == response_payload["data"]
        assert transformed.usage.prompt_tokens == 2
        assert transformed.usage.total_tokens == 2

    def test_provider_config_manager_routes_multimodal_models(self):
        import litellm
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_embedding_config(
            model="voyage-multimodal-3.5", provider=litellm.LlmProviders.VOYAGE
        )

        assert isinstance(config, VoyageMultimodalEmbeddingConfig)
