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
                        {"type": "image_url", "image_url": "https://example.com/a.png"},
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
        assert request["inputs"][0]["content"][2] == {
            "type": "image_url",
            "image_url": "https://example.com/a.png",
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

    def test_map_openai_params_dimensions(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        config = VoyageMultimodalEmbeddingConfig()
        assert config.get_supported_openai_params("voyage-multimodal-3.5") == [
            "dimensions"
        ]
        optional_params = config.map_openai_params(
            {"dimensions": 512}, {}, "voyage-multimodal-3.5", False
        )
        assert optional_params == {"output_dimension": 512}
        assert (
            config.map_openai_params({}, {}, "voyage-multimodal-3.5", False) == {}
        )

    def test_validate_environment_uses_api_key(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        config = VoyageMultimodalEmbeddingConfig()
        headers = config.validate_environment(
            {}, "voyage-multimodal-3.5", [], {}, {}, api_key="test-key"
        )
        assert headers == {"Authorization": "Bearer test-key"}

    def test_validate_environment_uses_secret_fallback(self, monkeypatch):
        import litellm.llms.voyage.embedding.transformation_multimodal as module
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        def fake_get_secret(name):
            return "secret-key" if name == "VOYAGE_AI_API_KEY" else None

        monkeypatch.setattr(module, "get_secret_str", fake_get_secret)
        config = VoyageMultimodalEmbeddingConfig()
        headers = config.validate_environment(
            {}, "voyage-multimodal-3.5", [], {}, {}, api_key=None
        )
        assert headers == {"Authorization": "Bearer secret-key"}

    def test_validate_environment_raises_without_api_key(self, monkeypatch):
        import litellm.llms.voyage.embedding.transformation_multimodal as module
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        monkeypatch.setattr(module, "get_secret_str", lambda name: None)
        config = VoyageMultimodalEmbeddingConfig()
        with pytest.raises(ValueError) as exc_info:
            config.validate_environment(
                {}, "voyage-multimodal-3.5", [], {}, {}, api_key=None
            )
        assert "VOYAGE_API_KEY" in str(exc_info.value)

    def test_normalize_image_url_dict_missing_url_raises(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        config = VoyageMultimodalEmbeddingConfig()
        with pytest.raises(ValueError) as exc_info:
            config._normalize_content_item({"type": "image_url", "image_url": {}})
        assert "image_url" in str(exc_info.value)

    def test_is_multimodal_embeddings_helper(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        assert VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings(
            "voyage-multimodal-3"
        )
        assert VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings(
            "VOYAGE-MULTIMODAL-3.5"
        )
        assert not VoyageMultimodalEmbeddingConfig.is_multimodal_embeddings(
            "voyage-3.5"
        )

    def test_utils_routing_via_provider_config_and_dimensions(self):
        import litellm
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )
        from litellm.utils import (
            ProviderConfigManager,
            get_optional_params_embeddings,
        )

        config = ProviderConfigManager.get_provider_embedding_config(
            model="voyage-multimodal-3.5", provider=litellm.LlmProviders.VOYAGE
        )
        assert isinstance(config, VoyageMultimodalEmbeddingConfig)

        optional_params = get_optional_params_embeddings(
            model="voyage-multimodal-3.5",
            dimensions=1024,
            custom_llm_provider="voyage",
            drop_params=True,
        )
        assert optional_params.get("output_dimension") == 1024

    def test_get_supported_openai_params_voyage_routes_multimodal(self):
        from litellm.litellm_core_utils.get_supported_openai_params import (
            get_supported_openai_params,
        )

        multimodal_params = get_supported_openai_params(
            model="voyage-multimodal-3.5",
            custom_llm_provider="voyage",
            request_type="embeddings",
        )
        assert multimodal_params == ["dimensions"]

        standard_params = get_supported_openai_params(
            model="voyage-3.5",
            custom_llm_provider="voyage",
            request_type="embeddings",
        )
        assert "dimensions" in standard_params
        assert "encoding_format" in standard_params

    def test_passthrough_non_content_input(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
        )

        config = VoyageMultimodalEmbeddingConfig()
        request = config.transform_embedding_request(
            "voyage-multimodal-3.5", [{"foo": "bar"}], {}, {}
        )
        assert request["inputs"] == [{"foo": "bar"}]

    def test_error_response_transformation_and_error_class(self):
        from litellm.llms.voyage.embedding.transformation_multimodal import (
            VoyageMultimodalEmbeddingConfig,
            VoyageMultimodalEmbeddingError,
        )
        from litellm.types.utils import EmbeddingResponse

        config = VoyageMultimodalEmbeddingConfig()
        raw_response = MagicMock()
        raw_response.json.side_effect = ValueError("not json")
        raw_response.status_code = 400
        raw_response.text = "bad request"

        with pytest.raises(VoyageMultimodalEmbeddingError) as exc_info:
            config.transform_embedding_response(
                "voyage-multimodal-3.5", raw_response, EmbeddingResponse(), MagicMock()
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.message == "bad request"

        error = config.get_error_class("rate limited", 429, {"x-test": "1"})
        assert isinstance(error, VoyageMultimodalEmbeddingError)
        assert error.status_code == 429
        assert error.message == "rate limited"
