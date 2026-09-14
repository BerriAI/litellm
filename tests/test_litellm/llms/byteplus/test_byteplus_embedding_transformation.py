import httpx

from litellm.llms.byteplus.embedding.transformation import BytePlusEmbeddingConfig


class TestBytePlusEmbeddingConfig:
    def test_get_supported_openai_params(self):
        config = BytePlusEmbeddingConfig()
        params = config.get_supported_openai_params("skylark-embedding-vision-250615")
        assert "encoding_format" in params
        assert "dimensions" in params
        assert "sparse_embedding" in params

    def test_get_complete_url_text(self):
        config = BytePlusEmbeddingConfig()
        url = config.get_complete_url(
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            api_key="key",
            model="doubao-embedding-text",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/embeddings"

    def test_get_complete_url_multimodal(self):
        config = BytePlusEmbeddingConfig()
        url = config.get_complete_url(
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            api_key="key",
            model="skylark-embedding-vision-250615",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/embeddings/multimodal"

    def test_transform_embedding_request_multimodal(self):
        config = BytePlusEmbeddingConfig()
        multimodal_input = [
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "text", "text": "What is in the video?"},
        ]
        data = config.transform_embedding_request(
            model="skylark-embedding-vision-250615",
            input=multimodal_input,
            optional_params={
                "encoding_format": "float",
                "dimensions": 1024,
                "sparse_embedding": {"type": "enabled"},
            },
            headers={},
        )
        assert data["model"] == "skylark-embedding-vision-250615"
        assert len(data["input"]) == 3
        assert data["encoding_format"] == "float"
        assert data["dimensions"] == 1024
        assert data["sparse_embedding"] == {"type": "enabled"}

    def test_transform_embedding_response_multimodal_object(self):
        config = BytePlusEmbeddingConfig()
        mock_json = {
            "created": 1743575029,
            "data": {
                "embedding": [-0.123, -0.355, 0.255],
                "sparse_embedding": [{"index": 1, "value": 0.088}],
                "object": "embedding",
            },
            "id": "req-123",
            "model": "skylark-embedding-vision-250615",
            "object": "list",
            "usage": {"prompt_tokens": 25, "total_tokens": 25},
        }
        raw_resp = httpx.Response(status_code=200, json=mock_json)
        res = config.transform_embedding_response(
            model="skylark-embedding-vision-250615",
            raw_response=raw_resp,
            model_response=None,
            logging_obj=None,
            api_key="key",
            request_data={},
            optional_params={},
            litellm_params={},
        )
        assert res.model == "skylark-embedding-vision-250615"
        assert len(res.data) == 1
        assert res.data[0]["embedding"] == [-0.123, -0.355, 0.255]

    def test_litellm_embedding_dispatch_byteplus(self, monkeypatch):
        import litellm
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        mock_json = {
            "created": 1743575029,
            "data": {
                "embedding": [0.1, 0.2, 0.3],
                "object": "embedding",
            },
            "id": "req-123",
            "model": "skylark-embedding-vision-250615",
            "object": "list",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_resp = httpx.Response(status_code=200, json=mock_json)

        def mock_post(*args, **kwargs):
            return mock_resp

        monkeypatch.setattr(HTTPHandler, "post", mock_post)
        monkeypatch.setattr(httpx.Client, "post", mock_post)

        res = litellm.embedding(
            model="byteplus/skylark-embedding-vision-250615",
            input="test text",
            api_key="mock-key",
        )
        assert res.model == "skylark-embedding-vision-250615"
        assert res.data[0]["embedding"] == [0.1, 0.2, 0.3]

    def test_transform_embedding_request_extra_body_reserved_fields(self):
        config = BytePlusEmbeddingConfig()
        data = config.transform_embedding_request(
            model="doubao-embedding-text",
            input="test text",
            optional_params={
                "extra_body": {
                    "model": "malicious-model",
                    "input": "malicious input",
                    "custom_param": "val",
                }
            },
            headers={},
        )
        assert data["model"] == "doubao-embedding-text"
        assert data["input"] == ["test text"]
        assert data["custom_param"] == "val"

    def test_extract_embedding_data_variations(self):
        from litellm.llms.byteplus.embedding.transformation import _extract_embedding_data

        # Test list
        data_list = [{"embedding": [0.1, 0.2]}]
        assert _extract_embedding_data(data_list) == data_list

        # Test other/empty
        assert _extract_embedding_data(None) == []
        assert _extract_embedding_data("invalid") == []

    def test_get_complete_url_variations(self):
        config = BytePlusEmbeddingConfig()
        url1 = config.get_complete_url(
            api_base="https://custom.com/embeddings",
            api_key="key",
            model="doubao-embedding-text",
            optional_params={},
            litellm_params={},
        )
        assert url1 == "https://custom.com/embeddings"

        url2 = config.get_complete_url(
            api_base="https://custom.com",
            api_key="key",
            model="doubao-embedding-text",
            optional_params={},
            litellm_params={},
        )
        assert url2 == "https://custom.com/api/v3/embeddings"

    def test_validate_environment_missing_key(self, monkeypatch):
        import pytest

        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        config = BytePlusEmbeddingConfig()
        with pytest.raises(ValueError, match="BytePlus API key is required"):
            config.validate_environment(
                headers={},
                model="doubao-embedding-text",
                messages=[],
                optional_params={},
                litellm_params={},
            )

        headers = config.validate_environment(
            headers={},
            model="doubao-embedding-text",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )
        assert headers.get("Authorization") == "Bearer test-key"

    def test_get_error_class(self):
        config = BytePlusEmbeddingConfig()
        err = config.get_error_class("embedding error", 400, headers={"x-request-id": "123"})
        assert err.status_code == 400
        assert "embedding error" in err.message

    def test_provider_config_manager_embedding(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_embedding_config(
            model="byteplus/doubao-embedding-text",
            provider=LlmProviders.BYTEPLUS,
        )
        assert isinstance(cfg, BytePlusEmbeddingConfig)

    def test_transform_embedding_response_invalid_json(self):
        import pytest
        from litellm.types.utils import EmbeddingResponse

        config = BytePlusEmbeddingConfig()
        raw_resp = httpx.Response(status_code=200, text="not json")
        with pytest.raises(ValueError, match="Failed to parse BytePlus response as JSON"):
            config.transform_embedding_response(
                model="doubao-embedding-text",
                raw_response=raw_resp,
                model_response=EmbeddingResponse(),
                logging_obj=None,
                api_key="key",
                request_data={},
                optional_params={},
                litellm_params={},
            )

    def test_litellm_embedding_byteplus(self, monkeypatch):
        import litellm
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        mock_json = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
            "model": "ep-20250101",
            "object": "list",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_req = httpx.Request("POST", "https://ark.ap-southeast.bytepluses.com/api/v3/embeddings")
        mock_resp = httpx.Response(status_code=200, json=mock_json, request=mock_req)

        def mock_post(*args, **kwargs):
            return mock_resp

        monkeypatch.setattr(HTTPHandler, "post", mock_post)
        monkeypatch.setattr(httpx.Client, "post", mock_post)
        monkeypatch.setattr(httpx.Client, "send", mock_post)

        res = litellm.embedding(
            model="byteplus/ep-20250101",
            input=["hello"],
            api_key="test-api-key",
            extra_headers={"X-Custom": "val"},
        )
        assert res.data[0]["embedding"] == [0.1, 0.2, 0.3]

    def test_litellm_embedding_byteplus_missing_key(self, monkeypatch):
        import pytest
        import litellm

        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None)

        with pytest.raises(Exception, match="Missing API key for Byteplus"):
            litellm.embedding(
                model="byteplus/ep-20250101",
                input=["hello"],
            )
