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
