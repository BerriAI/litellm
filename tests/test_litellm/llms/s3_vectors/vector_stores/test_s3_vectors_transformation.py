from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from litellm.llms.s3_vectors.vector_stores.transformation import (
    S3VectorsVectorStoreConfig,
)
from litellm.types.vector_stores import VectorStoreSearchResponse


def _mock_router(model_names, sync=False):
    """Router mock serving the given embedding model names."""
    router = MagicMock()
    router.get_model_list.return_value = [{"model_name": name} for name in model_names]
    embedding_response = Mock(data=[{"embedding": [0.1, 0.2, 0.3]}])
    if sync:
        router.embedding = MagicMock(return_value=embedding_response)
    else:
        router.aembedding = AsyncMock(return_value=embedding_response)
    return router


class TestS3VectorsVectorStoreConfig:
    def test_init(self):
        """Test that S3VectorsVectorStoreConfig initializes correctly"""
        config = S3VectorsVectorStoreConfig()
        assert config is not None

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are returned"""
        config = S3VectorsVectorStoreConfig()
        params = config.get_supported_openai_params("test-model")
        assert "max_num_results" in params

    def test_get_complete_url(self):
        """Test URL generation for S3 Vectors"""
        config = S3VectorsVectorStoreConfig()
        litellm_params = {"aws_region_name": "us-west-2"}
        url = config.get_complete_url(None, litellm_params)
        assert url == "https://s3vectors.us-west-2.api.aws"

    def test_get_complete_url_missing_region(self, monkeypatch):
        """Missing region falls back to the default region (parity with ingestion)"""
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        config = S3VectorsVectorStoreConfig()
        url = config.get_complete_url(None, {})
        assert url == "https://s3vectors.us-west-2.api.aws"

    def test_get_complete_url_uses_env_region(self, monkeypatch):
        """Missing region param resolves from AWS_REGION_NAME env var"""
        monkeypatch.setenv("AWS_REGION_NAME", "eu-west-1")
        monkeypatch.delenv("AWS_REGION", raising=False)
        config = S3VectorsVectorStoreConfig()
        url = config.get_complete_url(None, {})
        assert url == "https://s3vectors.eu-west-1.api.aws"

    def test_get_complete_url_invalid_region_format(self):
        """Invalid region format raises"""
        config = S3VectorsVectorStoreConfig()
        with pytest.raises(ValueError, match="Invalid AWS region format"):
            config.get_complete_url(None, {"aws_region_name": "Bad_Region!"})

    def test_transform_search_request(self):
        """Full request-body transformation with a router-injected embedding"""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}
        router = _mock_router(["text-embedding-3-small"], sync=True)

        url, request_body = config.transform_search_vector_store_request(
            vector_store_id="test-bucket:test-index",
            query="test query",
            vector_store_search_optional_params={"max_num_results": 7},
            api_base="https://s3vectors.us-west-2.api.aws",
            litellm_logging_obj=mock_logging_obj,
            litellm_params={},
            extra_body=None,
            router=router,
        )

        assert url == "https://s3vectors.us-west-2.api.aws/QueryVectors"
        assert request_body == {
            "vectorBucketName": "test-bucket",
            "indexName": "test-index",
            "queryVector": {"float32": [0.1, 0.2, 0.3]},
            "topK": 7,
            "returnDistance": True,
            "returnMetadata": True,
        }
        assert mock_logging_obj.model_call_details["query"] == "test query"

    @pytest.mark.asyncio
    async def test_atransform_search_uses_router_for_virtual_model(self):
        """Regression: router-served embedding models must resolve via the router,
        not a bare litellm.aembedding call (which has no deployment credentials)."""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}
        router = _mock_router(["my-embedding-model"])

        with patch("litellm.aembedding", new=AsyncMock()) as mock_bare_aembedding:
            url, request_body = await config.atransform_search_vector_store_request(
                vector_store_id="test-bucket:test-index",
                query="test query",
                vector_store_search_optional_params={},
                api_base="https://s3vectors.us-west-2.api.aws",
                litellm_logging_obj=mock_logging_obj,
                litellm_params={"embedding_model": "my-embedding-model"},
                extra_body=None,
                router=router,
            )

        router.aembedding.assert_awaited_once_with(model="my-embedding-model", input=["test query"])
        mock_bare_aembedding.assert_not_awaited()
        assert request_body["queryVector"]["float32"] == [0.1, 0.2, 0.3]
        assert request_body["topK"] == 5  # default

    @pytest.mark.asyncio
    async def test_atransform_search_falls_back_when_router_does_not_serve_model(self):
        """Router present but embedding_model is not a router deployment ->
        bare litellm.aembedding keeps working (provider-prefixed + env creds stores)."""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}
        router = _mock_router(["some-other-model"])

        mock_bare = AsyncMock(return_value=Mock(data=[{"embedding": [0.4, 0.5]}]))
        with patch("litellm.aembedding", new=mock_bare):
            _, request_body = await config.atransform_search_vector_store_request(
                vector_store_id="test-bucket:test-index",
                query="test query",
                vector_store_search_optional_params={},
                api_base="https://s3vectors.us-west-2.api.aws",
                litellm_logging_obj=mock_logging_obj,
                litellm_params={"embedding_model": "azure/text-embedding-3-small"},
                extra_body=None,
                router=router,
            )

        mock_bare.assert_awaited_once_with(model="azure/text-embedding-3-small", input=["test query"])
        router.aembedding.assert_not_awaited()
        assert request_body["queryVector"]["float32"] == [0.4, 0.5]

    @pytest.mark.asyncio
    async def test_atransform_search_without_router_uses_bare_embedding(self):
        """Backward compat: no router -> bare litellm.aembedding as before"""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        mock_bare = AsyncMock(return_value=Mock(data=[{"embedding": [0.6, 0.7]}]))
        with patch("litellm.aembedding", new=mock_bare):
            _, request_body = await config.atransform_search_vector_store_request(
                vector_store_id="test-bucket:test-index",
                query="test query",
                vector_store_search_optional_params={},
                api_base="https://s3vectors.us-west-2.api.aws",
                litellm_logging_obj=mock_logging_obj,
                litellm_params={},
                extra_body=None,
            )

        mock_bare.assert_awaited_once_with(model="text-embedding-3-small", input=["test query"])
        assert request_body["queryVector"]["float32"] == [0.6, 0.7]

    def test_transform_search_uses_router_for_virtual_model_sync(self):
        """Sync twin: router-served embedding model resolves via router.embedding"""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}
        router = _mock_router(["my-embedding-model"], sync=True)

        with patch("litellm.embedding", new=MagicMock()) as mock_bare_embedding:
            _, request_body = config.transform_search_vector_store_request(
                vector_store_id="test-bucket:test-index",
                query="test query",
                vector_store_search_optional_params={},
                api_base="https://s3vectors.us-west-2.api.aws",
                litellm_logging_obj=mock_logging_obj,
                litellm_params={"embedding_model": "my-embedding-model"},
                extra_body=None,
                router=router,
            )

        router.embedding.assert_called_once_with(model="my-embedding-model", input=["test query"])
        mock_bare_embedding.assert_not_called()
        assert request_body["queryVector"]["float32"] == [0.1, 0.2, 0.3]

    def test_transform_search_without_router_uses_bare_embedding_sync(self):
        """Sync twin: no router -> bare litellm.embedding as before"""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        mock_bare = MagicMock(return_value=Mock(data=[{"embedding": [0.8, 0.9]}]))
        with patch("litellm.embedding", new=mock_bare):
            _, request_body = config.transform_search_vector_store_request(
                vector_store_id="test-bucket:test-index",
                query="test query",
                vector_store_search_optional_params={},
                api_base="https://s3vectors.us-west-2.api.aws",
                litellm_logging_obj=mock_logging_obj,
                litellm_params={},
                extra_body=None,
            )

        mock_bare.assert_called_once_with(model="text-embedding-3-small", input=["test query"])
        assert request_body["queryVector"]["float32"] == [0.8, 0.9]

    def test_transform_search_request_invalid_vector_store_id(self):
        """Test that invalid vector_store_id format raises error"""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        with pytest.raises(
            ValueError,
            match="vector_store_id must be in format 'bucket_name:index_name'",
        ):
            config.transform_search_vector_store_request(
                vector_store_id="invalid-format",
                query="test query",
                vector_store_search_optional_params={},
                api_base="https://s3vectors.us-west-2.api.aws",
                litellm_logging_obj=mock_logging_obj,
                litellm_params={},
                extra_body=None,
            )

    def test_transform_search_response(self):
        """Test search response transformation"""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {"query": "test query"}

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "vectors": [
                {
                    "distance": 0.05,  # S3 Vectors returns distance, not score
                    "metadata": {
                        "source_text": "This is test content",
                        "chunk_index": "0",
                        "filename": "test.pdf",
                    },
                },
                {
                    "distance": 0.15,
                    "metadata": {
                        "source_text": "More test content",
                        "chunk_index": "1",
                    },
                },
            ]
        }
        mock_response.status_code = 200
        mock_response.headers = {}

        result = config.transform_search_vector_store_response(
            mock_response, mock_logging_obj
        )

        # VectorStoreSearchResponse is a TypedDict, so check structure instead of isinstance
        assert result["object"] == "vector_store.search_results.page"
        assert result["search_query"] == "test query"
        assert len(result["data"]) == 2
        # Score should be 1 - distance (cosine similarity)
        assert result["data"][0]["score"] == 0.95  # 1 - 0.05
        assert result["data"][0]["content"][0]["text"] == "This is test content"
        assert result["data"][0]["filename"] == "test.pdf"
        assert result["data"][1]["score"] == 0.85  # 1 - 0.15
        assert result["data"][1]["content"][0]["text"] == "More test content"

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping"""
        config = S3VectorsVectorStoreConfig()
        non_default_params = {"max_num_results": 5}
        optional_params = {}

        result = config.map_openai_params(non_default_params, optional_params, False)

        assert result["maxResults"] == 5
