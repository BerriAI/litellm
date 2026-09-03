from collections.abc import Mapping
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from litellm.llms.base_llm.vector_store.transformation import (
    RouterVectorStoreEmbeddingExecutor,
)
from litellm.llms.s3_vectors.vector_stores.transformation import (
    S3VectorsVectorStoreConfig,
)
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import VectorStoreSearchResponse

QUERY_VECTOR = [0.1, 0.2, 0.3]


def _embedding_response(vector):
    return EmbeddingResponse(data=[{"embedding": vector, "index": 0, "object": "embedding"}])


class _RecordingExecutor:
    """Executor double recording every (model, query, configuration) it was asked to embed."""

    def __init__(self, vector=QUERY_VECTOR):
        self.vector = vector
        self.calls = []

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        self.calls.append((model, query, dict(configuration)))
        return _embedding_response(self.vector)

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        self.calls.append((model, query, dict(configuration)))
        return _embedding_response(self.vector)


def _logging_obj():
    logging_obj = Mock()
    logging_obj.model_call_details = {}
    return logging_obj


def _search_kwargs(**overrides):
    kwargs = {
        "vector_store_id": "test-bucket:test-index",
        "query": "test query",
        "vector_store_search_optional_params": {},
        "api_base": "https://s3vectors.us-west-2.api.aws",
        "litellm_logging_obj": _logging_obj(),
        "litellm_params": {},
        "extra_body": None,
    }
    kwargs.update(overrides)
    return kwargs


class TestS3VectorsVectorStoreConfig:
    def test_init(self):
        config = S3VectorsVectorStoreConfig()
        assert config is not None

    def test_get_supported_openai_params(self):
        config = S3VectorsVectorStoreConfig()
        params = config.get_supported_openai_params("test-model")
        assert "max_num_results" in params

    def test_get_complete_url(self):
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
        config = S3VectorsVectorStoreConfig()
        with pytest.raises(ValueError, match="Invalid AWS region format"):
            config.get_complete_url(None, {"aws_region_name": "Bad_Region!"})

    def test_transform_search_request(self):
        """Full request-body transformation with the query embedded through the injected executor"""
        config = S3VectorsVectorStoreConfig()
        logging_obj = _logging_obj()
        executor = _RecordingExecutor()

        url, request_body = config.transform_search_vector_store_request(
            **_search_kwargs(
                vector_store_search_optional_params={"max_num_results": 7},
                litellm_logging_obj=logging_obj,
                embedding_executor=executor,
            )
        )

        assert url == "https://s3vectors.us-west-2.api.aws/QueryVectors"
        assert request_body == {
            "vectorBucketName": "test-bucket",
            "indexName": "test-index",
            "queryVector": {"float32": QUERY_VECTOR},
            "topK": 7,
            "returnDistance": True,
            "returnMetadata": True,
        }
        assert executor.calls == [("text-embedding-3-small", "test query", {})]
        assert logging_obj.model_call_details["query"] == "test query"

    @pytest.mark.parametrize(
        ("litellm_params", "expected_model"),
        [
            ({}, "text-embedding-3-small"),
            ({"embedding_model": ""}, "text-embedding-3-small"),
            ({"embedding_model": "my-embedding-model"}, "my-embedding-model"),
            ({"litellm_embedding_model": "shared-key-model"}, "shared-key-model"),
            (
                {"litellm_embedding_model": "shared-key-model", "embedding_model": "legacy-alias"},
                "shared-key-model",
            ),
        ],
    )
    def test_query_embedding_model_accepts_embedding_model_alias(self, litellm_params, expected_model):
        assert S3VectorsVectorStoreConfig.query_embedding_model(litellm_params) == expected_model

    @pytest.mark.asyncio
    async def test_atransform_search_embeds_alias_and_store_config_through_executor(self):
        """The store's embedding_model alias and litellm_embedding_config reach the executor unchanged"""
        config = S3VectorsVectorStoreConfig()
        executor = _RecordingExecutor(vector=[0.4, 0.5])

        _, request_body = await config.atransform_search_vector_store_request(
            **_search_kwargs(
                query=["test", "query"],
                litellm_params={
                    "embedding_model": "my-embedding-model",
                    "litellm_embedding_config": {"api_key": "store-key"},
                },
                embedding_executor=executor,
            )
        )

        assert executor.calls == [("my-embedding-model", "test query", {"api_key": "store-key"})]
        assert request_body["queryVector"]["float32"] == [0.4, 0.5]
        assert request_body["topK"] == 5

    @pytest.mark.asyncio
    async def test_atransform_search_router_executor_carries_request_metadata(self):
        """Regression (LIT-6750): a bare Router alias resolves through the Router with the request's
        team metadata on the embedding call, so the embedding is attributed to the calling key and team."""
        config = S3VectorsVectorStoreConfig()
        router = MagicMock()
        router.aembedding = AsyncMock(return_value=_embedding_response(QUERY_VECTOR))
        request_metadata = {"user_api_key_team_id": "team-a", "user_api_key": "hashed-key"}

        _, request_body = await config.atransform_search_vector_store_request(
            **_search_kwargs(
                litellm_params={"embedding_model": "team-embeddings"},
                embedding_executor=RouterVectorStoreEmbeddingExecutor(router=router, metadata=request_metadata),
            )
        )

        router.aembedding.assert_awaited_once_with(
            model="team-embeddings", input=["test query"], metadata=request_metadata
        )
        assert request_body["queryVector"]["float32"] == QUERY_VECTOR

    @pytest.mark.asyncio
    async def test_atransform_search_without_executor_uses_bare_embedding(self):
        """Backward compat: SDK callers without an executor keep embedding through litellm.aembedding"""
        config = S3VectorsVectorStoreConfig()

        mock_bare = AsyncMock(return_value=_embedding_response([0.6, 0.7]))
        with patch("litellm.aembedding", new=mock_bare):  # test-quality-ok: stubs the bare-embedding fallback whose request body the test asserts on
            _, request_body = await config.atransform_search_vector_store_request(**_search_kwargs())

        mock_bare.assert_awaited_once_with(model="text-embedding-3-small", input=["test query"])
        assert request_body["queryVector"]["float32"] == [0.6, 0.7]

    def test_transform_search_without_executor_uses_bare_embedding_sync(self):
        """Sync twin: no executor -> bare litellm.embedding as before"""
        config = S3VectorsVectorStoreConfig()

        mock_bare = MagicMock(return_value=_embedding_response([0.8, 0.9]))
        with patch("litellm.embedding", new=mock_bare):  # test-quality-ok: stubs the bare-embedding fallback whose request body the test asserts on
            _, request_body = config.transform_search_vector_store_request(
                **_search_kwargs(litellm_params={"embedding_model": "my-embedding-model"})
            )

        mock_bare.assert_called_once_with(model="my-embedding-model", input=["test query"])
        assert request_body["queryVector"]["float32"] == [0.8, 0.9]

    def test_transform_search_request_invalid_vector_store_id(self):
        """An unparseable vector_store_id raises before any embedding is generated"""
        config = S3VectorsVectorStoreConfig()
        executor = _RecordingExecutor()

        with pytest.raises(
            ValueError,
            match="vector_store_id must be in format 'bucket_name:index_name'",
        ):
            config.transform_search_vector_store_request(
                **_search_kwargs(vector_store_id="invalid-format", embedding_executor=executor)
            )

        assert executor.calls == []

    def test_transform_search_request_bucket_from_litellm_params(self):
        config = S3VectorsVectorStoreConfig()

        _, request_body = config.transform_search_vector_store_request(
            **_search_kwargs(
                vector_store_id="only-index",
                litellm_params={"vector_bucket_name": "params-bucket"},
                embedding_executor=_RecordingExecutor(),
            )
        )

        assert request_body["vectorBucketName"] == "params-bucket"
        assert request_body["indexName"] == "only-index"

    def test_transform_search_response(self):
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {"query": "test query"}

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "vectors": [
                {
                    "distance": 0.05,
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

        result = config.transform_search_vector_store_response(mock_response, mock_logging_obj)

        assert result["object"] == "vector_store.search_results.page"
        assert result["search_query"] == "test query"
        assert len(result["data"]) == 2
        assert result["data"][0]["score"] == 0.95
        assert result["data"][0]["content"][0]["text"] == "This is test content"
        assert result["data"][0]["filename"] == "test.pdf"
        assert result["data"][1]["score"] == 0.85
        assert result["data"][1]["content"][0]["text"] == "More test content"

    def test_map_openai_params(self):
        config = S3VectorsVectorStoreConfig()
        non_default_params = {"max_num_results": 5}
        optional_params = {}

        result = config.map_openai_params(non_default_params, optional_params, False)

        assert result["maxResults"] == 5
