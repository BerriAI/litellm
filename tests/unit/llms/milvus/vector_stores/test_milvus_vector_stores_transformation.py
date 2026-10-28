"""
Tests for Milvus Vector Store
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.llms.milvus.vector_stores.transformation import MilvusVectorStoreConfig
from litellm.types.utils import EmbeddingResponse
from litellm.vector_stores import asearch as vector_store_asearch
from litellm.vector_stores import search as vector_store_search

MOCK_MILVUS_SEARCH_RESPONSE = {
    "code": 0,
    "cost": 6,
    "data": [
        {
            "book_id": 0,
            "book_intro_text": "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8",
            "distance": 10.240219,
        },
        {
            "book_id": 1,
            "book_intro_text": "abababababa_9a13e8f3-bb1e-487f-b555-b8ae4b127243",
            "distance": 10.240219,
        },
        {
            "book_id": 2,
            "book_intro_text": "abababababa_870f47f1-23ec-4364-ad30-6d364ba8ddb5",
            "distance": 10.240219,
        },
        {
            "book_id": 1000,
            "book_intro_text": "abababababa_8ea2d76a-3fdf-49b3-8f16-a91638361bba",
            "distance": 8.531628,
        },
        {
            "book_id": 1001,
            "book_intro_text": "abababababa_24758251-e740-4183-8649-2f742f676ca0",
            "distance": 8.531628,
        },
        {
            "book_id": 1002,
            "book_intro_text": "abababababa_faa55789-220d-4ef1-b5bf-a72f2fbd061b",
            "distance": 8.531628,
        },
        {
            "book_id": 0,
            "book_intro_text": "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8",
            "distance": 8.236887,
        },
        {
            "book_id": 1,
            "book_intro_text": "abababababa_9a13e8f3-bb1e-487f-b555-b8ae4b127243",
            "distance": 8.236887,
        },
        {
            "book_id": 2,
            "book_intro_text": "abababababa_870f47f1-23ec-4364-ad30-6d364ba8ddb5",
            "distance": 8.236887,
        },
    ],
    "topks": [3, 3, 3],
}


class RecordingEmbeddingExecutor:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def embed(self, model, query, configuration):
        self.calls.append((model, query, dict(configuration)))
        return self.response

    async def aembed(self, model, query, configuration):
        self.calls.append((model, query, dict(configuration)))
        return self.response


ALIAS_QUERY_VECTOR = [0.5, -0.25, 0.125]
ALIAS_EMBEDDING_RESPONSE = EmbeddingResponse(
    data=[{"embedding": ALIAS_QUERY_VECTOR, "index": 0, "object": "embedding"}]
)
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
MILVUS_SEARCH_URL = "https://milvus.example/v2/vectordb/entities/search"
ZILLIZ_SEARCH_URL = (
    "https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com/v2/vectordb/entities/search"
)
ALIAS_SEARCH_KWARGS = {
    "query": "what is machine learning?",
    "vector_store_id": "book_2",
    "custom_llm_provider": "milvus",
    "api_base": "https://milvus.example",
    "api_key": "mock_milvus_api_key",
    "litellm_embedding_model": "multilingual-e5-large",
    "milvus_text_field": "book_intro_text",
}
ZILLIZ_SEARCH_KWARGS = {
    "query": "what is machine learning?",
    "vector_store_id": "book_2",
    "custom_llm_provider": "milvus",
    "api_base": "https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
    "api_key": "mock_milvus_api_key",
    "litellm_embedding_model": "text-embedding-3-large",
    "litellm_embedding_config": {
        "api_key": "mock_openai_api_key",
    },
    "outputFields": ["book_intro_text"],
    "annsField": "book_intro_vector",
    "milvus_text_field": "book_intro_text",
}


def _alias_router():
    return Router(
        model_list=[
            {
                "model_name": "multilingual-e5-large",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "deployment-key",
                },
            }
        ]
    )


def _mock_embedding_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(OPENAI_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": ALIAS_QUERY_VECTOR}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )


def _mock_search_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(MILVUS_SEARCH_URL).mock(return_value=httpx.Response(200, json=MOCK_MILVUS_SEARCH_RESPONSE))


def _mock_zilliz_search_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(ZILLIZ_SEARCH_URL).mock(return_value=httpx.Response(200, json=MOCK_MILVUS_SEARCH_RESPONSE))


def _search_request_body(search_route: respx.Route):
    return json.loads(search_route.calls.last.request.read())


class TestMilvusVectorStore:
    """Test Milvus Vector Store with mocked responses"""

    @pytest.mark.asyncio
    async def test_basic_search_with_mock_async(self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
        """Test basic vector search with mocked backend response (async)"""
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        embedding_route = _mock_embedding_route(respx_mock)
        search_route = _mock_zilliz_search_route(respx_mock)

        response = await vector_store_asearch(**ZILLIZ_SEARCH_KWARGS)

        embedding_request = embedding_route.calls.last.request
        assert embedding_request.headers["authorization"] == "Bearer mock_openai_api_key"
        embedding_body = json.loads(embedding_request.read())
        assert embedding_body["model"] == "text-embedding-3-large"
        assert embedding_body["input"] == ["what is machine learning?"]

        search_request = search_route.calls.last.request
        request_data = json.loads(search_request.read())
        assert request_data["collectionName"] == "book_2"
        assert request_data["data"] == [ALIAS_QUERY_VECTOR]
        assert request_data["annsField"] == "book_intro_vector"
        assert request_data["outputFields"] == ["book_intro_text"]
        assert "v2/vectordb/entities/search" in str(search_request.url)

        assert response is not None
        assert response["object"] == "vector_store.search_results.page"  # type: ignore
        assert len(response["data"]) == 9  # type: ignore

        first_result = response["data"][0]  # type: ignore
        assert first_result["score"] == 10.240219  # type: ignore
        assert len(first_result["content"]) > 0  # type: ignore
        assert first_result["content"][0]["type"] == "text"  # type: ignore
        assert first_result["content"][0]["text"] == "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8"  # type: ignore
        assert first_result["attributes"]["book_id"] == 0  # type: ignore
        assert "book_intro_text" not in first_result["attributes"]  # type: ignore

    def test_basic_search_with_mock_sync(self, respx_mock: respx.MockRouter):
        """Test basic vector search with mocked backend response (sync)"""
        embedding_route = _mock_embedding_route(respx_mock)
        search_route = _mock_zilliz_search_route(respx_mock)

        response = vector_store_search(**ZILLIZ_SEARCH_KWARGS)

        assert embedding_route.call_count == 1
        assert search_route.call_count == 1

        request_data = _search_request_body(search_route)
        assert request_data["collectionName"] == "book_2"
        assert request_data["data"] == [ALIAS_QUERY_VECTOR]
        assert request_data["annsField"] == "book_intro_vector"
        assert request_data["outputFields"] == ["book_intro_text"]

        assert response is not None
        assert response["object"] == "vector_store.search_results.page"  # type: ignore
        assert len(response["data"]) == 9  # type: ignore
        assert "search_query" in response  # type: ignore

        expected_results = [
            {
                "book_id": 0,
                "text": "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8",
                "distance": 10.240219,
            },
            {
                "book_id": 1,
                "text": "abababababa_9a13e8f3-bb1e-487f-b555-b8ae4b127243",
                "distance": 10.240219,
            },
            {
                "book_id": 2,
                "text": "abababababa_870f47f1-23ec-4364-ad30-6d364ba8ddb5",
                "distance": 10.240219,
            },
        ]

        for idx, expected in enumerate(expected_results):
            result = response["data"][idx]  # type: ignore
            assert result["score"] == expected["distance"]  # type: ignore
            assert len(result["content"]) > 0  # type: ignore
            assert result["content"][0]["type"] == "text"  # type: ignore
            assert result["content"][0]["text"] == expected["text"]  # type: ignore
            assert result["attributes"]["book_id"] == expected["book_id"]  # type: ignore
            assert "book_intro_text" not in result["attributes"]  # type: ignore

    def test_user_supplied_db_and_partition_are_dropped(self, respx_mock: respx.MockRouter):
        """User-supplied dbName / partitionNames must not be forwarded to Milvus."""
        _mock_embedding_route(respx_mock)
        search_route = _mock_zilliz_search_route(respx_mock)

        vector_store_search(
            **ZILLIZ_SEARCH_KWARGS,
            dbName="other_tenant_db",
            partitionNames=["other_tenant_partition"],
        )

        request_data = _search_request_body(search_route)
        assert "dbName" not in request_data
        assert "partitionNames" not in request_data
        assert request_data["collectionName"] == "book_2"
        assert request_data["annsField"] == "book_intro_vector"
        assert request_data["outputFields"] == ["book_intro_text"]

    def test_backend_configured_db_and_partition_are_forwarded(self, respx_mock: respx.MockRouter):
        """milvus_db_name / milvus_partition_names from litellm_params must be sent."""
        _mock_embedding_route(respx_mock)
        search_route = _mock_zilliz_search_route(respx_mock)

        vector_store_search(
            **ZILLIZ_SEARCH_KWARGS,
            milvus_db_name="tenant_a_db",
            milvus_partition_names=["tenant_a_partition"],
        )

        request_data = _search_request_body(search_route)
        assert request_data["dbName"] == "tenant_a_db"
        assert request_data["partitionNames"] == ["tenant_a_partition"]

    def test_user_params_cannot_override_backend_db_and_partition(self, respx_mock: respx.MockRouter):
        """Backend-config dbName/partitionNames must win over user-supplied values."""
        _mock_embedding_route(respx_mock)
        search_route = _mock_zilliz_search_route(respx_mock)

        vector_store_search(
            **ZILLIZ_SEARCH_KWARGS,
            milvus_db_name="tenant_a_db",
            milvus_partition_names=["tenant_a_partition"],
            dbName="other_tenant_db",
            partitionNames=["other_tenant_partition"],
        )

        request_data = _search_request_body(search_route)
        assert request_data["dbName"] == "tenant_a_db"
        assert request_data["partitionNames"] == ["tenant_a_partition"]


def _assert_alias_resolved(embedding_route: respx.Route, search_route: respx.Route, response):
    embedding_request = embedding_route.calls.last.request
    assert embedding_request.headers["authorization"] == "Bearer deployment-key"
    embedding_body = json.loads(embedding_request.read())
    assert embedding_body["model"] == "text-embedding-3-small"
    assert embedding_body["input"] == ["what is machine learning?"]
    search_request = search_route.calls.last.request
    assert search_request.headers["authorization"] == "Bearer mock_milvus_api_key"
    assert json.loads(search_request.read())["data"] == [ALIAS_QUERY_VECTOR]
    assert len(response["data"]) == len(MOCK_MILVUS_SEARCH_RESPONSE["data"])
    assert response["data"][0]["content"][0]["text"] == MOCK_MILVUS_SEARCH_RESPONSE["data"][0]["book_intro_text"]


def test_router_search_resolves_bare_embedding_alias_sync(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding_route = _mock_embedding_route(respx_mock)
    search_route = _mock_search_route(respx_mock)

    response = _alias_router().vector_store_search(**ALIAS_SEARCH_KWARGS)

    _assert_alias_resolved(embedding_route, search_route, response)


@pytest.mark.asyncio
async def test_router_search_resolves_bare_embedding_alias_async(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding_route = _mock_embedding_route(respx_mock)
    search_route = _mock_search_route(respx_mock)

    response = await _alias_router().avector_store_search(**ALIAS_SEARCH_KWARGS)

    _assert_alias_resolved(embedding_route, search_route, response)


def test_sdk_search_with_router_kwarg_resolves_bare_embedding_alias_sync(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding_route = _mock_embedding_route(respx_mock)
    search_route = _mock_search_route(respx_mock)

    response = litellm.vector_stores.search(router=_alias_router(), **ALIAS_SEARCH_KWARGS)

    _assert_alias_resolved(embedding_route, search_route, response)


@pytest.mark.asyncio
async def test_sdk_search_with_router_kwarg_resolves_bare_embedding_alias_async(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding_route = _mock_embedding_route(respx_mock)
    search_route = _mock_search_route(respx_mock)

    response = await litellm.vector_stores.asearch(router=_alias_router(), **ALIAS_SEARCH_KWARGS)

    _assert_alias_resolved(embedding_route, search_route, response)


def _team_alias_router():
    return Router(
        model_list=[
            {
                "model_name": "team-a-embedder",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "deployment-key",
                },
                "model_info": {"team_id": "team-a", "team_public_model_name": "multilingual-e5-large"},
            }
        ]
    )


@pytest.mark.asyncio
async def test_sdk_search_with_router_kwarg_resolves_team_alias_from_request_metadata(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding_route = _mock_embedding_route(respx_mock)
    search_route = _mock_search_route(respx_mock)

    response = await litellm.vector_stores.asearch(
        router=_team_alias_router(), metadata={"user_api_key_team_id": "team-a"}, **ALIAS_SEARCH_KWARGS
    )

    _assert_alias_resolved(embedding_route, search_route, response)


@pytest.mark.asyncio
async def test_sdk_search_with_router_kwarg_rejects_team_alias_without_team_metadata(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding_route = _mock_embedding_route(respx_mock)
    _mock_search_route(respx_mock)

    with pytest.raises(litellm.APIConnectionError):
        await litellm.vector_stores.asearch(router=_team_alias_router(), **ALIAS_SEARCH_KWARGS)

    assert embedding_route.call_count == 0


@pytest.mark.asyncio
async def test_transform_uses_injected_executor_without_embedding_config(respx_mock: respx.MockRouter):
    executor = RecordingEmbeddingExecutor(ALIAS_EMBEDDING_RESPONSE)
    config = MilvusVectorStoreConfig()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    transform_kwargs = {
        "vector_store_id": "book_2",
        "query": ["what is", "milvus?"],
        "vector_store_search_optional_params": {"limit": 3},
        "api_base": "https://milvus.example",
        "litellm_logging_obj": logging_obj,
        "litellm_params": {"litellm_embedding_model": "multilingual-e5-large", "milvus_db_name": "docs"},
        "embedding_executor": executor,
    }

    url, sync_body = config.transform_search_vector_store_request(**transform_kwargs)
    _, async_body = await config.atransform_search_vector_store_request(**transform_kwargs)

    assert respx_mock.calls.call_count == 0
    assert executor.calls == [("multilingual-e5-large", "what is milvus?", {})] * 2
    assert url == MILVUS_SEARCH_URL
    assert sync_body == async_body
    assert sync_body == {
        "collectionName": "book_2",
        "data": [ALIAS_QUERY_VECTOR],
        "annsField": "book_intro_vector",
        "limit": 3,
        "dbName": "docs",
    }
    assert logging_obj.model_call_details["input"] == "what is milvus?"
    assert logging_obj.model_call_details["embedding_model"] == "multilingual-e5-large"


def test_transform_falls_back_to_sdk_embedding_without_executor_or_config(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    embedding_route = _mock_embedding_route(respx_mock)
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    _, body = MilvusVectorStoreConfig().transform_search_vector_store_request(
        vector_store_id="book_2",
        query="q",
        vector_store_search_optional_params={},
        api_base="https://milvus.example",
        litellm_logging_obj=logging_obj,
        litellm_params={"litellm_embedding_model": "openai/text-embedding-3-small"},
    )

    embedding_request = embedding_route.calls.last.request
    assert embedding_request.headers["authorization"] == "Bearer env-key"
    assert json.loads(embedding_request.read())["input"] == ["q"]
    assert body["data"] == [ALIAS_QUERY_VECTOR]


def test_transform_requires_embedding_model():
    with pytest.raises(ValueError, match="litellm_embedding_model is required"):
        MilvusVectorStoreConfig().transform_search_vector_store_request(
            vector_store_id="book_2",
            query="q",
            vector_store_search_optional_params={},
            api_base="https://milvus.example",
            litellm_logging_obj=MagicMock(),
            litellm_params={"litellm_embedding_config": {"api_key": "store-key"}},
            embedding_executor=RecordingEmbeddingExecutor(ALIAS_EMBEDDING_RESPONSE),
        )
