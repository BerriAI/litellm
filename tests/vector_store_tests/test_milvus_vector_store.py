"""
Tests for Milvus Vector Store
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.llms.milvus.vector_stores.transformation import MilvusVectorStoreConfig
from litellm.types.utils import EmbeddingResponse
from litellm.vector_stores import asearch as vector_store_asearch
from litellm.vector_stores import search as vector_store_search

# Mock response from actual Milvus API
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
# Mock embedding response from OpenAI
MOCK_EMBEDDING_RESPONSE = MagicMock()
MOCK_EMBEDDING_RESPONSE.data = [
    {
        "embedding": [
            0.023,
            -0.019,
            0.045,
            -0.012,
            0.067,
            -0.034,
            0.089,
            -0.056,
        ]
        * 128  # Simulate 1024-dimensional embedding
    }
]


class TestMilvusVectorStore:
    """Test Milvus Vector Store with mocked responses"""

    @pytest.mark.asyncio
    async def test_basic_search_with_mock_async(self):
        """Test basic vector search with mocked backend response (async)"""

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post:
                mock_post.return_value = mock_response

                # Make the search request
                response = await vector_store_asearch(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                )

                print("Response:", json.dumps(response, indent=2, default=str))

                # Verify embedding was called with correct parameters
                mock_embedding.assert_called_once()
                embedding_call_args = mock_embedding.call_args
                assert embedding_call_args[1]["model"] == "text-embedding-3-large"
                assert embedding_call_args[1]["input"] == ["what is machine learning?"]
                assert embedding_call_args[1]["api_key"] == "mock_openai_api_key"

                # Verify the API was called
                mock_post.assert_called_once()

                # Verify the request payload
                call_args = mock_post.call_args
                print(f"call_args: {call_args}")
                print(f"call_args.kwargs: {call_args.kwargs}")

                # The post method is called with 'data' parameter (JSON string) not 'json' parameter
                request_data_str = call_args.kwargs.get("data")
                if request_data_str:
                    request_data = json.loads(request_data_str)
                else:
                    # Fallback: check for json kwarg or in args
                    request_data = call_args.kwargs.get("json")
                    if request_data is None and len(call_args.args) > 0 and isinstance(call_args.args[0], dict):
                        request_data = call_args.args[0]

                assert request_data is not None, f"Could not extract request data. Call args: {call_args}"
                print("Request data:", json.dumps(request_data, indent=2, default=str))

                # Validate request structure
                assert "collectionName" in request_data
                assert request_data["collectionName"] == "book_2"
                assert "data" in request_data
                assert isinstance(request_data["data"], list)
                assert len(request_data["data"]) == 1  # Single query vector
                assert "annsField" in request_data
                assert request_data["annsField"] == "book_intro_vector"
                assert "outputFields" in request_data
                assert request_data["outputFields"] == ["book_intro_text"]

                # Verify the URL format
                url = call_args.kwargs.get("url", "")
                assert "v2/vectordb/entities/search" in url

                # Validate the response structure (LiteLLM standard format)
                assert response is not None
                assert response["object"] == "vector_store.search_results.page"  # type: ignore
                assert "data" in response
                assert len(response["data"]) == 9  # type: ignore  # 9 results in mock response

                # Validate first result
                first_result = response["data"][0]  # type: ignore
                assert "score" in first_result
                assert first_result["score"] == 10.240219  # type: ignore
                assert "content" in first_result
                assert "attributes" in first_result

                # Validate content structure
                assert len(first_result["content"]) > 0  # type: ignore
                assert first_result["content"][0]["type"] == "text"  # type: ignore
                assert "text" in first_result["content"][0]  # type: ignore
                assert (
                    first_result["content"][0]["text"]  # type: ignore
                    == "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8"
                )

                # Validate attributes contain book_id but NOT book_intro_text (it's in content)
                assert "book_id" in first_result["attributes"]  # type: ignore
                assert first_result["attributes"]["book_id"] == 0  # type: ignore
                assert "book_intro_text" not in first_result["attributes"]  # type: ignore  # Should be in content, not attributes

    def test_basic_search_with_mock_sync(self):
        """Test basic vector search with mocked backend response (sync)"""

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
                mock_post.return_value = mock_response

                # Make the search request
                response = vector_store_search(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                )

                print("Response:", json.dumps(response, indent=2, default=str))

                # Verify embedding was called
                mock_embedding.assert_called_once()

                # Verify the API was called
                mock_post.assert_called_once()

                # Verify the request payload
                call_args = mock_post.call_args

                # The post method is called with 'data' parameter (JSON string) not 'json' parameter
                request_data_str = call_args.kwargs.get("data")
                if request_data_str:
                    request_data = json.loads(request_data_str)
                else:
                    # Fallback: check for json kwarg or in args
                    request_data = call_args.kwargs.get("json")
                    if request_data is None and len(call_args.args) > 0 and isinstance(call_args.args[0], dict):
                        request_data = call_args.args[0]

                assert request_data is not None, f"Could not extract request data. Call args: {call_args}"

                # Validate request structure
                assert "collectionName" in request_data
                assert request_data["collectionName"] == "book_2"
                assert "data" in request_data
                assert isinstance(request_data["data"], list)
                assert "annsField" in request_data
                assert "outputFields" in request_data

                # Validate the response structure
                assert response is not None
                assert response["object"] == "vector_store.search_results.page"  # type: ignore
                assert "data" in response  # type: ignore
                assert len(response["data"]) == 9  # type: ignore  # 9 results in mock response
                assert "search_query" in response  # type: ignore

                # Validate first few results
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
                    assert "score" in result
                    assert result["score"] == expected["distance"]  # type: ignore
                    assert "content" in result
                    assert len(result["content"]) > 0  # type: ignore
                    assert result["content"][0]["type"] == "text"  # type: ignore
                    assert "text" in result["content"][0]  # type: ignore
                    assert result["content"][0]["text"] == expected["text"]  # type: ignore
                    assert "attributes" in result
                    assert result["attributes"]["book_id"] == expected["book_id"]  # type: ignore
                    assert "book_intro_text" not in result["attributes"]  # type: ignore  # Should be in content, not attributes

    def _extract_request_body(self, mock_post):
        call_args = mock_post.call_args
        request_data_str = call_args.kwargs.get("data")
        if request_data_str:
            return json.loads(request_data_str)
        request_data = call_args.kwargs.get("json")
        if request_data is None and len(call_args.args) > 0 and isinstance(call_args.args[0], dict):
            request_data = call_args.args[0]
        return request_data

    def test_user_supplied_db_and_partition_are_dropped(self):
        """User-supplied dbName / partitionNames must not be forwarded to Milvus."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
                mock_post.return_value = mock_response

                vector_store_search(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                    dbName="other_tenant_db",
                    partitionNames=["other_tenant_partition"],
                )

                mock_post.assert_called_once()
                request_data = self._extract_request_body(mock_post)
                assert request_data is not None
                assert "dbName" not in request_data
                assert "partitionNames" not in request_data
                assert request_data["collectionName"] == "book_2"
                assert request_data["annsField"] == "book_intro_vector"
                assert request_data["outputFields"] == ["book_intro_text"]

    def test_backend_configured_db_and_partition_are_forwarded(self):
        """milvus_db_name / milvus_partition_names from litellm_params must be sent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
                mock_post.return_value = mock_response

                vector_store_search(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                    milvus_db_name="tenant_a_db",
                    milvus_partition_names=["tenant_a_partition"],
                )

                mock_post.assert_called_once()
                request_data = self._extract_request_body(mock_post)
                assert request_data is not None
                assert request_data["dbName"] == "tenant_a_db"
                assert request_data["partitionNames"] == ["tenant_a_partition"]

    def test_user_params_cannot_override_backend_db_and_partition(self):
        """Backend-config dbName/partitionNames must win over user-supplied values."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
                mock_post.return_value = mock_response

                vector_store_search(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                    milvus_db_name="tenant_a_db",
                    milvus_partition_names=["tenant_a_partition"],
                    dbName="other_tenant_db",
                    partitionNames=["other_tenant_partition"],
                )

                mock_post.assert_called_once()
                request_data = self._extract_request_body(mock_post)
                assert request_data is not None
                assert request_data["dbName"] == "tenant_a_db"
                assert request_data["partitionNames"] == ["tenant_a_partition"]


# @pytest.mark.parametrize("sync_mode", [True, False])
# @pytest.mark.asyncio
# async def test_basic_search_vector_store(sync_mode):
#     """Integration test with real Milvus API (requires credentials)"""
#     litellm._turn_on_debug()
#     litellm.set_verbose = True
#     base_request_args = {
#         "vector_store_id": "book_2",
#         "custom_llm_provider": "milvus",
#         "api_base": "https://in03-18505f064ffbc6f.serverless.aws-eu-central-1.cloud.zilliz.com",
#         "litellm_embedding_model": "text-embedding-3-large",
#         "litellm_embedding_config": {
#             "api_key": os.getenv("OPENAI_API_KEY"),
#         },
#         "default_output_fields": [
#             "book_intro_text"
#         ],  # field containing the text to return in the response
#         "default_anns_field": "book_intro_vector",
#     }
#     default_query = base_request_args.pop("query", "Basic ping")
#     print(f"base_request_args: {base_request_args}")
#     try:
#         if sync_mode:
#             response = vector_store_search(query=default_query, **base_request_args)
#         else:
#             response = await vector_store_asearch(
#                 query=default_query, **base_request_args
#             )
#     except litellm.InternalServerError:
#         pytest.skip("Skipping test due to litellm.InternalServerError")

#     print("litellm response=", json.dumps(response, indent=4, default=str))
#     assert len(response["data"]) > 0  # type: ignore


if __name__ == "__main__":
    # Run tests
    import asyncio

    test = TestMilvusVectorStore()

    print("Running async mock test...")
    asyncio.run(test.test_basic_search_with_mock_async())

    print("\nRunning sync mock test...")
    test.test_basic_search_with_mock_sync()

    print("\n✅ All mock tests passed!")


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
ALIAS_SEARCH_KWARGS = {
    "query": "what is machine learning?",
    "vector_store_id": "book_2",
    "custom_llm_provider": "milvus",
    "api_base": "https://milvus.example",
    "api_key": "mock_milvus_api_key",
    "litellm_embedding_model": "multilingual-e5-large",
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
