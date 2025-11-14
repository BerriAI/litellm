"""
Tests for Milvus Vector Store
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
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

        with patch("litellm.embedding") as mock_embedding:
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
                    if (
                        request_data is None
                        and len(call_args.args) > 0
                        and isinstance(call_args.args[0], dict)
                    ):
                        request_data = call_args.args[0]

                assert (
                    request_data is not None
                ), f"Could not extract request data. Call args: {call_args}"
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

            with patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
            ) as mock_post:
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
                    if (
                        request_data is None
                        and len(call_args.args) > 0
                        and isinstance(call_args.args[0], dict)
                    ):
                        request_data = call_args.args[0]

                assert (
                    request_data is not None
                ), f"Could not extract request data. Call args: {call_args}"

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
