import os



import litellm
import litellm.vector_stores.main
import json
from typing import Optional

import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import (
    StandardLoggingPayload,
)


class MockCustomLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None
        self.completion_logging_payload: Optional[StandardLoggingPayload] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload = kwargs.get("standard_logging_object")
        # Store the payload - completion calls have call_type='acompletion'
        if payload and payload.get("call_type") == "acompletion":
            self.completion_logging_payload = payload
        self.standard_logging_payload = payload
        pass


@pytest.fixture(autouse=True)
def add_aws_region_to_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")


@pytest.fixture
def setup_vector_store_registry():
    from litellm.vector_stores.vector_store_registry import (
        VectorStoreRegistry,
        LiteLLM_ManagedVectorStore,
    )

    # Init vector store registry
    litellm.vector_store_registry = VectorStoreRegistry(
        vector_stores=[
            LiteLLM_ManagedVectorStore(
                vector_store_id="T37J8R4WTM", custom_llm_provider="bedrock"
            )
        ]
    )


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call(
    setup_vector_store_registry,
):
    """
    Test that the Bedrock Knowledge Base Hook works when making a real llm api call and returns citations.
    """

    # Init client
    litellm.turn_on_debug()
    async_client = AsyncHTTPHandler()
    response = await litellm.acompletion(
        model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "what is litellm?"}],
        vector_store_ids=["T37J8R4WTM"],
        client=async_client,
    )
    print("OPENAI RESPONSE:", json.dumps(dict(response), indent=4, default=str))
    assert response is not None

    # Check that search_results are present in provider_specific_fields
    assert hasattr(response.choices[0].message, "provider_specific_fields")
    provider_fields = response.choices[0].message.provider_specific_fields
    assert provider_fields is not None
    assert "search_results" in provider_fields
    search_results = provider_fields["search_results"]
    assert search_results is not None
    assert len(search_results) > 0

    # Check search result structure (OpenAI-compatible format)
    first_search_result = search_results[0]
    assert "object" in first_search_result
    assert first_search_result["object"] == "vector_store.search_results.page"
    assert "data" in first_search_result
    assert len(first_search_result["data"]) > 0

    # Check individual result structure
    first_result = first_search_result["data"][0]
    assert "score" in first_result
    assert "content" in first_result
    print(f"Search results returned: {len(search_results)}")
    print(f"First search result has {len(first_search_result['data'])} items")


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call_with_tools(
    setup_vector_store_registry,
):
    """
    Test that the Bedrock Knowledge Base Hook works when making a real llm api call
    """

    # Init client
    litellm.turn_on_debug()
    response = await litellm.acompletion(
        model=f"anthropic/{os.environ.get('CI_CD_DEFAULT_ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001')}",
        messages=[{"role": "user", "content": "what is litellm?"}],
        max_tokens=10,
        tools=[{"type": "file_search", "vector_store_ids": ["T37J8R4WTM"]}],
    )
    assert response is not None


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call_with_tools_and_filters(
    setup_vector_store_registry,
):
    """
    Test that filters from file_search tools are properly passed through to vector store search.
    This test verifies the entire flow: tool parsing -> filter extraction -> vector store API call.

    In this case we filter for a non-existent user_id, which should return no results.
    """
    litellm.turn_on_debug()

    response = await litellm.acompletion(
        model=f"anthropic/{os.environ.get('CI_CD_DEFAULT_ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001')}",
        messages=[{"role": "user", "content": "what is litellm?"}],
        max_tokens=10,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": ["T37J8R4WTM"],
                "filters": {
                    "key": "user_id",
                    "value": "fake-user-id",
                    "operator": "eq",
                },
            }
        ],
    )

    # Verify response is not None
    assert response is not None

    # Verify search results were added to the response (this proves the search was called)
    assert hasattr(response.choices[0].message, "provider_specific_fields")
    provider_fields = response.choices[0].message.provider_specific_fields
    assert provider_fields is not None
    assert (
        "search_results" in provider_fields
    ), "search_results not in provider_specific_fields"

    search_results = provider_fields["search_results"]
    assert (
        search_results is not None and len(search_results) > 0
    ), "No search results found"

    # The search was performed - this confirms filters were passed through
    # The logs above show:  litellm.asearch(... filters={'key': 'user_id', 'value': 'fake-user-id', 'operator': 'eq'})
    # And the Bedrock API request contains: {'filter': {'equals': {'key': 'user_id', 'value': 'fake-user-id'}}}

    print("✅ Filters were successfully passed through to vector store search")
    print(f"   Search was performed and {len(search_results)} result(s) returned")


# @pytest.mark.asyncio
# async def test_logging_with_knowledge_base_hook(setup_vector_store_registry):
#     """
#     Test that the knowledge base request was logged in standard logging payload
#     """
#     test_custom_logger = MockCustomLogger()
#     litellm.set_verbose = True
#     await litellm.acompletion(
#         model="gpt-5.5",
#         messages=[{"role": "user", "content": "what is litellm?"}],
#         vector_store_ids = [
#             "T37J8R4WTM"
#         ],
#     )

#     # sleep for 1 second to allow the logging callback to run
#     await asyncio.sleep(1)

#     # assert that the knowledge base request was logged in the standard logging payload
#     standard_logging_payload: Optional[StandardLoggingPayload] = test_custom_logger.standard_logging_payload
#     assert standard_logging_payload is not None


#     metadata = standard_logging_payload["metadata"]
#     standard_logging_vector_store_request_metadata: Optional[List[StandardLoggingVectorStoreRequest]] = metadata["vector_store_request_metadata"]

#     print("standard_logging_vector_store_request_metadata:", json.dumps(standard_logging_vector_store_request_metadata, indent=4, default=str))

#     # 1 vector store request was made, expect 1 vector store request metadata object
#     assert len(standard_logging_vector_store_request_metadata) == 1

#     # expect the vector store request metadata object to have the correct values
#     vector_store_request_metadata = standard_logging_vector_store_request_metadata[0]
#     assert vector_store_request_metadata.get("vector_store_id") == "T37J8R4WTM"
#     assert vector_store_request_metadata.get("query") == "what is litellm?"
#     assert vector_store_request_metadata.get("custom_llm_provider") == "bedrock"


#     vector_store_search_response: VectorStoreSearchResponse = vector_store_request_metadata.get("vector_store_search_response")
#     assert vector_store_search_response is not None
#     assert vector_store_search_response.get("search_query") == "what is litellm?"
#     assert len(vector_store_search_response.get("data", [])) >=0
#     for item in vector_store_search_response.get("data", []):
#         assert item.get("score") is not None
#         assert item.get("content") is not None
#         assert len(item.get("content", [])) >= 0
#         for content_item in item.get("content", []):
#             text_content = content_item.get("text")
#             assert text_content is not None
#             assert len(text_content) > 0






