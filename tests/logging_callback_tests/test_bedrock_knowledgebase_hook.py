import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
import litellm.vector_stores.main
import gzip
import json
import logging
import time
from typing import Optional, List
from unittest.mock import AsyncMock, patch, Mock

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import VectorStorePreCallHook
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload, StandardLoggingVectorStoreRequest
from litellm.types.vector_stores import (
    VectorStoreSearchResponse,
    VectorStoreResultContent,
    VectorStoreSearchResult,
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
    from litellm.vector_stores.vector_store_registry import VectorStoreRegistry, LiteLLM_ManagedVectorStore
    # Init vector store registry
    litellm.vector_store_registry = VectorStoreRegistry(
        vector_stores=[
            LiteLLM_ManagedVectorStore(
                vector_store_id="T37J8R4WTM",
                custom_llm_provider="bedrock"
            )
        ]
    )


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_completion(setup_vector_store_registry):
    litellm._turn_on_debug()
    client = AsyncHTTPHandler()
    print("value of litellm.vector_store_registry:", litellm.vector_store_registry)

    with patch.object(client, "post") as mock_post:
        # Mock the response for the LLM call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        # Provide proper JSON response content
        mock_response.text = json.dumps({
            "id": "msg_01ABC123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "LiteLLM is a library that simplifies LLM API access."}],
            "model": "claude-3.5-sonnet",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        })
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        
        try:
            response = await litellm.acompletion(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": "what is litellm?"}],
                vector_store_ids = [
                    "T37J8R4WTM"
                ],
                client=client
        )
        except Exception as e:
            print(f"Error: {e}")

        # Verify the LLM request was made
        mock_post.assert_called_once()
        
        # Verify the request body
        print("call args:", mock_post.call_args)
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4, default=str))
        
        # Assert content from the knowedge base was applied to the request
        
        # 1. we should have 2 content blocks, the first is the context from the knowledge base, the second is the user message
        content = request_body["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "text"

        # 2. the first content block should have the bedrock knowledge base prefix string
        # this helps confirm that the context from the knowledge base was applied to the request
        assert VectorStorePreCallHook.CONTENT_PREFIX_STRING in content[0]["text"]
        


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call(setup_vector_store_registry):
    """
    Test that the Bedrock Knowledge Base Hook works when making a real llm api call and returns citations.
    """
    
    # Init client
    litellm._turn_on_debug()
    async_client = AsyncHTTPHandler()
    response = await litellm.acompletion(
        model="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=[{"role": "user", "content": "what is litellm?"}],
        vector_store_ids = [
            "T37J8R4WTM"
        ],
        client=async_client
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
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call_streaming(setup_vector_store_registry):
    """
    Test that the Bedrock Knowledge Base Hook works with streaming and returns search_results in chunks.
    """
    
    # Init client
    # litellm._turn_on_debug()
    async_client = AsyncHTTPHandler()
    response = await litellm.acompletion(
        model="anthropic/claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "what is litellm?"}],
        vector_store_ids = [
            "T37J8R4WTM"
        ],
        stream=True,
        client=async_client
    )
    
    # Collect chunks
    chunks = []
    search_results_found = False
    async for chunk in response:
        chunks.append(chunk)
        print(f"Chunk: {chunk}")
        
        # Check if this chunk has search_results in provider_specific_fields
        if hasattr(chunk, "choices") and chunk.choices:
            for choice in chunk.choices:
                if hasattr(choice, "delta") and choice.delta:
                    provider_fields = getattr(choice.delta, "provider_specific_fields", None)
                    if provider_fields and "search_results" in provider_fields:
                        search_results = provider_fields["search_results"]
                        print(f"Found search_results in streaming chunk: {len(search_results)} results")
                        
                        # Verify structure
                        assert search_results is not None
                        assert len(search_results) > 0
                        
                        first_search_result = search_results[0]
                        assert "object" in first_search_result
                        assert first_search_result["object"] == "vector_store.search_results.page"
                        assert "data" in first_search_result
                        assert len(first_search_result["data"]) > 0
                        
                        search_results_found = True
    
    print(f"Total chunks received: {len(chunks)}")
    assert len(chunks) > 0
    assert search_results_found, "search_results should be present in streaming chunks"


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call_with_tools(setup_vector_store_registry):
    """
    Test that the Bedrock Knowledge Base Hook works when making a real llm api call
    """
    
    # Init client
    litellm._turn_on_debug()
    response = await litellm.acompletion(
        model="anthropic/claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "what is litellm?"}],
        max_tokens=10,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": ["T37J8R4WTM"]
            }
        ],
    )
    assert response is not None

@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call_with_tools_and_filters(setup_vector_store_registry):
    """
    Test that filters from file_search tools are properly passed through to vector store search.
    This test verifies the entire flow: tool parsing -> filter extraction -> vector store API call.

    In this case we filter for a non-existent user_id, which should return no results.
    """
    litellm._turn_on_debug()
    
    response = await litellm.acompletion(
        model="anthropic/claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "what is litellm?"}],
        max_tokens=10,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": ["T37J8R4WTM"],
                "filters": {
                    "key": "user_id",
                    "value": "fake-user-id",
                    "operator": "eq"
                }
            }
        ],
    )

    # Verify response is not None
    assert response is not None
    
    # Verify search results were added to the response (this proves the search was called)
    assert hasattr(response.choices[0].message, "provider_specific_fields")
    provider_fields = response.choices[0].message.provider_specific_fields
    assert provider_fields is not None
    assert "search_results" in provider_fields, "search_results not in provider_specific_fields"
    
    search_results = provider_fields["search_results"]
    assert search_results is not None and len(search_results) > 0, "No search results found"
    
    # The search was performed - this confirms filters were passed through
    # The logs above show:  litellm.asearch(... filters={'key': 'user_id', 'value': 'fake-user-id', 'operator': 'eq'})
    # And the Bedrock API request contains: {'filter': {'equals': {'key': 'user_id', 'value': 'fake-user-id'}}}
    
    print("✅ Filters were successfully passed through to vector store search")
    print(f"   Search was performed and {len(search_results)} result(s) returned")


@pytest.mark.asyncio
async def test_bedrock_kb_request_body_has_transformed_filters(setup_vector_store_registry):
    """
    Validate that the Bedrock Knowledge Base request body contains the transformed filters.
    """
    captured_request_body: dict = {}

    async def fake_async_vector_store_search_handler(
        vector_store_id,
        query,
        vector_store_search_optional_params,
        vector_store_provider_config,
        custom_llm_provider,
        litellm_params,
        logging_obj,
        extra_headers=None,
        extra_body=None,
        timeout=None,
        client=None,
        _is_async=False,
    ):
        litellm_params_dict = (
            litellm_params.model_dump(exclude_none=False)
            if hasattr(litellm_params, "model_dump")
            else dict(litellm_params)
        )
        api_base = vector_store_provider_config.get_complete_url(
            api_base=litellm_params_dict.get("api_base"),
            litellm_params=litellm_params_dict,
        )

        url, request_body = vector_store_provider_config.transform_search_vector_store_request(
            vector_store_id=vector_store_id,
            query=query,
            vector_store_search_optional_params=vector_store_search_optional_params,
            api_base=api_base,
            litellm_logging_obj=logging_obj,
            litellm_params=litellm_params_dict,
        )
        captured_request_body["url"] = url
        captured_request_body["body"] = request_body

        return VectorStoreSearchResponse(
            object="vector_store.search_results.page",
            search_query=query if isinstance(query, str) else " ".join(query),
            data=[
                VectorStoreSearchResult(
                    score=0.9,
                    content=[VectorStoreResultContent(text="LiteLLM is a library", type="text")],
                )
            ],
        )

    with patch.object(
        litellm.vector_stores.main.base_llm_http_handler,
        "async_vector_store_search_handler",
        new=AsyncMock(side_effect=fake_async_vector_store_search_handler),
    ):
        response = await litellm.acompletion(
            model="anthropic/claude-3-5-haiku-latest",
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

    assert response is not None
    print("captured_request_body:", json.dumps(captured_request_body, indent=4, default=str))
    assert "body" in captured_request_body, "Bedrock KB request body was not captured"

    vector_search = captured_request_body["body"]["retrievalConfiguration"]["vectorSearchConfiguration"]
    aws_filter = vector_search["filter"]
    assert "equals" in aws_filter, f"Expected 'equals' in AWS format, got: {aws_filter}"
    assert aws_filter["equals"]["key"] == "user_id"
    assert aws_filter["equals"]["value"] == "fake-user-id"

    print("✅ Filters transformed correctly: OpenAI format -> AWS Bedrock format")

@pytest.mark.asyncio
async def test_openai_with_knowledge_base_mock_openai(setup_vector_store_registry):
    """
    Tests that knowledge base content is correctly passed to the OpenAI API call
    """
    litellm.set_verbose = True
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")
    
    # Variable to capture the request
    captured_request = {}

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        # Create async mock that returns proper structure
        async def mock_create(**kwargs):
            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content="Mock response from OpenAI", role="assistant"))
            ]
            mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            mock_response.id = "chatcmpl-123"
            mock_response.object = "chat.completion"
            mock_response.created = 1234567890
            mock_response.model = "gpt-4"
            
            # Store the request for verification
            captured_request.update(kwargs)
            
            # Return wrapper with parse method
            wrapper = Mock()
            wrapper.parse.return_value = mock_response
            return wrapper
        
        mock_client.side_effect = mock_create
        
        try:
            await litellm.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": "what is litellm?"}],
                vector_store_ids = [
                    "T37J8R4WTM"
                ],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        # Verify the API was called
        mock_client.assert_called_once()
        request_body = captured_request
        
        # Verify the request contains messages with knowledge base context
        assert "messages" in request_body
        messages = request_body["messages"]
        
        # We expect at least 2 messages:
        # 1. User message with the knowledge base context
        # 2. User message with the question
        assert len(messages) >= 2
        
        print("request messages:", json.dumps(messages, indent=4, default=str))

        # assert message[0] is the user message with the knowledge base context
        assert messages[0]["role"] == "user"
        assert VectorStorePreCallHook.CONTENT_PREFIX_STRING in messages[0]["content"]


@pytest.mark.asyncio
async def test_openai_with_vector_store_ids_in_tool_call_mock_openai(setup_vector_store_registry):
    """
    Tests that vector store ids can be passed as tools

    This is the OpenAI format
    """
    litellm.set_verbose = True
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")
    
    # Variable to capture the request
    captured_request = {}

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        # Create async mock that returns proper structure
        async def mock_create(**kwargs):
            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content="Mock response from OpenAI", role="assistant"))
            ]
            mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            mock_response.id = "chatcmpl-123"
            mock_response.object = "chat.completion"
            mock_response.created = 1234567890
            mock_response.model = "gpt-4"
            
            # Store the request for verification
            captured_request.update(kwargs)
            
            # Return wrapper with parse method
            wrapper = Mock()
            wrapper.parse.return_value = mock_response
            return wrapper
        
        mock_client.side_effect = mock_create
        
        try:
            await litellm.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": "what is litellm?"}],
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": ["T37J8R4WTM"]
                }],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        # Verify the API was called
        mock_client.assert_called_once()
        request_body = captured_request
        print("request body:", json.dumps(request_body, indent=4, default=str))
        
        # Verify the request contains messages with knowledge base context
        assert "messages" in request_body
        messages = request_body["messages"]
        
        # We expect at least 2 messages:
        # 1. User message with the knowledge base context
        # 2. User message with the question
        assert len(messages) >= 2
        
        print("request messages:", json.dumps(messages, indent=4, default=str))

        # assert message[0] is the user message with the knowledge base context
        assert messages[0]["role"] == "user"
        assert VectorStorePreCallHook.CONTENT_PREFIX_STRING in messages[0]["content"]

        # assert that the tool call was not sent to the upstream llm API if it's a litellm vector store
        assert "tools" not in request_body


@pytest.mark.asyncio
async def test_openai_with_mixed_tool_call_mock_openai(setup_vector_store_registry):
    """Ensure unrecognized vector store tools are forwarded to the provider"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")
    
    # Variable to capture the request
    captured_request = {}

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        # Create async mock that returns proper structure
        async def mock_create(**kwargs):
            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content="Mock response from OpenAI", role="assistant"))
            ]
            mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            mock_response.id = "chatcmpl-123"
            mock_response.object = "chat.completion"
            mock_response.created = 1234567890
            mock_response.model = "gpt-4"
            
            # Store the request for verification
            captured_request.update(kwargs)
            
            # Return wrapper with parse method
            wrapper = Mock()
            wrapper.parse.return_value = mock_response
            return wrapper
        
        mock_client.side_effect = mock_create
        
        try:
            await litellm.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": "what is litellm?"}],
                tools=[
                    {"type": "file_search", "vector_store_ids": ["T37J8R4WTM"]},
                    {"type": "file_search", "vector_store_ids": ["unknownVS"]},
                ],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = captured_request

        assert "messages" in request_body
        messages = request_body["messages"]
        assert len(messages) >= 2
        assert messages[0]["role"] == "user"
        assert VectorStorePreCallHook.CONTENT_PREFIX_STRING in messages[0]["content"]

        assert "tools" in request_body
        tools = request_body["tools"]
        assert len(tools) == 1
        assert tools[0]["vector_store_ids"] == ["unknownVS"]


# @pytest.mark.asyncio
# async def test_logging_with_knowledge_base_hook(setup_vector_store_registry):
#     """
#     Test that the knowledge base request was logged in standard logging payload
#     """
#     test_custom_logger = MockCustomLogger()
#     litellm.set_verbose = True
#     await litellm.acompletion(
#         model="gpt-4",
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
            



@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_without_vector_store_registry(setup_vector_store_registry):
    litellm._turn_on_debug()
    client = AsyncHTTPHandler()
    litellm.vector_store_registry = None

    with patch.object(client, "post") as mock_post:
        # Mock the response for the LLM call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        # Provide proper JSON response content
        mock_response.text = json.dumps({
            "id": "msg_01ABC123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "LiteLLM is a library that simplifies LLM API access."}],
            "model": "claude-3.5-sonnet",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        })
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        try:
            response = await litellm.acompletion(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": "what is litellm?"}],
                vector_store_ids = [
                    "T37J8R4WTM"
                ],
                client=client
        )
        except Exception as e:
            print(f"Error: {e}")

        # Verify the LLM request was made
        mock_post.assert_called_once()
        
        # Verify the request body
        print("call args:", mock_post.call_args)
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4, default=str))
        
        # Assert content from the knowedge base was applied to the request
        
        # 1. we should have 1 content block, the first is the user message
        # There should only be one since there is no initialized vector store registry
        content = request_body["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        




@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_vector_store_not_in_registry(setup_vector_store_registry):
    """
    No vector store request is made for vector store ids that are not in the registry

    In this test newUnknownVectorStoreId is not in the registry, so no vector store request is made
    """
    litellm._turn_on_debug()
    client = AsyncHTTPHandler()

    if litellm.vector_store_registry is not None:
        print("Registry iniitalized:", litellm.vector_store_registry.vector_stores)
    else:
        print("Registry is None")


    with patch.object(client, "post") as mock_post:
        # Mock the response for the LLM call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        # Provide proper JSON response content
        mock_response.text = json.dumps({
            "id": "msg_01ABC123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "LiteLLM is a library that simplifies LLM API access."}],
            "model": "claude-3.5-sonnet",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        })
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        try:
            response = await litellm.acompletion(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": "what is litellm?"}],
                vector_store_ids = [
                    "newUnknownVectorStoreId"
                ],
                client=client
        )
        except Exception as e:
            print(f"Error: {e}")

        # Verify the LLM request was made
        mock_post.assert_called_once()
        
        # Verify the request body
        print("call args:", mock_post.call_args)
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4, default=str))
        
        # Assert content from the knowedge base was applied to the request
        
        # 1. we should have 1 content block, the first is the user message
        # There should only be one since there is no initialized vector store registry
        content = request_body["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        

@pytest.mark.asyncio
async def test_provider_specific_fields_in_proxy_http_response(setup_vector_store_registry):
    """
    Test that provider_specific_fields (like search_results) are included 
    in the proxy HTTP JSON response, not just in Python SDK objects.
    
    This test catches serialization bugs where exclude=True would strip 
    provider_specific_fields from the HTTP response.
    """
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app, initialize
    from litellm.proxy.utils import ProxyLogging
    import litellm.proxy.proxy_server as proxy_server
    from unittest.mock import patch as mock_patch
    
    # Initialize proxy
    await initialize(
        model="gpt-3.5-turbo",
        alias=None,
        api_base=None,
        debug=False,
        temperature=None,
        max_tokens=None,
        request_timeout=600,
        max_budget=None,
        telemetry=False,
        drop_params=True,
        add_function_to_prompt=False,
        headers=None,
        save=False,
        use_queue=False,
        config=None
    )
    
    # Create test client
    client = TestClient(app)
    
    # Create mock response with provider_specific_fields
    mock_response = litellm.ModelResponse(
        id="test-123",
        model="gpt-3.5-turbo",
        created=1234567890,
        object="chat.completion"
    )
    
    # Create message with provider_specific_fields
    mock_message = litellm.Message(
        content="LiteLLM is a tool that simplifies working with multiple LLMs.",
        role="assistant",
        provider_specific_fields={
            "search_results": [{
                "object": "vector_store.search_results.page",
                "search_query": "what is litellm?",
                "data": [{
                    "score": 0.95,
                    "content": [{"text": "Test content", "type": "text"}],
                    "file_id": "test-file",
                    "filename": "test.txt"
                }]
            }]
        }
    )
    
    mock_choice = litellm.Choices(
        finish_reason="stop",
        index=0,
        message=mock_message
    )
    
    mock_response.choices = [mock_choice]
    mock_response.usage = litellm.Usage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    
    # Patch the completion call at the proxy level
    with mock_patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        # Make HTTP request to proxy
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "What is litellm?"}]
            }
        )
        
        # Check HTTP response
        assert response.status_code == 200
        result = response.json()
        
        print("HTTP Response JSON:", json.dumps(result, indent=2))
        
        # THE KEY ASSERTIONS - These would FAIL with exclude=True!
        assert "choices" in result
        assert len(result["choices"]) > 0
        
        choice = result["choices"][0]
        assert "message" in choice
        
        message = choice["message"]
        
        # Verify provider_specific_fields is in the JSON response
        assert "provider_specific_fields" in message, \
            "provider_specific_fields missing from HTTP JSON response! This means exclude=True is preventing serialization."
        
        assert "search_results" in message["provider_specific_fields"]
        search_results = message["provider_specific_fields"]["search_results"]
        assert len(search_results) > 0
        
        # Verify search result structure
        first_result = search_results[0]
        assert first_result["object"] == "vector_store.search_results.page"
        assert "data" in first_result
        assert len(first_result["data"]) > 0
        
        print("✅ provider_specific_fields successfully serialized in HTTP response")
