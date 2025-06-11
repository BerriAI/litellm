import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
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
from litellm.integrations.vector_stores.bedrock_vector_store import BedrockVectorStore
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload, StandardLoggingVectorStoreRequest
from litellm.types.vector_stores import VectorStoreSearchResponse

class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_payload = kwargs.get("standard_logging_object")
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
async def test_basic_bedrock_knowledgebase_retrieval(setup_vector_store_registry):

    bedrock_knowledgebase_hook = BedrockVectorStore(aws_region_name="us-west-2")
    response = await bedrock_knowledgebase_hook.make_bedrock_kb_retrieve_request(
        knowledge_base_id="T37J8R4WTM",
        query="what is litellm?",
    )
    assert response is not None


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
        
        # 1. we should have 2 content blocks, the first is the user message, the second is the context from the knowledge base
        content = request_body["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "text"

        # 2. the message with the context should have the bedrock knowledge base prefix string
        # this helps confirm that the context from the knowledge base was applied to the request
        assert BedrockVectorStore.CONTENT_PREFIX_STRING in content[1]["text"]
        


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call(setup_vector_store_registry):
    """
    Test that the Bedrock Knowledge Base Hook works when making a real llm api call
    """
    
    # Init client
    litellm._turn_on_debug()
    async_client = AsyncHTTPHandler()
    litellm.callbacks = [BedrockVectorStore(aws_region_name="us-west-2")]
    response = await litellm.acompletion(
        model="anthropic/claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "what is litellm?"}],
        vector_store_ids = [
            "T37J8R4WTM"
        ],
        client=async_client
    )
    assert response is not None


@pytest.mark.asyncio
async def test_openai_with_knowledge_base_mock_openai(setup_vector_store_registry):
    """
    Tests that knowledge base content is correctly passed to the OpenAI API call
    """
    litellm.callbacks = [BedrockVectorStore(aws_region_name="us-west-2")]
    litellm.set_verbose = True
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
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
        request_body = mock_client.call_args.kwargs
        
        # Verify the request contains messages with knowledge base context
        assert "messages" in request_body
        messages = request_body["messages"]
        
        # We expect at least 2 messages:
        # 1. User message with the question
        # 2. User message with the knowledge base context
        assert len(messages) >= 2
        
        print("request messages:", json.dumps(messages, indent=4, default=str))

        # assert message[1] is the user message with the knowledge base context
        assert messages[1]["role"] == "user"
        assert BedrockVectorStore.CONTENT_PREFIX_STRING in messages[1]["content"]


@pytest.mark.asyncio
async def test_openai_with_vector_store_ids_in_tool_call_mock_openai(setup_vector_store_registry):
    """
    Tests that vector store ids can be passed as tools

    This is the OpenAI format
    """
    litellm.callbacks = [BedrockVectorStore(aws_region_name="us-west-2")]
    litellm.set_verbose = True
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
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
        request_body = mock_client.call_args.kwargs
        print("request body:", json.dumps(request_body, indent=4, default=str))
        
        # Verify the request contains messages with knowledge base context
        assert "messages" in request_body
        messages = request_body["messages"]
        
        # We expect at least 2 messages:
        # 1. User message with the question
        # 2. User message with the knowledge base context
        assert len(messages) >= 2
        
        print("request messages:", json.dumps(messages, indent=4, default=str))

        # assert message[1] is the user message with the knowledge base context
        assert messages[1]["role"] == "user"
        assert BedrockVectorStore.CONTENT_PREFIX_STRING in messages[1]["content"]

        # assert that the tool call was not sent to the upstream llm API if it's a litellm vector store
        assert "tools" not in request_body


@pytest.mark.asyncio
async def test_openai_with_mixed_tool_call_mock_openai(setup_vector_store_registry):
    """Ensure unrecognized vector store tools are forwarded to the provider"""
    litellm.callbacks = [BedrockVectorStore(aws_region_name="us-west-2")]
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
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
        request_body = mock_client.call_args.kwargs

        assert "messages" in request_body
        messages = request_body["messages"]
        assert len(messages) >= 2
        assert messages[1]["role"] == "user"
        assert BedrockVectorStore.CONTENT_PREFIX_STRING in messages[1]["content"]

        assert "tools" in request_body
        tools = request_body["tools"]
        assert len(tools) == 1
        assert tools[0]["vector_store_ids"] == ["unknownVS"]


@pytest.mark.asyncio
async def test_logging_with_knowledge_base_hook(setup_vector_store_registry):
    """
    Test that the knowledge base request was logged in standard logging payload
    """
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [BedrockVectorStore(aws_region_name="us-west-2"), test_custom_logger]
    litellm.set_verbose = True
    await litellm.acompletion(
        model="gpt-4",
        messages=[{"role": "user", "content": "what is litellm?"}],
        vector_store_ids = [
            "T37J8R4WTM"
        ],
    )

    # sleep for 1 second to allow the logging callback to run
    await asyncio.sleep(1)

    # assert that the knowledge base request was logged in the standard logging payload
    standard_logging_payload: Optional[StandardLoggingPayload] = test_custom_logger.standard_logging_payload
    assert standard_logging_payload is not None


    metadata = standard_logging_payload["metadata"]
    standard_logging_vector_store_request_metadata: Optional[List[StandardLoggingVectorStoreRequest]] = metadata["vector_store_request_metadata"]

    print("standard_logging_vector_store_request_metadata:", json.dumps(standard_logging_vector_store_request_metadata, indent=4, default=str))

    # 1 vector store request was made, expect 1 vector store request metadata object
    assert len(standard_logging_vector_store_request_metadata) == 1

    # expect the vector store request metadata object to have the correct values
    vector_store_request_metadata = standard_logging_vector_store_request_metadata[0]
    assert vector_store_request_metadata.get("vector_store_id") == "T37J8R4WTM"
    assert vector_store_request_metadata.get("query") == "what is litellm?"
    assert vector_store_request_metadata.get("custom_llm_provider") == "bedrock"


    vector_store_search_response: VectorStoreSearchResponse = vector_store_request_metadata.get("vector_store_search_response")
    assert vector_store_search_response is not None
    assert vector_store_search_response.get("search_query") == "what is litellm?"
    assert len(vector_store_search_response.get("data", [])) >=0
    for item in vector_store_search_response.get("data", []):
        assert item.get("score") is not None
        assert item.get("content") is not None
        assert len(item.get("content", [])) >= 0
        for content_item in item.get("content", []):
            text_content = content_item.get("text")
            assert text_content is not None
            assert len(text_content) > 0
            

@pytest.mark.asyncio
async def test_logging_with_knowledge_base_hook_no_vector_store_registry(setup_vector_store_registry):
    """
    Test that the knowledge base request was logged in standard logging payload
    """
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [BedrockVectorStore(aws_region_name="us-west-2"), test_custom_logger]
    litellm.vector_store_registry = None
    await litellm.acompletion(
        model="gpt-4",
        messages=[{"role": "user", "content": "what is litellm?"}],
    )



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

    print("Registry iniitalized:", litellm.vector_store_registry.vector_stores)


    with patch.object(client, "post") as mock_post:
        # Mock the response for the LLM call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
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
        
