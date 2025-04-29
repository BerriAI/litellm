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
from unittest.mock import AsyncMock, patch, Mock

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.rag_hooks.bedrock_knowledgebase import BedrockKnowledgeBaseHook
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler


@pytest.mark.asyncio
async def test_basic_bedrock_knowledgebase_retrieval():

    bedrock_knowledgebase_hook = BedrockKnowledgeBaseHook()
    response = await bedrock_knowledgebase_hook.make_bedrock_kb_retrieve_request(
        knowledge_base_id="test_knowledge_base_id",
        query="what is litellm?",
    )
    assert response is not None


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_completion():
    litellm._turn_on_debug()
    client = AsyncHTTPHandler()

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
                knowledge_bases = [
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
        assert BedrockKnowledgeBaseHook.CONTENT_PREFIX_STRING in content[1]["text"]
        


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_llm_api_call():
    """
    Test that the Bedrock Knowledge Base Hook works when making a real llm api call
    """
    litellm._turn_on_debug()
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "what is litellm?"}],
        knowledge_bases = [
            "T37J8R4WTM"
        ]
        
    )
    assert response is not None

