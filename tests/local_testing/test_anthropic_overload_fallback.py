import os
import sys
from typing import List
from dotenv import load_dotenv
from litellm.exceptions import ServiceUnavailableError
from litellm.types.llms.openai import AllMessageValues
from litellm.router import Router

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

litellm.set_verbose = True
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages: List[AllMessageValues] = [{"content": user_message, "role": "user"}]

@pytest.mark.asyncio
async def test_anthropic_overload_fallback():
    """
    Test that when an Anthropic model fails mid-stream, it can fallback to another model
    """
    # Create a router with Claude model and a fallback
    _router = Router(
        model_list=[
            {
                "model_name": "claude-anthropic",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20240620",
                    "api_key": os.environ.get("ANTHROPIC_API_KEY", "fake-key"),
                },
            },
            {
                "model_name": "claude-aws",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "api_key": os.environ.get("AWS_ACCESS_KEY_ID", "fake-key"),
                },
            },
        ],
        fallbacks=[{"claude-anthropic": ["claude-aws"]}],
    )

    # Messages for testing
    messages = [{"role": "user", "content": "Tell me about yourself"}]

    # Mock the primary model to fail after a few tokens
    with patch('litellm.llms.anthropic.chat.handler.AnthropicChatCompletion.acompletion_stream_function') as mock_anthropic_stream:
        # Create a generator that raises an exception after yielding a few tokens
        async def failing_stream(*args, **kwargs):
            # First yield a few tokens
            for i in range(3):
                yield {
                    "choices": [{"delta": {"content": f"Token {i} ", "role": "assistant" if i == 0 else None}, "index": 0}],
                    "model": "anthropic/claude-3-5-sonnet-20240620",
                    "object": "chat.completion.chunk"
                }
            # Raise an exception after a few tokens - use ServiceUnavailableError with "overloaded" message
            raise ServiceUnavailableError(
                message="AnthropicException - Overloaded. Handle with litellm.InternalServerError.",
                model="anthropic/claude-3-5-sonnet-20240620",
                llm_provider="anthropic",
            )
        
        mock_anthropic_stream.side_effect = failing_stream
        
        # Mock the fallback model to return successful responses
        with patch('boto3.Session') as mock_session_class:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client
            mock_session_class.return_value = mock_session
            
            # Configure the mock client to return proper string values
            mock_client.meta.region_name = "us-east-1"
            
            with patch('litellm.llms.bedrock.chat.invoke_handler.BedrockLLM.async_streaming') as mock_bedrock_stream:
                async def mock_bedrock_stream_func(*args, **kwargs):
                    for i in range(3):
                        yield {
                            "choices": [{"delta": {"content": f"Fallback token {i} ", "role": "assistant" if i == 0 else None}, "index": 0}],
                            "model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                            "object": "chat.completion.chunk"
                        }
                
                mock_bedrock_stream.return_value = mock_bedrock_stream_func()
                
                # Collect all chunks to verify fallback worked
                chunks = []
                full_content = ""
                
                async for chunk in await _router.acompletion(
                    model="claude-anthropic",
                    messages=messages,
                    stream=True,
                ):
                    chunks.append(chunk)
                    if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                
                # Verify we got chunks from both models
                assert "Token" in full_content, "Should contain content from the original model"
                assert "Fallback token" in full_content, "Should contain content from the fallback model"
                
                # Verify fallback headers
                assert any(
                    hasattr(chunk, "_hidden_params") and 
                    chunk._hidden_params.get("additional_headers", {}).get("x-litellm-fallback-used", False)
                    for chunk in chunks if hasattr(chunk, "_hidden_params")
                ), "Should have fallback headers"
                
                print(f"Test passed! Mid-stream fallback worked correctly. Full content: {full_content}")