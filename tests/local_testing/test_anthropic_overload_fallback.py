import os
import sys
from typing import List
from dotenv import load_dotenv
from litellm.exceptions import ServiceUnavailableError, MidStreamFallbackError
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
from litellm import completion, completion_cost, embedding
from litellm.types.utils import Delta, StreamingChoices

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

    # Patch acompletion to simulate both the error and the fallback
    with patch('litellm.acompletion') as mock_acompletion:
        call_count = 0
        
        async def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            model = kwargs.get("model", "")
            is_fallback = kwargs.get("metadata", {}).get("mid_stream_fallback", False)
            
            if call_count == 1:  # First call - original model
                # Return a generator that will raise an error
                async def error_generator():
                    # First yield some content
                    for i in range(3):
                        chunk = litellm.ModelResponse(
                            id=f"chatcmpl-test-{i}",
                            choices=[
                                StreamingChoices(
                                    delta=Delta(
                                        content=f"Token {i} ",
                                        role="assistant" if i == 0 else None,
                                    ),
                                    index=0,
                                )
                            ],
                            model="anthropic/claude-3-5-sonnet-20240620",
                        )
                        yield chunk
                    
                    # Then raise the error
                    raise ServiceUnavailableError(
                        message="AnthropicException - Overloaded. Handle with litellm.InternalServerError.",
                        model="anthropic/claude-3-5-sonnet-20240620",
                        llm_provider="anthropic",
                    )
                
                return error_generator()
            
            else:  # Second call - fallback model
                # Return a successful generator
                async def success_generator():
                    for i in range(3):
                        chunk = litellm.ModelResponse(
                            id=f"chatcmpl-fallback-{i}",
                            choices=[
                                StreamingChoices(
                                    delta=Delta(
                                        content=f"Fallback token {i} ",
                                        role="assistant" if i == 0 else None,
                                    ),
                                    index=0,
                                )
                            ],
                            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                        )
                        # Add fallback header
                        chunk._hidden_params = {
                            "additional_headers": {
                                "x-litellm-fallback-used": True
                            }
                        }
                        yield chunk
                
                return success_generator()
        
        mock_acompletion.side_effect = mock_completion
        
        # Execute the test
        chunks = []
        full_content = ""
        
        try:
            async for chunk in await _router.acompletion(
                model="claude-anthropic",
                messages=messages,
                stream=True,
            ):
                chunks.append(chunk)
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
                    
            # Verify we got chunks from both models
            print(f"Full content: {full_content}")
            assert "Token" in full_content, "Should contain content from the original model"
            assert "Fallback token" in full_content, "Should contain content from the fallback model"
            
            # Verify at least one chunk has fallback headers
            has_fallback_header = False
            for chunk in chunks:
                if (hasattr(chunk, "_hidden_params") and 
                    chunk._hidden_params.get("additional_headers", {}).get("x-litellm-fallback-used", False)):
                    has_fallback_header = True
                    break
            
            assert has_fallback_header, "Should have fallback headers"
            
            print(f"Test passed! Mid-stream fallback worked correctly. Full content: {full_content}")
            
        except Exception as e:
            print(f"Error during streaming: {e}")
            # Print additional information for debugging
            print(f"Number of chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i}: {chunk}")
            raise