import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
import respx
from respx import MockRouter

import os
from contextlib import contextmanager
from typing import Generator, AsyncGenerator
import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion

@contextmanager
def no_env_var(*vars: str) -> Generator[None, None, None]:
    original_values = {}
    try:
        for var in vars:
            if var in os.environ:
                original_values[var] = os.environ.pop(var)
        yield
    finally:
        for var, val in original_values.items():
            os.environ[var] = val

location_tool = {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius"]},
                    },
                    "required": ["location"],
                },
            },
        }
@pytest.fixture(params=["nvidia", "nvidia_nim"])
def provider(request):
    return request.param

@pytest.fixture(scope="module")
def check_nvidia_api_keys():
    """Fixture to check if NVIDIA API keys are available and skip tests if not."""
    if not any(key in os.environ for key in ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"]):
        pytest.skip("Either NVIDIA_API_KEY or NVIDIA_NIM_API_KEY environment variable is not set.")


def test_completion_missing_key(provider):
    with no_env_var("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"):
        with pytest.raises(litellm.exceptions.AuthenticationError):
            completion(
                model=f"{provider}/meta/llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today in Fahrenheit?",
                    }
                ],
                presence_penalty=0.5,
                frequency_penalty=0.1,
            )

def test_completion_bogus_key(provider):
    with pytest.raises(litellm.exceptions.AuthenticationError):
        completion(
            api_key="bogus-key",
            model=f"{provider}/meta/llama-3.1-70b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
            presence_penalty=0.5,
            frequency_penalty=0.1,
        )

def test_completion_invalid_provider(check_nvidia_api_keys):
    with pytest.raises(litellm.exceptions.BadRequestError) as err_msg:
        completion(
            model="invalid_model",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
            presence_penalty=0.5,
            frequency_penalty=0.1,
        )
    assert "LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=invalid_model" in str(err_msg.value)


@pytest.mark.respx
def test_completion_nvidia(respx_mock: MockRouter, provider):
    litellm.set_verbose = True
    mock_response = ModelResponse(
        id="cmpl-mock",
        choices=[Choices(message=Message(content="Mocked response", role="assistant"))],
        created=int(datetime.now().timestamp()),
        model=f"{provider}/nvidia/meta/llama-3.1-70b-instruct"
    )
    model_name=f"{provider}/meta/llama-3.1-70b-instruct"

    mock_request = respx_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions"
    ).mock(return_value=httpx.Response(200, json=mock_response.dict()))
    try:
        response = completion(
            api_key="bogus-key",
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
            presence_penalty=0.5,
            frequency_penalty=0.1,
        )
        # Add any assertions here to check the response
        print(response)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

        assert mock_request.called
        request_body = json.loads(mock_request.calls[0].request.content)

        print("request_body: ", request_body)

        assert request_body == {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
            "model": "meta/llama-3.1-70b-instruct",
            "frequency_penalty": 0.1,
            "presence_penalty": 0.5,
        }
    except litellm.exceptions.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

## ---------------------------------------- Embedding test cases ----------------------------------------

@pytest.mark.respx
def test_embedding_nvidia(respx_mock: MockRouter, provider):
    litellm.set_verbose = True
    mock_response = EmbeddingResponse(
        model=f"{provider}/meta/llama-3.1-70b-instruct",
        data=[
            {
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
        ),
    )
    mock_request = respx_mock.post(
        "https://integrate.api.nvidia.com/v1/embeddings"
    ).mock(return_value=httpx.Response(200, json=mock_response.dict()))
    _ = litellm.embedding(
        api_key="bogus-key",
        model=f"{provider}/nvidia/nv-embedqa-e5-v5",
        input="What is the meaning of life?",
        input_type="passage",
    )
    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)
    print("request_body: ", request_body)
    
    assert request_body["input"] == "What is the meaning of life?"
    assert request_body["model"] == "nvidia/nv-embedqa-e5-v5"
    assert request_body["input_type"] == "passage"


## ---------------------------------------- aCompletion test cases ----------------------------------------

@pytest.mark.asyncio
async def test_async_completion_missing_key(provider):
    """
    Test async completion with missing API key raises AuthenticationError
    """
    with no_env_var("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"):
        with pytest.raises(litellm.exceptions.AuthenticationError):
            await litellm.acompletion(
                model=f"{provider}/meta/llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today in Fahrenheit?",
                    }
                ],
                presence_penalty=0.5,
                frequency_penalty=0.1,
            )

@pytest.mark.asyncio
@pytest.mark.respx
async def test_async_completion_nvidia(respx_mock, check_nvidia_api_keys):
    """
    Test successful async completion with NVIDIA API
    """
    litellm.set_verbose = True
    
    # Create a mock response similar to the sync test
    mock_response = {
        "id": "cmpl-mock-async",
        "choices": [{
            "message": {
                "content": "Mocked async response", 
                "role": "assistant"
            }
        }],
        "created": int(datetime.now().timestamp()),
        "model": "meta/llama-3.1-70b-instruct"
    }
    
    model_name = "nvidia/meta/llama-3.1-70b-instruct"
    
    # Mock the async POST request
    mock_request = respx_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions"
    ).mock(return_value=httpx.Response(200, json=mock_response))
    
    try:
        response = await litellm.acompletion(
            api_key="test-async-key",
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
            presence_penalty=0.5,
            frequency_penalty=0.1,
        )
        
        # Assertions
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        
        assert mock_request.called
        request_body = json.loads(mock_request.calls[0].request.content)
        
        assert request_body == {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
            "model": "meta/llama-3.1-70b-instruct",
            "frequency_penalty": 0.1,
            "presence_penalty": 0.5,
        }
    except Exception as e:
        pytest.fail(f"Async completion test failed: {e}")

@pytest.mark.asyncio
async def test_async_completion_timeout(provider):
    """
    Test async completion with simulated timeout
    """
    # Mock the acompletion method to raise a timeout
    with pytest.raises(litellm.exceptions.Timeout):
        await litellm.acompletion(
            model=f"{provider}/meta/llama-3.1-70b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": "Simulate a timeout scenario",
                }
            ],
            timeout=0.001  # Very short timeout to force a timeout error
        )

@pytest.mark.asyncio
@pytest.mark.respx
async def test_async_completion_with_stream(respx_mock: respx.MockRouter, provider):
    """
    Test async streaming completion for NVIDIA API
    """
    # Prepare streaming chunks as a list of dictionaries
    streaming_chunks = [
        {
            "id": "cmpl-stream-1",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": "meta/llama-3.1-70b-instruct",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "Streaming"},
                "finish_reason": None
            }]
        },
        {
            "id": "cmpl-stream-2",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": "meta/llama-3.1-70b-instruct",
            "choices": [{
                "index": 0,
                "delta": {"content": " response"},
                "finish_reason": None
            }]
        },
        {
            "id": "cmpl-stream-3",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": "meta/llama-3.1-70b-instruct",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
    ]
    
    # Mock the endpoint to return streaming chunks
    mock_request = respx_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions"
    ).mock(
        side_effect=[
            httpx.Response(200, json=chunk) 
            for chunk in streaming_chunks
        ]
    )
    
    try:
        response = await litellm.acompletion(
            api_key="test-async-stream-key",
            model=f"{provider}/meta/llama-3.1-70b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": "Generate a streaming response",
                }
            ],
            stream=True
        )
        
        # Collect streamed content
        full_content = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        
        # assert full_content == "Streaming response"
        assert mock_request.called
        assert len(mock_request.calls) > 0
    except Exception as e:
        pytest.fail(f"Async streaming completion test failed: {e}")

## ---------------------------------------- Tool Calling test cases ----------------------------------------
@pytest.mark.respx
@pytest.mark.parametrize("stream", [False, True])
def test_nvidia_tool_use(provider, check_nvidia_api_keys, stream):
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    tools = [location_tool]
    model = f"{provider}/meta/llama-3.1-405b-instruct"

    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        
        if stream:
            kwargs["stream"] = True
            kwargs["stop"] = ["\n\n"]
        
        response = completion(**kwargs)
        
        if stream:
            # For streaming response, check the first chunk
            for chunk in response:
                # Check if chunk has tool_calls
                has_tool_calls = chunk.choices[0].delta.tool_calls is not None
                
                # If no tool_calls in delta, check if content contains function call indicators
                if not has_tool_calls:
                    has_function_content = (
                        chunk.choices[0].delta.content and 
                        ("<function" in chunk.choices[0].delta.content or 
                         "get_current_weather" in chunk.choices[0].delta.content)
                    )
                    
                    assert has_function_content, "No tool calls or function call content found in stream chunk"
                else:
                    assert chunk.choices[0].delta.tool_calls[0].function.name == "get_current_weather"
                
                # Only check the first chunk
                break
        else:
            # For non-streaming response
            assert response.choices[0].message.tool_calls
            assert response.choices[0].message.tool_calls[0].function.name == "get_current_weather"
            assert response.choices[0].finish_reason == "tool_calls"
            
    except litellm.InternalServerError:
        pass

@pytest.mark.asyncio
@pytest.mark.respx
@pytest.mark.parametrize("stream", [False, True])
async def test_async_nvidia_tool_use(provider, check_nvidia_api_keys, stream):
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    tools = [location_tool]
    model=f"{provider}/meta/llama-3.1-405b-instruct"

    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        
        if stream:
            kwargs["stream"] = True
            kwargs["stop"] = ["\n\n"]
        
        response = await litellm.acompletion(**kwargs)
        
        if stream:
            # For streaming response, collect chunks
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                # Break after collecting a few chunks to keep test fast
                if len(chunks) > 5:
                    break
            
            # Check if any chunk has tool_calls
            has_tool_calls = any(
                chunk.choices[0].delta.tool_calls is not None 
                for chunk in chunks
            )
            
            # If no tool_calls in delta, check if content contains function call indicators
            if not has_tool_calls:
                has_function_content = any(
                    chunk.choices[0].delta.content and 
                    ("<function" in chunk.choices[0].delta.content or 
                     "get_current_weather" in chunk.choices[0].delta.content)
                    for chunk in chunks
                )
                
                assert has_function_content, "No tool calls or function call content found in stream chunks"
        else:
            # For non-streaming response, check directly
            assert response.choices[0].message.tool_calls
            assert response.choices[0].message.tool_calls[0].function.name == "get_current_weather"
            assert response.choices[0].finish_reason == "tool_calls"
        
    except litellm.InternalServerError:
        pass