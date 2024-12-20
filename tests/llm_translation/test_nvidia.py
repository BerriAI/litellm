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

def test_completion_missing_key():
    with no_env_var("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"):
        with pytest.raises(litellm.exceptions.AuthenticationError):
            completion(
                model="nvidia/meta/llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today in Fahrenheit?",
                    }
                ],
                presence_penalty=0.5,
                frequency_penalty=0.1,
            )

def test_completion_bogus_key():
    with pytest.raises(litellm.exceptions.AuthenticationError):
        completion(
            api_key="bogus-key",
            model="nvidia/meta/llama-3.1-70b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
            presence_penalty=0.5,
            frequency_penalty=0.1,
        )

@pytest.mark.skipif(not any(key in os.environ for key in ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"]), reason="Either NVIDIA_API_KEY or NVIDIA_NIM_API_KEY environment variable is not set.")
def test_completion_invalid_model():
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
def test_completion_nvidia(respx_mock: MockRouter):
    litellm.set_verbose = True
    mock_response = ModelResponse(
        id="cmpl-mock",
        choices=[Choices(message=Message(content="Mocked response", role="assistant"))],
        created=int(datetime.now().timestamp()),
        model="nvidia/meta/llama-3.1-70b-instruct"
    )
    model_name="nvidia/meta/llama-3.1-70b-instruct"

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


@pytest.mark.respx
def test_embedding_nvidia(respx_mock: MockRouter):
    litellm.set_verbose = True
    mock_response = EmbeddingResponse(
        model="nvidia/meta/llama-3.1-70b-instruct",
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
    response = litellm.embedding(
        api_key="bogus-key",
        model="nvidia/nvidia/nv-embedqa-e5-v5",
        input="What is the meaning of life?",
        input_type="passage",
    )
    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)
    print("request_body: ", request_body)
    assert request_body == {
        "input": "What is the meaning of life?",
        "model": "nvidia/nv-embedqa-e5-v5",
        "input_type": "passage",
    }


## ---------------------------------------- ACompletion test cases ----------------------------------------

@pytest.mark.asyncio
async def test_async_completion_missing_key():
    """
    Test async completion with missing API key raises AuthenticationError
    """
    with no_env_var("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"):
        with pytest.raises(litellm.exceptions.AuthenticationError):
            await litellm.acompletion(
                model="nvidia/meta/llama-3.1-70b-instruct",
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
@pytest.mark.skipif(not any(key in os.environ for key in ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"]), reason="Either NVIDIA_API_KEY or NVIDIA_NIM_API_KEY environment variable is not set.")
async def test_async_completion_nvidia(respx_mock):
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
async def test_async_completion_timeout():
    """
    Test async completion with simulated timeout
    """
    # Mock the acompletion method to raise a timeout
    with pytest.raises(litellm.exceptions.Timeout):
        await litellm.acompletion(
            model="nvidia/meta/llama-3.1-70b-instruct",
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
async def test_async_completion_with_stream(respx_mock: respx.MockRouter):
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
            model="nvidia/meta/llama-3.1-70b-instruct",
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
@pytest.mark.skipif(not any(key in os.environ for key in ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"]), reason="Either NVIDIA_API_KEY or NVIDIA_NIM_API_KEY environment variable is not set.")
def test_nvidia_tool_use():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    tools = [location_tool]
    model="nvidia/meta/llama-3.1-70b-instruct"

    try:
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
        )
        assert response.choices[0].message.tool_calls
        assert (
            response.choices[0].message.tool_calls[0].function.name
            == "get_current_weather"
        )
        assert response.choices[0].finish_reason == "tool_calls"
    except litellm.InternalServerError:
        pass

@pytest.mark.respx
@pytest.mark.skipif(not any(key in os.environ for key in ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"]), reason="Either NVIDIA_API_KEY or NVIDIA_NIM_API_KEY environment variable is not set.")
def test_nvidia_tool_use_stream():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    tools = [location_tool]
    model="nvidia/meta/llama-3.1-70b-instruct"

    try:
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
            stop=["\n\n"]
        )
        for chunk in response:
            assert chunk.choices[0].delta.tool_calls
            assert (
                chunk.choices[0].delta.tool_calls[0].function.name
                == "get_current_weather"
            )
            assert chunk.choices[0].finish_reason == "tool_calls"
            break
    except litellm.InternalServerError:
        pass

@pytest.mark.asyncio
@pytest.mark.respx
@pytest.mark.skipif(not any(key in os.environ for key in ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"]), reason="Either NVIDIA_API_KEY or NVIDIA_NIM_API_KEY environment variable is not set.")
async def test_async_nvidia_tool_use():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    tools = [location_tool]
    model="nvidia/meta/llama-3.1-70b-instruct"

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
        )
        assert response.choices[0].message.tool_calls
        assert (
            response.choices[0].message.tool_calls[0].function.name
            == "get_current_weather"
        )
        assert response.choices[0].finish_reason == "tool_calls"
    except litellm.InternalServerError:
        pass

@pytest.mark.asyncio
@pytest.mark.respx
@pytest.mark.skipif(not any(key in os.environ for key in ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"]), reason="Either NVIDIA_API_KEY or NVIDIA_NIM_API_KEY environment variable is not set.")
async def test_async_nvidia_tool_use_stream():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    tools = [location_tool]
    model="nvidia/meta/llama-3.1-70b-instruct"

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
            stop=["\n\n"]
        )
        async for chunk in response:
            assert chunk.choices[0].delta.tool_calls
            assert (
                chunk.choices[0].delta.tool_calls[0].function.name
                == "get_current_weather"
            )
            assert chunk.choices[0].finish_reason == "tool_calls"
            break
        
    except litellm.InternalServerError:
        pass