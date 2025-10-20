import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import json

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from unittest.mock import AsyncMock, patch
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

litellm.num_retries = 3


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_chat_completion_cohere_citations(stream):
    try:
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "Which penguins are the tallest?",
            },
        ]
        response = await litellm.acompletion(
            model="cohere_chat/command-r",
            messages=messages,
            documents=[
                {"title": "Tall penguins", "text": "Emperor penguins are the tallest."},
                {
                    "title": "Penguin habitats",
                    "text": "Emperor penguins only live in Antarctica.",
                },
            ],
            stream=stream,
        )

        if stream:
            citations_chunk = False
            async for chunk in response:
                print("received chunk", chunk)
                if "citations" in chunk:
                    citations_chunk = True
                    break
            assert citations_chunk
        else:
            assert response.citations is not None
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_cohere_command_r_plus_function_call():
    litellm.set_verbose = True
    tools = [
        {
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
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]
    try:
        # test without max tokens
        response = completion(
            model="command-r-plus",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        # Add any assertions, here to check response args
        print(response)
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert isinstance(
            response.choices[0].message.tool_calls[0].function.arguments, str
        )
    except litellm.Timeout:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# @pytest.mark.skip(reason="flaky test, times out frequently")
@pytest.mark.flaky(retries=6, delay=1)
def test_completion_cohere():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {"role": "assistant", "content": [{"text": "2", "type": "text"}]},
            {"role": "assistant", "content": [{"text": "3", "type": "text"}]},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="command-r",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# FYI - cohere_chat looks quite unstable, even when testing locally
@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.flaky(retries=3, delay=1)
async def test_chat_completion_cohere(sync_mode):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        if sync_mode is False:
            response = await litellm.acompletion(
                model="cohere_chat/command-r",
                messages=messages,
                max_tokens=10,
            )
        else:
            response = completion(
                model="cohere_chat/command-r",
                messages=messages,
                max_tokens=10,
            )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [False])
async def test_chat_completion_cohere_stream(sync_mode):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        if sync_mode is False:
            response = await litellm.acompletion(
                model="cohere_chat/command-r",
                messages=messages,
                max_tokens=10,
                stream=True,
            )
            print("async cohere stream response", response)
            async for chunk in response:
                print(chunk)
        else:
            response = completion(
                model="cohere_chat/command-r",
                messages=messages,
                max_tokens=10,
                stream=True,
            )
            print(response)
            for chunk in response:
                print(chunk)
    except litellm.APIConnectionError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_cohere_request_body_with_allowed_params():
    """
    Test to validate that when allowed_openai_params is provided, the request body contains
    the correct response_format and reasoning_effort values.
    """
    # Define test parameters
    test_response_format = {"type": "json"}
    test_reasoning_effort = "low"
    test_tools = [{
        "type": "function", 
        "function": {
            "name": "get_current_time", 
            "description": "Get the current time in a given location.", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "location": {"type": "string", "description": "The city name, e.g. San Francisco"}
                },
                "required": ["location"]
            }
        }
    }]

    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "I am Command, a language model developed by Cohere.",
        "generation_id": "mock-generation-id",
        "finish_reason": "COMPLETE"
    }

    # Mock the AsyncHTTPHandler.post method at the module level
    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", return_value=mock_response) as mock_post:
        try:
            await litellm.acompletion(
                model="cohere/command",
                messages=[{"content": "what llm are you", "role": "user"}],
                allowed_openai_params=["tools", "response_format", "reasoning_effort"],
                response_format=test_response_format,
                reasoning_effort=test_reasoning_effort,
                tools=test_tools
            )
        except Exception:
            pass  # We only care about the request body validation

        # Verify the API call was made
        mock_post.assert_called_once()
        
        # Get and parse the request body
        request_data = json.loads(mock_post.call_args.kwargs["data"])
        print(f"request_data: {request_data}")
        
        # Validate request contains our specified parameters
        assert "allowed_openai_params" not in request_data
        assert request_data["response_format"] == test_response_format
        assert request_data["reasoning_effort"] == test_reasoning_effort


def test_cohere_embedding_outout_dimensions():
    litellm._turn_on_debug()
    response = embedding(model="cohere/embed-v4.0", input="Hello, world!", dimensions=512)
    print(f"response: {response}\n")
    assert len(response.data[0]["embedding"]) == 512


# Comprehensive Cohere Embed v4 tests
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embed_v4_basic_text(sync_mode):
    """Test basic text embedding functionality with Cohere Embed v4."""
    try:
        data = {
            "model": "cohere/embed-v4.0",
            "input": ["Hello world!", "This is a test sentence."],
            "input_type": "search_document"
        }
        
        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)
        
        # Validate response structure
        assert response.model is not None
        assert len(response.data) == 2
        assert response.data[0]['object'] == 'embedding'
        assert len(response.data[0]['embedding']) > 0
        assert response.usage.prompt_tokens > 0
        assert isinstance(response.usage, litellm.Usage)
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embed_v4_with_dimensions(sync_mode):
    """Test Cohere Embed v4 with specific dimension parameter."""
    try:
        data = {
            "model": "cohere/embed-v4.0",
            "input": ["Test with custom dimensions"],
            "dimensions": 512,
            "input_type": "search_query"
        }
        
        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)
        
        # Validate dimension
        assert len(response.data[0]['embedding']) == 512
        assert isinstance(response.usage, litellm.Usage)
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embed_v4_image_embedding(sync_mode):
    """Test Cohere Embed v4 image embedding functionality (multimodal)."""
    try:
        import base64
        
        # 1x1 pixel red PNG (base64 encoded)
        test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
        test_image_b64 = base64.b64encode(test_image_data).decode('utf-8')
        
        data = {
            "model": "cohere/embed-v4.0",
            "input": [test_image_b64],
            "input_type": "image"
        }
        
        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)
        
        # Validate response structure for image embedding
        assert response.model is not None
        assert len(response.data) == 1
        assert response.data[0]['object'] == 'embedding'
        assert len(response.data[0]['embedding']) > 0
        assert isinstance(response.usage, litellm.Usage)
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("input_type", ["search_document", "search_query", "classification", "clustering"])
@pytest.mark.asyncio
async def test_cohere_embed_v4_input_types(input_type):
    """Test Cohere Embed v4 with different input types."""
    try:
        response = await litellm.aembedding(
            model="cohere/embed-v4.0",
            input=[f"Test text for {input_type}"],
            input_type=input_type
        )
        
        assert response.model is not None
        assert len(response.data) == 1
        assert response.data[0]['object'] == 'embedding'
        assert len(response.data[0]['embedding']) > 0
        assert isinstance(response.usage, litellm.Usage)
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_embed_v4_encoding_format():
    """Test Cohere Embed v4 with different encoding formats."""
    try:
        response = embedding(
            model="cohere/embed-v4.0",
            input=["Test encoding format"],
            encoding_format="float"
        )
        
        assert response.model is not None
        assert len(response.data) == 1
        assert response.data[0]['object'] == 'embedding'
        assert len(response.data[0]['embedding']) > 0
        # Validate that embeddings are floats
        assert all(isinstance(x, float) for x in response.data[0]['embedding'])
        assert isinstance(response.usage, litellm.Usage)
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_embed_v4_error_handling():
    """Test error handling for Cohere Embed v4 with invalid inputs."""
    try:
        # Test with empty input - should raise an error
        try:
            response = embedding(
                model="cohere/embed-v4.0",
                input=[]  # Empty input
            )
            pytest.fail("Should have failed with empty input")
        except Exception:
            pass  # Expected to fail
        
        # Test with None input - should raise an error
        try:
            response = embedding(
                model="cohere/embed-v4.0",
                input=None
            )
            pytest.fail("Should have failed with None input")
        except Exception:
            pass  # Expected to fail
            
    except Exception as e:
        pytest.fail(f"Error in error handling test: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embed_v4_multiple_texts(sync_mode):
    """Test Cohere Embed v4 with multiple text inputs."""
    try:
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming the world",
            "Python is a versatile programming language",
            "Natural language processing enables human-computer interaction"
        ]
        
        data = {
            "model": "cohere/embed-v4.0",
            "input": texts,
            "input_type": "search_document"
        }
        
        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)
        
        # Validate response structure
        assert response.model is not None
        assert len(response.data) == len(texts)
        
        for i, data_item in enumerate(response.data):
            assert data_item['object'] == 'embedding'
            assert data_item['index'] == i
            assert len(data_item['embedding']) > 0
            assert all(isinstance(x, float) for x in data_item['embedding'])
        
        assert isinstance(response.usage, litellm.Usage)
        assert response.usage.prompt_tokens > 0
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_embed_v4_with_optional_params():
    """Test Cohere Embed v4 with various optional parameters."""
    try:
        response = embedding(
            model="cohere/embed-v4.0",
            input=["Test with optional parameters"],
            input_type="search_query",
            dimensions=256,
            encoding_format="float"
        )
        
        # Validate response
        assert response.model is not None
        assert len(response.data) == 1
        assert response.data[0]['object'] == 'embedding'
        assert len(response.data[0]['embedding']) == 256  # Custom dimensions
        assert all(isinstance(x, float) for x in response.data[0]['embedding'])
        assert isinstance(response.usage, litellm.Usage)
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")