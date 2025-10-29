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
            model="cohere_chat/v1/command-r",
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
            model="cohere_chat/v1/command-r-plus",
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
            model="cohere_chat/v1/command-r",
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
                model="cohere_chat/v1/command-r",
                messages=messages,
                max_tokens=10,
            )
        else:
            response = completion(
                model="cohere_chat/v1/command-r",
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
                model="cohere_chat/v1/command-r",
                messages=messages,
                max_tokens=10,
                stream=True,
            )
            print("async cohere stream response", response)
            async for chunk in response:
                print(chunk)
        else:
            response = completion(
                model="cohere_chat/v1/command-r",
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
                model="cohere/v1/command",
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


# ==================== COHERE V2 API TESTS ====================

@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_cohere_v2_chat_completion(sync_mode):
    """Test basic Cohere v2 chat completion functionality."""
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        if sync_mode:
            response = completion(
                model="cohere_chat/v2/command-a-03-2025",
                messages=messages,
                max_tokens=50
            )
        else:
            response = await litellm.acompletion(
                model="cohere_chat/v2/command-a-03-2025",
                messages=messages,
                max_tokens=50
            )
        
        # Validate response structure
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert response.usage is not None
        assert response.usage.total_tokens > 0
        print(f"Cohere v2 response: {response}")
        
    except litellm.ServiceUnavailableError:
        pass  # Skip if service is unavailable
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_cohere_v2_streaming(stream):
    """Test Cohere v2 streaming functionality."""
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "user", "content": "Tell me a short story about a robot."}
        ]
        
        response = await litellm.acompletion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            max_tokens=100,
            stream=stream
        )
        
        if stream:
            # Test streaming response
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                if len(chunks) >= 3:  # Test first few chunks
                    break
            assert len(chunks) > 0
            print(f"Received {len(chunks)} streaming chunks")
        else:
            # Test non-streaming response
            assert response.choices is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None
            print(f"Non-streaming response: {response.choices[0].message.content}")
            
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_v2_tool_calling():
    """Test Cohere v2 tool calling functionality."""
    try:
        litellm.set_verbose = True
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "What's the weather like in New York?"}
        ]
        
        response = completion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=100
        )
        
        # Validate tool calling response
        assert response.choices is not None
        assert len(response.choices) > 0
        message = response.choices[0].message
        
        # Check if tool calls are present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            assert len(message.tool_calls) > 0
            tool_call = message.tool_calls[0]
            assert tool_call.function.name == "get_weather"
            assert tool_call.function.arguments is not None
            print(f"Tool call: {tool_call.function.name} - {tool_call.function.arguments}")
        else:
            # If no tool calls, check that we got a regular response
            assert message.content is not None
            print(f"Regular response: {message.content}")
            
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_cohere_v2_annotations(stream):
    """Test Cohere v2 annotations functionality (replaces citations)."""
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "user", "content": "What are the benefits of renewable energy?"}
        ]
        
        documents = [
            {
                "data": {
                    "title": "Renewable Energy Benefits Document", 
                    "snippet": "Renewable energy sources like solar and wind power provide clean electricity while reducing greenhouse gas emissions and dependence on fossil fuels."
                }
            },
            {
                "data": {
                    "title": "Environmental Impact Study", 
                    "snippet": "Studies show that renewable energy significantly reduces carbon footprint and helps combat climate change."
                }
            }
        ]
        
        response = await litellm.acompletion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            documents=documents,
            max_tokens=100,
            stream=stream
        )
        
        if stream:
            # Test streaming with annotations
            annotations_found = False
            async for chunk in response:
                # Check if chunk has a message with annotations
                if (hasattr(chunk, 'choices') and chunk.choices and 
                    len(chunk.choices) > 0 and 
                    hasattr(chunk.choices[0], 'message') and 
                    hasattr(chunk.choices[0].message, 'annotations') and 
                    chunk.choices[0].message.annotations):
                    annotations_found = True
                    print(f"Streaming annotations: {chunk.choices[0].message.annotations}")
                    break
            # Note: Annotations might not appear in every chunk during streaming
        else:
            # Test non-streaming with annotations
            assert response.choices is not None
            assert len(response.choices) > 0
            
            # Check for annotations in message
            message = response.choices[0].message
            if hasattr(message, 'annotations') and message.annotations:
                assert len(message.annotations) > 0
                print(f"Annotations found: {len(message.annotations)}")
                
                # Validate annotation structure
                for annotation in message.annotations:
                    assert annotation.get('type') == 'url_citation', f"Expected type 'url_citation', got {annotation.get('type')}"
                    assert 'url_citation' in annotation, "Missing url_citation field"
                    url_citation = annotation['url_citation']
                    assert 'start_index' in url_citation, "Missing start_index"
                    assert 'end_index' in url_citation, "Missing end_index"
                    assert 'title' in url_citation, "Missing title"
                    assert 'url' in url_citation, "Missing url"
                
                print(f"First annotation: {message.annotations[0]}")
            else:
                # Annotations might not always be present depending on the response
                print("No annotations in this response")
            
            # Ensure citations field is NOT present (removed backward compatibility)
            assert not hasattr(response, 'citations'), "Citations field should be removed - no backward compatibility"
                
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_v2_parameter_mapping():
    """Test Cohere v2 parameter mapping and validation."""
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "user", "content": "Generate a creative story."}
        ]
        
        # Test various parameters that should be mapped correctly
        response = completion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            temperature=0.7,
            max_tokens=50,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END", "STOP"],
            seed=42
        )
        
        # Validate response
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert response.usage is not None
        print(f"Parameter mapping test response: {response.choices[0].message.content}")
        
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_cohere_v2_error_handling():
    """Test Cohere v2 error handling with invalid parameters."""
    try:
        # Test with invalid model name
        try:
            response = completion(
                model="cohere_chat/v2/invalid-model",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            # If we get here, the test should fail
            pytest.fail("Should have failed with invalid model")
        except Exception as e:
            # Expected to fail with invalid model
            print(f"Expected error with invalid model: {e}")
            
        # Test with empty messages
        try:
            response = completion(
                model="cohere_chat/v2/command-a-03-2025",
                messages=[],  # Empty messages
                max_tokens=10
            )
            pytest.fail("Should have failed with empty messages")
        except Exception as e:
            # Expected to fail with empty messages
            print(f"Expected error with empty messages: {e}")
            
    except Exception as e:
        pytest.fail(f"Unexpected error in error handling test: {e}")


@pytest.mark.asyncio
async def test_cohere_documents_options_in_request_body():
    """
    Test that documents parameters is properly included 
    in the request body after transformation (sent via extra_body).
    """
    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Test response with citations",
        "generation_id": "mock-generation-id",
        "finish_reason": "COMPLETE"
    }

    # Mock the AsyncHTTPHandler.post method
    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", return_value=mock_response) as mock_post:
        try:
            # Test documents and citation_options parameters
            test_documents = [
                {
                    "data": {
                        "title": "Test Document 1", 
                        "snippet": "This is test content 1"
                    }
                },
                {
                    "data": {
                        "title": "Test Document 2", 
                        "snippet": "This is test content 2"
                    }
                }
            ]
            await litellm.acompletion(
                model="cohere_chat/command-a-03-2025",
                messages=[{"role": "user", "content": "Test message"}],
                documents=test_documents,
            )
        except Exception:
            pass  # We only care about the request body validation

        # Verify the API call was made
        mock_post.assert_called_once()
        
        # Get and parse the request body
        request_data = json.loads(mock_post.call_args.kwargs["data"])
        print(f"Request body: {request_data}")
        
        # Validate that documents and citation_options are in the request body
        assert "documents" in request_data
        assert request_data["documents"] == test_documents


@pytest.mark.asyncio
async def test_cohere_v2_conversation_history():
    """Test Cohere v2 with conversation history."""
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"}
        ]
        
        response = await litellm.acompletion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            max_tokens=50
        )
        
        # Validate response with conversation history
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        print(f"Conversation history response: {response.choices[0].message.content}")
        
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")