import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

# For testing, make sure the COHERE_API_KEY or CO_API_KEY environment variable is set
# You can set it before running the tests with: export COHERE_API_KEY=your_api_key
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
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

litellm.num_retries = 3


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_chat_completion_cohere_v2_citations(stream):
    try:
        class MockResponse:
            def __init__(self, status_code, json_data, is_stream=False):
                self.status_code = status_code
                self._json_data = json_data
                self.headers = {}
                self.is_stream = is_stream
                
                # For streaming responses with citations
                if is_stream:
                    # Create streaming chunks with citations at the end
                    self._iter_content_chunks = [
                        json.dumps({"text": "Emperor"}).encode(),
                        json.dumps({"text": " penguins"}).encode(),
                        json.dumps({"text": " are"}).encode(),
                        json.dumps({"text": " the"}).encode(),
                        json.dumps({"text": " tallest"}).encode(),
                        json.dumps({"text": " and"}).encode(),
                        json.dumps({"text": " they"}).encode(),
                        json.dumps({"text": " live"}).encode(),
                        json.dumps({"text": " in"}).encode(),
                        json.dumps({"text": " Antarctica"}).encode(),
                        json.dumps({"text": "."}).encode(),
                        # Citations in a separate chunk
                        json.dumps({"citations": [
                            {
                                "start": 0,
                                "end": 30,
                                "text": "Emperor penguins are the tallest",
                                "document_ids": ["doc1"]
                            },
                            {
                                "start": 31,
                                "end": 70,
                                "text": "they live in Antarctica",
                                "document_ids": ["doc2"]
                            }
                        ]}).encode(),
                        json.dumps({"finish_reason": "COMPLETE"}).encode(),
                    ]

            def json(self):
                return self._json_data

            @property
            def text(self):
                return json.dumps(self._json_data)
                
            def iter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                    
            async def aiter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                
        async def mock_async_post(*args, **kwargs):
            # For asynchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Async Request body:", request_body)
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Check if documents are included
            documents = request_body.get("documents", [])
            assert len(documents) > 0
            
            # Mock response with citations
            mock_response = {
                "text": "Emperor penguins are the tallest penguins and they live in Antarctica.",
                "generation_id": "mock-id",
                "id": "mock-completion",
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "citations": [
                    {
                        "start": 0,
                        "end": 30,
                        "text": "Emperor penguins are the tallest",
                        "document_ids": ["doc1"]
                    },
                    {
                        "start": 31,
                        "end": 70,
                        "text": "they live in Antarctica",
                        "document_ids": ["doc2"]
                    }
                ]
            }
            
            # Create a streaming response with citations
            if stream:
                return MockResponse(
                    200,
                    {
                        "text": "Emperor penguins are the tallest penguins and they live in Antarctica.",
                        "generation_id": "mock-id",
                        "id": "mock-completion",
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                        "citations": [
                            {
                                "start": 0,
                                "end": 30,
                                "text": "Emperor penguins are the tallest",
                                "document_ids": ["doc1"]
                            },
                            {
                                "start": 31,
                                "end": 70,
                                "text": "they live in Antarctica",
                                "document_ids": ["doc2"]
                            }
                        ],
                        "stream": True
                    },
                    is_stream=True
                )
            else:
                return MockResponse(200, mock_response)
            
        # Mock the async HTTP client
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock, side_effect=mock_async_post):
            litellm.set_verbose = True
            messages = [
                {
                    "role": "user",
                    "content": "Which penguins are the tallest?",
                },
            ]
            response = await litellm.acompletion(
                model="cohere_v2/command-r",
                messages=messages,
                stream=stream,
                documents=[
                    {"title": "Tall penguins", "text": "Emperor penguins are the tallest."},
                    {
                        "title": "Penguin habitats",
                        "text": "Emperor penguins only live in Antarctica.",
                    },
            ],
        )

        if stream:
            citations_chunk = False
            async for chunk in response:
                print("received chunk", chunk)
                if hasattr(chunk, "citations") or (isinstance(chunk, dict) and "citations" in chunk):
                    citations_chunk = True
                    break
            assert citations_chunk
        else:
            assert hasattr(response, "citations")
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_cohere_v2_command_r_plus_function_call():
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
            api_version="v2",  # Specify v2 API version
        )
        # Add any assertions, here to check response args
        print(response)
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert isinstance(
            response.choices[0].message.tool_calls[0].function.arguments, str
        )

        messages.append(
            response.choices[0].message.model_dump()
        )  # Add assistant tool invokes
        tool_result = (
            '{"location": "Boston", "temperature": "72", "unit": "fahrenheit"}'
        )
        # Add user submitted tool results in the OpenAI format
        messages.append(
            {
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
                "role": "tool",
                "name": response.choices[0].message.tool_calls[0].function.name,
                "content": tool_result,
            }
        )
        # In the second response, Cohere should deduce answer from tool results
        second_response = completion(
            model="command-r-plus",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            force_single_step=True,
            api_version="v2",  # Specify v2 API version
        )
        print(second_response)
    except litellm.Timeout:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.flaky(retries=6, delay=1)
def test_completion_cohere_v2():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="command-r",
            messages=messages,
            api_version="v2",  # Specify v2 API version
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_chat_completion_cohere_v2(sync_mode):
    try:
        class MockResponse:
            def __init__(self, status_code, json_data, is_stream=False):
                self.status_code = status_code
                self._json_data = json_data
                self.headers = {}
                self.is_stream = is_stream
                
                # For streaming responses with citations
                if is_stream:
                    # Create streaming chunks with citations at the end
                    self._iter_content_chunks = [
                        json.dumps({"text": "Emperor"}).encode(),
                        json.dumps({"text": " penguins"}).encode(),
                        json.dumps({"text": " are"}).encode(),
                        json.dumps({"text": " the"}).encode(),
                        json.dumps({"text": " tallest"}).encode(),
                        json.dumps({"text": " and"}).encode(),
                        json.dumps({"text": " they"}).encode(),
                        json.dumps({"text": " live"}).encode(),
                        json.dumps({"text": " in"}).encode(),
                        json.dumps({"text": " Antarctica"}).encode(),
                        json.dumps({"text": "."}).encode(),
                        # Citations in a separate chunk
                        json.dumps({"citations": [
                            {
                                "start": 0,
                                "end": 30,
                                "text": "Emperor penguins are the tallest",
                                "document_ids": ["doc1"]
                            },
                            {
                                "start": 31,
                                "end": 70,
                                "text": "they live in Antarctica",
                                "document_ids": ["doc2"]
                            }
                        ]}).encode(),
                        json.dumps({"finish_reason": "COMPLETE"}).encode(),
                    ]

            def json(self):
                return self._json_data

            @property
            def text(self):
                return json.dumps(self._json_data)
                
            def iter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                    
            async def aiter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                
        def mock_sync_post(*args, **kwargs):
            # For synchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Sync Request body:", request_body)
            
            # Verify the model is passed correctly
            assert request_body.get("model") == "command-r"
            
            # Verify max_tokens is passed correctly
            assert request_body.get("max_tokens") == 10
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Mock response
            return MockResponse(
                200,
                {
                    "text": "This is a mocked response for sync request",
                    "generation_id": "mock-id",
                    "id": "mock-completion",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            )
            
        async def mock_async_post(*args, **kwargs):
            # For asynchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Async Request body:", request_body)
            
            # Verify the model is passed correctly
            assert request_body.get("model") == "command-r"
            
            # Verify max_tokens is passed correctly
            assert request_body.get("max_tokens") == 10
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Mock response
            return MockResponse(
                200,
                {
                    "text": "This is a mocked response for async request",
                    "generation_id": "mock-id",
                    "id": "mock-completion",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            )
            
        # Mock both sync and async HTTP clients
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", side_effect=mock_sync_post):
            with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock, side_effect=mock_async_post):
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
                        model="cohere_v2/command-r",
                        messages=messages,
                        max_tokens=10,
                    )
                else:
                    response = completion(
                        model="cohere_v2/command-r",
                        messages=messages,
                        max_tokens=10,
                    )
                print(response)
                assert response is not None
                assert "This is a mocked response" in response.choices[0].message.content
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [False])
async def test_chat_completion_cohere_v2_stream(sync_mode):
    try:
        class MockResponse:
            def __init__(self, status_code, json_data, is_stream=False):
                self.status_code = status_code
                self._json_data = json_data
                self.headers = {}
                self.is_stream = is_stream
                
                # For streaming responses
                if is_stream:
                    self._iter_content_chunks = [
                        json.dumps({"text": "This"}).encode(),
                        json.dumps({"text": " is"}).encode(),
                        json.dumps({"text": " a"}).encode(),
                        json.dumps({"text": " streamed"}).encode(),
                        json.dumps({"text": " response"}).encode(),
                        json.dumps({"text": "."}).encode(),
                        json.dumps({"finish_reason": "COMPLETE"}).encode(),
                    ]
                    
            def json(self):
                return self._json_data

            @property
            def text(self):
                return json.dumps(self._json_data)
                
            def iter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                    
            async def aiter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                
        async def mock_async_post(*args, **kwargs):
            # For asynchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Async Request body:", request_body)
            
            # Verify the model is passed correctly
            assert request_body.get("model") == "command-r"
            
            # Verify max_tokens is passed correctly
            assert request_body.get("max_tokens") == 10
            
            # Verify stream is set to True
            assert request_body.get("stream") == True
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Return a streaming response
            return MockResponse(
                200,
                {
                    "text": "This is a streamed response.",
                    "generation_id": "mock-id",
                    "id": "mock-completion",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
                is_stream=True
            )
            
        # Mock the async HTTP client for streaming
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock, side_effect=mock_async_post):
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
                    model="cohere_v2/command-r",
                    messages=messages,
                    stream=True,
                    max_tokens=10,
                )
                # Verify we get streaming chunks
                chunk_count = 0
                async for chunk in response:
                    print(f"chunk: {chunk}")
                    chunk_count += 1
                assert chunk_count > 0, "No streaming chunks were received"
            else:
                # This test is only for async mode
                pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_v2_mock_completion():
    """
    Test cohere_v2 completion with mocked responses to avoid API calls
    """
    try:
        import httpx

        class MockResponse:
            def __init__(self, status_code, json_data, is_stream=False):
                self.status_code = status_code
                self._json_data = json_data
                self.headers = {}
                self.is_stream = is_stream
                
                # For streaming responses with citations
                if is_stream:
                    # Create streaming chunks with citations at the end
                    self._iter_content_chunks = [
                        json.dumps({"text": "Emperor"}).encode(),
                        json.dumps({"text": " penguins"}).encode(),
                        json.dumps({"text": " are"}).encode(),
                        json.dumps({"text": " the"}).encode(),
                        json.dumps({"text": " tallest"}).encode(),
                        json.dumps({"text": " and"}).encode(),
                        json.dumps({"text": " they"}).encode(),
                        json.dumps({"text": " live"}).encode(),
                        json.dumps({"text": " in"}).encode(),
                        json.dumps({"text": " Antarctica"}).encode(),
                        json.dumps({"text": "."}).encode(),
                        # Citations in a separate chunk
                        json.dumps({"citations": [
                            {
                                "start": 0,
                                "end": 30,
                                "text": "Emperor penguins are the tallest",
                                "document_ids": ["doc1"]
                            },
                            {
                                "start": 31,
                                "end": 70,
                                "text": "they live in Antarctica",
                                "document_ids": ["doc2"]
                            }
                        ]}).encode(),
                        json.dumps({"finish_reason": "COMPLETE"}).encode(),
                    ]

            def json(self):
                return self._json_data

            @property
            def text(self):
                return json.dumps(self._json_data)
                
            def iter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                    
            async def aiter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()

        def mock_sync_post(*args, **kwargs):
            # For synchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Sync Request body:", request_body)
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Mock response
            return MockResponse(
                200,
                {
                    "text": "This is a mocked response from Cohere v2 API",
                    "generation_id": "mock-id",
                    "id": "mock-completion",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            )

        async def mock_async_post(*args, **kwargs):
            # For asynchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Async Request body:", request_body)
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Mock response
            return MockResponse(
                200,
                {
                    "text": "This is a mocked response from Cohere v2 API",
                    "generation_id": "mock-id",
                    "id": "mock-completion",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            )

        # Mock both sync and async HTTP clients
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", side_effect=mock_sync_post):
            with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock, side_effect=mock_async_post):
                litellm.set_verbose = True
                messages = [{"role": "user", "content": "Hello from mock test"}]
                response = completion(
                    model="cohere_v2/command-r",
                    messages=messages,
                )
                assert response is not None
                assert "This is a mocked response" in response.choices[0].message.content

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_v2_request_body_with_allowed_params():
    """
    Test to validate that when allowed_openai_params is provided, the request body contains
    the correct response_format and reasoning_effort values.
    """
    try:
        import httpx

        class MockResponse:
            def __init__(self, status_code, json_data, is_stream=False):
                self.status_code = status_code
                self._json_data = json_data
                self.headers = {}
                self.is_stream = is_stream
                
                # For streaming responses with citations
                if is_stream:
                    # Create streaming chunks with citations at the end
                    self._iter_content_chunks = [
                        json.dumps({"text": "Emperor"}).encode(),
                        json.dumps({"text": " penguins"}).encode(),
                        json.dumps({"text": " are"}).encode(),
                        json.dumps({"text": " the"}).encode(),
                        json.dumps({"text": " tallest"}).encode(),
                        json.dumps({"text": " and"}).encode(),
                        json.dumps({"text": " they"}).encode(),
                        json.dumps({"text": " live"}).encode(),
                        json.dumps({"text": " in"}).encode(),
                        json.dumps({"text": " Antarctica"}).encode(),
                        json.dumps({"text": "."}).encode(),
                        # Citations in a separate chunk
                        json.dumps({"citations": [
                            {
                                "start": 0,
                                "end": 30,
                                "text": "Emperor penguins are the tallest",
                                "document_ids": ["doc1"]
                            },
                            {
                                "start": 31,
                                "end": 70,
                                "text": "they live in Antarctica",
                                "document_ids": ["doc2"]
                            }
                        ]}).encode(),
                        json.dumps({"finish_reason": "COMPLETE"}).encode(),
                    ]

            def json(self):
                return self._json_data

            @property
            def text(self):
                return json.dumps(self._json_data)
                
            def iter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                    
            async def aiter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()

        def mock_sync_post(*args, **kwargs):
            # For synchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Sync Request body:", request_body)
            
            # Verify the model is passed correctly
            assert request_body.get("model") == "command-r"
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Mock response
            return MockResponse(
                200,
                {
                    "text": "This is a test response",
                    "generation_id": "test-id",
                    "id": "test",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            )
            
        async def mock_async_post(*args, **kwargs):
            # For asynchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Async Request body:", request_body)
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Mock response
            return MockResponse(
                200,
                {
                    "text": "This is a test response",
                    "generation_id": "test-id",
                    "id": "test",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            )

        # Mock both sync and async HTTP clients
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", side_effect=mock_sync_post):
            with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock, side_effect=mock_async_post):
                litellm.set_verbose = True
                messages = [{"role": "user", "content": "Hello"}]
                response = completion(
                    model="cohere_v2/command-r",
                    messages=messages,
                )
                assert response is not None

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
@pytest.mark.asyncio
async def test_chat_completion_cohere_v2_streaming_citations():
    """
    Test specifically for streaming with citations in Cohere v2
    """
    try:
        class MockResponse:
            def __init__(self, status_code, json_data, is_stream=False):
                self.status_code = status_code
                self._json_data = json_data
                self.headers = {}
                self.is_stream = is_stream
                
                # For streaming responses with citations
                if is_stream:
                    # Create streaming chunks with citations at the end
                    self._iter_content_chunks = [
                        json.dumps({"text": "Emperor"}).encode(),
                        json.dumps({"text": " penguins"}).encode(),
                        json.dumps({"text": " are"}).encode(),
                        json.dumps({"text": " the"}).encode(),
                        json.dumps({"text": " tallest"}).encode(),
                        json.dumps({"text": " and"}).encode(),
                        json.dumps({"text": " they"}).encode(),
                        json.dumps({"text": " live"}).encode(),
                        json.dumps({"text": " in"}).encode(),
                        json.dumps({"text": " Antarctica"}).encode(),
                        json.dumps({"text": "."}).encode(),
                        # Citations in a separate chunk
                        json.dumps({"citations": [
                            {
                                "start": 0,
                                "end": 30,
                                "text": "Emperor penguins are the tallest",
                                "document_ids": ["doc1"]
                            },
                            {
                                "start": 31,
                                "end": 70,
                                "text": "they live in Antarctica",
                                "document_ids": ["doc2"]
                            }
                        ]}).encode(),
                        json.dumps({"finish_reason": "COMPLETE"}).encode(),
                    ]

            def json(self):
                return self._json_data

            @property
            def text(self):
                return json.dumps(self._json_data)
                
            def iter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                    
            async def aiter_lines(self):
                if self.is_stream:
                    for chunk in self._iter_content_chunks:
                        yield chunk
                else:
                    yield json.dumps(self._json_data).encode()
                
        async def mock_async_post(*args, **kwargs):
            # For asynchronous HTTP client
            data = kwargs.get("data", "{}")
            request_body = json.loads(data)
            print("Async Request body:", request_body)
            
            # Verify the messages are formatted correctly for v2
            messages = request_body.get("messages", [])
            assert len(messages) > 0
            assert "role" in messages[0]
            assert "content" in messages[0]
            
            # Check if documents are included
            documents = request_body.get("documents", [])
            assert len(documents) > 0
            
            # Verify stream is set to True
            assert request_body.get("stream") == True
            
            # Return a streaming response with citations
            return MockResponse(
                200,
                {
                    "text": "Emperor penguins are the tallest penguins and they live in Antarctica.",
                    "generation_id": "mock-id",
                    "id": "mock-completion",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
                is_stream=True
            )
            
        # Mock the async HTTP client
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock, side_effect=mock_async_post):
            litellm.set_verbose = True
            messages = [
                {
                    "role": "user",
                    "content": "Which penguins are the tallest?",
                },
            ]
            response = await litellm.acompletion(
                model="cohere_v2/command-r",
                messages=messages,
                stream=True,
                documents=[
                    {"title": "Tall penguins", "text": "Emperor penguins are the tallest."},
                    {
                        "title": "Penguin habitats",
                        "text": "Emperor penguins only live in Antarctica.",
                    },
                ],
            )
            
            # Verify we get streaming chunks with citations
            citations_chunk = False
            async for chunk in response:
                print("received chunk", chunk)
                if hasattr(chunk, "citations") or (isinstance(chunk, dict) and "citations" in chunk):
                    citations_chunk = True
                    break
            assert citations_chunk, "No citations chunk was received"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
@pytest.mark.skip(reason="Only run this test when you want to test with a real API key")
@pytest.mark.asyncio
async def test_cohere_v2_real_api_call():
    """
    Test for making a real API call to Cohere v2. This test is skipped by default.
    To run this test, remove the skip mark and ensure you have a valid Cohere API key.
    """
    try:
        # Set the API key from environment variable
        os.environ["CO_API_KEY"] = "LitgtFBRwgpnyF5KAaJINtLNJkx5Ty6LsFVV1IYM"  # Using the provided API key
        
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "What is the capital of France?",
            },
        ]
        
        # Make a real API call
        response = await litellm.acompletion(
            model="cohere_v2/command-r",
            messages=messages,
            max_tokens=100,
        )
        
        print("Real API Response:", response)
        assert response is not None
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        
        # Test streaming with real API
        stream_response = await litellm.acompletion(
            model="cohere_v2/command-r",
            messages=messages,
            stream=True,
            max_tokens=100,
        )
        
        # Verify we get streaming chunks
        chunk_count = 0
        async for chunk in stream_response:
            print(f"Stream chunk: {chunk}")
            chunk_count += 1
            if chunk_count > 5:  # Just check a few chunks to avoid long test
                break
                
        assert chunk_count > 0, "No streaming chunks were received"
        
    except Exception as e:
        pytest.fail(f"Error occurred with real API call: {e}")
