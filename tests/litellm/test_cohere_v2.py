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
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
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
