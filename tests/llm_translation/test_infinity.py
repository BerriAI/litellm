import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


import litellm

import json
import os
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from test_rerank import assert_response_shape
import litellm

from base_embedding_unit_tests import BaseLLMEmbeddingTest
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.types.utils import EmbeddingResponse, Usage


@pytest.mark.asyncio()
async def test_infinity_rerank():
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "usage": {"prompt_tokens": 100, "total_tokens": 150},
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    expected_payload = {
        "model": "rerank-model",
        "query": "hello",
        "top_n": 3,
        "documents": ["hello", "world"],
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="infinity/rerank-model",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
            api_base="https://api.infinity.ai",
        )

        print("async re rank response: ", response)

        # Assert
        mock_post.assert_called_once()
        print("call args", mock_post.call_args)
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("Arguments passed to API=", args_to_api)
        print("url = ", _url)
        assert _url == "https://api.infinity.ai/rerank"

        request_data = json.loads(args_to_api)
        assert request_data["query"] == expected_payload["query"]
        assert request_data["documents"] == expected_payload["documents"]
        assert request_data["top_n"] == expected_payload["top_n"]
        assert request_data["model"] == expected_payload["model"]

        assert response.id is not None
        assert response.results is not None
        assert response.meta["tokens"]["input_tokens"] == 100
        assert (
            response.meta["tokens"]["output_tokens"] == 50
        )  # total_tokens - prompt_tokens

        assert_response_shape(response, custom_llm_provider="infinity")


@pytest.mark.asyncio()
async def test_infinity_rerank_with_return_documents():
    mock_response = AsyncMock()

    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95, "document": "hello"}],
            "usage": {"prompt_tokens": 100, "total_tokens": 150},
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="infinity/rerank-model",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
            return_documents=True,
            api_base="https://api.infinity.ai",
        )
        assert response.results[0]["document"] == {"text": "hello"}
        assert_response_shape(response, custom_llm_provider="infinity")


@pytest.mark.asyncio()
async def test_infinity_rerank_with_env(monkeypatch):
    # Set up mock response
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "usage": {"prompt_tokens": 100, "total_tokens": 150},
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    # Set environment variable
    monkeypatch.setenv("INFINITY_API_BASE", "https://env.infinity.ai")

    expected_payload = {
        "model": "rerank-model",
        "query": "hello",
        "top_n": 3,
        "documents": ["hello", "world"],
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="infinity/rerank-model",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
        )

        print("async re rank response: ", response)

        # Assert
        mock_post.assert_called_once()
        print("call args", mock_post.call_args)
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("Arguments passed to API=", args_to_api)
        print("url = ", _url)
        assert _url == "https://env.infinity.ai/rerank"

        request_data = json.loads(args_to_api)
        assert request_data["query"] == expected_payload["query"]
        assert request_data["documents"] == expected_payload["documents"]
        assert request_data["top_n"] == expected_payload["top_n"]
        assert request_data["model"] == expected_payload["model"]

        assert response.id is not None
        assert response.results is not None
        assert response.meta["tokens"]["input_tokens"] == 100
        assert (
            response.meta["tokens"]["output_tokens"] == 50
        )  # total_tokens - prompt_tokens

        assert_response_shape(response, custom_llm_provider="infinity")

#### Embedding Tests
@pytest.mark.asyncio()
async def test_infinity_embedding():
    mock_response = AsyncMock()

    def return_val():
        return {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"prompt_tokens": 100, "total_tokens": 150},
            "model": "custom-model/embedding-v1",
            "object": "list"
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    expected_payload = {
        "model": "custom-model/embedding-v1",
        "input": ["hello world"],
        "encoding_format": "float",
        "output_dimension": 512
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.aembedding(
            model="infinity/custom-model/embedding-v1",
            input=["hello world"],
            dimensions=512,
            encoding_format="float",
            api_base="https://api.infinity.ai/embeddings",
            
        )

        # Assert
        mock_post.assert_called_once()
        print("call args", mock_post.call_args)
        request_data = mock_post.call_args.kwargs["json"]
        _url = mock_post.call_args.kwargs["url"]
        assert _url == "https://api.infinity.ai/embeddings"

        assert request_data["input"] == expected_payload["input"]
        assert request_data["model"] == expected_payload["model"]
        assert request_data["output_dimension"] == expected_payload["output_dimension"]
        assert request_data["encoding_format"] == expected_payload["encoding_format"]

        assert response.data is not None
        assert response.usage.prompt_tokens == 100
        assert response.usage.total_tokens == 150
        assert response.model == "custom-model/embedding-v1"
        assert response.object == "list"


@pytest.mark.asyncio()
async def test_infinity_embedding_with_env(monkeypatch):
    # Set up mock response
    mock_response = AsyncMock()

    def return_val():
        return {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"prompt_tokens": 100, "total_tokens": 150},
            "model": "custom-model/embedding-v1",
            "object": "list"
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    expected_payload = {
        "model": "custom-model/embedding-v1",
        "input": ["hello world"],
        "encoding_format": "float",
        "output_dimension": 512
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.aembedding(
            model="infinity/custom-model/embedding-v1",
            input=["hello world"],
            dimensions=512,
            encoding_format="float",
            api_base="https://api.infinity.ai/embeddings",
        )

        # Assert
        mock_post.assert_called_once()
        print("call args", mock_post.call_args)
        request_data = mock_post.call_args.kwargs["json"]
        _url = mock_post.call_args.kwargs["url"]
        assert _url == "https://api.infinity.ai/embeddings"

        assert request_data["input"] == expected_payload["input"]
        assert request_data["model"] == expected_payload["model"]
        assert request_data["output_dimension"] == expected_payload["output_dimension"]
        assert request_data["encoding_format"] == expected_payload["encoding_format"]

        assert response.data is not None
        assert response.usage.prompt_tokens == 100
        assert response.usage.total_tokens == 150
        assert response.model == "custom-model/embedding-v1"
        assert response.object == "list"


@pytest.mark.asyncio()
async def test_infinity_embedding_extra_params():
    mock_response = AsyncMock()

    def return_val():
        return {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"prompt_tokens": 100, "total_tokens": 150},
            "model": "custom-model/embedding-v1",
            "object": "list"
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.aembedding(
            model="infinity/custom-model/embedding-v1",
            input=["test input"],
            dimensions=512,
            encoding_format="float",
            modality="text",
            api_base="https://api.infinity.ai/embeddings",
        )

        mock_post.assert_called_once()
        request_data = mock_post.call_args.kwargs["json"]

        # Assert the request parameters
        assert request_data["input"] == ["test input"]
        assert request_data["model"] == "custom-model/embedding-v1"
        assert request_data["output_dimension"] == 512
        assert request_data["encoding_format"] == "float"
        assert request_data["modality"] == "text"


@pytest.mark.asyncio()
async def test_infinity_embedding_prompt_token_mapping():
    mock_response = AsyncMock()

    def return_val():
        return {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"total_tokens": 1, "prompt_tokens": 1},
            "model": "custom-model/embedding-v1",
            "object": "list"
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.aembedding(
            model="infinity/custom-model/embedding-v1",
            input=["a"],
            dimensions=512,
            encoding_format="float",
            api_base="https://api.infinity.ai/embeddings",
        )

        mock_post.assert_called_once()
        # Assert the response
        assert response.usage.prompt_tokens == 1
        assert response.usage.total_tokens == 1
